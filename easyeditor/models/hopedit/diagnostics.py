from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import torch
import torch.nn.functional as F


ROUTE_EVAL_EVENTS = {"rewrite", "rephrase", "post_edit"}


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _normalize_prompt(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def _scalar(metric_value: Any) -> float | None:
    if metric_value is None:
        return None
    if isinstance(metric_value, list):
        if not metric_value:
            return None
        return float(sum(metric_value) / len(metric_value))
    return float(metric_value)


def infer_case_to_edit_id(route_logs: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for entry in route_logs:
        if entry.get("route_event") != "post_edit":
            continue
        case_id = entry.get("case_id")
        target_edit_id = entry.get("target_edit_id")
        if case_id is not None and target_edit_id is not None:
            mapping[int(case_id)] = str(target_edit_id)
    for metric in metrics:
        case_id = metric.get("case_id")
        if case_id is None or case_id in mapping:
            continue
        mapping[int(case_id)] = f"hopedit_{int(case_id):05d}"
    return mapping


def build_prompt_annotation_index(metrics: list[dict[str, Any]], case_to_edit_id: dict[int, str]) -> dict[str, deque[dict[str, Any]]]:
    annotations: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for metric in metrics:
        request = metric.get("requested_rewrite", {})
        case_id = metric.get("case_id")
        if case_id is None:
            continue
        case_id = int(case_id)
        target_new = request.get("target_new", "")
        target_edit_id = case_to_edit_id.get(case_id)
        prompt = request.get("prompt")
        if prompt:
            annotations[_normalize_prompt(f"{prompt} {target_new}".strip())].append(
                {
                    "case_id": case_id,
                    "event_type": "rewrite",
                    "expected_edit_id": target_edit_id,
                    "prompt_role": "rewrite",
                }
            )
        rephrase_prompt = request.get("rephrase_prompt")
        if rephrase_prompt:
            annotations[_normalize_prompt(f"{rephrase_prompt} {target_new}".strip())].append(
                {
                    "case_id": case_id,
                    "event_type": "rephrase",
                    "expected_edit_id": target_edit_id,
                    "prompt_role": "rephrase",
                }
            )
        for locality_key, locality_entry in request.get("locality", {}).items():
            locality_prompt = locality_entry.get("prompt")
            locality_gt = locality_entry.get("ground_truth", "")
            if locality_prompt:
                annotations[_normalize_prompt(f"{locality_prompt} {locality_gt}".strip())].append(
                    {
                        "case_id": case_id,
                        "event_type": "locality",
                        "locality_key": locality_key,
                        "expected_edit_id": None,
                        "prompt_role": f"locality:{locality_key}",
                    }
                )
    return annotations


def annotate_route_logs(route_logs: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_to_edit_id = infer_case_to_edit_id(route_logs, metrics)
    prompt_annotations = build_prompt_annotation_index(metrics, case_to_edit_id)

    annotated_logs: list[dict[str, Any]] = []
    for route_entry in route_logs:
        annotated = dict(route_entry)
        annotated["top1_prob"] = float(route_entry.get("top_scores", [0.0])[0]) if route_entry.get("top_scores") else 0.0
        annotated["num_candidates"] = len(route_entry.get("top_edit_ids", []))

        if route_entry.get("route_event") == "post_edit":
            case_id = route_entry.get("case_id")
            expected_edit_id = route_entry.get("target_edit_id") or case_to_edit_id.get(case_id)
            annotated.update(
                {
                    "event_type": "post_edit",
                    "expected_edit_id": expected_edit_id,
                    "correct_route": route_entry.get("chosen_edit_id") == expected_edit_id,
                    "false_activation": False,
                }
            )
        else:
            prompt_key = _normalize_prompt(route_entry.get("prompt"))
            match = prompt_annotations[prompt_key].popleft() if prompt_key in prompt_annotations and prompt_annotations[prompt_key] else None
            if match is None:
                annotated.update(
                    {
                        "event_type": "unknown",
                        "expected_edit_id": None,
                        "correct_route": None,
                        "false_activation": route_entry.get("chosen_edit_id") is not None,
                    }
                )
            else:
                expected_edit_id = match.get("expected_edit_id")
                event_type = match.get("event_type", "unknown")
                chosen_edit_id = route_entry.get("chosen_edit_id")
                correct_route = None
                false_activation = False
                if event_type in ROUTE_EVAL_EVENTS:
                    correct_route = chosen_edit_id == expected_edit_id
                elif event_type == "locality":
                    false_activation = chosen_edit_id is not None
                annotated.update(match)
                annotated.update(
                    {
                        "expected_edit_id": expected_edit_id,
                        "correct_route": correct_route,
                        "false_activation": false_activation,
                    }
                )
        annotated_logs.append(annotated)
    return annotated_logs


def summarize_route_diagnostics(annotated_logs: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in annotated_logs:
        grouped[entry.get("event_type", "unknown")].append(entry)

    route_summary: dict[str, Any] = {}
    for event_type, entries in grouped.items():
        top1_probs = [float(entry.get("top1_prob", 0.0)) for entry in entries]
        route_margins = [float(entry.get("route_margin", 0.0)) for entry in entries]
        coverage = [1.0 if entry.get("chosen_edit_id") is not None else 0.0 for entry in entries]
        event_summary = {
            "count": len(entries),
            "coverage": _safe_mean(coverage),
            "top1_prob_mean": _safe_mean(top1_probs),
            "route_margin_mean": _safe_mean(route_margins),
        }
        if event_type in ROUTE_EVAL_EVENTS:
            correct = [1.0 if entry.get("correct_route") else 0.0 for entry in entries if entry.get("correct_route") is not None]
            event_summary["route_accuracy"] = _safe_mean(correct)
        if event_type == "locality":
            false_activation = [1.0 if entry.get("false_activation") else 0.0 for entry in entries]
            event_summary["false_activation_rate"] = _safe_mean(false_activation)
            event_summary["no_edit_rate"] = _safe_mean([1.0 if entry.get("chosen_edit_id") is None else 0.0 for entry in entries])
        route_summary[event_type] = event_summary

    per_case = []
    for metric in metrics:
        request = metric.get("requested_rewrite", {})
        case_id = metric.get("case_id")
        pre = metric.get("pre", {})
        post = metric.get("post", {})
        pre_rewrite = _scalar(pre.get("rewrite_acc"))
        post_rewrite = _scalar(post.get("rewrite_acc"))
        per_case.append(
            {
                "case_id": case_id,
                "subject": request.get("subject"),
                "prompt": request.get("prompt"),
                "pre_rewrite_acc": pre_rewrite,
                "post_rewrite_acc": post_rewrite,
                "rewrite_delta": None if pre_rewrite is None or post_rewrite is None else post_rewrite - pre_rewrite,
                "pre_rephrase_acc": _scalar(pre.get("rephrase_acc")),
                "post_rephrase_acc": _scalar(post.get("rephrase_acc")),
                "locality": {
                    key: _scalar(value)
                    for key, value in post.get("locality", {}).items()
                    if key.endswith("_acc")
                },
            }
        )

    retention_values = [entry["post_rewrite_acc"] for entry in per_case if entry["post_rewrite_acc"] is not None]
    cutoff = max(1, len(retention_values) // 2) if retention_values else 0
    retention_summary = {
        "post_rewrite_mean": _safe_mean(retention_values),
        "first_half_post_rewrite_mean": _safe_mean(retention_values[:cutoff]),
        "second_half_post_rewrite_mean": _safe_mean(retention_values[cutoff:]),
    }
    if retention_summary["first_half_post_rewrite_mean"] is not None and retention_summary["second_half_post_rewrite_mean"] is not None:
        retention_summary["early_late_gap"] = retention_summary["first_half_post_rewrite_mean"] - retention_summary["second_half_post_rewrite_mean"]
    else:
        retention_summary["early_late_gap"] = None

    return {
        "summary": {
            "num_logged_events": len(annotated_logs),
            "num_cases": len(metrics),
        },
        "routing": route_summary,
        "retention": retention_summary,
        "per_case": per_case,
    }


def _entry_views(entry: dict[str, Any]) -> list[dict[str, Any]]:
    view_records = entry.get("view_keys") or []
    if view_records:
        return view_records
    return [
        {
            "view_name": "anchor",
            "text": entry.get("prompt"),
            "raw_semantic_key": entry.get("raw_semantic_key"),
            "raw_activation_key": entry.get("raw_activation_key"),
            "semantic_key": entry.get("semantic_key"),
            "activation_key": entry.get("activation_key"),
        }
    ]


def export_memory_snapshot(memory_entries: list[dict[str, Any]], include_keys: bool = False) -> list[dict[str, Any]]:
    snapshot = []
    for entry in memory_entries:
        view_records = _entry_views(entry)
        row = {
            "edit_id": entry.get("edit_id"),
            "prompt": entry.get("prompt"),
            "subject": entry.get("subject"),
            "rephrase_prompt": entry.get("rephrase_prompt"),
            "target_new": entry.get("target_new"),
            "raw_semantic_norm": float(entry["raw_semantic_key"].norm().item()) if isinstance(entry.get("raw_semantic_key"), torch.Tensor) else None,
            "raw_activation_norm": float(entry["raw_activation_key"].norm().item()) if isinstance(entry.get("raw_activation_key"), torch.Tensor) else None,
            "semantic_norm": float(entry["semantic_key"].norm().item()) if isinstance(entry.get("semantic_key"), torch.Tensor) else None,
            "activation_norm": float(entry["activation_key"].norm().item()) if isinstance(entry.get("activation_key"), torch.Tensor) else None,
            "num_views": len(view_records),
            "view_names": [view.get("view_name") for view in view_records],
            "conflict_neighbors": entry.get("conflict_neighbors", []),
        }
        if include_keys:
            for key_name in ["raw_semantic_key", "raw_activation_key", "semantic_key", "activation_key"]:
                value = entry.get(key_name)
                row[key_name] = value.tolist() if isinstance(value, torch.Tensor) else value
            row["view_keys"] = []
            for view in view_records:
                view_row = {"view_name": view.get("view_name"), "text": view.get("text")}
                for key_name in ["raw_semantic_key", "raw_activation_key", "semantic_key", "activation_key"]:
                    value = view.get(key_name)
                    view_row[key_name] = value.tolist() if isinstance(value, torch.Tensor) else value
                row["view_keys"].append(view_row)
        snapshot.append(row)
    return snapshot


def _matrix_to_list(matrix: torch.Tensor | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return [[float(value) for value in row] for row in matrix.tolist()]


def _offdiag_values(matrix: torch.Tensor | None) -> list[float]:
    if matrix is None:
        return []
    values = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            values.append(float(matrix[i, j].item()))
    return values


def _best_pairwise_view_conflict(entry_a: dict[str, Any], entry_b: dict[str, Any], semantic_weight: float, activation_weight: float) -> dict[str, Any]:
    best = None
    for view_a in _entry_views(entry_a):
        sem_a = view_a.get("semantic_key")
        act_a = view_a.get("activation_key")
        raw_sem_a = view_a.get("raw_semantic_key")
        raw_act_a = view_a.get("raw_activation_key")
        if not isinstance(sem_a, torch.Tensor) or not isinstance(act_a, torch.Tensor):
            continue
        sem_a = F.normalize(sem_a.float(), dim=-1)
        act_a = F.normalize(act_a.float(), dim=-1)
        raw_sem_a = F.normalize(raw_sem_a.float(), dim=-1) if isinstance(raw_sem_a, torch.Tensor) else None
        raw_act_a = F.normalize(raw_act_a.float(), dim=-1) if isinstance(raw_act_a, torch.Tensor) else None
        for view_b in _entry_views(entry_b):
            sem_b = view_b.get("semantic_key")
            act_b = view_b.get("activation_key")
            raw_sem_b = view_b.get("raw_semantic_key")
            raw_act_b = view_b.get("raw_activation_key")
            if not isinstance(sem_b, torch.Tensor) or not isinstance(act_b, torch.Tensor):
                continue
            sem_b = F.normalize(sem_b.float(), dim=-1)
            act_b = F.normalize(act_b.float(), dim=-1)
            raw_sem_b = F.normalize(raw_sem_b.float(), dim=-1) if isinstance(raw_sem_b, torch.Tensor) else None
            raw_act_b = F.normalize(raw_act_b.float(), dim=-1) if isinstance(raw_act_b, torch.Tensor) else None

            semantic_score = float(torch.dot(sem_a, sem_b).item())
            activation_score = float(torch.dot(act_a, act_b).item())
            combined_score = semantic_weight * semantic_score + activation_weight * activation_score
            candidate = {
                "combined_conflict": combined_score,
                "semantic_cosine": semantic_score,
                "activation_cosine": activation_score,
                "raw_semantic_cosine": float(torch.dot(raw_sem_a, raw_sem_b).item()) if raw_sem_a is not None and raw_sem_b is not None else None,
                "raw_activation_cosine": float(torch.dot(raw_act_a, raw_act_b).item()) if raw_act_a is not None and raw_act_b is not None else None,
                "view_name_a": view_a.get("view_name"),
                "view_name_b": view_b.get("view_name"),
            }
            if best is None or candidate["combined_conflict"] > best["combined_conflict"]:
                best = candidate
    if best is None:
        return {
            "combined_conflict": None,
            "semantic_cosine": None,
            "activation_cosine": None,
            "raw_semantic_cosine": None,
            "raw_activation_cosine": None,
            "view_name_a": None,
            "view_name_b": None,
        }
    return best


def summarize_conflicts(memory_entries: list[dict[str, Any]], semantic_weight: float, activation_weight: float) -> dict[str, Any]:
    if not memory_entries:
        return {
            "num_edits": 0,
            "edit_ids": [],
            "raw_semantic_cosine": None,
            "raw_activation_cosine": None,
            "semantic_cosine": None,
            "activation_cosine": None,
            "combined_conflict": None,
            "mean_semantic_offdiag": None,
            "mean_activation_offdiag": None,
            "mean_raw_activation_offdiag": None,
            "mean_combined_offdiag": None,
            "mean_max_offdiag_conflict": None,
            "max_pair_conflict": None,
            "hardest_pairs": [],
        }

    n = len(memory_entries)
    raw_semantic_cosine = torch.eye(n, dtype=torch.float32)
    raw_activation_cosine = torch.eye(n, dtype=torch.float32)
    semantic_cosine = torch.eye(n, dtype=torch.float32)
    activation_cosine = torch.eye(n, dtype=torch.float32)
    combined_conflict = torch.eye(n, dtype=torch.float32)

    hardest_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            best = _best_pairwise_view_conflict(memory_entries[i], memory_entries[j], semantic_weight, activation_weight)
            semantic_cosine[i, j] = semantic_cosine[j, i] = float(best["semantic_cosine"])
            activation_cosine[i, j] = activation_cosine[j, i] = float(best["activation_cosine"])
            combined_conflict[i, j] = combined_conflict[j, i] = float(best["combined_conflict"])
            if best["raw_semantic_cosine"] is not None:
                raw_semantic_cosine[i, j] = raw_semantic_cosine[j, i] = float(best["raw_semantic_cosine"])
            if best["raw_activation_cosine"] is not None:
                raw_activation_cosine[i, j] = raw_activation_cosine[j, i] = float(best["raw_activation_cosine"])
            hardest_pairs.append(
                {
                    "edit_id_a": memory_entries[i]["edit_id"],
                    "edit_id_b": memory_entries[j]["edit_id"],
                    "combined_conflict": float(best["combined_conflict"]),
                    "semantic_cosine": float(best["semantic_cosine"]),
                    "activation_cosine": float(best["activation_cosine"]),
                    "raw_activation_cosine": float(best["raw_activation_cosine"]) if best["raw_activation_cosine"] is not None else None,
                    "raw_semantic_cosine": float(best["raw_semantic_cosine"]) if best["raw_semantic_cosine"] is not None else None,
                    "view_name_a": best.get("view_name_a"),
                    "view_name_b": best.get("view_name_b"),
                }
            )
    hardest_pairs.sort(key=lambda item: item["combined_conflict"], reverse=True)

    offdiag_values = _offdiag_values(combined_conflict)
    max_offdiag_per_edit = []
    for i in range(n):
        row_values = [float(combined_conflict[i, j].item()) for j in range(n) if j != i]
        if row_values:
            max_offdiag_per_edit.append(max(row_values))

    return {
        "num_edits": len(memory_entries),
        "edit_ids": [entry["edit_id"] for entry in memory_entries],
        "view_counts": [len(_entry_views(entry)) for entry in memory_entries],
        "raw_semantic_cosine": _matrix_to_list(raw_semantic_cosine),
        "raw_activation_cosine": _matrix_to_list(raw_activation_cosine),
        "semantic_cosine": _matrix_to_list(semantic_cosine),
        "activation_cosine": _matrix_to_list(activation_cosine),
        "combined_conflict": _matrix_to_list(combined_conflict),
        "mean_semantic_offdiag": _safe_mean(_offdiag_values(semantic_cosine)),
        "mean_activation_offdiag": _safe_mean(_offdiag_values(activation_cosine)),
        "mean_raw_activation_offdiag": _safe_mean(_offdiag_values(raw_activation_cosine)),
        "mean_combined_offdiag": _safe_mean(offdiag_values),
        "mean_max_offdiag_conflict": _safe_mean(max_offdiag_per_edit),
        "max_pair_conflict": max(offdiag_values) if offdiag_values else None,
        "hardest_pairs": hardest_pairs[: min(10, len(hardest_pairs))],
    }
