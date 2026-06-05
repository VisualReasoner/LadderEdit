import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor
from easyeditor.models.hopedit.hopedit_main import HopEditController
from examples.analyze_hopedit_factor_scores import (
    analyze_rows,
    backfill_relation_ids,
    load_relation_ids_from_memory,
    read_jsonl,
    split_eval_rows,
)
from examples.edit_experiment_utils import resolve_hparams_class


def load_indices(index_file: str | None) -> list[int] | None:
    if not index_file:
        return None
    payload = json.loads(Path(index_file).read_text())
    if isinstance(payload, dict):
        payload = payload.get("selected_indices") or payload.get("indices")
    if payload is None:
        raise ValueError(f"No indices found in {index_file}")
    return [int(idx) for idx in payload]


def trace_number(trace_id: str) -> int:
    return int(str(trace_id).rsplit("_", 1)[-1])


def orthogonalize_relation(relation: torch.Tensor | None, subject: torch.Tensor | None) -> torch.Tensor | None:
    if not isinstance(relation, torch.Tensor):
        return None
    relation = relation.detach().float().cpu()
    if not isinstance(subject, torch.Tensor) or int(subject.numel()) != int(relation.numel()):
        return F.normalize(relation, p=2, dim=-1)
    subject = subject.detach().float().cpu()
    denominator = torch.dot(subject, subject)
    if float(denominator.item()) <= 1.0e-12:
        return F.normalize(relation, p=2, dim=-1)
    residual = relation - (torch.dot(relation, subject) / denominator) * subject
    if float(residual.norm().item()) <= 1.0e-12:
        return F.normalize(relation, p=2, dim=-1)
    return F.normalize(residual, p=2, dim=-1)


def build_relation_whitener(
    relation_by_trace: dict[str, torch.Tensor],
    trace_ids: list[str],
    *,
    eps: float,
) -> dict[str, Any]:
    relations = [relation_by_trace[trace_id].detach().double().cpu() for trace_id in trace_ids if trace_id in relation_by_trace]
    if len(relations) < 2:
        raise ValueError("Need at least two relation vectors to build a whitener.")
    matrix = torch.stack(relations, dim=0)
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    keep = singular_values > float(eps)
    if not bool(keep.any().item()):
        raise ValueError("All singular values were below the whitening eps.")
    basis = vh[keep].T.contiguous()
    scale = (float(max(1, matrix.shape[0] - 1)) ** 0.5) / (singular_values[keep] + float(eps))
    return {
        "mean": mean,
        "basis": basis,
        "scale": scale,
        "rank": int(keep.sum().item()),
        "num_vectors": int(matrix.shape[0]),
        "eps": float(eps),
        "singular_min_kept": float(singular_values[keep].min().item()),
        "singular_max": float(singular_values.max().item()),
    }


def build_vector_whitener(
    vectors: list[torch.Tensor],
    *,
    eps: float,
) -> dict[str, Any]:
    if len(vectors) < 2:
        raise ValueError("Need at least two vectors to build a whitener.")
    matrix = torch.stack([vector.detach().double().cpu() for vector in vectors], dim=0)
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    keep = singular_values > float(eps)
    if not bool(keep.any().item()):
        raise ValueError("All singular values were below the whitening eps.")
    basis = vh[keep].T.contiguous()
    scale = (float(max(1, matrix.shape[0] - 1)) ** 0.5) / (singular_values[keep] + float(eps))
    return {
        "mean": mean,
        "basis": basis,
        "scale": scale,
        "rank": int(keep.sum().item()),
        "num_vectors": int(matrix.shape[0]),
        "eps": float(eps),
        "singular_min_kept": float(singular_values[keep].min().item()),
        "singular_max": float(singular_values.max().item()),
    }


def apply_relation_whitener(relation: torch.Tensor | None, whitener: dict[str, Any]) -> torch.Tensor | None:
    if not isinstance(relation, torch.Tensor):
        return None
    relation = relation.detach().double().cpu()
    whitened = ((relation - whitener["mean"]) @ whitener["basis"]) * whitener["scale"]
    norm = float(whitened.norm().item())
    if norm <= 1.0e-12:
        return whitened.float()
    return (whitened / norm).float()


def build_cluster_whiteners(
    trace_relation_transformed: dict[str, torch.Tensor],
    relation_by_trace: dict[str, object],
    *,
    eps: float,
    min_cluster_size: int,
) -> dict[str, dict[str, Any]]:
    by_relation: dict[str, list[torch.Tensor]] = {}
    for trace_id, relation in trace_relation_transformed.items():
        relation_id = relation_by_trace.get(str(trace_id))
        if relation_id is None or not isinstance(relation, torch.Tensor):
            continue
        by_relation.setdefault(str(relation_id), []).append(relation)
    whiteners: dict[str, dict[str, Any]] = {}
    for relation_id, vectors in by_relation.items():
        if len(vectors) < int(min_cluster_size):
            continue
        try:
            whiteners[relation_id] = build_vector_whitener(vectors, eps=eps)
        except ValueError:
            continue
    return whiteners


def build_cluster_centers(
    trace_relation_transformed: dict[str, torch.Tensor],
    relation_by_trace: dict[str, object],
    *,
    min_cluster_size: int,
) -> dict[str, dict[str, Any]]:
    by_relation: dict[str, list[torch.Tensor]] = {}
    for trace_id, relation in trace_relation_transformed.items():
        relation_id = relation_by_trace.get(str(trace_id))
        if relation_id is None or not isinstance(relation, torch.Tensor):
            continue
        by_relation.setdefault(str(relation_id), []).append(relation.detach().float().cpu())
    centers: dict[str, dict[str, Any]] = {}
    for relation_id, vectors in by_relation.items():
        if len(vectors) < int(min_cluster_size):
            continue
        matrix = torch.stack(vectors, dim=0)
        center = matrix.mean(dim=0)
        centers[relation_id] = {
            "center": center,
            "num_vectors": int(matrix.shape[0]),
            "center_norm": float(center.norm().item()),
        }
    return centers


def apply_cluster_centering(relation: torch.Tensor | None, center_payload: dict[str, Any]) -> torch.Tensor | None:
    if not isinstance(relation, torch.Tensor):
        return None
    centered = relation.detach().float().cpu() - center_payload["center"]
    norm = float(centered.norm().item())
    if norm <= 1.0e-12:
        return centered
    return centered / norm


def extract_factor_rows(
    controller: HopEditController,
    prompts: list[str],
    subjects: list[str | None],
    targets: list[str | None],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, len(prompts), batch_size):
        stop = start + batch_size
        batch = controller._extract_batched_factored_address_keys(
            prompts[start:stop],
            subjects[start:stop],
            targets[start:stop],
        )
        rows.extend(batch)
    return rows


def replace_relation_scores(
    log_rows: list[dict[str, Any]],
    trace_relation_transformed: dict[str, torch.Tensor],
    query_rows: list[dict[str, Any]],
    *,
    transform_name: str,
    whitener: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    transformed = []
    transformed_count = 0
    for row, query in zip(log_rows, query_rows):
        copied = dict(row)
        trace_ids = row.get("factor_score_trace_ids")
        if whitener is None:
            query_relation_transformed = orthogonalize_relation(query.get("relation_factor"), query.get("subject_factor"))
        else:
            query_relation_transformed = apply_relation_whitener(query.get("relation_factor"), whitener)
        if isinstance(trace_ids, list) and isinstance(query_relation_transformed, torch.Tensor):
            relation_scores = []
            for trace_id in trace_ids:
                trace_transformed = trace_relation_transformed.get(str(trace_id))
                relation_scores.append(
                    None
                    if not isinstance(trace_transformed, torch.Tensor)
                    else float(torch.dot(query_relation_transformed, trace_transformed).item())
                )
            copied["factor_relation_scores"] = relation_scores
            copied["factor_relation_score_transform"] = transform_name
            transformed_count += 1
        transformed.append(copied)
    return transformed, transformed_count


def replace_relation_scores_with_cluster_whiteners(
    log_rows: list[dict[str, Any]],
    trace_relation_global: dict[str, torch.Tensor],
    query_rows: list[dict[str, Any]],
    *,
    global_whitener: dict[str, Any],
    relation_by_trace: dict[str, object],
    cluster_whiteners: dict[str, dict[str, Any]],
    transform_name: str,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    transformed = []
    transformed_count = 0
    stats = {
        "candidate_scores": 0,
        "cluster_whitened_candidate_scores": 0,
        "fallback_global_candidate_scores": 0,
        "missing_trace_vectors": 0,
    }
    query_cache: dict[tuple[int, str], torch.Tensor | None] = {}
    trace_cache: dict[tuple[str, str], torch.Tensor | None] = {}
    for row_idx, (row, query) in enumerate(zip(log_rows, query_rows)):
        copied = dict(row)
        trace_ids = row.get("factor_score_trace_ids")
        relation_scores = []
        if isinstance(trace_ids, list):
            query_relation_global = apply_relation_whitener(query.get("relation_factor"), global_whitener)
            for trace_id_value in trace_ids:
                trace_id = str(trace_id_value)
                trace_global = trace_relation_global.get(trace_id)
                if not isinstance(trace_global, torch.Tensor):
                    relation_scores.append(None)
                    stats["missing_trace_vectors"] += 1
                    continue
                relation_id_value = relation_by_trace.get(trace_id)
                relation_id = None if relation_id_value is None else str(relation_id_value)
                whitener = None if relation_id is None else cluster_whiteners.get(relation_id)
                stats["candidate_scores"] += 1
                if whitener is None:
                    query_transformed = query_relation_global
                    trace_transformed = trace_global
                    stats["fallback_global_candidate_scores"] += 1
                else:
                    query_key = (row_idx, relation_id)
                    trace_key = (trace_id, relation_id)
                    if query_key not in query_cache:
                        query_cache[query_key] = apply_relation_whitener(query_relation_global, whitener)
                    if trace_key not in trace_cache:
                        trace_cache[trace_key] = apply_relation_whitener(trace_global, whitener)
                    query_transformed = query_cache[query_key]
                    trace_transformed = trace_cache[trace_key]
                    stats["cluster_whitened_candidate_scores"] += 1
                relation_scores.append(
                    None
                    if not isinstance(query_transformed, torch.Tensor) or not isinstance(trace_transformed, torch.Tensor)
                    else float(torch.dot(query_transformed, trace_transformed).item())
                )
            copied["factor_relation_scores"] = relation_scores
            copied["factor_relation_score_transform"] = transform_name
            transformed_count += 1
        transformed.append(copied)
    return transformed, transformed_count, stats


def replace_relation_scores_with_cluster_centers(
    log_rows: list[dict[str, Any]],
    trace_relation_global: dict[str, torch.Tensor],
    query_rows: list[dict[str, Any]],
    *,
    global_whitener: dict[str, Any],
    relation_by_trace: dict[str, object],
    cluster_centers: dict[str, dict[str, Any]],
    transform_name: str,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    transformed = []
    transformed_count = 0
    stats = {
        "candidate_scores": 0,
        "cluster_centered_candidate_scores": 0,
        "fallback_global_candidate_scores": 0,
        "missing_trace_vectors": 0,
    }
    query_cache: dict[tuple[int, str], torch.Tensor | None] = {}
    trace_cache: dict[tuple[str, str], torch.Tensor | None] = {}
    for row_idx, (row, query) in enumerate(zip(log_rows, query_rows)):
        copied = dict(row)
        trace_ids = row.get("factor_score_trace_ids")
        relation_scores = []
        if isinstance(trace_ids, list):
            query_relation_global = apply_relation_whitener(query.get("relation_factor"), global_whitener)
            for trace_id_value in trace_ids:
                trace_id = str(trace_id_value)
                trace_global = trace_relation_global.get(trace_id)
                if not isinstance(trace_global, torch.Tensor):
                    relation_scores.append(None)
                    stats["missing_trace_vectors"] += 1
                    continue
                relation_id_value = relation_by_trace.get(trace_id)
                relation_id = None if relation_id_value is None else str(relation_id_value)
                center_payload = None if relation_id is None else cluster_centers.get(relation_id)
                stats["candidate_scores"] += 1
                if center_payload is None:
                    query_transformed = query_relation_global
                    trace_transformed = trace_global
                    stats["fallback_global_candidate_scores"] += 1
                else:
                    query_key = (row_idx, relation_id)
                    trace_key = (trace_id, relation_id)
                    if query_key not in query_cache:
                        query_cache[query_key] = apply_cluster_centering(query_relation_global, center_payload)
                    if trace_key not in trace_cache:
                        trace_cache[trace_key] = apply_cluster_centering(trace_global, center_payload)
                    query_transformed = query_cache[query_key]
                    trace_transformed = trace_cache[trace_key]
                    stats["cluster_centered_candidate_scores"] += 1
                relation_scores.append(
                    None
                    if not isinstance(query_transformed, torch.Tensor) or not isinstance(trace_transformed, torch.Tensor)
                    else float(torch.dot(query_transformed, trace_transformed).item())
                )
            copied["factor_relation_scores"] = relation_scores
            copied["factor_relation_score_transform"] = transform_name
            transformed_count += 1
        transformed.append(copied)
    return transformed, transformed_count, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument(
        "--transform",
        choices=["orthogonalize", "whiten", "global_then_per_cluster_whiten", "global_then_cluster_center"],
        default="orthogonalize",
    )
    parser.add_argument("--whiten_eps", default=1.0e-4, type=float)
    parser.add_argument("--cluster_whiten_eps", default=1.0e-4, type=float)
    parser.add_argument("--cluster_min_size", default=5, type=int)
    args = parser.parse_args()

    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    memory = json.loads((args.run_dir / "memory_snapshot.json").read_text())
    log_path = args.run_dir / "annotated_route_logs.jsonl"
    if not log_path.exists():
        log_path = args.run_dir / "route_logs.jsonl"
    log_rows = read_jsonl(log_path)
    log_rows = backfill_relation_ids(log_rows, load_relation_ids_from_memory(args.run_dir))

    hparams_path = run_config["hparams_path"]
    hparams_class = resolve_hparams_class("HOPEDIT")
    hparams = hparams_class.from_hparams(hparams_path)
    editor = BaseEditor.from_hparams(hparams)
    controller = HopEditController(model=editor.model, tok=editor.tok, hparams=hparams)

    trace_entries = sorted(memory, key=lambda row: trace_number(row["trace_id"]))
    trace_prompts = [str(row.get("prompt") or "") for row in trace_entries]
    trace_subjects = [row.get("subject") for row in trace_entries]
    trace_targets = [row.get("target_new") for row in trace_entries]
    trace_factor_rows = extract_factor_rows(
        controller,
        trace_prompts,
        trace_subjects,
        trace_targets,
        batch_size=args.batch_size,
    )
    trace_subject_by_id: dict[str, torch.Tensor] = {}
    trace_relation_by_id: dict[str, torch.Tensor] = {}
    relation_id_by_trace: dict[str, object] = {}
    target_by_trace = {}
    subject_by_trace = {}
    for entry, factor_row in zip(trace_entries, trace_factor_rows):
        trace_id = str(entry["trace_id"])
        subject = factor_row.get("subject_factor")
        relation = factor_row.get("relation_factor")
        if isinstance(subject, torch.Tensor):
            trace_subject_by_id[trace_id] = subject.detach().float().cpu()
        if isinstance(relation, torch.Tensor):
            trace_relation_by_id[trace_id] = relation.detach().float().cpu()
        relation_id_by_trace[trace_id] = entry.get("relation_id")
        target_by_trace[trace_id] = entry.get("target_new")
        subject_by_trace[trace_id] = entry.get("subject")

    query_prompts = [str(row.get("prompt") or "") for row in log_rows]
    query_subjects = []
    query_targets = []
    for row in log_rows:
        trace_id = row.get("expected_edit_id") or row.get("target_edit_id")
        query_subjects.append(row.get("subject") or subject_by_trace.get(str(trace_id)))
        query_targets.append(target_by_trace.get(str(trace_id)))
    query_factor_rows = extract_factor_rows(
        controller,
        query_prompts,
        query_subjects,
        query_targets,
        batch_size=args.batch_size,
    )

    trace_ids = [str(entry["trace_id"]) for entry in trace_entries]
    whitener = None
    cluster_whiteners = None
    cluster_centers = None
    cluster_score_stats = None
    if args.transform == "whiten":
        whitener = build_relation_whitener(trace_relation_by_id, trace_ids, eps=args.whiten_eps)
        trace_relation_transformed_by_id = {
            trace_id: transformed
            for trace_id in trace_ids
            if (transformed := apply_relation_whitener(trace_relation_by_id.get(trace_id), whitener)) is not None
        }
        transform_name = "embedding_relation_whitened"
    elif args.transform == "global_then_per_cluster_whiten":
        whitener = build_relation_whitener(trace_relation_by_id, trace_ids, eps=args.whiten_eps)
        trace_relation_transformed_by_id = {
            trace_id: transformed
            for trace_id in trace_ids
            if (transformed := apply_relation_whitener(trace_relation_by_id.get(trace_id), whitener)) is not None
        }
        cluster_whiteners = build_cluster_whiteners(
            trace_relation_transformed_by_id,
            relation_id_by_trace,
            eps=args.cluster_whiten_eps,
            min_cluster_size=args.cluster_min_size,
        )
        transform_name = "embedding_relation_global_then_per_cluster_whitened"
    elif args.transform == "global_then_cluster_center":
        whitener = build_relation_whitener(trace_relation_by_id, trace_ids, eps=args.whiten_eps)
        trace_relation_transformed_by_id = {
            trace_id: transformed
            for trace_id in trace_ids
            if (transformed := apply_relation_whitener(trace_relation_by_id.get(trace_id), whitener)) is not None
        }
        cluster_centers = build_cluster_centers(
            trace_relation_transformed_by_id,
            relation_id_by_trace,
            min_cluster_size=args.cluster_min_size,
        )
        transform_name = "embedding_relation_global_then_cluster_centered"
    else:
        trace_relation_transformed_by_id = {}
        for trace_id in trace_ids:
            relation_orth = orthogonalize_relation(trace_relation_by_id.get(trace_id), trace_subject_by_id.get(trace_id))
            if isinstance(relation_orth, torch.Tensor):
                trace_relation_transformed_by_id[trace_id] = relation_orth
        transform_name = "embedding_relation_gram_schmidt_against_subject"

    if args.transform == "global_then_per_cluster_whiten":
        transformed_rows, transformed_count, cluster_score_stats = replace_relation_scores_with_cluster_whiteners(
            log_rows,
            trace_relation_transformed_by_id,
            query_factor_rows,
            global_whitener=whitener,
            relation_by_trace=relation_id_by_trace,
            cluster_whiteners=cluster_whiteners or {},
            transform_name=transform_name,
        )
    elif args.transform == "global_then_cluster_center":
        transformed_rows, transformed_count, cluster_score_stats = replace_relation_scores_with_cluster_centers(
            log_rows,
            trace_relation_transformed_by_id,
            query_factor_rows,
            global_whitener=whitener,
            relation_by_trace=relation_id_by_trace,
            cluster_centers=cluster_centers or {},
            transform_name=transform_name,
        )
    else:
        transformed_rows, transformed_count = replace_relation_scores(
            log_rows,
            trace_relation_transformed_by_id,
            query_factor_rows,
            transform_name=transform_name,
            whitener=whitener,
        )
    ds_size = int(run_config.get("stream_length") or run_config.get("requested_ds_size") or len(trace_entries))
    splits = split_eval_rows(transformed_rows, ds_size)
    diagnostics = {
        "run_dir": str(args.run_dir),
        "log_path": str(log_path),
        "ds_size": ds_size,
        "score_transform": {
            "name": transform_name,
            "transformed_rows": transformed_count,
            "total_rows": len(transformed_rows),
            "whitener": None
            if whitener is None
            else {
                "rank": whitener["rank"],
                "num_vectors": whitener["num_vectors"],
                "eps": whitener["eps"],
                "singular_min_kept": whitener["singular_min_kept"],
                "singular_max": whitener["singular_max"],
            },
            "cluster_whitening": None
            if cluster_whiteners is None
            else {
                "cluster_count": len(cluster_whiteners),
                "cluster_min_size": int(args.cluster_min_size),
                "cluster_whiten_eps": float(args.cluster_whiten_eps),
                "score_stats": cluster_score_stats,
                "clusters": {
                    relation_id: {
                        "rank": whitener_payload["rank"],
                        "num_vectors": whitener_payload["num_vectors"],
                        "singular_min_kept": whitener_payload["singular_min_kept"],
                        "singular_max": whitener_payload["singular_max"],
                    }
                    for relation_id, whitener_payload in sorted(cluster_whiteners.items())
                },
            },
            "cluster_centering": None
            if cluster_centers is None
            else {
                "cluster_count": len(cluster_centers),
                "cluster_min_size": int(args.cluster_min_size),
                "score_stats": cluster_score_stats,
                "clusters": {
                    relation_id: {
                        "num_vectors": center_payload["num_vectors"],
                        "center_norm": center_payload["center_norm"],
                    }
                    for relation_id, center_payload in sorted(cluster_centers.items())
                },
            },
        },
        "available_splits": sorted(splits.keys()),
        "splits": {},
    }
    for name, split_rows in splits.items():
        diagnostics["splits"][name] = analyze_rows(split_rows, name)

    output = args.output or (args.run_dir / f"factor_score_diagnostics_{transform_name}.json")
    output.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
