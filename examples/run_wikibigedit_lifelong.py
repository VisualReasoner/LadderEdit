import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor
from easyeditor.editors.utils import _prepare_requests
from easyeditor.evaluate import compute_edit_quality
from easyeditor.models.hopedit.diagnostics import (
    annotate_route_logs,
    export_memory_snapshot,
    summarize_conflicts,
    summarize_route_diagnostics,
)

from examples.edit_experiment_utils import (
    backbone_slug,
    mean_optional,
    metric_mean,
    nested_acc_mean,
    method_name,
    placeholder_artifact,
    resolve_hparams_class,
    summarize_run,
    write_json,
    write_jsonl,
)


INCREMENTS = [
    "20240201_20240220",
    "20240220_20240301",
    "20240301_20240320",
    "20240320_20240401",
    "20240401_20240501",
    "20240501_20240601",
    "20240601_20240620",
    "20240620_20240701",
]

REWRITE_SUCCESS_THRESHOLD = 0.5
LOCALITY_SUCCESS_THRESHOLD = 0.95
VALID_EVALUATION_MODES = {"teacher_forcing", "wild_em", "wild_llm_judge"}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text_for_match(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def string_match(lhs: str | None, rhs: str | None) -> float:
    return float(normalize_text_for_match(lhs) == normalize_text_for_match(rhs))


def load_increment_records(increment_dir: Path, increment: str) -> tuple[list[dict], Path]:
    path = increment_dir / f"wiki_big_edit_{increment}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing WikiBigEdit increment file: {path}")
    payload = json.loads(path.read_text())
    return payload, path


def sample_records(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(records) <= limit:
        return list(records)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(records)), limit))
    return [records[idx] for idx in chosen]


def _clean_optional_text(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _clean_required_text(value, field_name: str):
    value = _clean_optional_text(value)
    if value is None:
        raise ValueError(f"WikiBigEdit required field {field_name} is missing or NaN")
    return value


def normalize_wikibigedit_records(records: list[dict], base_index: int = 0) -> list[dict]:
    normalized = []
    for offset, item in enumerate(records):
        prompt = _clean_optional_text(item.get("update") or item.get("prompt"))
        if prompt is None:
            raise KeyError(f"WikiBigEdit record {base_index + offset} is missing an update/prompt field")

        target_new = _clean_required_text(item.get("ans") or item.get("target_new"), "ans/target_new")
        ground_truth = _clean_optional_text(item.get("ground_truth")) or "<|endoftext|>"
        locality_prompt = _clean_optional_text(item.get("loc") or item.get("locality"))
        locality_ground_truth = _clean_optional_text(item.get("loc_ans") or item.get("locality_ans"))

        locality = {}
        if locality_prompt is not None and locality_ground_truth is not None:
            locality["locality"] = {
                "prompt": [locality_prompt],
                "ground_truth": [locality_ground_truth],
            }

        portability = {}
        personas_prompt = _clean_optional_text(item.get("personas") or item.get("portability_personas"))
        if personas_prompt is not None and target_new is not None:
            portability["personas"] = {
                "prompt": [personas_prompt],
                "ground_truth": [target_new],
            }
        mhop_prompt = _clean_optional_text(item.get("mhop") or item.get("portability_hop"))
        mhop_answer = _clean_optional_text(item.get("mhop_ans") or item.get("portability_hop_ans"))
        if mhop_prompt is not None and mhop_answer is not None:
            portability["mhop"] = {
                "prompt": [mhop_prompt],
                "ground_truth": [mhop_answer],
            }

        normalized.append(
            {
                "source_index": base_index + offset,
                "prompt": prompt,
                "subject": _clean_optional_text(item.get("subject")),
                "rephrase_prompt": _clean_optional_text(item.get("rephrase")),
                "target_new": target_new,
                "ground_truth": ground_truth,
                "locality": locality,
                "portability": portability,
                "tag": item.get("tag"),
            }
        )
    return normalized


def build_editor_inputs(records: list[dict]):
    prompts = [record["prompt"] for record in records]
    subject = [record["subject"] for record in records]
    target_new = [record["target_new"] for record in records]
    ground_truth = [record["ground_truth"] for record in records]
    rephrase_values = [record.get("rephrase_prompt") for record in records]
    rephrase_prompts = rephrase_values if any(value is not None for value in rephrase_values) else None

    locality_inputs = None
    locality_keys = sorted({key for record in records for key in (record.get("locality") or {}).keys()})
    if locality_keys:
        locality_inputs = {}
        for key in locality_keys:
            locality_inputs[key] = {"prompt": [], "ground_truth": []}
            for record in records:
                bucket = (record.get("locality") or {}).get(key)
                locality_inputs[key]["prompt"].append(bucket.get("prompt") if bucket is not None else None)
                locality_inputs[key]["ground_truth"].append(bucket.get("ground_truth") if bucket is not None else None)

    portability_inputs = None
    portability_keys = sorted({key for record in records for key in (record.get("portability") or {}).keys()})
    if portability_keys:
        portability_inputs = {}
        for key in portability_keys:
            portability_inputs[key] = {"prompt": [], "ground_truth": []}
            for record in records:
                bucket = (record.get("portability") or {}).get(key)
                portability_inputs[key]["prompt"].append(bucket.get("prompt") if bucket is not None else None)
                portability_inputs[key]["ground_truth"].append(bucket.get("ground_truth") if bucket is not None else None)

    requests = _prepare_requests(
        prompts,
        target_new,
        ground_truth,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        subject=subject,
    )
    for idx, request in enumerate(requests):
        request.setdefault("case_id", idx)
    return requests, "token_em"


def compute_pre_metrics(editor: BaseEditor, requests: list[dict], eval_metric: str):
    rows = []
    for request in requests:
        rows.append(
            {
                "pre": compute_edit_quality(
                    editor.model,
                    editor.model_name,
                    editor.hparams,
                    editor.tok,
                    request,
                    editor.hparams.device,
                    eval_metric=eval_metric,
                    test_generation=False,
                )
            }
        )
    return rows


def evaluate_requests(editor: BaseEditor, requests: list[dict], eval_metric: str):
    rows = []
    for idx, request in enumerate(requests):
        post = compute_edit_quality(
            editor.model,
            editor.model_name,
            editor.hparams,
            editor.tok,
            request,
            editor.hparams.device,
            eval_metric=eval_metric,
            test_generation=False,
        )
        rows.append(
            {
                "case_id": idx,
                "requested_rewrite": request,
                "post": post,
                "time": None,
            }
        )
    return rows


def attach_pre_metrics(metrics: list[dict], pre_metrics: list[dict] | None):
    if pre_metrics is None:
        return metrics
    for metric, pre in zip(metrics, pre_metrics):
        metric["pre"] = pre["pre"]
    return metrics


def finalize_locality(metrics: list[dict], hparams):
    for metric in metrics:
        pre = metric.get("pre", {})
        post = metric.get("post", {})
        if "locality" not in post:
            continue
        for locality_key in list(metric.get("requested_rewrite", {}).get("locality", {}).keys()):
            output_key = f"{locality_key}_output"
            if output_key not in post["locality"]:
                continue
            if output_key not in pre.get("locality", {}):
                continue
            locality_result = []
            post_outputs = _extract_locality_outputs(post["locality"][output_key])
            pre_outputs = _extract_locality_outputs(pre["locality"][output_key])
            for ans, label in zip(post_outputs, pre_outputs):
                if isinstance(ans, str) or isinstance(label, str):
                    locality_result.append(string_match(ans, label))
                else:
                    locality_result.append(float(np.mean(np.equal(ans, label))))
            post["locality"][f"{locality_key}_acc"] = locality_result
            post["locality"].pop(output_key, None)
        if "locality" in pre:
            pre.pop("locality")
    return metrics


def _extract_locality_outputs(value):
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], list) and isinstance(value[1], list):
        # LLM-judge / generation path returns (scores, responses).
        if value[1] and all(isinstance(item, str) for item in value[1]):
            return value[1]
        return value[0]
    return value


def apply_single_edit(editor: BaseEditor, request: dict):
    method = editor.alg_name
    if method in {"IKE", "ICE"}:
        edited_model, _weights_copy, _icl_examples = editor.model, {}, editor.apply_algo(
            editor.model,
            editor.tok,
            [request],
            editor.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
            train_ds=None,
        )
        return edited_model

    edited_model, _weights_copy = editor.apply_algo(
        editor.model,
        editor.tok,
        [request],
        editor.hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
        train_ds=None,
    )

    if method in {"HOPEDIT", "LoRA", "QLoRA", "DPO", "MELO", "SERAC"}:
        editor.model = edited_model
    return edited_model


def clear_controller_logs(editor: BaseEditor):
    controller = editor.model if hasattr(editor.model, "route_logs") else None
    if controller is not None:
        controller.route_logs = []


def collect_controller_state(editor: BaseEditor):
    controller = editor.model if hasattr(editor.model, "route_logs") else None
    route_logs = []
    memory_entries = []
    memory_snapshot = []
    if controller is not None:
        route_logs = list(getattr(controller, "route_logs", []))
        memory_entries = list(getattr(controller, "memory_entries", []))
        if hasattr(controller, "export_memory_snapshot"):
            memory_snapshot = controller.export_memory_snapshot(include_keys=False)
        else:
            memory_snapshot = export_memory_snapshot(memory_entries, include_keys=False)
    return route_logs, memory_entries, memory_snapshot


def safe_mean(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def percentile(values: list[float | None], q: float) -> float | None:
    filtered = sorted(float(v) for v in values if v is not None)
    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]
    rank = (len(filtered) - 1) * q
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return filtered[low]
    weight = rank - low
    return float(filtered[low] * (1.0 - weight) + filtered[high] * weight)


def primary_route_entry_by_case(annotated_logs: list[dict]) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    priority = {"rewrite": 0, "rephrase": 1, "post_edit": 2}
    for entry in annotated_logs:
        case_id = entry.get("case_id")
        if case_id is None:
            continue
        event_type = entry.get("event_type")
        if event_type not in priority:
            continue
        case_id = int(case_id)
        existing = mapping.get(case_id)
        if existing is None or priority[event_type] < priority.get(existing.get("event_type"), 99):
            mapping[case_id] = entry
    return mapping


def build_failure_decomposition(
    run_config: dict,
    metrics: list[dict],
    summary: dict,
    annotated_logs: list[dict],
) -> dict:
    route_by_case = primary_route_entry_by_case(annotated_logs)
    exact_supported = bool(route_by_case)
    cases = []
    for row in summary.get("per_case", []):
        case_id = int(row.get("case_id", 0))
        route_entry = route_by_case.get(case_id)
        rewrite_acc = row.get("post_rewrite_acc")
        locality_acc = row.get("post_locality_acc")
        rewrite_ok = rewrite_acc is not None and rewrite_acc >= REWRITE_SUCCESS_THRESHOLD
        locality_ok = locality_acc is None or locality_acc >= LOCALITY_SUCCESS_THRESHOLD
        route_observed = route_entry is not None and route_entry.get("correct_route") is not None

        if route_observed:
            correct_route = bool(route_entry.get("correct_route"))
            chosen_edit_id = route_entry.get("chosen_edit_id")
            if (not correct_route) or chosen_edit_id is None:
                label = "retrieval_failure"
            elif correct_route and (not rewrite_ok) and locality_ok:
                label = "update_failure"
            elif correct_route and rewrite_ok and (not locality_ok):
                label = "locality_failure"
            elif correct_route and (not rewrite_ok) and (not locality_ok):
                label = "mixed_failure"
            elif correct_route and rewrite_ok:
                label = "correct"
            else:
                label = "unknown"
        else:
            if rewrite_ok and locality_ok:
                label = "correct"
            elif (not rewrite_ok) and locality_ok:
                label = "update_failure_proxy"
            elif rewrite_ok and (not locality_ok):
                label = "locality_failure_proxy"
            elif (not rewrite_ok) and (not locality_ok):
                label = "mixed_failure_proxy"
            else:
                label = "unknown"

        cases.append(
            {
                "case_id": case_id,
                "source_index": row.get("source_index"),
                "subject": row.get("subject"),
                "prompt": row.get("prompt"),
                "label": label,
                "post_rewrite_acc": rewrite_acc,
                "post_rephrase_acc": row.get("post_rephrase_acc"),
                "post_locality_acc": locality_acc,
                "route_observed": route_observed,
                "correct_route": None if route_entry is None else route_entry.get("correct_route"),
                "chosen_edit_id": None if route_entry is None else route_entry.get("chosen_edit_id"),
                "expected_edit_id": None if route_entry is None else route_entry.get("expected_edit_id"),
                "top1_prob": None if route_entry is None else route_entry.get("top1_prob"),
                "route_margin": None if route_entry is None else route_entry.get("route_margin"),
            }
        )

    counts: dict[str, int] = {}
    for row in cases:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    total = len(cases)
    rates = {key: (value / total if total else None) for key, value in counts.items()}
    return {
        "applicable": True,
        "editing_method": run_config["editing_method"],
        "run_name": run_config["run_name"],
        "exact_supported": exact_supported,
        "mode": "exact" if exact_supported else "proxy",
        "rewrite_success_threshold": REWRITE_SUCCESS_THRESHOLD,
        "locality_success_threshold": LOCALITY_SUCCESS_THRESHOLD,
        "counts": counts,
        "rates": rates,
        "per_case": cases,
    }


def build_theory_metrics(
    run_config: dict,
    summary: dict,
    route_diagnostics: dict,
    conflict_diagnostics: dict,
    failure_decomposition: dict,
) -> dict:
    rw_losses = []
    rp_losses = []
    loc_losses = []
    primary_distortions = []
    extended_distortions = []
    for row in summary.get("per_case", []):
        rewrite_acc = row.get("post_rewrite_acc")
        rephrase_acc = row.get("post_rephrase_acc")
        locality_acc = row.get("post_locality_acc")
        rw_loss = None if rewrite_acc is None else 1.0 - float(rewrite_acc)
        rp_loss = None if rephrase_acc is None else 1.0 - float(rephrase_acc)
        loc_loss = None if locality_acc is None else 1.0 - float(locality_acc)
        rw_losses.append(rw_loss)
        rp_losses.append(rp_loss)
        loc_losses.append(loc_loss)
        if rw_loss is not None or loc_loss is not None:
            primary_distortions.append(safe_mean([rw_loss, loc_loss]))
        if rw_loss is not None or rp_loss is not None or loc_loss is not None:
            extended_distortions.append(safe_mean([rw_loss, rp_loss, loc_loss]))

    routing_summary = route_diagnostics.get("routing", {}) if isinstance(route_diagnostics, dict) else {}
    rewrite_route = routing_summary.get("rewrite", {}) if isinstance(routing_summary, dict) else {}
    locality_route = routing_summary.get("locality", {}) if isinstance(routing_summary, dict) else {}

    return {
        "applicable": True,
        "editing_method": run_config["editing_method"],
        "run_name": run_config["run_name"],
        "distortion": {
            "primary_definition": "mean(1-rewrite_acc, 1-locality_acc) over available channels",
            "extended_definition": "mean(1-rewrite_acc, 1-rephrase_acc, 1-locality_acc) over available channels",
            "rewrite_loss_mean": safe_mean(rw_losses),
            "rephrase_loss_mean": safe_mean(rp_losses),
            "locality_loss_mean": safe_mean(loc_losses),
            "primary_mean": safe_mean(primary_distortions),
            "primary_p50": percentile(primary_distortions, 0.5),
            "primary_p90": percentile(primary_distortions, 0.9),
            "primary_max": max((float(v) for v in primary_distortions if v is not None), default=None),
            "extended_mean": safe_mean(extended_distortions),
        },
        "performance": {
            "post_rewrite_mean": summary.get("post_rewrite_mean"),
            "post_rephrase_mean": summary.get("post_rephrase_mean"),
            "post_locality_mean": summary.get("post_locality_mean"),
            "rewrite_delta_mean": summary.get("rewrite_delta_mean"),
            "rephrase_delta_mean": summary.get("rephrase_delta_mean"),
            "early_late_gap": summary.get("early_late_gap"),
        },
        "routing": {
            "applicable": bool(rewrite_route),
            "rewrite_route_accuracy": rewrite_route.get("route_accuracy"),
            "rewrite_route_margin_mean": rewrite_route.get("route_margin_mean"),
            "rewrite_top1_prob_mean": rewrite_route.get("top1_prob_mean"),
            "rewrite_coverage": rewrite_route.get("coverage"),
            "locality_false_activation_rate": locality_route.get("false_activation_rate"),
            "locality_no_edit_rate": locality_route.get("no_edit_rate"),
        },
        "conflict": {
            "applicable": bool(conflict_diagnostics) and conflict_diagnostics.get("applicable", True),
            "num_edits": conflict_diagnostics.get("num_edits") if isinstance(conflict_diagnostics, dict) else None,
            "mean_combined_offdiag": conflict_diagnostics.get("mean_combined_offdiag") if isinstance(conflict_diagnostics, dict) else None,
            "mean_max_offdiag_conflict": conflict_diagnostics.get("mean_max_offdiag_conflict") if isinstance(conflict_diagnostics, dict) else None,
            "max_pair_conflict": conflict_diagnostics.get("max_pair_conflict") if isinstance(conflict_diagnostics, dict) else None,
        },
        "failure": {
            "mode": failure_decomposition.get("mode"),
            "exact_supported": failure_decomposition.get("exact_supported"),
            "rates": failure_decomposition.get("rates"),
        },
    }


def build_efficiency_metrics(
    run_config: dict,
    summary: dict,
    route_diagnostics: dict,
    memory_snapshot: list[dict] | dict,
) -> dict:
    stream_length = run_config.get("stream_length") or 0
    cumulative_edits = run_config.get("cumulative_edits") or 0
    eval_wall = run_config.get("eval_wall_time_seconds")
    edit_wall = run_config.get("increment_edit_wall_time_seconds")
    total_wall = run_config.get("wall_time_seconds")
    if isinstance(memory_snapshot, list):
        memory_entries_final = len(memory_snapshot)
    else:
        memory_entries_final = None
    return {
        "applicable": True,
        "editing_method": run_config["editing_method"],
        "run_name": run_config["run_name"],
        "increment": run_config.get("increment"),
        "stream_type": run_config.get("stream_type"),
        "stream_length": stream_length,
        "cumulative_edits": cumulative_edits,
        "edit_wall_time_seconds": edit_wall,
        "eval_wall_time_seconds": eval_wall,
        "total_wall_time_seconds": total_wall,
        "edit_seconds_per_edit": None if not edit_wall or not stream_length else float(edit_wall / stream_length),
        "eval_seconds_per_case": None if not eval_wall or not stream_length else float(eval_wall / stream_length),
        "cumulative_seconds_per_edit": None if not total_wall or not cumulative_edits else float(total_wall / cumulative_edits),
        "current_eval_cases_per_second": None if not eval_wall or not stream_length else float(stream_length / eval_wall),
        "memory_entries_final": memory_entries_final,
        "memory_entries_per_edit": None if memory_entries_final is None or not cumulative_edits else float(memory_entries_final / cumulative_edits),
        "mean_case_time_reported": summary.get("mean_time"),
        "route_events_logged": route_diagnostics.get("summary", {}).get("num_logged_events") if isinstance(route_diagnostics, dict) else None,
    }


def write_checkpoint_artifacts(
    output_dir: Path,
    run_config: dict,
    metrics: list[dict],
    eval_records: list[dict],
    route_logs: list[dict],
    memory_entries: list[dict],
    memory_snapshot: list[dict],
    hparams,
):
    write_json(output_dir / "metrics.json", metrics)
    annotated_logs = annotate_route_logs(route_logs, metrics) if route_logs else []
    write_jsonl(output_dir / "annotated_route_logs.jsonl", annotated_logs)
    route_diagnostics = summarize_route_diagnostics(annotated_logs, metrics) if route_logs else placeholder_artifact("route_diagnostics", run_config)
    conflict_diagnostics = (
        summarize_conflicts(memory_entries, hparams.semantic_weight, hparams.activation_weight)
        if memory_entries and hasattr(hparams, "semantic_weight") and hasattr(hparams, "activation_weight")
        else placeholder_artifact("conflict_diagnostics", run_config)
    )
    write_json(output_dir / "route_diagnostics.json", route_diagnostics)
    write_json(output_dir / "conflict_diagnostics.json", conflict_diagnostics)
    write_json(output_dir / "memory_snapshot.json", memory_snapshot if memory_snapshot else placeholder_artifact("memory_snapshot", run_config))
    summary = summarize_run(metrics, eval_records, run_config, memory_snapshot if isinstance(memory_snapshot, list) else [])
    write_json(output_dir / "summary.json", summary)
    failure_decomposition = build_failure_decomposition(run_config, metrics, summary, annotated_logs)
    theory_metrics = build_theory_metrics(run_config, summary, route_diagnostics, conflict_diagnostics, failure_decomposition)
    efficiency_metrics = build_efficiency_metrics(run_config, summary, route_diagnostics, memory_snapshot if memory_snapshot else [])
    write_json(output_dir / "failure_decomposition.json", failure_decomposition)
    write_json(output_dir / "theory_metrics.json", theory_metrics)
    write_json(output_dir / "efficiency_metrics.json", efficiency_metrics)
    return summary


def evaluate_and_write(
    editor: BaseEditor,
    hparams,
    method: str,
    backbone: str,
    output_dir: Path,
    eval_records: list[dict],
    eval_requests: list[dict],
    eval_metric: str,
    run_config: dict,
    pre_metrics: list[dict] | None = None,
):
    clear_controller_logs(editor)
    eval_start = time.time()
    metrics = evaluate_requests(editor, eval_requests, eval_metric)
    eval_seconds = time.time() - eval_start
    metrics = attach_pre_metrics(metrics, pre_metrics)
    metrics = finalize_locality(metrics, hparams)
    route_logs, memory_entries, memory_snapshot = collect_controller_state(editor)
    run_config["eval_wall_time_seconds"] = eval_seconds
    write_json(output_dir / "run_config.json", run_config)
    summary = write_checkpoint_artifacts(
        output_dir,
        run_config,
        metrics,
        eval_records,
        route_logs,
        memory_entries,
        memory_snapshot,
        hparams,
    )
    if method == "HOPEDIT":
        write_json(output_dir / "hopedit_route_diagnostics.json", json.loads((output_dir / "route_diagnostics.json").read_text()))
        write_json(output_dir / "hopedit_conflict_diagnostics.json", json.loads((output_dir / "conflict_diagnostics.json").read_text()))
        write_json(output_dir / "hopedit_memory_snapshot.json", json.loads((output_dir / "memory_snapshot.json").read_text()))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", required=True, type=str)
    parser.add_argument("--hparams_dir", required=True, type=str)
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"), type=str)
    parser.add_argument("--increment_dir", default=None, type=str)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "wikibigedit_lifelong"), type=str)
    parser.add_argument("--increments", default=" ".join(INCREMENTS), type=str)
    parser.add_argument("--ds_size_per_increment", default=0, type=int)
    parser.add_argument("--past_eval_max_samples", default=0, type=int)
    parser.add_argument("--checkpoint_interval", default=0, type=int)
    parser.add_argument("--evaluation_mode", default="teacher_forcing", choices=sorted(VALID_EVALUATION_MODES), type=str)
    parser.add_argument("--api_key", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    seed_everything(args.seed)

    method = method_name(args.editing_method)
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    if not hasattr(hparams, "sequential_edit"):
        setattr(hparams, "sequential_edit", True)
    else:
        hparams.sequential_edit = True
    if args.evaluation_mode == "teacher_forcing":
        if hasattr(hparams, "evaluation_type"):
            delattr(hparams, "evaluation_type")
        if hasattr(hparams, "api_key"):
            delattr(hparams, "api_key")
    elif args.evaluation_mode == "wild_em":
        hparams.evaluation_type = "LLM-judge"
        hparams.api_key = None
    elif args.evaluation_mode == "wild_llm_judge":
        hparams.evaluation_type = "LLM-judge"
        hparams.api_key = args.api_key

    increment_dir = Path(args.increment_dir) if args.increment_dir else Path(args.data_dir) / "wikibigedit"
    requested_increments = [token for token in args.increments.split() if token]
    backbone = backbone_slug(hparams.model_name)

    output_root = Path(args.output_root) / f"{method.lower()}_{backbone}"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "editing_method": method,
        "alg_name": hparams.alg_name,
        "model_name": hparams.model_name,
        "backbone": backbone,
        "increment_dir": str(increment_dir.resolve()),
        "increments": requested_increments,
        "ds_size_per_increment": args.ds_size_per_increment,
        "past_eval_max_samples": args.past_eval_max_samples,
        "checkpoint_interval": args.checkpoint_interval,
        "seed": args.seed,
        "evaluation_mode": args.evaluation_mode,
        "hparams_path": str(Path(args.hparams_dir).resolve()),
    }
    write_json(output_root / "run_manifest.json", manifest)

    editor = BaseEditor.from_hparams(hparams)
    all_increment_records: dict[str, list[dict]] = {}
    cumulative_edits = 0
    start_time = time.time()

    for increment_idx, increment in enumerate(requested_increments):
        raw_records, dataset_file = load_increment_records(increment_dir, increment)
        sampled_records = sample_records(raw_records, args.ds_size_per_increment, args.seed + increment_idx)
        current_records = normalize_wikibigedit_records(sampled_records)
        current_requests, eval_metric = build_editor_inputs(current_records)

        increment_dir_out = output_root / f"increment_{increment}"
        increment_dir_out.mkdir(parents=True, exist_ok=True)
        write_json(
            increment_dir_out / "increment_manifest.json",
            {
                "increment": increment,
                "dataset_file": str(dataset_file.resolve()),
                "records_in_source_file": len(raw_records),
                "records_used": len(current_records),
            },
        )

        print(json.dumps({"phase": "pre_eval", "increment": increment, "records": len(current_records)}), flush=True)
        pre_metrics = compute_pre_metrics(editor, current_requests, eval_metric)

        print(json.dumps({"phase": "edit", "increment": increment, "records": len(current_requests)}), flush=True)
        edit_start = time.time()
        checkpoint_steps = []
        if args.checkpoint_interval > 0:
            checkpoint_steps = list(range(args.checkpoint_interval, len(current_requests) + 1, args.checkpoint_interval))
            if checkpoint_steps and checkpoint_steps[-1] != len(current_requests):
                checkpoint_steps.append(len(current_requests))
            elif not checkpoint_steps and len(current_requests) > 0:
                checkpoint_steps = [len(current_requests)]
        next_checkpoint_idx = 0

        for request_idx, request in enumerate(current_requests, start=1):
            apply_single_edit(editor, request)
            if next_checkpoint_idx < len(checkpoint_steps) and request_idx == checkpoint_steps[next_checkpoint_idx]:
                seen_records = current_records[:request_idx]
                seen_requests = current_requests[:request_idx]
                seen_pre_metrics = pre_metrics[:request_idx]
                checkpoint_output_dir = increment_dir_out / f"checkpoint_{request_idx:06d}" / "current"
                checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_run_config = {
                    "run_name": f"{method.lower()}_{backbone}_wikibigedit_{increment}_current_{request_idx:06d}",
                    "editing_method": method,
                    "alg_name": hparams.alg_name,
                    "model_name": hparams.model_name,
                    "backbone": backbone,
                    "data_type": "WikiBigEdit",
                    "dataset_file": str(dataset_file.resolve()),
                    "stream_type": "increment_current_checkpoint",
                    "seed": args.seed,
                    "evaluation_mode": args.evaluation_mode,
                    "sequential_edit": True,
                    "stream_length": len(seen_records),
                    "requested_ds_size": len(seen_records),
                    "hparams_path": str(Path(args.hparams_dir).resolve()),
                    "output_dir": str(checkpoint_output_dir.resolve()),
                    "wall_time_seconds": time.time() - start_time,
                    "increment_edit_wall_time_seconds": time.time() - edit_start,
                    "increment": increment,
                    "increment_index": increment_idx,
                    "cumulative_edits": cumulative_edits + request_idx,
                    "checkpoint_step": request_idx,
                    "checkpoint_interval": args.checkpoint_interval,
                }
                checkpoint_summary = evaluate_and_write(
                    editor,
                    hparams,
                    method,
                    backbone,
                    checkpoint_output_dir,
                    seen_records,
                    seen_requests,
                    eval_metric,
                    checkpoint_run_config,
                    pre_metrics=seen_pre_metrics,
                )
                print(
                    json.dumps(
                        {
                            "phase": "checkpoint_eval_complete",
                            "increment": increment,
                            "checkpoint_step": request_idx,
                            "post_rewrite_mean": checkpoint_summary.get("post_rewrite_mean"),
                            "post_rephrase_mean": checkpoint_summary.get("post_rephrase_mean"),
                            "post_locality_mean": checkpoint_summary.get("post_locality_mean"),
                            "cumulative_edits": cumulative_edits + request_idx,
                        }
                    ),
                    flush=True,
                )
                next_checkpoint_idx += 1
        edit_seconds = time.time() - edit_start
        cumulative_edits += len(current_requests)
        all_increment_records[increment] = current_records

        current_output_dir = increment_dir_out / "current"
        current_output_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "run_name": f"{method.lower()}_{backbone}_wikibigedit_{increment}_current",
            "editing_method": method,
            "alg_name": hparams.alg_name,
            "model_name": hparams.model_name,
            "backbone": backbone,
            "data_type": "WikiBigEdit",
            "dataset_file": str(dataset_file.resolve()),
            "stream_type": "increment_current",
            "seed": args.seed,
            "evaluation_mode": args.evaluation_mode,
            "sequential_edit": True,
            "stream_length": len(current_records),
            "requested_ds_size": len(current_records),
            "hparams_path": str(Path(args.hparams_dir).resolve()),
            "output_dir": str(current_output_dir.resolve()),
            "wall_time_seconds": time.time() - start_time,
            "increment_edit_wall_time_seconds": edit_seconds,
            "increment": increment,
            "increment_index": increment_idx,
            "cumulative_edits": cumulative_edits,
        }
        current_summary = evaluate_and_write(
            editor,
            hparams,
            method,
            backbone,
            current_output_dir,
            current_records,
            current_requests,
            eval_metric,
            run_config,
            pre_metrics=pre_metrics,
        )

        print(
            json.dumps(
                {
                    "phase": "current_eval_complete",
                    "increment": increment,
                    "post_rewrite_mean": current_summary.get("post_rewrite_mean"),
                    "post_rephrase_mean": current_summary.get("post_rephrase_mean"),
                    "post_locality_mean": current_summary.get("post_locality_mean"),
                    "cumulative_edits": cumulative_edits,
                }
            ),
            flush=True,
        )

        past_eval_summaries = {}
        for previous_increment in requested_increments[:increment_idx]:
            previous_records = list(all_increment_records[previous_increment])
            if args.past_eval_max_samples > 0 and len(previous_records) > args.past_eval_max_samples:
                previous_records = sample_records(previous_records, args.past_eval_max_samples, args.seed + increment_idx + len(previous_increment))
            previous_requests, previous_metric = build_editor_inputs(previous_records)
            prev_output_dir = increment_dir_out / "past_eval" / previous_increment
            prev_output_dir.mkdir(parents=True, exist_ok=True)
            prev_config = {
                "run_name": f"{method.lower()}_{backbone}_wikibigedit_{increment}_past_{previous_increment}",
                "editing_method": method,
                "alg_name": hparams.alg_name,
                "model_name": hparams.model_name,
                "backbone": backbone,
                "data_type": "WikiBigEdit",
                "dataset_file": previous_increment,
                "stream_type": "increment_past",
                "seed": args.seed,
                "evaluation_mode": args.evaluation_mode,
                "sequential_edit": True,
                "stream_length": len(previous_records),
                "requested_ds_size": len(previous_records),
                "hparams_path": str(Path(args.hparams_dir).resolve()),
                "output_dir": str(prev_output_dir.resolve()),
                "wall_time_seconds": time.time() - start_time,
                "increment": increment,
                "evaluated_past_increment": previous_increment,
                "increment_index": increment_idx,
                "cumulative_edits": cumulative_edits,
            }
            summary = evaluate_and_write(
                editor,
                hparams,
                method,
                backbone,
                prev_output_dir,
                previous_records,
                previous_requests,
                previous_metric,
                prev_config,
                pre_metrics=None,
            )
            past_eval_summaries[previous_increment] = summary

        write_json(increment_dir_out / "past_eval_summaries.json", past_eval_summaries)


if __name__ == "__main__":
    main()
