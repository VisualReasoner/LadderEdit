import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.edit_experiment_utils import infer_family_bucket, load_normalized_records, resolve_hparams_class, write_json


def seed_everything(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in re.split(r"[:,\s]+", str(value)) if part.strip()]


def _parse_poolings(value: str) -> list[str]:
    return [str(part).strip().lower() for part in re.split(r"[:,\s]+", str(value)) if str(part).strip()]


def _hparam_float(hparams, name: str, default: float) -> float:
    value = getattr(hparams, name, None)
    return float(default) if value is None else float(value)


def _compute_route(rows, query_row, index, subject_margin_threshold, relation_margin_threshold, subject_energy_threshold, relation_energy_threshold):
    subject_factor = query_row.get("subject_factor")
    relation_factor = query_row.get("relation_factor")
    subject_rows = []
    relation_rows = []
    for trace_row in rows[: index + 1]:
        trace_subject = trace_row.get("subject_factor")
        trace_relation = trace_row.get("relation_factor")
        if subject_factor is not None and trace_subject is not None:
            score = float(subject_factor.dot(trace_subject).item())
            subject_rows.append({"edit_id": trace_row["edit_id"], "score": score})
        if relation_factor is not None and trace_relation is not None:
            score = float(relation_factor.dot(trace_relation).item())
            relation_rows.append({"edit_id": trace_row["edit_id"], "score": score})
    subject_rows.sort(key=lambda row: row["score"], reverse=True)
    relation_rows.sort(key=lambda row: row["score"], reverse=True)
    subject_top = subject_rows[0] if subject_rows else None
    relation_top = relation_rows[0] if relation_rows else None
    subject_runner = subject_rows[1] if len(subject_rows) > 1 else None
    relation_runner = relation_rows[1] if len(relation_rows) > 1 else None
    subject_margin = None if subject_top is None else float(subject_top["score"] - (subject_runner["score"] if subject_runner else 0.0))
    relation_margin = None if relation_top is None else float(relation_top["score"] - (relation_runner["score"] if relation_runner else 0.0))
    subject_energy = None if subject_top is None else float(-subject_top["score"])
    relation_energy = None if relation_top is None else float(-relation_top["score"])
    same_trace = bool(subject_top is not None and relation_top is not None and subject_top["edit_id"] == relation_top["edit_id"])
    subject_pass = bool(
        subject_energy is not None
        and subject_margin is not None
        and subject_energy < subject_energy_threshold
        and subject_margin > subject_margin_threshold
    )
    relation_pass = bool(
        relation_energy is not None
        and relation_margin is not None
        and relation_energy < relation_energy_threshold
        and relation_margin > relation_margin_threshold
    )
    chosen_edit_id = subject_top["edit_id"] if same_trace and subject_pass and relation_pass else None
    return {
        "subject_top_edit_id": None if subject_top is None else subject_top["edit_id"],
        "relation_top_edit_id": None if relation_top is None else relation_top["edit_id"],
        "subject_margin": subject_margin,
        "relation_margin": relation_margin,
        "subject_energy": subject_energy,
        "relation_energy": relation_energy,
        "same_trace": same_trace,
        "subject_pass": subject_pass,
        "relation_pass": relation_pass,
        "chosen_edit_id": chosen_edit_id,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT")
    parser.add_argument("--hparams_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_type", default="CounterFact")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ds_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subject_layers", type=str, default="8,10,12,16")
    parser.add_argument("--subject_poolings", type=str, default="last,mean")
    parser.add_argument("--relation_layer", type=int, default=16)
    args = parser.parse_args()

    from easyeditor import BaseEditor
    from easyeditor.models.hopedit.hopedit_main import HopEditController

    seed_everything(args.seed)
    hparams_class = resolve_hparams_class(args.editing_method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)
    if hasattr(editor, "model") and hasattr(editor.model, "_extract_batched_factored_address_keys"):
        controller = editor.model
    else:
        controller = HopEditController(model=editor.model, tok=editor.tok, hparams=hparams)

    records, dataset_file = load_normalized_records(args.data_dir, args.data_type, args.ds_size)
    subject_counts = {}
    for record in records:
        subject = " ".join(str(record.get("subject") or "").strip().lower().split())
        if subject:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

    prompts = [str(record.get("prompt") or "") for record in records]
    rephrases = [str(record.get("rephrase_prompt") or record.get("prompt") or "") for record in records]
    subjects = [record.get("subject") for record in records]
    objects = [record.get("target_new") for record in records]

    subject_layers = _parse_layers(args.subject_layers)
    subject_poolings = _parse_poolings(args.subject_poolings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_rows = []
    for subject_layer in subject_layers:
        for subject_pooling in subject_poolings:
            prompt_rows = controller._extract_batched_factored_address_keys(
                prompts,
                subjects,
                objects,
                subject_layer_override=subject_layer,
                relation_layer_override=args.relation_layer,
                subject_pooling_override=subject_pooling,
            )
            query_rows = controller._extract_batched_factored_address_keys(
                prompts,
                subjects,
                [None for _ in prompts],
                subject_layer_override=subject_layer,
                relation_layer_override=args.relation_layer,
                subject_pooling_override=subject_pooling,
            )
            rephrase_rows = controller._extract_batched_factored_address_keys(
                rephrases,
                subjects,
                objects,
                subject_layer_override=subject_layer,
                relation_layer_override=args.relation_layer,
                subject_pooling_override=subject_pooling,
            )
            trace_rows = []
            for idx, (record, prompt_row, rephrase_row) in enumerate(zip(records, prompt_rows, rephrase_rows)):
                trace_rows.append(
                    {
                        "edit_id": f"hopedit_{idx:05d}",
                        "case_id": idx,
                        "prompt": record.get("prompt"),
                        "subject": record.get("subject"),
                        "family_bucket": infer_family_bucket(record, args.data_type, subject_counts),
                        "subject_factor": prompt_row.get("subject_factor"),
                        "relation_factor": prompt_row.get("relation_factor"),
                        "subject_found": bool(prompt_row.get("subject_found")),
                        "relation_token_count": int(prompt_row.get("relation_token_count") or 0),
                        "subject_agreement": None
                        if prompt_row.get("subject_factor") is None or rephrase_row.get("subject_factor") is None
                        else float(prompt_row["subject_factor"].dot(rephrase_row["subject_factor"]).item()),
                        "relation_agreement": None
                        if prompt_row.get("relation_factor") is None or rephrase_row.get("relation_factor") is None
                        else float(prompt_row["relation_factor"].dot(rephrase_row["relation_factor"]).item()),
                    }
                )

            per_case = []
            for idx, trace_row in enumerate(trace_rows):
                route = _compute_route(
                    trace_rows,
                    query_rows[idx],
                    idx,
                    subject_margin_threshold=_hparam_float(hparams, "factored_subject_margin_threshold", 0.03),
                    relation_margin_threshold=_hparam_float(hparams, "factored_relation_margin_threshold", 0.03),
                    subject_energy_threshold=_hparam_float(hparams, "factored_subject_energy_threshold", 0.0),
                    relation_energy_threshold=_hparam_float(hparams, "factored_relation_energy_threshold", 0.0),
                )
                expected_edit_id = trace_row["edit_id"]
                row = {
                    "case_id": idx,
                    "edit_id": expected_edit_id,
                    "prompt": trace_row["prompt"],
                    "subject": trace_row["subject"],
                    "family_bucket": trace_row["family_bucket"],
                    "subject_found": trace_row["subject_found"],
                    "relation_token_count": trace_row["relation_token_count"],
                    "subject_agreement": trace_row["subject_agreement"],
                    "relation_agreement": trace_row["relation_agreement"],
                    "subject_top_correct": route["subject_top_edit_id"] == expected_edit_id,
                    "relation_top_correct": route["relation_top_edit_id"] == expected_edit_id,
                    "both_top_correct": (
                        route["subject_top_edit_id"] == expected_edit_id
                        and route["relation_top_edit_id"] == expected_edit_id
                        and route["same_trace"]
                    ),
                    "hard_and_correct": route["chosen_edit_id"] == expected_edit_id,
                    "hard_and_fired": route["chosen_edit_id"] is not None,
                    "same_trace": route["same_trace"],
                    "subject_margin": route["subject_margin"],
                    "relation_margin": route["relation_margin"],
                    "subject_energy": route["subject_energy"],
                    "relation_energy": route["relation_energy"],
                    "factor_failure_partition": (
                        "none"
                        if route["same_trace"] and route["subject_pass"] and route["relation_pass"]
                        else "both"
                        if ((not route["same_trace"]) or (not route["subject_pass"])) and ((not route["same_trace"]) or (not route["relation_pass"]))
                        else "subject"
                        if ((not route["same_trace"]) or (not route["subject_pass"]))
                        else "relation"
                    ),
                }
                per_case.append(row)

            family_buckets = {}
            for bucket in sorted({row["family_bucket"] for row in per_case}):
                bucket_rows = [row for row in per_case if row["family_bucket"] == bucket]
                family_buckets[bucket] = {
                    "count": len(bucket_rows),
                    "subject_top1_rate": _safe_mean([1.0 if row["subject_top_correct"] else 0.0 for row in bucket_rows]),
                    "relation_top1_rate": _safe_mean([1.0 if row["relation_top_correct"] else 0.0 for row in bucket_rows]),
                    "both_top1_rate": _safe_mean([1.0 if row["both_top_correct"] else 0.0 for row in bucket_rows]),
                    "hard_and_correct_rate": _safe_mean([1.0 if row["hard_and_correct"] else 0.0 for row in bucket_rows]),
                    "hard_and_fire_rate": _safe_mean([1.0 if row["hard_and_fired"] else 0.0 for row in bucket_rows]),
                    "same_trace_rate": _safe_mean([1.0 if row["same_trace"] else 0.0 for row in bucket_rows]),
                }

            config_rows.append(
                {
                    "subject_layer": subject_layer,
                    "relation_layer": args.relation_layer,
                    "subject_pooling": subject_pooling,
                    "subject_top1_rate": _safe_mean([1.0 if row["subject_top_correct"] else 0.0 for row in per_case]),
                    "relation_top1_rate": _safe_mean([1.0 if row["relation_top_correct"] else 0.0 for row in per_case]),
                    "both_top1_rate": _safe_mean([1.0 if row["both_top_correct"] else 0.0 for row in per_case]),
                    "hard_and_correct_rate": _safe_mean([1.0 if row["hard_and_correct"] else 0.0 for row in per_case]),
                    "hard_and_fire_rate": _safe_mean([1.0 if row["hard_and_fired"] else 0.0 for row in per_case]),
                    "same_trace_rate": _safe_mean([1.0 if row["same_trace"] else 0.0 for row in per_case]),
                    "subject_found_rate": _safe_mean([1.0 if row["subject_found"] else 0.0 for row in per_case]),
                    "subject_agreement_mean": _safe_mean([row["subject_agreement"] for row in per_case]),
                    "relation_agreement_mean": _safe_mean([row["relation_agreement"] for row in per_case]),
                    "family_buckets": family_buckets,
                    "per_case": per_case,
                }
            )

    summary = {
        "run_name": f"v43_subject_probe_{args.data_type.lower()}_{args.ds_size}_seed{args.seed}",
        "dataset_file": str(dataset_file),
        "data_type": args.data_type,
        "ds_size": args.ds_size,
        "seed": args.seed,
        "relation_layer": args.relation_layer,
        "thresholds": {
            "subject_margin": _hparam_float(hparams, "factored_subject_margin_threshold", 0.03),
            "relation_margin": _hparam_float(hparams, "factored_relation_margin_threshold", 0.03),
            "subject_energy": _hparam_float(hparams, "factored_subject_energy_threshold", 0.0),
            "relation_energy": _hparam_float(hparams, "factored_relation_energy_threshold", 0.0),
        },
        "configs": config_rows,
    }
    write_json(output_dir / "subject_probe_summary.json", summary)

    compact = []
    for row in config_rows:
        compact.append(
            {
                "subject_layer": row["subject_layer"],
                "subject_pooling": row["subject_pooling"],
                "subject_top1_rate": row["subject_top1_rate"],
                "relation_top1_rate": row["relation_top1_rate"],
                "both_top1_rate": row["both_top1_rate"],
                "hard_and_correct_rate": row["hard_and_correct_rate"],
                "hard_and_fire_rate": row["hard_and_fire_rate"],
                "subject_found_rate": row["subject_found_rate"],
                "subject_agreement_mean": row["subject_agreement_mean"],
                "relation_agreement_mean": row["relation_agreement_mean"],
                "family_buckets": row["family_buckets"],
            }
        )
    write_json(output_dir / "subject_probe_compact.json", compact)


if __name__ == "__main__":
    main()
