import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor

from examples.edit_experiment_utils import (
    backbone_slug,
    mean_optional,
    metric_mean,
    method_name,
    nested_acc_mean,
    resolve_hparams_class,
    write_json,
)
from examples.run_wikibigedit_lifelong import (
    INCREMENTS,
    build_editor_inputs,
    compute_pre_metrics,
    configure_evaluation_mode,
    load_increment_records,
    normalize_wikibigedit_records,
    sample_records,
    seed_everything,
    VALID_EVALUATION_MODES,
)


def summarize_pre_metrics(records: list[dict], pre_metrics: list[dict]) -> dict:
    per_case = []
    for idx, (record, metric) in enumerate(zip(records, pre_metrics)):
        pre = metric.get("pre", {})
        locality_keys = []
        if isinstance(pre.get("locality"), dict):
            locality_keys = sorted(pre["locality"].keys())
        per_case.append(
            {
                "case_id": idx,
                "source_index": record.get("source_index", idx),
                "subject": record.get("subject"),
                "prompt": record.get("prompt"),
                "target_new": record.get("target_new"),
                "pre_rewrite_acc": metric_mean(pre.get("rewrite_acc")),
                "pre_rephrase_acc": metric_mean(pre.get("rephrase_acc")),
                "pre_portability_acc": nested_acc_mean(pre.get("portability")),
                "pre_locality_reference_keys": locality_keys,
            }
        )

    return {
        "stream_length": len(per_case),
        "pre_rewrite_mean": mean_optional([row["pre_rewrite_acc"] for row in per_case]),
        "pre_rephrase_mean": mean_optional([row["pre_rephrase_acc"] for row in per_case]),
        "pre_portability_mean": mean_optional([row["pre_portability_acc"] for row in per_case]),
        "pre_locality_reference_cases": sum(1 for row in per_case if row["pre_locality_reference_keys"]),
        "per_case": per_case,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", required=True, type=str)
    parser.add_argument("--hparams_dir", required=True, type=str)
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"), type=str)
    parser.add_argument("--increment_dir", default=None, type=str)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "wikibigedit_pre_eval"), type=str)
    parser.add_argument("--increments", default=" ".join(INCREMENTS), type=str)
    parser.add_argument("--ds_size_per_increment", default=0, type=int)
    parser.add_argument("--evaluation_mode", default="teacher_forcing", choices=sorted(VALID_EVALUATION_MODES), type=str)
    parser.add_argument("--eval_batch_size", default=32, type=int)
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
    hparams.eval_batch_size = args.eval_batch_size
    configure_evaluation_mode(hparams, args.evaluation_mode, args.api_key)

    increment_dir = Path(args.increment_dir) if args.increment_dir else Path(args.data_dir) / "wikibigedit"
    requested_increments = [token for token in args.increments.split() if token]
    backbone = backbone_slug(hparams.model_name)
    output_root = Path(args.output_root) / f"{method.lower()}_{backbone}"
    output_root.mkdir(parents=True, exist_ok=True)

    write_json(
        output_root / "run_manifest.json",
        {
            "editing_method": method,
            "alg_name": hparams.alg_name,
            "model_name": hparams.model_name,
            "backbone": backbone,
            "increment_dir": str(increment_dir.resolve()),
            "increments": requested_increments,
            "ds_size_per_increment": args.ds_size_per_increment,
            "evaluation_mode": args.evaluation_mode,
            "eval_batch_size": args.eval_batch_size,
            "seed": args.seed,
            "hparams_path": str(Path(args.hparams_dir).resolve()),
        },
    )

    editor = BaseEditor.from_hparams(hparams)
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
                "evaluation_mode": args.evaluation_mode,
            },
        )

        print(
            json.dumps(
                {
                    "phase": "pre_eval_only",
                    "increment": increment,
                    "records": len(current_records),
                    "evaluation_mode": args.evaluation_mode,
                }
            ),
            flush=True,
        )

        eval_start = time.time()
        pre_metrics = compute_pre_metrics(editor, current_requests, eval_metric)
        eval_seconds = time.time() - eval_start
        write_json(increment_dir_out / "pre_metrics.json", pre_metrics)
        write_json(
            increment_dir_out / "pre_eval_summary.json",
            {
                **summarize_pre_metrics(current_records, pre_metrics),
                "eval_wall_time_seconds": eval_seconds,
                "wall_time_seconds": time.time() - start_time,
            },
        )
        print(
            json.dumps(
                {
                    "phase": "pre_eval_only_complete",
                    "increment": increment,
                    "records": len(pre_metrics),
                    "eval_wall_time_seconds": eval_seconds,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
