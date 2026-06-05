import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor
from easyeditor.evaluate import compute_edit_quality
from easyeditor.models.hopedit.hopedit_main import load_hopedit_runtime_checkpoint

from examples.edit_experiment_utils import backbone_slug, method_name, resolve_hparams_class
from examples.run_wikibigedit_lifelong import (
    build_editor_inputs,
    evaluate_requests,
    load_increment_records,
    normalize_wikibigedit_records,
    sample_records,
    seed_everything,
)


def evaluate_requests_serial(editor: BaseEditor, requests: list[dict], eval_metric: str):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT", type=str)
    parser.add_argument("--hparams_dir", required=True, type=str)
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"), type=str)
    parser.add_argument("--increment", required=True, type=str)
    parser.add_argument("--runtime_checkpoint", required=True, type=str)
    parser.add_argument("--sample_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    seed_everything(args.seed)

    method = method_name(args.editing_method)
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    hparams.sequential_edit = True
    hparams.eval_batch_size = args.eval_batch_size
    if hasattr(hparams, "evaluation_type"):
        delattr(hparams, "evaluation_type")
    if hasattr(hparams, "api_key"):
        delattr(hparams, "api_key")

    editor = BaseEditor.from_hparams(hparams)
    editor.model = load_hopedit_runtime_checkpoint(
        editor.model,
        editor.tok,
        hparams,
        args.runtime_checkpoint,
        is_trainable=True,
    )

    increment_dir = Path(args.data_dir) / "wikibigedit"
    raw_records, _dataset_file = load_increment_records(increment_dir, args.increment)
    sampled_records = sample_records(raw_records, args.sample_size, args.seed)
    current_records = normalize_wikibigedit_records(sampled_records)
    current_requests, eval_metric = build_editor_inputs(current_records)

    start = time.time()
    serial_metrics = evaluate_requests_serial(editor, current_requests, eval_metric)
    serial_seconds = time.time() - start

    start = time.time()
    batched_metrics = evaluate_requests(editor, current_requests, eval_metric)
    batched_seconds = time.time() - start

    rewrite_pairs = []
    for serial_row, batched_row in zip(serial_metrics, batched_metrics):
        rewrite_pairs.append(
            {
                "serial": serial_row["post"].get("rewrite_acc"),
                "batched": batched_row["post"].get("rewrite_acc"),
            }
        )

    print(
        json.dumps(
            {
                "editing_method": method,
                "backbone": backbone_slug(hparams.model_name),
                "increment": args.increment,
                "runtime_checkpoint": str(Path(args.runtime_checkpoint).resolve()),
                "sample_size": len(current_requests),
                "eval_batch_size": args.eval_batch_size,
                "serial_seconds": serial_seconds,
                "serial_seconds_per_case": serial_seconds / max(len(current_requests), 1),
                "batched_seconds": batched_seconds,
                "batched_seconds_per_case": batched_seconds / max(len(current_requests), 1),
                "speedup_over_serial": None if batched_seconds <= 0 else serial_seconds / batched_seconds,
                "rewrite_examples": rewrite_pairs[:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
