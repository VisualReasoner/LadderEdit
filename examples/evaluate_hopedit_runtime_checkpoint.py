import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor
from easyeditor.models.hopedit.hopedit_main import load_hopedit_runtime_checkpoint

from examples.edit_experiment_utils import backbone_slug, method_name, resolve_hparams_class, write_json
from examples.run_wikibigedit_lifelong import (
    evaluate_and_write,
    load_increment_records,
    normalize_wikibigedit_records,
    build_editor_inputs,
    configure_evaluation_mode,
    resolve_pre_metrics_path,
    seed_everything,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT", type=str)
    parser.add_argument("--hparams_dir", required=True, type=str)
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"), type=str)
    parser.add_argument("--runtime_checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--increment", default=None, type=str)
    parser.add_argument("--checkpoint_step", default=None, type=int)
    parser.add_argument("--evaluation_mode", default="teacher_forcing", type=str)
    parser.add_argument("--pre_metrics_root", default=None, type=str)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--api_key", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    seed_everything(args.seed)

    runtime_checkpoint = Path(args.runtime_checkpoint)
    runner_state = json.loads((runtime_checkpoint / "runner_state.json").read_text())
    increment = args.increment or runner_state["increment"]
    checkpoint_step = args.checkpoint_step or int(runner_state["checkpoint_step"])

    method = method_name(args.editing_method)
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    hparams.sequential_edit = True
    hparams.eval_batch_size = args.eval_batch_size
    configure_evaluation_mode(hparams, args.evaluation_mode, args.api_key)

    editor = BaseEditor.from_hparams(hparams)
    editor.model = load_hopedit_runtime_checkpoint(
        editor.model,
        editor.tok,
        hparams,
        str(runtime_checkpoint),
        is_trainable=True,
    )

    increment_dir = Path(args.data_dir) / "wikibigedit"
    raw_records, dataset_file = load_increment_records(increment_dir, increment)
    current_records = normalize_wikibigedit_records(raw_records)[:checkpoint_step]
    current_requests, eval_metric = build_editor_inputs(current_records)

    pre_metrics = None
    if args.pre_metrics_root is not None:
        pre_metrics_path = resolve_pre_metrics_path(Path(args.pre_metrics_root), method, backbone_slug(hparams.model_name), increment)
        pre_metrics = json.loads(pre_metrics_path.read_text())[:checkpoint_step]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "run_name": f"{method.lower()}_{backbone_slug(hparams.model_name)}_{increment}_deferred_eval_{checkpoint_step:06d}",
        "editing_method": method,
        "alg_name": hparams.alg_name,
        "model_name": hparams.model_name,
        "backbone": backbone_slug(hparams.model_name),
        "data_type": "WikiBigEdit",
        "dataset_file": str(dataset_file.resolve()),
        "stream_type": "deferred_checkpoint_eval",
        "seed": args.seed,
        "evaluation_mode": args.evaluation_mode,
        "sequential_edit": True,
        "stream_length": len(current_records),
        "requested_ds_size": len(current_records),
        "hparams_path": str(Path(args.hparams_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "wall_time_seconds": None,
        "increment_edit_wall_time_seconds": None,
        "increment": increment,
        "increment_index": None,
        "cumulative_edits": checkpoint_step,
        "checkpoint_step": checkpoint_step,
        "runtime_checkpoint": str(runtime_checkpoint.resolve()),
    }

    eval_start = time.time()
    summary = evaluate_and_write(
        editor,
        hparams,
        method,
        backbone_slug(hparams.model_name),
        output_dir,
        current_records,
        current_requests,
        eval_metric,
        run_config,
        pre_metrics=pre_metrics,
    )
    write_json(
        output_dir / "deferred_eval_status.json",
        {
            "increment": increment,
            "checkpoint_step": checkpoint_step,
            "evaluation_mode": args.evaluation_mode,
            "eval_wall_time_seconds": time.time() - eval_start,
            "summary_path": str((output_dir / "summary.json").resolve()),
            "summary": summary,
        },
    )


if __name__ == "__main__":
    main()
