import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor
from easyeditor.editors.utils import _prepare_requests
from easyeditor.models.hopedit.hopedit_main import load_hopedit_runtime_checkpoint

from examples.checkpoint_suite_utils import update_checkpoint_manifest_status, write_json
from examples.edit_experiment_utils import (
    backbone_slug,
    build_editor_inputs,
    load_normalized_records,
    method_name,
    resolve_hparams_class,
)
from examples.run_wikibigedit_lifelong import (
    build_editor_inputs as build_wikibigedit_inputs,
    configure_evaluation_mode,
    evaluate_and_write,
    load_increment_records,
    normalize_wikibigedit_records,
    resolve_pre_metrics_path,
    seed_everything,
)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_indices(index_file: str | None) -> list[int] | None:
    if not index_file:
        return None
    payload = json.loads(Path(index_file).read_text())
    if isinstance(payload, dict):
        payload = payload.get("selected_indices") or payload.get("indices")
    if payload is None:
        raise ValueError(f"No indices found in {index_file}")
    return [int(idx) for idx in payload]


def build_requests_from_manifest(payload: dict):
    data_type = payload["data_type"]
    checkpoint_step = int(payload["checkpoint_step"])
    if data_type == "WikiBigEdit":
        increment_dir = Path(payload["increment_dir"])
        increment = payload["increment"]
        raw_records, dataset_file = load_increment_records(increment_dir, increment)
        records = normalize_wikibigedit_records(raw_records)[:checkpoint_step]
        requests, eval_metric = build_wikibigedit_inputs(records)
    else:
        indices = load_indices(payload.get("index_file"))
        if indices is not None:
            indices = indices[:checkpoint_step]
        records, dataset_file = load_normalized_records(
            payload["data_dir"],
            data_type,
            checkpoint_step,
            indices=indices,
            data_file=payload.get("data_file"),
        )
        editor_inputs = build_editor_inputs(records, data_type)
        requests = _prepare_requests(
            editor_inputs["prompts"],
            editor_inputs["target_new"],
            editor_inputs["ground_truth"],
            rephrase_prompts=editor_inputs["rephrase_prompts"],
            locality_inputs=editor_inputs["locality_inputs"],
            portability_inputs=editor_inputs["portability_inputs"],
            subject=editor_inputs["subject"],
        )
        for idx, request in enumerate(requests):
            request.setdefault("case_id", idx)
        eval_metric = editor_inputs["eval_metric"]
    return records, requests, eval_metric, dataset_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deferred_manifest", required=True, type=str)
    parser.add_argument("--evaluation_mode", default=None, type=str)
    parser.add_argument("--api_key", default=None, type=str)
    args = parser.parse_args()

    manifest_path = Path(args.deferred_manifest)
    payload = load_manifest(manifest_path)
    evaluation_mode = args.evaluation_mode or payload.get("evaluation_mode", "teacher_forcing")

    seed_everything(int(payload.get("seed", 0)))

    method = method_name(payload["editing_method"])
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(payload["hparams_path"])
    hparams.sequential_edit = True
    hparams.eval_batch_size = int(payload.get("eval_batch_size", 32))
    configure_evaluation_mode(hparams, evaluation_mode, args.api_key)

    editor = BaseEditor.from_hparams(hparams)
    runtime_checkpoint = Path(payload["runtime_checkpoint"])
    editor.model = load_hopedit_runtime_checkpoint(
        editor.model,
        editor.tok,
        hparams,
        str(runtime_checkpoint),
        is_trainable=True,
    )

    records, requests, eval_metric, dataset_file = build_requests_from_manifest(payload)

    pre_metrics = None
    pre_metrics_root = payload.get("pre_metrics_root")
    if pre_metrics_root:
        if payload["data_type"] == "WikiBigEdit":
            increment = payload["increment"]
            pre_metrics_path = resolve_pre_metrics_path(
                Path(pre_metrics_root),
                method,
                backbone_slug(hparams.model_name),
                increment,
            )
            pre_metrics = json.loads(pre_metrics_path.read_text())[: int(payload["checkpoint_step"])]

    checkpoint_dir = manifest_path.parent
    run_config = {
        "run_name": f"{method.lower()}_{backbone_slug(hparams.model_name)}_{payload['data_type'].lower()}_deferred_{int(payload['checkpoint_step']):06d}",
        "editing_method": method,
        "alg_name": hparams.alg_name,
        "model_name": hparams.model_name,
        "backbone": backbone_slug(hparams.model_name),
        "data_type": payload["data_type"],
        "dataset_file": str(Path(dataset_file).resolve()),
        "stream_type": "deferred_checkpoint_eval",
        "seed": int(payload.get("seed", 0)),
        "evaluation_mode": evaluation_mode,
        "sequential_edit": True,
        "stream_length": len(records),
        "requested_ds_size": len(records),
        "hparams_path": str(Path(payload["hparams_path"]).resolve()),
        "output_dir": str(checkpoint_dir.resolve()),
        "checkpoint_step": int(payload["checkpoint_step"]),
        "runtime_checkpoint": str(runtime_checkpoint.resolve()),
    }

    eval_start = time.time()
    summary = evaluate_and_write(
        editor,
        hparams,
        method,
        backbone_slug(hparams.model_name),
        checkpoint_dir,
        records,
        requests,
        eval_metric,
        run_config,
        pre_metrics=pre_metrics,
    )
    status_path = checkpoint_dir / "deferred_eval_status.json"
    write_json(
        status_path,
        {
            "dataset": payload["data_type"],
            "checkpoint_step": int(payload["checkpoint_step"]),
            "evaluation_mode": evaluation_mode,
            "eval_wall_time_seconds": time.time() - eval_start,
            "runtime_checkpoint": str(runtime_checkpoint.resolve()),
            "summary_path": str((checkpoint_dir / "summary.json").resolve()),
            "summary": summary,
        },
    )
    update_checkpoint_manifest_status(
        checkpoint_dir,
        evaluation_mode=evaluation_mode,
        status="done",
        summary_path=checkpoint_dir / "summary.json",
        eval_status_path=status_path,
    )


if __name__ == "__main__":
    main()
