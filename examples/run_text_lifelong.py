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

from examples.checkpoint_suite_utils import (
    build_checkpoint_manifest,
    infer_memory_semantics,
    update_checkpoint_manifest_status,
    write_checkpoint_manifest,
)
from examples.edit_experiment_utils import (
    backbone_slug,
    build_editor_inputs,
    load_normalized_records,
    method_name,
    resolve_hparams_class,
    write_json,
)
from examples.run_wikibigedit_lifelong import (
    VALID_EVALUATION_MODES,
    VALID_EVAL_POLICIES,
    VALID_PRE_EVAL_MODES,
    apply_single_edit,
    compute_pre_metrics,
    configure_evaluation_mode,
    evaluate_and_write,
    maybe_gpu_keepalive,
    restore_evaluation_state,
    sample_records,
    save_runtime_checkpoint_if_supported,
    seed_everything,
    snapshot_evaluation_state,
    write_deferred_eval_manifest,
    directory_size_bytes,
)


def parse_checkpoint_counts(raw: str, ds_size: int) -> list[int]:
    counts = sorted({int(token) for token in raw.split(",") if token.strip()})
    return [count for count in counts if 0 < count <= ds_size]


def load_indices(index_file: str | None) -> list[int] | None:
    if index_file is None:
        return None
    payload = json.loads(Path(index_file).read_text())
    if isinstance(payload, dict):
        payload = payload.get("selected_indices") or payload.get("indices")
    if payload is None:
        raise ValueError(f"No indices found in {index_file}")
    return [int(idx) for idx in payload]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", required=True, type=str)
    parser.add_argument("--hparams_dir", required=True, type=str)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--data_type", required=True, choices=["ZsRE", "CounterFact", "Hallucination", "MQuAKE"], type=str)
    parser.add_argument("--data_file", default=None, type=str)
    parser.add_argument("--index_file", default=None, type=str)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "text_lifelong"), type=str)
    parser.add_argument("--ds_size", default=1000, type=int)
    parser.add_argument("--checkpoint_counts", default="1000", type=str)
    parser.add_argument("--evaluation_mode", default="teacher_forcing", choices=sorted(VALID_EVALUATION_MODES), type=str)
    parser.add_argument("--pre_eval_mode", default="match", choices=sorted(VALID_PRE_EVAL_MODES), type=str)
    parser.add_argument("--resume_runtime_checkpoint", default=None, type=str)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint_eval_policy", default="skip", choices=sorted(VALID_EVAL_POLICIES), type=str)
    parser.add_argument("--current_eval_policy", default="skip", choices=sorted(VALID_EVAL_POLICIES), type=str)
    parser.add_argument("--gpu_keepalive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_keepalive_interval_seconds", default=10.0, type=float)
    parser.add_argument("--gpu_keepalive_burst_seconds", default=0.5, type=float)
    parser.add_argument("--gpu_keepalive_matrix_size", default=2048, type=int)
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
    pre_eval_mode = args.evaluation_mode if args.pre_eval_mode == "match" else args.pre_eval_mode

    indices = load_indices(args.index_file)
    records, dataset_file = load_normalized_records(
        args.data_dir,
        args.data_type,
        args.ds_size,
        indices=indices,
        data_file=args.data_file,
    )
    editor_inputs = build_editor_inputs(records, args.data_type)
    editor_requests = _prepare_requests(
        editor_inputs["prompts"],
        editor_inputs["target_new"],
        editor_inputs["ground_truth"],
        rephrase_prompts=editor_inputs["rephrase_prompts"],
        locality_inputs=editor_inputs["locality_inputs"],
        portability_inputs=editor_inputs["portability_inputs"],
        subject=editor_inputs["subject"],
    )
    for idx, request in enumerate(editor_requests):
        request.setdefault("case_id", idx)
    checkpoint_counts = parse_checkpoint_counts(args.checkpoint_counts, len(records))

    backbone = backbone_slug(hparams.model_name)
    output_root = Path(args.output_root) / f"{method.lower()}_{backbone}_{args.data_type.lower()}"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "editing_method": method,
        "alg_name": hparams.alg_name,
        "model_name": hparams.model_name,
        "backbone": backbone,
        "data_type": args.data_type,
        "data_file": str(dataset_file.resolve()),
        "requested_ds_size": args.ds_size,
        "records_used": len(records),
        "checkpoint_counts": checkpoint_counts,
        "seed": args.seed,
        "evaluation_mode": args.evaluation_mode,
        "pre_eval_mode": pre_eval_mode,
        "eval_batch_size": args.eval_batch_size,
        "checkpoint_eval_policy": args.checkpoint_eval_policy,
        "current_eval_policy": args.current_eval_policy,
        "hparams_path": str(Path(args.hparams_dir).resolve()),
        "index_file": None if args.index_file is None else str(Path(args.index_file).resolve()),
        "eval_status": {"teacher_forcing": "pending", "non_teacher_forcing": "pending"},
    }
    write_json(output_root / "run_manifest.json", manifest)

    editor = BaseEditor.from_hparams(hparams)
    resumed_edits = 0
    if args.resume_runtime_checkpoint is not None:
        checkpoint_dir = Path(args.resume_runtime_checkpoint)
        runner_state = json.loads((checkpoint_dir / "runner_state.json").read_text())
        resumed_edits = int(runner_state.get("checkpoint_step") or 0)
        if method != "HOPEDIT":
            raise ValueError("--resume_runtime_checkpoint is currently supported only for HOPEDIT")
        resume_start = time.time()
        with maybe_gpu_keepalive(args, hparams, f"resume_load_{args.data_type}_{resumed_edits:06d}"):
            editor.model = load_hopedit_runtime_checkpoint(editor.model, editor.tok, hparams, str(checkpoint_dir), is_trainable=True)
        manifest["resume_runtime_checkpoint"] = str(checkpoint_dir.resolve())
        manifest["resume_runtime_checkpoint_load_seconds"] = time.time() - resume_start
        manifest["resume_runtime_checkpoint_size_bytes"] = directory_size_bytes(checkpoint_dir)
        write_json(output_root / "run_manifest.json", manifest)

    pre_metrics = None
    if pre_eval_mode != "skip":
        pre_eval_state = snapshot_evaluation_state(editor.hparams)
        configure_evaluation_mode(editor.hparams, pre_eval_mode, args.api_key)
        try:
            with maybe_gpu_keepalive(args, editor.hparams, f"pre_eval_{args.data_type}"):
                pre_metrics = compute_pre_metrics(editor, editor_requests, editor_inputs["eval_metric"])
        finally:
            restore_evaluation_state(editor.hparams, pre_eval_state)
        write_json(output_root / "pre_metrics.json", pre_metrics)

    start_time = time.time()
    next_checkpoint_idx = 0
    while next_checkpoint_idx < len(checkpoint_counts) and checkpoint_counts[next_checkpoint_idx] <= resumed_edits:
        next_checkpoint_idx += 1

    for request_idx, request in enumerate(editor_requests[resumed_edits:], start=resumed_edits + 1):
        apply_single_edit(editor, request)
        if next_checkpoint_idx < len(checkpoint_counts) and request_idx == checkpoint_counts[next_checkpoint_idx]:
            checkpoint_output_dir = output_root / f"checkpoint_{request_idx:06d}" / "current"
            checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
            with maybe_gpu_keepalive(args, editor.hparams, f"checkpoint_{args.data_type}_{request_idx:06d}"):
                runtime = save_runtime_checkpoint_if_supported(
                    editor,
                    checkpoint_output_dir,
                    {
                        "data_type": args.data_type,
                        "checkpoint_step": request_idx,
                        "checkpoint_counts": checkpoint_counts,
                    },
                )
                semantics = infer_memory_semantics(
                    method,
                    controller=editor.model if hasattr(editor, "model") else None,
                )
                checkpoint_manifest = build_checkpoint_manifest(
                    dataset=args.data_type,
                    split_or_increment="main",
                    backbone=backbone,
                    method=method,
                    hopedit_mode=getattr(editor.model, "hopedit_mode", None),
                    assignment_policy=getattr(editor.model, "cell_assignment_policy", None),
                    cell_budget=getattr(editor.hparams, "cell_budget", None),
                    edit_count=request_idx,
                    checkpoint_path=checkpoint_output_dir,
                    checkpoint_size_bytes=None if runtime is None else runtime.get("checkpoint_size_bytes"),
                    checkpoint_load_seconds=manifest.get("resume_runtime_checkpoint_load_seconds"),
                    runtime_checkpoint_path=None if runtime is None else runtime.get("checkpoint_dir"),
                    saved_memory_semantics=semantics,
                    evaluation_mode=args.evaluation_mode,
                    extra={
                        "dataset_file": str(dataset_file.resolve()),
                        "hparams_path": str(Path(args.hparams_dir).resolve()),
                        "seed": args.seed,
                    },
                )
                write_checkpoint_manifest(checkpoint_output_dir, checkpoint_manifest)
                if args.checkpoint_eval_policy == "full":
                    checkpoint_summary = evaluate_and_write(
                        editor,
                        hparams,
                        method,
                        backbone,
                        checkpoint_output_dir,
                        records[:request_idx],
                        editor_requests[:request_idx],
                        editor_inputs["eval_metric"],
                        {
                            "run_name": f"{method.lower()}_{backbone}_{args.data_type.lower()}_{request_idx:06d}",
                            "editing_method": method,
                            "alg_name": hparams.alg_name,
                            "model_name": hparams.model_name,
                            "backbone": backbone,
                            "data_type": args.data_type,
                            "dataset_file": str(dataset_file.resolve()),
                            "stream_type": "checkpoint_eval",
                            "seed": args.seed,
                            "evaluation_mode": args.evaluation_mode,
                            "sequential_edit": True,
                            "stream_length": request_idx,
                            "requested_ds_size": request_idx,
                            "hparams_path": str(Path(args.hparams_dir).resolve()),
                            "output_dir": str(checkpoint_output_dir.resolve()),
                            "wall_time_seconds": time.time() - start_time,
                            "checkpoint_step": request_idx,
                            "checkpoint_save_seconds": None if runtime is None else runtime.get("checkpoint_save_seconds"),
                            "checkpoint_size_bytes": None if runtime is None else runtime.get("checkpoint_size_bytes"),
                            "checkpoint_load_seconds": manifest.get("resume_runtime_checkpoint_load_seconds"),
                        },
                        pre_metrics=None if pre_metrics is None else pre_metrics[:request_idx],
                    )
                    update_checkpoint_manifest_status(
                        checkpoint_output_dir,
                        evaluation_mode=args.evaluation_mode,
                        status="done",
                        summary_path=checkpoint_output_dir / "summary.json",
                        eval_status_path=checkpoint_output_dir / "deferred_eval_status.json",
                    )
                else:
                    write_deferred_eval_manifest(
                        checkpoint_output_dir,
                        {
                            "phase": "checkpoint_eval_deferred",
                            "editing_method": method,
                            "hparams_path": str(Path(args.hparams_dir).resolve()),
                            "data_type": args.data_type,
                            "data_dir": str(Path(args.data_dir).resolve()),
                            "data_file": str(dataset_file.resolve()),
                            "checkpoint_step": request_idx,
                            "runtime_checkpoint": None if runtime is None else str(runtime["checkpoint_dir"].resolve()),
                            "eval_batch_size": args.eval_batch_size,
                            "evaluation_mode": args.evaluation_mode,
                            "seed": args.seed,
                            "index_file": None if args.index_file is None else str(Path(args.index_file).resolve()),
                        },
                    )
            next_checkpoint_idx += 1

    current_output_dir = output_root / "current"
    current_output_dir.mkdir(parents=True, exist_ok=True)
    with maybe_gpu_keepalive(args, editor.hparams, f"current_{args.data_type}"):
        current_runtime = save_runtime_checkpoint_if_supported(
            editor,
            current_output_dir,
            {
                "data_type": args.data_type,
                "checkpoint_step": len(records),
                "checkpoint_counts": checkpoint_counts,
            },
        )
        current_manifest = build_checkpoint_manifest(
            dataset=args.data_type,
            split_or_increment="main",
            backbone=backbone,
            method=method,
            hopedit_mode=getattr(editor.model, "hopedit_mode", None),
            assignment_policy=getattr(editor.model, "cell_assignment_policy", None),
            cell_budget=getattr(editor.hparams, "cell_budget", None),
            edit_count=len(records),
            checkpoint_path=current_output_dir,
            checkpoint_size_bytes=None if current_runtime is None else current_runtime.get("checkpoint_size_bytes"),
            checkpoint_load_seconds=manifest.get("resume_runtime_checkpoint_load_seconds"),
            runtime_checkpoint_path=None if current_runtime is None else current_runtime.get("checkpoint_dir"),
            saved_memory_semantics=infer_memory_semantics(method, controller=editor.model if hasattr(editor, "model") else None),
            evaluation_mode=args.evaluation_mode,
            extra={
                "dataset_file": str(dataset_file.resolve()),
                "hparams_path": str(Path(args.hparams_dir).resolve()),
                "seed": args.seed,
            },
        )
        write_checkpoint_manifest(current_output_dir, current_manifest)
        if args.current_eval_policy == "full":
            evaluate_and_write(
                editor,
                hparams,
                method,
                backbone,
                current_output_dir,
                records,
                editor_requests,
                editor_inputs["eval_metric"],
                {
                    "run_name": f"{method.lower()}_{backbone}_{args.data_type.lower()}_current",
                    "editing_method": method,
                    "alg_name": hparams.alg_name,
                    "model_name": hparams.model_name,
                    "backbone": backbone,
                    "data_type": args.data_type,
                    "dataset_file": str(dataset_file.resolve()),
                    "stream_type": "current",
                    "seed": args.seed,
                    "evaluation_mode": args.evaluation_mode,
                    "sequential_edit": True,
                    "stream_length": len(records),
                    "requested_ds_size": len(records),
                    "hparams_path": str(Path(args.hparams_dir).resolve()),
                    "output_dir": str(current_output_dir.resolve()),
                    "wall_time_seconds": time.time() - start_time,
                    "checkpoint_step": len(records),
                    "checkpoint_save_seconds": None if current_runtime is None else current_runtime.get("checkpoint_save_seconds"),
                    "checkpoint_size_bytes": None if current_runtime is None else current_runtime.get("checkpoint_size_bytes"),
                    "checkpoint_load_seconds": manifest.get("resume_runtime_checkpoint_load_seconds"),
                },
                pre_metrics=pre_metrics,
            )
            update_checkpoint_manifest_status(
                current_output_dir,
                evaluation_mode=args.evaluation_mode,
                status="done",
                summary_path=current_output_dir / "summary.json",
                eval_status_path=current_output_dir / "deferred_eval_status.json",
            )
        else:
            write_deferred_eval_manifest(
                current_output_dir,
                {
                    "phase": "current_eval_deferred",
                    "editing_method": method,
                    "hparams_path": str(Path(args.hparams_dir).resolve()),
                    "data_type": args.data_type,
                    "data_dir": str(Path(args.data_dir).resolve()),
                    "data_file": str(dataset_file.resolve()),
                    "checkpoint_step": len(records),
                    "runtime_checkpoint": None if current_runtime is None else str(current_runtime["checkpoint_dir"].resolve()),
                    "eval_batch_size": args.eval_batch_size,
                    "evaluation_mode": args.evaluation_mode,
                    "seed": args.seed,
                    "index_file": None if args.index_file is None else str(Path(args.index_file).resolve()),
                },
            )


if __name__ == "__main__":
    main()
