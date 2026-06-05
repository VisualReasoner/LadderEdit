import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
MEMOIR_ROOT = WORKSPACE_ROOT / "MEMOIR"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.edit_experiment_utils import (  # noqa: E402
    backbone_slug,
    build_editor_inputs,
    canonical_run_name,
    load_normalized_records,
    summarize_run,
    write_json,
    write_jsonl,
)


def seed_everything(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _import_memoir_easyeditor():
    if str(MEMOIR_ROOT) not in sys.path:
        sys.path.insert(0, str(MEMOIR_ROOT))
    if "easyeditor" in sys.modules:
        del sys.modules["easyeditor"]
    from easyeditor import BaseEditor, MEMOIRHyperParams  # type: ignore

    return BaseEditor, MEMOIRHyperParams


def _memory_snapshot(edited_model):
    if edited_model is None:
        return []
    if not hasattr(edited_model, "get_adapter_layer"):
        return []
    try:
        adapter = edited_model.get_adapter_layer()
    except Exception:
        return []
    masks = getattr(adapter, "masks_for_edited_samples", None)
    if masks is None:
        return []
    if hasattr(masks, "shape") and len(getattr(masks, "shape", [])) >= 2:
        count = int(masks.shape[0])
    elif isinstance(masks, list):
        count = len(masks)
    else:
        count = 0
    return [{"entry_id": idx, "kind": "memoir_mask"} for idx in range(count)]


def _memoir_memory_audit(edited_model, hparams, stream_length: int):
    audit = {
        "applicable": False,
        "memory_model": "memoir_dense_residual_plus_retrieval_masks",
        "stream_length": int(stream_length),
        "top_k": int(getattr(hparams, "top_k", 0) or 0),
    }
    if edited_model is None or not hasattr(edited_model, "get_adapter_layer"):
        return audit
    try:
        adapter = edited_model.get_adapter_layer()
    except Exception as exc:
        audit["error"] = repr(exc)
        return audit

    new_weight = getattr(adapter, "new_weight", None)
    masks = getattr(adapter, "masks_for_edited_samples", None)
    dense_params = int(new_weight.numel()) if hasattr(new_weight, "numel") else 0
    dense_nonzero_params = int((new_weight.detach() != 0).sum().item()) if hasattr(new_weight, "detach") else None
    feature_dim = int(new_weight.shape[1]) if hasattr(new_weight, "shape") and len(new_weight.shape) >= 2 else None
    output_dim = int(new_weight.shape[0]) if hasattr(new_weight, "shape") and len(new_weight.shape) >= 2 else None
    if hasattr(masks, "shape") and len(getattr(masks, "shape", [])) >= 2:
        mask_count = int(masks.shape[0])
        mask_dim = int(masks.shape[1])
        dense_mask_bits = int(mask_count * mask_dim)
    elif isinstance(masks, list):
        mask_count = len(masks)
        mask_dim = feature_dim
        dense_mask_bits = None if mask_dim is None else int(mask_count * mask_dim)
    else:
        mask_count = 0
        mask_dim = feature_dim
        dense_mask_bits = 0

    top_k = int(getattr(hparams, "top_k", 0) or 0)
    sparse_mask_indices = int(mask_count * top_k)
    # Conservative logical accounting in scalar slots. We keep dense mask bits
    # separate because they are bit/boolean storage, not fp params.
    audit.update(
        {
            "applicable": True,
            "edited_layer": (getattr(hparams, "inner_params", None) or [None])[0],
            "dense_residual_params": dense_params,
            "dense_residual_nonzero_params": dense_nonzero_params,
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "mask_count": mask_count,
            "mask_dim": mask_dim,
            "dense_mask_bits": dense_mask_bits,
            "sparse_mask_index_count": sparse_mask_indices,
            "logical_params_dense_residual_only": dense_params,
            "logical_params_dense_residual_plus_sparse_indices": int(dense_params + sparse_mask_indices),
            "memory_note": (
                "MEMOIR stores a shared dense residual matrix for the edited layer plus retrieval masks. "
                "Mask storage is reported both as dense bits and sparse top-k indices; compare with care "
                "against LoRA parameter counts."
            ),
        }
    )
    return audit


def build_efficiency_metrics(run_config, summary, memory_snapshot):
    stream_length = run_config.get("stream_length")
    wall_time_seconds = run_config.get("wall_time_seconds")
    memory_entries_final = len(memory_snapshot)
    return {
        "applicable": True,
        "editing_method": run_config["editing_method"],
        "run_name": run_config["run_name"],
        "stream_length": stream_length,
        "wall_time_seconds": wall_time_seconds,
        "edit_seconds_per_edit": None
        if not wall_time_seconds or not stream_length
        else float(wall_time_seconds / stream_length),
        "memory_unit": "edit_mask",
        "memory_entries_final": memory_entries_final,
        "retained_units_final": memory_entries_final,
        "checkpoint_save_seconds": None,
        "checkpoint_load_seconds": None,
        "checkpoint_size_bytes": None,
        "bytes_per_retained_edit": None,
        "bytes_per_retained_unit": None,
        "route_events_logged": None,
        "mean_case_time_reported": summary.get("mean_time"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_dir", required=True, type=str)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--data_type", required=True, type=str, choices=["ZsRE", "CounterFact"])
    parser.add_argument("--data_file", default=None, type=str)
    parser.add_argument("--output_root", default="./outputs/memoir_baselines", type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--ds_size", default=16, type=int)
    parser.add_argument("--stream_type", default="budget_poc", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--sequential_edit", action="store_true")
    args = parser.parse_args()

    BaseEditor, MEMOIRHyperParams = _import_memoir_easyeditor()
    seed_everything(args.seed)
    hparams = MEMOIRHyperParams.from_hparams(args.hparams_dir)

    records, dataset_file = load_normalized_records(
        args.data_dir,
        args.data_type,
        args.ds_size,
        data_file=args.data_file,
    )
    editor_inputs = build_editor_inputs(records, args.data_type)

    stream_length = len(records)
    run_name = args.run_name or canonical_run_name("MEMOIR", hparams.model_name, args.data_type, args.stream_type, stream_length, args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_name": run_name,
        "editing_method": "MEMOIR",
        "alg_name": hparams.alg_name,
        "model_name": hparams.model_name,
        "backbone": backbone_slug(hparams.model_name),
        "data_type": args.data_type,
        "dataset_file": str(dataset_file),
        "stream_type": args.stream_type,
        "seed": args.seed,
        "sequential_edit": bool(args.sequential_edit),
        "stream_length": stream_length,
        "requested_ds_size": args.ds_size,
        "hparams_path": str(Path(args.hparams_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    write_json(output_dir / "run_config.json", run_config)

    editor = BaseEditor.from_hparams(hparams)
    start_time = time.time()
    metrics, edited_model, _ = editor.edit(
        prompts=editor_inputs["prompts"],
        target_new=editor_inputs["target_new"],
        ground_truth=editor_inputs["ground_truth"],
        rephrase_prompts=editor_inputs["rephrase_prompts"],
        loc_prompts=editor_inputs["loc_prompts"],
        subject=editor_inputs["subject"],
        locality_inputs=editor_inputs["locality_inputs"],
        portability_inputs=editor_inputs["portability_inputs"],
        sequential_edit=args.sequential_edit,
        eval_metric=editor_inputs["eval_metric"],
    )
    run_config["wall_time_seconds"] = float(time.time() - start_time)
    write_json(output_dir / "run_config.json", run_config)

    memory_snapshot = _memory_snapshot(edited_model)
    summary = summarize_run(metrics, records, run_config, memory_snapshot=memory_snapshot)
    memory_audit = _memoir_memory_audit(edited_model, hparams, stream_length)
    efficiency = build_efficiency_metrics(run_config, summary, memory_snapshot)
    efficiency["memoir_memory_audit"] = memory_audit

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "efficiency_metrics.json", efficiency)
    write_json(output_dir / "memory_audit.json", memory_audit)
    write_json(output_dir / "memoir_raw_metrics.json", metrics)
    write_jsonl(output_dir / "memory_snapshot.jsonl", memory_snapshot)

    print(f"Summary written to {output_dir / 'summary.json'}")
    print(f"Efficiency written to {output_dir / 'efficiency_metrics.json'}")
