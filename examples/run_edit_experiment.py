import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.edit_experiment_utils import (
    backbone_slug,
    build_editor_inputs,
    canonical_run_name,
    load_normalized_records,
    method_name,
    placeholder_artifact,
    resolve_hparams_class,
    summarize_run,
    write_json,
    write_jsonl,
)
from examples.checkpoint_suite_utils import infer_memory_semantics

FULL_EXACT_PARAMS_PER_LORA = 5_046_272


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


def build_theory_metrics(
    run_config,
    summary,
    route_diagnostics,
    conflict_diagnostics,
    prototype_diagnostics,
    stability_diagnostics=None,
    hierarchy_diagnostics=None,
    gate_diagnostics=None,
    slot_diagnostics=None,
    state_diagnostics=None,
    factor_space_diagnostics=None,
):
    routing = route_diagnostics.get("routing", {}) if isinstance(route_diagnostics, dict) else {}
    cross_view = route_diagnostics.get("cross_view", {}) if isinstance(route_diagnostics, dict) else {}
    rw_losses = []
    rp_losses = []
    loc_losses = []
    distortions = []
    for row in summary.get("per_case", []):
        rw = row.get("post_rewrite_acc")
        rp = row.get("post_rephrase_acc")
        loc = row.get("post_locality_acc")
        rw_loss = None if rw is None else 1.0 - float(rw)
        rp_loss = None if rp is None else 1.0 - float(rp)
        loc_loss = None if loc is None else 1.0 - float(loc)
        rw_losses.append(rw_loss)
        rp_losses.append(rp_loss)
        loc_losses.append(loc_loss)
        distortions.append(_safe_mean([rw_loss, rp_loss, loc_loss]))
    return {
        "applicable": True,
        "editing_method": run_config["editing_method"],
        "run_name": run_config["run_name"],
        "performance": {
            "post_rewrite_mean": summary.get("post_rewrite_mean"),
            "post_rephrase_mean": summary.get("post_rephrase_mean"),
            "post_locality_mean": summary.get("post_locality_mean"),
            "rewrite_delta_mean": summary.get("rewrite_delta_mean"),
            "rephrase_delta_mean": summary.get("rephrase_delta_mean"),
            "early_late_gap": summary.get("early_late_gap"),
        },
        "distortion": {
            "rewrite_loss_mean": _safe_mean(rw_losses),
            "rephrase_loss_mean": _safe_mean(rp_losses),
            "locality_loss_mean": _safe_mean(loc_losses),
            "extended_mean": _safe_mean(distortions),
        },
        "routing": {
            "rewrite_route_accuracy": routing.get("rewrite", {}).get("route_accuracy"),
            "rephrase_route_accuracy": routing.get("rephrase", {}).get("route_accuracy"),
            "cross_view_route_gap": cross_view.get("cross_view_route_gap"),
            "rewrite_coverage": routing.get("rewrite", {}).get("coverage"),
            "rephrase_coverage": routing.get("rephrase", {}).get("coverage"),
            "route_stage_counts": route_diagnostics.get("summary", {}).get("route_stage_counts") if isinstance(route_diagnostics, dict) else None,
        },
        "conflict": {
            "num_cells": conflict_diagnostics.get("num_cells") if isinstance(conflict_diagnostics, dict) else None,
            "within_cell_conflict_mean": conflict_diagnostics.get("within_cell_conflict_mean") if isinstance(conflict_diagnostics, dict) else None,
            "within_cell_conflict_max": conflict_diagnostics.get("within_cell_conflict_max") if isinstance(conflict_diagnostics, dict) else None,
        },
        "prototypes": prototype_diagnostics if isinstance(prototype_diagnostics, dict) else None,
        "stability": stability_diagnostics if isinstance(stability_diagnostics, dict) else None,
        "hierarchy": hierarchy_diagnostics if isinstance(hierarchy_diagnostics, dict) else None,
        "gate": gate_diagnostics if isinstance(gate_diagnostics, dict) else None,
        "slots": slot_diagnostics if isinstance(slot_diagnostics, dict) else None,
        "states": state_diagnostics if isinstance(state_diagnostics, dict) else None,
        "factor_space": factor_space_diagnostics if isinstance(factor_space_diagnostics, dict) else None,
    }


def build_efficiency_metrics(run_config, summary, route_diagnostics, memory_snapshot):
    semantics = infer_memory_semantics(run_config["editing_method"], controller=None, memory_snapshot=memory_snapshot)
    checkpoint_size_bytes = run_config.get("checkpoint_size_bytes")
    retained_units_final = semantics.get("retained_units_final")
    memory_entries_final = len(memory_snapshot) if isinstance(memory_snapshot, list) else None
    return {
        "applicable": True,
        "editing_method": run_config["editing_method"],
        "run_name": run_config["run_name"],
        "stream_length": run_config.get("stream_length"),
        "wall_time_seconds": run_config.get("wall_time_seconds"),
        "edit_seconds_per_edit": None
        if not run_config.get("wall_time_seconds") or not run_config.get("stream_length")
        else float(run_config["wall_time_seconds"] / run_config["stream_length"]),
        "memory_unit": semantics.get("memory_unit"),
        "memory_entries_final": memory_entries_final,
        "retained_units_final": retained_units_final,
        "checkpoint_save_seconds": run_config.get("checkpoint_save_seconds"),
        "checkpoint_load_seconds": run_config.get("checkpoint_load_seconds"),
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "bytes_per_retained_edit": None
        if not checkpoint_size_bytes or not memory_entries_final
        else float(checkpoint_size_bytes / max(1, memory_entries_final)),
        "bytes_per_retained_unit": None
        if not checkpoint_size_bytes or not retained_units_final
        else float(checkpoint_size_bytes / max(1, retained_units_final)),
        "route_events_logged": route_diagnostics.get("summary", {}).get("num_logged_events") if isinstance(route_diagnostics, dict) else None,
        "mean_case_time_reported": summary.get("mean_time"),
    }


def _numel(value):
    if hasattr(value, "numel"):
        return int(value.numel())
    return 0


def _tensor_list_numel(value):
    if value is None:
        return 0
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return int(sum(_tensor_list_numel(item) for item in value))
    return 0


def _find_named_module(model, dotted_name: str):
    current = model
    for part in dotted_name.replace("[", ".").replace("]", "").split("."):
        if part == "":
            continue
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _count_adapter_like_params(model) -> int:
    total = 0
    for name, param in model.named_parameters():
        lower = name.lower()
        if any(token in lower for token in ("lora_", "ranknum", "melo", "grace", "defer", "predict_values")):
            total += int(param.numel())
    return total


def build_logical_memory_audit(*, edited_model, hparams, run_config, stream_length: int) -> dict:
    """Best-effort deployed-representation memory audit for stock baselines.

    The residual-ladder paper compares against an exact rank-8 Qwen LoRA bank,
    whose per-edit parameter count is fixed by the current main configuration.
    This audit reports method-native logical storage and normalizes it to that
    exact-bank denominator.  For methods that directly overwrite model weights
    (MEMIT/AlphaEdit), we count the dense layer deltas needed to reproduce the
    edited model from the base model.
    """
    method = str(run_config.get("editing_method") or "").upper()
    exact_total = int(stream_length * FULL_EXACT_PARAMS_PER_LORA)
    audit = {
        "applicable": True,
        "memory_accounting": "inference_extra_logical_deployed_representation",
        "memory_scope": "incremental_inference_storage_over_frozen_base_model",
        "excluded_from_memory": [
            "frozen base model weights",
            "optimizer state",
            "training activations",
            "temporary SVD/workspace tensors",
            "transient evaluator materialized banks",
        ],
        "normalizer": "rank8_qwen_exact_lora_bank",
        "params_per_exact_lora": FULL_EXACT_PARAMS_PER_LORA,
        "exact_total_params": exact_total,
        "method": method,
        "stream_length": int(stream_length),
        "notes": [],
    }

    logical_params = None
    terms = {}
    model = edited_model
    if hasattr(model, "model") and method in {"DEFER", "WISE", "GRACE"}:
        # Wrapper modules keep side-memory around a base model; inspect wrapper
        # first, but fall back to nested model traversal below.
        pass

    if method in {"LORA", "QLORA", "MELO"}:
        adapter_params = _count_adapter_like_params(model)
        trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        logical_params = adapter_params or trainable_params
        terms = {"adapter_like_params": adapter_params, "trainable_params": trainable_params}
        audit["notes"].append("Counts LoRA/MELO adapter tensors in the final edited model.")

    elif method == "DEFER":
        side_params = 0
        for name, param in model.named_parameters():
            lower = name.lower()
            if "defer" in lower or "predict_values" in lower:
                side_params += int(param.numel())
        logical_params = side_params
        terms = {"defer_side_params": side_params}
        audit["notes"].append("Counts learned defer gate and value-predictor parameters.")

    elif method == "GRACE":
        side_params = 0
        key_count = 0
        for module in model.modules():
            if hasattr(module, "keys"):
                side_params += _numel(getattr(module, "keys", None))
                side_params += _numel(getattr(module, "values", None))
                side_params += _numel(getattr(module, "epsilons", None))
                key_count += int(getattr(module, "keys").shape[0]) if hasattr(getattr(module, "keys"), "shape") else 0
        logical_params = side_params
        terms = {"grace_key_value_params": side_params, "grace_key_count": key_count}
        audit["notes"].append("Counts GRACE key/value/epsilon side-memory tensors.")

    elif method == "WISE":
        side_params = 0
        memory_weight_count = 0
        for module in model.modules():
            if hasattr(module, "new_weight"):
                side_params += _numel(getattr(module, "new_weight", None))
            if hasattr(module, "memory_weight"):
                memory_weight = getattr(module, "memory_weight", None)
                memory_weight_count += len(memory_weight) if isinstance(memory_weight, list) else 0
                side_params += _tensor_list_numel(memory_weight)
            for attr in ("editing_activation", "memory_mean_act", "activation_mask"):
                if hasattr(module, attr):
                    side_params += _tensor_list_numel(getattr(module, attr))
        logical_params = side_params
        terms = {"wise_side_params": side_params, "wise_memory_weight_count": memory_weight_count}
        audit["notes"].append("Counts WISE side-memory tensors visible on the final adapter.")

    elif method in {"MEMIT", "ALPHAEDIT"}:
        dense_delta_params = 0
        layers = list(getattr(hparams, "layers", []) or [])
        rewrite_template = getattr(hparams, "rewrite_module_tmp", None)
        target_shapes = []
        if rewrite_template:
            base_model = model.model if hasattr(model, "model") else model
            for layer in layers:
                try:
                    module = _find_named_module(base_model, rewrite_template.format(layer))
                    weight = getattr(module, "weight", None)
                    params = _numel(weight)
                    dense_delta_params += params
                    target_shapes.append({"layer": int(layer), "params": params})
                except Exception as exc:
                    target_shapes.append({"layer": int(layer), "error": repr(exc)})
        logical_params = dense_delta_params
        terms = {"dense_edited_layer_delta_params": dense_delta_params, "edited_layers": target_shapes}
        audit["notes"].append("Counts dense edited-layer deltas needed to reproduce the edited model from the base.")

    else:
        trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        logical_params = trainable_params
        terms = {"trainable_params_final": trainable_params}
        audit["notes"].append("Fallback: counts final trainable parameters; inspect before using as paper memory.")

    audit["terms"] = terms
    audit["logical_params"] = None if logical_params is None else int(logical_params)
    audit["inference_extra_params"] = None if logical_params is None else int(logical_params)
    audit["memory_fraction_vs_exact"] = None if not logical_params or exact_total <= 0 else float(logical_params / exact_total)
    audit["compression_ratio_vs_exact"] = None if not logical_params else float(exact_total / logical_params)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, choices=['ZsRE', 'CounterFact', 'WikiBigEdit'])
    parser.add_argument('--data_file', default=None, type=str)
    parser.add_argument('--output_root', default='./outputs/experiments', type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--run_name', default=None, type=str)
    parser.add_argument('--route_log_path', default=None, type=str)
    parser.add_argument('--hparams_override_json', default=None, type=str)
    parser.add_argument('--ds_size', default=16, type=int)
    parser.add_argument('--index_file', default=None, type=str)
    parser.add_argument('--stream_type', default='standard', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--sequential_edit', action='store_true')
    args = parser.parse_args()

    from easyeditor import BaseEditor
    from easyeditor.models.hopedit.diagnostics import (
        annotate_route_logs,
        export_memory_snapshot,
        summarize_conflicts,
        summarize_route_diagnostics,
        summarize_trace_confusion_audit,
    )

    seed_everything(args.seed)
    method = method_name(args.editing_method)
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    if args.hparams_override_json:
        override_source = args.hparams_override_json
        if override_source.lstrip().startswith("{"):
            overrides = json.loads(override_source)
        else:
            overrides = json.loads(Path(override_source).read_text())
        if not isinstance(overrides, dict):
            raise ValueError('--hparams_override_json must decode to a JSON object')
        for key, value in overrides.items():
            setattr(hparams, key, value)
    if not hasattr(hparams, 'sequential_edit'):
        setattr(hparams, 'sequential_edit', bool(args.sequential_edit))
    elif args.sequential_edit:
        hparams.sequential_edit = True

    indices = None
    if args.index_file is not None:
        index_payload = json.loads(Path(args.index_file).read_text())
        if isinstance(index_payload, dict):
            indices = index_payload.get('selected_indices') or index_payload.get('indices')
        else:
            indices = index_payload
        if indices is None:
            raise ValueError(f'No indices found in {args.index_file}')
        indices = [int(idx) for idx in indices]

    records, dataset_file = load_normalized_records(
        args.data_dir,
        args.data_type,
        args.ds_size,
        indices=indices,
        data_file=args.data_file,
    )
    editor_inputs = build_editor_inputs(records, args.data_type)

    stream_length = len(records)
    run_name = args.run_name or canonical_run_name(method, hparams.model_name, args.data_type, args.stream_type, stream_length, args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(hparams, 'route_log_dir'):
        hparams.route_log_dir = str(output_dir)

    run_config = {
        'run_name': run_name,
        'editing_method': method,
        'alg_name': hparams.alg_name,
        'model_name': hparams.model_name,
        'backbone': backbone_slug(hparams.model_name),
        'data_type': args.data_type,
        'dataset_file': str(dataset_file),
        'stream_type': args.stream_type,
        'seed': args.seed,
        'sequential_edit': bool(args.sequential_edit),
        'stream_length': stream_length,
        'requested_ds_size': args.ds_size,
        'index_file': args.index_file,
        'hparams_path': str(Path(args.hparams_dir).resolve()),
        'output_dir': str(output_dir.resolve()),
    }
    write_json(output_dir / 'run_config.json', run_config)

    editor = BaseEditor.from_hparams(hparams)
    start_time = time.time()
    edit_kwargs = {
        'prompts': editor_inputs['prompts'],
        'target_new': editor_inputs['target_new'],
        'ground_truth': editor_inputs['ground_truth'],
        'rephrase_prompts': editor_inputs['rephrase_prompts'],
        'loc_prompts': editor_inputs['loc_prompts'],
        'subject': editor_inputs['subject'],
        'locality_inputs': editor_inputs['locality_inputs'],
        'portability_inputs': editor_inputs['portability_inputs'],
        'sequential_edit': args.sequential_edit,
        'eval_metric': editor_inputs['eval_metric'],
    }
    for optional_key in ('address_rephrase_prompts', 'relation_id', 'source_index'):
        optional_value = editor_inputs.get(optional_key)
        if optional_value is None:
            continue
        request_key = 'address_rephrase_prompt' if optional_key == 'address_rephrase_prompts' else optional_key
        edit_kwargs[request_key] = optional_value
    metrics, edited_model, _ = editor.edit(**edit_kwargs)
    wall_time_seconds = time.time() - start_time
    run_config['wall_time_seconds'] = wall_time_seconds
    write_json(output_dir / 'run_config.json', run_config)

    metrics_path = output_dir / 'metrics.json'
    write_json(metrics_path, metrics)
    if method == 'HOPEDIT':
        legacy_metrics = output_dir / f"HOPEDIT_{hparams.model_name.split('/')[-1]}_{args.data_type}.json"
        write_json(legacy_metrics, metrics)

    controller = edited_model if hasattr(edited_model, 'route_logs') else editor.model if hasattr(editor.model, 'route_logs') else None
    route_logs = []
    memory_entries = []
    memory_snapshot = []

    if controller is not None and hasattr(controller, 'save_route_logs'):
        route_log_path = Path(args.route_log_path) if args.route_log_path else output_dir / 'route_logs.jsonl'
        controller.save_route_logs(str(route_log_path))
        route_logs = list(getattr(controller, 'route_logs', []))
        memory_entries = list(getattr(controller, 'memory_entries', []))
        if hasattr(controller, 'export_memory_snapshot'):
            memory_snapshot = controller.export_memory_snapshot(include_keys=False)
        else:
            memory_snapshot = export_memory_snapshot(memory_entries, include_keys=False)
    else:
        route_log_path = Path(args.route_log_path) if args.route_log_path else output_dir / 'route_logs.jsonl'
        write_jsonl(route_log_path, [])

    annotated_logs = annotate_route_logs(route_logs, metrics) if route_logs else []
    annotated_log_path = output_dir / 'annotated_route_logs.jsonl'
    write_jsonl(annotated_log_path, annotated_logs)

    route_diagnostics = summarize_route_diagnostics(annotated_logs, metrics) if route_logs else placeholder_artifact('route_diagnostics', run_config)
    conflict_diagnostics = summarize_conflicts(memory_entries, hparams.semantic_weight, hparams.activation_weight) if memory_entries and hasattr(hparams, 'semantic_weight') and hasattr(hparams, 'activation_weight') else placeholder_artifact('conflict_diagnostics', run_config)
    confusion_audit = summarize_trace_confusion_audit(annotated_logs, memory_snapshot) if annotated_logs and memory_snapshot else placeholder_artifact('trace_confusion_audit', run_config)
    prototype_diagnostics = controller.export_prototype_diagnostics() if controller is not None and hasattr(controller, 'export_prototype_diagnostics') else placeholder_artifact('prototype_diagnostics', run_config)
    stability_diagnostics = controller.export_stability_diagnostics() if controller is not None and hasattr(controller, 'export_stability_diagnostics') else placeholder_artifact('stability_diagnostics', run_config)
    hierarchy_diagnostics = controller.export_hierarchy_diagnostics() if controller is not None and hasattr(controller, 'export_hierarchy_diagnostics') else placeholder_artifact('hierarchy_diagnostics', run_config)
    gate_diagnostics = controller.export_gate_diagnostics() if controller is not None and hasattr(controller, 'export_gate_diagnostics') else placeholder_artifact('gate_diagnostics', run_config)
    slot_diagnostics = controller.export_slot_diagnostics() if controller is not None and hasattr(controller, 'export_slot_diagnostics') else placeholder_artifact('slot_diagnostics', run_config)
    state_diagnostics = controller.export_state_diagnostics() if controller is not None and hasattr(controller, 'export_state_diagnostics') else placeholder_artifact('state_diagnostics', run_config)
    factor_space_diagnostics = controller.export_factor_space_diagnostics() if controller is not None and hasattr(controller, 'export_factor_space_diagnostics') else placeholder_artifact('factor_space_diagnostics', run_config)
    shard_diagnostics = controller.export_shard_diagnostics() if controller is not None and hasattr(controller, 'export_shard_diagnostics') else placeholder_artifact('shard_diagnostics', run_config)
    support_diagnostics = controller.export_support_diagnostics() if controller is not None and hasattr(controller, 'export_support_diagnostics') else placeholder_artifact('support_diagnostics', run_config)
    realization_diagnostics = controller.export_realization_diagnostics() if controller is not None and hasattr(controller, 'export_realization_diagnostics') else placeholder_artifact('realization_diagnostics', run_config)
    write_json(output_dir / 'route_diagnostics.json', route_diagnostics)
    write_json(output_dir / 'conflict_diagnostics.json', conflict_diagnostics)
    write_json(output_dir / 'trace_confusion_audit.json', confusion_audit)
    write_json(output_dir / 'prototype_diagnostics.json', prototype_diagnostics)
    write_json(output_dir / 'stability_diagnostics.json', stability_diagnostics)
    write_json(output_dir / 'hierarchy_diagnostics.json', hierarchy_diagnostics)
    write_json(output_dir / 'gate_diagnostics.json', gate_diagnostics)
    write_json(output_dir / 'slot_diagnostics.json', slot_diagnostics)
    write_json(output_dir / 'state_diagnostics.json', state_diagnostics)
    write_json(output_dir / 'factor_space_diagnostics.json', factor_space_diagnostics)
    write_json(output_dir / 'shard_diagnostics.json', shard_diagnostics)
    write_json(output_dir / 'support_diagnostics.json', support_diagnostics)
    write_json(output_dir / 'realization_diagnostics.json', realization_diagnostics)
    write_json(output_dir / 'cross_view_route_gap.json', route_diagnostics.get('cross_view', {}) if isinstance(route_diagnostics, dict) else placeholder_artifact('cross_view_route_gap', run_config))
    write_json(output_dir / 'memory_snapshot.json', memory_snapshot if memory_snapshot else placeholder_artifact('memory_snapshot', run_config))

    if method == 'HOPEDIT':
        write_json(output_dir / 'hopedit_route_diagnostics.json', route_diagnostics)
        write_json(output_dir / 'hopedit_conflict_diagnostics.json', conflict_diagnostics)
        write_json(output_dir / 'hopedit_trace_confusion_audit.json', confusion_audit)
        write_json(output_dir / 'hopedit_memory_snapshot.json', memory_snapshot)
        if route_logs:
            (output_dir / 'hopedit_route_logs.jsonl').write_text(route_log_path.read_text())
            (output_dir / 'hopedit_route_logs_annotated.jsonl').write_text(annotated_log_path.read_text())

    summary = summarize_run(metrics, records, run_config, memory_snapshot if isinstance(memory_snapshot, list) else [])
    write_json(output_dir / 'summary.json', summary)
    write_json(output_dir / 'family_buckets.json', summary.get('family_buckets', {}))
    failure_ladder = {
        'run_name': run_name,
        'memory_unit': memory_snapshot[0].get('memory_unit') if isinstance(memory_snapshot, list) and memory_snapshot else None,
        'post_rewrite_mean': summary.get('post_rewrite_mean'),
        'post_rephrase_mean': summary.get('post_rephrase_mean'),
        'post_locality_mean': summary.get('post_locality_mean'),
        'cross_view_route_gap': route_diagnostics.get('cross_view', {}).get('cross_view_route_gap') if isinstance(route_diagnostics, dict) else None,
        'within_conflict_mean': conflict_diagnostics.get('within_cell_conflict_mean') if isinstance(conflict_diagnostics, dict) else None,
        'slot_transfer_attempts': slot_diagnostics.get('slot_transfer_attempts') if isinstance(slot_diagnostics, dict) else None,
    }
    write_json(output_dir / 'failure_ladder.json', failure_ladder)
    theory_metrics = build_theory_metrics(run_config, summary, route_diagnostics, conflict_diagnostics, prototype_diagnostics, stability_diagnostics, hierarchy_diagnostics, gate_diagnostics, slot_diagnostics, state_diagnostics, factor_space_diagnostics)
    efficiency_metrics = build_efficiency_metrics(run_config, summary, route_diagnostics, memory_snapshot if isinstance(memory_snapshot, list) else [])
    logical_memory_audit = build_logical_memory_audit(
        edited_model=edited_model,
        hparams=hparams,
        run_config=run_config,
        stream_length=stream_length,
    )
    write_json(output_dir / 'theory_metrics.json', theory_metrics)
    write_json(output_dir / 'efficiency_metrics.json', efficiency_metrics)
    write_json(output_dir / 'memory_audit.json', logical_memory_audit)

    if len(metrics) > 0:
        print('Metrics Summary:', {
            'pre_rewrite_mean': summary.get('pre_rewrite_mean'),
            'post_rewrite_mean': summary.get('post_rewrite_mean'),
            'rewrite_delta_mean': summary.get('rewrite_delta_mean'),
            'pre_rephrase_mean': summary.get('pre_rephrase_mean'),
            'post_rephrase_mean': summary.get('post_rephrase_mean'),
            'post_locality_mean': summary.get('post_locality_mean'),
            'post_portability_mean': summary.get('post_portability_mean'),
            'early_late_gap': summary.get('early_late_gap'),
        })
    print(f'Run config written to {output_dir / "run_config.json"}')
    print(f'Metrics written to {metrics_path}')
    print(f'Summary written to {output_dir / "summary.json"}')
    print(f'Route diagnostics written to {output_dir / "route_diagnostics.json"}')
    print(f'Conflict diagnostics written to {output_dir / "conflict_diagnostics.json"}')
    print(f'Prototype diagnostics written to {output_dir / "prototype_diagnostics.json"}')
    print(f'Stability diagnostics written to {output_dir / "stability_diagnostics.json"}')
    print(f'Hierarchy diagnostics written to {output_dir / "hierarchy_diagnostics.json"}')
    print(f'Gate diagnostics written to {output_dir / "gate_diagnostics.json"}')
    print(f'Slot diagnostics written to {output_dir / "slot_diagnostics.json"}')
    print(f'State diagnostics written to {output_dir / "state_diagnostics.json"}')
    print(f'Factor-space diagnostics written to {output_dir / "factor_space_diagnostics.json"}')
    print(f'Shard diagnostics written to {output_dir / "shard_diagnostics.json"}')
    print(f'Support diagnostics written to {output_dir / "support_diagnostics.json"}')
    print(f'Realization diagnostics written to {output_dir / "realization_diagnostics.json"}')
    print(f'Memory snapshot written to {output_dir / "memory_snapshot.json"}')
    print(f'Logical memory audit written to {output_dir / "memory_audit.json"}')
