import argparse
import json
import os
import random
import sys
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


def seed_everything(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, choices=['ZsRE', 'CounterFact'])
    parser.add_argument('--data_file', default=None, type=str)
    parser.add_argument('--output_root', default='./outputs/experiments', type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--run_name', default=None, type=str)
    parser.add_argument('--route_log_path', default=None, type=str)
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
    )

    seed_everything(args.seed)
    method = method_name(args.editing_method)
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
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
    metrics, edited_model, _ = editor.edit(
        prompts=editor_inputs['prompts'],
        target_new=editor_inputs['target_new'],
        ground_truth=editor_inputs['ground_truth'],
        rephrase_prompts=editor_inputs['rephrase_prompts'],
        loc_prompts=editor_inputs['loc_prompts'],
        subject=editor_inputs['subject'],
        locality_inputs=editor_inputs['locality_inputs'],
        portability_inputs=editor_inputs['portability_inputs'],
        sequential_edit=args.sequential_edit,
        eval_metric=editor_inputs['eval_metric'],
    )

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
    write_json(output_dir / 'route_diagnostics.json', route_diagnostics)
    write_json(output_dir / 'conflict_diagnostics.json', conflict_diagnostics)
    write_json(output_dir / 'memory_snapshot.json', memory_snapshot if memory_snapshot else placeholder_artifact('memory_snapshot', run_config))

    if method == 'HOPEDIT':
        write_json(output_dir / 'hopedit_route_diagnostics.json', route_diagnostics)
        write_json(output_dir / 'hopedit_conflict_diagnostics.json', conflict_diagnostics)
        write_json(output_dir / 'hopedit_memory_snapshot.json', memory_snapshot)
        if route_logs:
            (output_dir / 'hopedit_route_logs.jsonl').write_text(route_log_path.read_text())
            (output_dir / 'hopedit_route_logs_annotated.jsonl').write_text(annotated_log_path.read_text())

    summary = summarize_run(metrics, records, run_config, memory_snapshot if isinstance(memory_snapshot, list) else [])
    write_json(output_dir / 'summary.json', summary)

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
    print(f'Memory snapshot written to {output_dir / "memory_snapshot.json"}')
