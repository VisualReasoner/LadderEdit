import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from easyeditor import HopEditHyperParams, BaseEditor, summary_metrics
from easyeditor.models.hopedit.diagnostics import (
    annotate_route_logs,
    export_memory_snapshot,
    summarize_conflicts,
    summarize_route_diagnostics,
)


def load_dataset(data_dir: str, data_type: str, ds_size: int, indices: list[int] | None = None):
    if data_type == 'ZsRE':
        edit_data = json.load(open(f'{data_dir}/{data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))
        eval_metric = 'token em'
    elif data_type == 'hallucination':
        edit_data = json.load(open(f'{data_dir}/{data_type}/hallucination-edit.json', 'r', encoding='utf-8'))
        eval_metric = 'ppl'
    elif data_type == 'temporal':
        edit_data = json.load(open(f'{data_dir}/{data_type}/temporal-edit.json', 'r', encoding='utf-8'))
        eval_metric = 'ood_ppl'
    else:
        raise NotImplementedError(f'Unsupported data_type: {data_type}')
    if indices is not None:
        edit_data = [edit_data[idx] for idx in indices]
    else:
        edit_data = edit_data[:ds_size]

    if data_type == 'ZsRE':
        prompts = [item['src'] for item in edit_data]
        subject = [item['subject'] for item in edit_data]
        rephrase_prompts = [item['rephrase'] for item in edit_data]
        target_new = [item['alt'] for item in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': [item['loc'] for item in edit_data],
                'ground_truth': [item['loc_ans'] for item in edit_data],
            }
        }
    elif data_type == 'hallucination':
        prompts = [item['prompt'] for item in edit_data]
        subject = [item['subject'] for item in edit_data]
        rephrase_prompts = None
        target_new = [item['target_new'] for item in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': [item['locality_prompt'] for item in edit_data],
                'ground_truth': [item['locality_ground_truth'] for item in edit_data],
            }
        }
    elif data_type == 'temporal':
        prompts = [item['prompt'] for item in edit_data]
        subject = [item['subject'] for item in edit_data]
        rephrase_prompts = [item['ood_rephrase'] for item in edit_data]
        target_new = [item['target_new'] for item in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': [item['locality_prompt'] for item in edit_data],
                'ground_truth': [item['locality_ground_truth'] for item in edit_data],
            }
        }
    else:
        raise NotImplementedError(f'Unsupported data_type: {data_type}')
    return prompts, subject, rephrase_prompts, target_new, locality_inputs, eval_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, choices=['ZsRE', 'temporal', 'hallucination'])
    parser.add_argument('--output_dir', default='./outputs/hopedit', type=str)
    parser.add_argument('--route_log_path', default=None, type=str)
    parser.add_argument('--ds_size', default=3, type=int)
    parser.add_argument('--index_file', default=None, type=str)
    parser.add_argument('--sequential_edit', action='store_true')
    args = parser.parse_args()

    indices = None
    if args.index_file is not None:
        index_payload = json.load(open(args.index_file, 'r', encoding='utf-8'))
        if isinstance(index_payload, dict):
            indices = index_payload.get('selected_indices') or index_payload.get('indices')
        else:
            indices = index_payload
        if indices is None:
            raise ValueError(f'No indices found in {args.index_file}')
        indices = [int(idx) for idx in indices]

    prompts, subject, rephrase_prompts, target_new, locality_inputs, eval_metric = load_dataset(
        args.data_dir,
        args.data_type,
        args.ds_size,
        indices=indices,
    )

    hparams = HopEditHyperParams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        eval_metric=eval_metric,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_suffix = hparams.model_name.split('/')[-1]
    metrics_path = os.path.join(args.output_dir, f'HOPEDIT_{model_suffix}_{args.data_type}.json')
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    route_log_path = args.route_log_path or os.path.join(args.output_dir, 'hopedit_route_logs.jsonl')
    controller = edited_model if hasattr(edited_model, 'route_logs') else editor.model if hasattr(editor.model, 'route_logs') else None
    route_logs = []
    if hasattr(edited_model, 'save_route_logs'):
        edited_model.save_route_logs(route_log_path)
        route_logs = list(getattr(edited_model, 'route_logs', []))
    elif hasattr(editor.model, 'save_route_logs'):
        editor.model.save_route_logs(route_log_path)
        route_logs = list(getattr(editor.model, 'route_logs', []))

    annotated_logs = annotate_route_logs(route_logs, metrics)
    annotated_route_log_path = os.path.join(args.output_dir, 'hopedit_route_logs_annotated.jsonl')
    with open(annotated_route_log_path, 'w', encoding='utf-8') as handle:
        for entry in annotated_logs:
            handle.write(json.dumps(entry) + '\n')

    memory_entries = list(getattr(controller, 'memory_entries', [])) if controller is not None else []
    if controller is not None and hasattr(controller, 'export_memory_snapshot'):
        memory_snapshot = controller.export_memory_snapshot(include_keys=False)
    else:
        memory_snapshot = export_memory_snapshot(memory_entries, include_keys=False)
    memory_snapshot_path = os.path.join(args.output_dir, 'hopedit_memory_snapshot.json')
    with open(memory_snapshot_path, 'w', encoding='utf-8') as handle:
        json.dump(memory_snapshot, handle, indent=2)

    route_diagnostics = summarize_route_diagnostics(annotated_logs, metrics)
    route_diagnostics_path = os.path.join(args.output_dir, 'hopedit_route_diagnostics.json')
    with open(route_diagnostics_path, 'w', encoding='utf-8') as handle:
        json.dump(route_diagnostics, handle, indent=2)

    conflict_diagnostics = summarize_conflicts(memory_entries, hparams.semantic_weight, hparams.activation_weight)
    conflict_diagnostics_path = os.path.join(args.output_dir, 'hopedit_conflict_diagnostics.json')
    with open(conflict_diagnostics_path, 'w', encoding='utf-8') as handle:
        json.dump(conflict_diagnostics, handle, indent=2)

    if len(metrics) > 0:
        summary_metrics(metrics)
    print(f'Metrics written to {metrics_path}')
    print(f'Route logs written to {route_log_path}')
    print(f'Annotated route logs written to {annotated_route_log_path}')
    print(f'Route diagnostics written to {route_diagnostics_path}')
    print(f'Conflict diagnostics written to {conflict_diagnostics_path}')
    print(f'Memory snapshot written to {memory_snapshot_path}')
