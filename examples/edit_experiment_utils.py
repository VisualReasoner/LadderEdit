import json
import math
import re
from pathlib import Path
from typing import Any

METHOD_HPARAMS = {
    'ALPHAEDIT': ('easyeditor.models.alphaedit', 'AlphaEditHyperParams'),
    'FT': ('easyeditor.models.ft', 'FTHyperParams'),
    'GRACE': ('easyeditor.models.grace', 'GraceHyperParams'),
    'HOPEDIT': ('easyeditor.models.hopedit', 'HopEditHyperParams'),
    'LORA': ('easyeditor.models.lora', 'LoRAHyperParams'),
    'MELO': ('easyeditor.models.melo', 'MELOHyperParams'),
    'MEMIT': ('easyeditor.models.memit', 'MEMITHyperParams'),
    'QLORA': ('easyeditor.models.qlora', 'QLoRAHyperParams'),
    'ROME': ('easyeditor.models.rome', 'ROMEHyperParams'),
    'SERAC': ('easyeditor.models.serac', 'SERACHparams'),
    'SIMIE': ('easyeditor.models.simie', 'SimIEHyperParams'),
    'WISE': ('easyeditor.models.wise', 'WISEHyperParams'),
}


SLUG_RE = re.compile(r'[^a-z0-9]+')


def method_name(name: str) -> str:
    return name.strip().upper()


def resolve_hparams_class(method: str):
    method = method_name(method)
    if method not in METHOD_HPARAMS:
        raise NotImplementedError(f'Unsupported editing method: {method}')
    module_name, class_name = METHOD_HPARAMS[method]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def slugify(text: str) -> str:
    text = (text or '').strip().lower()
    text = SLUG_RE.sub('-', text)
    return text.strip('-')


def backbone_slug(model_name: str) -> str:
    value = model_name.split('/')[-1]
    return slugify(value)


def canonical_run_name(method: str, model_name: str, data_type: str, stream_type: str, stream_length: int, seed: int) -> str:
    return f"{slugify(method)}_{backbone_slug(model_name)}_{slugify(data_type)}_{slugify(stream_type)}_{stream_length}_seed{seed}"


DATASET_CANDIDATES = {
    'ZsRE': [
        'ZsRE/zsre_mend_edit.json',
        'zsre/zsre_mend_edit.json',
    ],
    'CounterFact': [
        'counterfact/counterfact-edit.json',
        'CounterFact/counterfact-edit.json',
        'counterfact.json',
        'counterfact/counterfact.json',
    ],
    'WikiBigEdit': [
        'wikibigedit_eval_17k.json',
        'WikiBigEdit/wikibigedit_eval_17k.json',
        'wikibigedit/wikibigedit_eval_17k.json',
        'wikibigedit.json',
    ],
}


def resolve_dataset_file(data_dir: str, data_type: str, data_file: str | None = None) -> Path:
    if data_file is not None:
        path = Path(data_file)
        if not path.exists():
            raise FileNotFoundError(f'Explicit --data_file does not exist: {path}')
        return path
    root = Path(data_dir)
    candidates = DATASET_CANDIDATES.get(data_type, [])
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path

    present_entries = []
    if root.exists():
        present_entries = sorted(str(p.relative_to(root)) for p in root.glob('*'))
    candidate_lines = ', '.join(str(root / candidate) for candidate in candidates) if candidates else '<none configured>'
    present_lines = ', '.join(present_entries[:20]) if present_entries else '<empty or missing data root>'
    raise FileNotFoundError(
        f'Could not resolve a dataset file for {data_type} under {data_dir}. '
        f'Searched: {candidate_lines}. '
        f'Present top-level entries: {present_lines}. '
        f'Pass --data_file /abs/path/to/the_dataset.json if your file lives elsewhere.'
    )


def _string_target(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get('str') or value.get('text') or value.get('target')
    return str(value)


def _first_or_none(value: Any):
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


def load_normalized_records(data_dir: str, data_type: str, ds_size: int, indices: list[int] | None = None, data_file: str | None = None):
    dataset_file = resolve_dataset_file(data_dir, data_type, data_file=data_file)
    payload = json.loads(dataset_file.read_text())
    if indices is not None:
        selected = [payload[idx] for idx in indices]
        source_indices = list(indices)
    else:
        selected = payload[:ds_size]
        source_indices = list(range(min(ds_size, len(payload))))

    records = []
    for source_index, item in zip(source_indices, selected):
        if data_type == 'ZsRE':
            records.append({
                'source_index': source_index,
                'prompt': item['src'],
                'subject': item['subject'],
                'rephrase_prompt': item.get('rephrase'),
                'target_new': item['alt'],
                'ground_truth': '<|endoftext|>',
                'locality_prompt': item.get('loc'),
                'locality_ground_truth': item.get('loc_ans'),
                'portability_prompt': item.get('portability', {}).get('New Question') if isinstance(item.get('portability'), dict) else None,
                'portability_ground_truth': item.get('portability', {}).get('New Answer') if isinstance(item.get('portability'), dict) else None,
            })
            continue

        if data_type == 'CounterFact':
            rewrite = item.get('requested_rewrite', {})
            subject = rewrite.get('subject') or item.get('subject')
            prompt_template = rewrite.get('prompt') or item.get('prompt')
            if prompt_template is None:
                raise KeyError(f'CounterFact record {source_index} is missing a prompt')
            prompt = prompt_template.format(subject) if '{}' in prompt_template else prompt_template
            target_new = _string_target(rewrite.get('target_new') or item.get('target_new'))
            ground_truth = _string_target(rewrite.get('target_true') or item.get('ground_truth') or item.get('target_true')) or '<|endoftext|>'
            locality_prompt = item.get('neighborhood_prompts')
            if locality_prompt is not None and not isinstance(locality_prompt, list):
                locality_prompt = [locality_prompt]
            locality_ground_truth = None
            if locality_prompt is not None:
                locality_ground_truth = [ground_truth] * len(locality_prompt)
            records.append({
                'source_index': source_index,
                'prompt': prompt,
                'subject': subject,
                'rephrase_prompt': _first_or_none(item.get('paraphrase_prompts')),
                'target_new': target_new,
                'ground_truth': ground_truth,
                'locality_prompt': locality_prompt,
                'locality_ground_truth': locality_ground_truth,
                'portability_prompt': None,
                'portability_ground_truth': None,
            })
            continue

        if data_type == 'WikiBigEdit':
            prompt = item.get('update') or item.get('prompt')
            if prompt is None:
                raise KeyError(f'WikiBigEdit record {source_index} is missing an update/prompt field')
            target_new = _string_target(item.get('ans') or item.get('target_new'))
            ground_truth = _string_target(item.get('ground_truth')) or '<|endoftext|>'
            locality_prompt = item.get('loc') or item.get('locality')
            locality_ground_truth = item.get('loc_ans') or item.get('locality_ans')

            portability = {}
            personas_prompt = item.get('personas') or item.get('portability_personas')
            if personas_prompt is not None and target_new is not None:
                portability['personas'] = {
                    'prompt': [personas_prompt],
                    'ground_truth': [target_new],
                }
            mhop_prompt = item.get('mhop') or item.get('portability_hop')
            mhop_answer = item.get('mhop_ans') or item.get('portability_hop_ans')
            if mhop_prompt is not None and mhop_answer is not None:
                portability['mhop'] = {
                    'prompt': [mhop_prompt],
                    'ground_truth': [mhop_answer],
                }

            locality = {}
            if locality_prompt is not None and locality_ground_truth is not None:
                locality['locality'] = {
                    'prompt': [locality_prompt],
                    'ground_truth': [locality_ground_truth],
                }

            records.append({
                'source_index': source_index,
                'prompt': prompt,
                'subject': item.get('subject'),
                'rephrase_prompt': item.get('rephrase'),
                'target_new': target_new,
                'ground_truth': ground_truth,
                'locality': locality,
                'portability': portability,
            })
            continue

        raise NotImplementedError(f'Unsupported data_type: {data_type}')

    return records, dataset_file


def build_editor_inputs(records: list[dict[str, Any]], data_type: str):
    prompts = [record['prompt'] for record in records]
    subject = [record['subject'] for record in records]
    target_new = [record['target_new'] for record in records]
    ground_truth = [record['ground_truth'] for record in records]

    rephrase_values = [record.get('rephrase_prompt') for record in records]
    rephrase_prompts = rephrase_values if any(value is not None for value in rephrase_values) else None

    locality_inputs = None
    if any(record.get('locality') for record in records):
        locality_inputs = {}
        locality_keys = sorted({key for record in records for key in (record.get('locality') or {}).keys()})
        for key in locality_keys:
            locality_inputs[key] = {
                'prompt': [],
                'ground_truth': [],
            }
            for record in records:
                bucket = (record.get('locality') or {}).get(key)
                locality_inputs[key]['prompt'].append(bucket.get('prompt') if bucket is not None else None)
                locality_inputs[key]['ground_truth'].append(bucket.get('ground_truth') if bucket is not None else None)
    else:
        locality_prompts = [record.get('locality_prompt') for record in records]
        locality_answers = [record.get('locality_ground_truth') for record in records]
        if any(value is not None for value in locality_prompts):
            locality_inputs = {
                'neighborhood': {
                    'prompt': locality_prompts,
                    'ground_truth': locality_answers,
                }
            }

    portability_inputs = None
    if any(record.get('portability') for record in records):
        portability_inputs = {}
        portability_keys = sorted({key for record in records for key in (record.get('portability') or {}).keys()})
        for key in portability_keys:
            portability_inputs[key] = {
                'prompt': [],
                'ground_truth': [],
            }
            for record in records:
                bucket = (record.get('portability') or {}).get(key)
                portability_inputs[key]['prompt'].append(bucket.get('prompt') if bucket is not None else None)
                portability_inputs[key]['ground_truth'].append(bucket.get('ground_truth') if bucket is not None else None)
    else:
        portability_prompts = [record.get('portability_prompt') for record in records]
        portability_answers = [record.get('portability_ground_truth') for record in records]
        if any(value is not None for value in portability_prompts):
            portability_inputs = {
                'one_hop': {
                    'prompt': portability_prompts,
                    'ground_truth': portability_answers,
                }
            }

    loc_prompts = []
    for record in records:
        locality_prompt = record.get('locality_prompt')
        locality_ground_truth = record.get('locality_ground_truth')
        if locality_prompt is None or locality_ground_truth is None:
            locality_bucket = (record.get('locality') or {}).get('locality')
            if locality_bucket is not None:
                locality_prompt = locality_bucket.get('prompt')
                locality_ground_truth = locality_bucket.get('ground_truth')
        if isinstance(locality_prompt, list):
            locality_prompt = _first_or_none(locality_prompt)
        if isinstance(locality_ground_truth, list):
            locality_ground_truth = _first_or_none(locality_ground_truth)

        if locality_prompt is not None and locality_ground_truth is not None:
            loc_prompts.append(f'{locality_prompt} {locality_ground_truth}')
        elif locality_prompt is not None:
            loc_prompts.append(str(locality_prompt))
        elif record.get('subject') is not None:
            loc_prompts.append(str(record['subject']))
        else:
            loc_prompts.append(str(record['prompt']))

    eval_metric = 'token_em'
    if data_type == 'ZsRE':
        eval_metric = 'token em'
    return {
        'prompts': prompts,
        'subject': subject,
        'target_new': target_new,
        'ground_truth': ground_truth,
        'rephrase_prompts': rephrase_prompts,
        'locality_inputs': locality_inputs,
        'portability_inputs': portability_inputs,
        'loc_prompts': loc_prompts,
        'eval_metric': eval_metric,
    }


def metric_mean(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        flat = [float(v) for v in value if v is not None]
        if not flat:
            return None
        return float(sum(flat) / len(flat))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def nested_acc_mean(section: dict[str, Any] | None) -> float | None:
    if not isinstance(section, dict) or not section:
        return None
    values = []
    for key, value in section.items():
        if isinstance(value, dict):
            nested = nested_acc_mean(value)
            if nested is not None:
                values.append(nested)
            continue
        if key.endswith('acc'):
            maybe = metric_mean(value)
            if maybe is not None:
                values.append(maybe)
    if not values:
        return None
    return float(sum(values) / len(values))


def mean_optional(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def summarize_run(metrics: list[dict[str, Any]], records: list[dict[str, Any]], run_config: dict[str, Any], memory_snapshot: list[dict[str, Any]] | None = None):
    per_case = []
    for idx, metric in enumerate(metrics):
        request = metric.get('requested_rewrite', {})
        record = records[idx] if idx < len(records) else {}
        pre = metric.get('pre', {})
        post = metric.get('post', {})
        row = {
            'case_id': metric.get('case_id', idx),
            'source_index': record.get('source_index', idx),
            'subject': request.get('subject', record.get('subject')),
            'prompt': request.get('prompt', record.get('prompt')),
            'target_new': request.get('target_new', record.get('target_new')),
            'pre_rewrite_acc': metric_mean(pre.get('rewrite_acc')),
            'post_rewrite_acc': metric_mean(post.get('rewrite_acc')),
            'pre_rephrase_acc': metric_mean(pre.get('rephrase_acc')),
            'post_rephrase_acc': metric_mean(post.get('rephrase_acc')),
            'post_locality_acc': nested_acc_mean(post.get('locality')),
            'post_portability_acc': nested_acc_mean(post.get('portability')),
            'time': metric.get('time'),
        }
        row['rewrite_delta'] = None if row['pre_rewrite_acc'] is None or row['post_rewrite_acc'] is None else row['post_rewrite_acc'] - row['pre_rewrite_acc']
        row['rephrase_delta'] = None if row['pre_rephrase_acc'] is None or row['post_rephrase_acc'] is None else row['post_rephrase_acc'] - row['pre_rephrase_acc']
        per_case.append(row)

    num_cases = len(per_case)
    midpoint = max(1, math.ceil(num_cases / 2)) if num_cases else 0
    early_post_rewrite = mean_optional([row['post_rewrite_acc'] for row in per_case[:midpoint]])
    late_post_rewrite = mean_optional([row['post_rewrite_acc'] for row in per_case[midpoint:]])
    early_late_gap = None
    if early_post_rewrite is not None and late_post_rewrite is not None:
        early_late_gap = early_post_rewrite - late_post_rewrite

    memory_snapshot = memory_snapshot or []
    memory_entries_final = len(memory_snapshot)
    summary = {
        'method': run_config['editing_method'],
        'alg_name': run_config['alg_name'],
        'model_name': run_config['model_name'],
        'backbone': run_config['backbone'],
        'dataset': run_config['data_type'],
        'stream_type': run_config['stream_type'],
        'stream_length': num_cases,
        'seed': run_config['seed'],
        'post_rewrite_mean': mean_optional([row['post_rewrite_acc'] for row in per_case]),
        'pre_rewrite_mean': mean_optional([row['pre_rewrite_acc'] for row in per_case]),
        'rewrite_delta_mean': mean_optional([row['rewrite_delta'] for row in per_case]),
        'post_rephrase_mean': mean_optional([row['post_rephrase_acc'] for row in per_case]),
        'pre_rephrase_mean': mean_optional([row['pre_rephrase_acc'] for row in per_case]),
        'rephrase_delta_mean': mean_optional([row['rephrase_delta'] for row in per_case]),
        'post_locality_mean': mean_optional([row['post_locality_acc'] for row in per_case]),
        'post_portability_mean': mean_optional([row['post_portability_acc'] for row in per_case]),
        'mean_time': mean_optional([row['time'] for row in per_case]),
        'wall_time_seconds': run_config.get('wall_time_seconds'),
        'edits_per_second': None if not run_config.get('wall_time_seconds') or num_cases == 0 else float(num_cases / run_config['wall_time_seconds']),
        'early_post_rewrite_mean': early_post_rewrite,
        'late_post_rewrite_mean': late_post_rewrite,
        'early_late_gap': early_late_gap,
        'memory_entries_final': memory_entries_final,
        'memory_entries_per_edit': None if num_cases == 0 else float(memory_entries_final / num_cases),
        'per_case': per_case,
    }
    return summary


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def placeholder_artifact(kind: str, run_config: dict[str, Any]):
    return {
        'applicable': False,
        'kind': kind,
        'editing_method': run_config['editing_method'],
        'alg_name': run_config['alg_name'],
        'run_name': run_config['run_name'],
    }
