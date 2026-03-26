import argparse
import json
import sys
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from examples.edit_experiment_utils import load_normalized_records


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokens(text):
    return set(TOKEN_RE.findall((text or '').lower()))


def jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def make_template(prompt, subject):
    prompt = (prompt or '').lower()
    subject = (subject or '').lower().strip()
    if subject and subject in prompt:
        prompt = prompt.replace(subject, '<subj>')
    return prompt


def normalize_prompt_key(prompt):
    prompt = (prompt or '').strip().lower()
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt


def pair_score(item_a, item_b):
    prompt_overlap = jaccard(tokens(item_a['prompt']), tokens(item_b['prompt']))
    subject_overlap = jaccard(tokens(item_a['subject']), tokens(item_b['subject']))
    rephrase_overlap = jaccard(tokens(item_a.get('rephrase_prompt') or ''), tokens(item_b.get('rephrase_prompt') or ''))
    template_a = make_template(item_a['prompt'], item_a['subject'])
    template_b = make_template(item_b['prompt'], item_b['subject'])
    template_match = 1.0 if template_a == template_b else jaccard(tokens(template_a), tokens(template_b))
    score = 0.40 * prompt_overlap + 0.25 * subject_overlap + 0.20 * template_match + 0.15 * rephrase_overlap
    return {
        'score': score,
        'prompt_overlap': prompt_overlap,
        'subject_overlap': subject_overlap,
        'template_overlap': template_match,
        'rephrase_overlap': rephrase_overlap,
        'same_template': template_a == template_b,
    }


def parse_subset_sizes(args):
    if args.subset_sizes:
        return [int(value.strip()) for value in args.subset_sizes.split(',') if value.strip()]
    return [args.subset_size]


def candidate_keys(item):
    subject = (item.get('subject') or '').lower().strip()
    prompt = item.get('prompt') or ''
    rephrase = item.get('rephrase_prompt') or ''
    template = make_template(prompt, item.get('subject') or '')
    prompt_tokens = sorted(tokens(prompt))
    rephrase_tokens = sorted(tokens(rephrase))
    keys = set()
    if subject:
        keys.add(('subject', subject))
    if template:
        keys.add(('template', template))
    if prompt_tokens:
        keys.add(('prompt_head', ' '.join(prompt_tokens[:6])))
    if rephrase_tokens:
        keys.add(('rephrase_head', ' '.join(rephrase_tokens[:6])))
    if subject and prompt_tokens:
        keys.add(('subject_prompt', subject, ' '.join(prompt_tokens[:4])))
    return keys


def build_candidate_pairs(records, max_bucket_size):
    buckets = defaultdict(list)
    for idx, item in enumerate(records):
        for key in candidate_keys(item):
            buckets[key].append(idx)

    candidate_pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > max_bucket_size:
            members = members[:max_bucket_size]
        for i, j in combinations(members, 2):
            candidate_pairs.add((i, j) if i < j else (j, i))
    return sorted(candidate_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--data_type', required=True, choices=['ZsRE', 'CounterFact'])
    parser.add_argument('--data_file', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--subset_size', type=int, default=16)
    parser.add_argument('--subset_sizes', default=None, help='Comma-separated subset sizes, e.g. 16,64,128')
    parser.add_argument('--top_pairs', type=int, default=200)
    parser.add_argument('--max_bucket_size', type=int, default=128)
    parser.add_argument('--version', default='v1')
    parser.add_argument(
        '--allow_duplicate_prompts',
        action='store_true',
        help='Allow multiple selected examples with the exact same normalized rewrite prompt.',
    )
    args = parser.parse_args()

    subset_sizes = parse_subset_sizes(args)
    records, dataset_file = load_normalized_records(args.data_dir, args.data_type, ds_size=10**9, indices=None, data_file=args.data_file)
    pairs = []
    candidate_pairs = build_candidate_pairs(records, args.max_bucket_size)
    for i, j in candidate_pairs:
        if records[i]['target_new'] == records[j]['target_new']:
            continue
        score_info = pair_score(records[i], records[j])
        if score_info['score'] <= 0:
            continue
        pairs.append({
            'i': i,
            'j': j,
            **score_info,
            'source_index_i': records[i]['source_index'],
            'source_index_j': records[j]['source_index'],
            'subject_i': records[i]['subject'],
            'subject_j': records[j]['subject'],
            'prompt_i': records[i]['prompt'],
            'prompt_j': records[j]['prompt'],
        })
    pairs.sort(key=lambda x: x['score'], reverse=True)
    top_pairs = pairs[:args.top_pairs]

    outputs = []
    for subset_size in subset_sizes:
        selected = []
        used = set()
        used_prompt_keys = set()
        for pair in top_pairs:
            for idx in [pair['i'], pair['j']]:
                prompt_key = normalize_prompt_key(records[idx]['prompt'])
                if idx not in used:
                    if not args.allow_duplicate_prompts and prompt_key in used_prompt_keys:
                        continue
                    used.add(idx)
                    used_prompt_keys.add(prompt_key)
                    selected.append(idx)
                if len(selected) >= subset_size:
                    break
            if len(selected) >= subset_size:
                break
        selected = selected[:subset_size]
        selected_set = set(selected)

        stats = defaultdict(lambda: {'max_pair_score': 0.0, 'mean_pair_score': 0.0, 'pair_count': 0})
        for pair in top_pairs:
            if pair['i'] in selected_set and pair['j'] in selected_set:
                for idx in [pair['i'], pair['j']]:
                    stats[idx]['max_pair_score'] = max(stats[idx]['max_pair_score'], pair['score'])
                    stats[idx]['mean_pair_score'] += pair['score']
                    stats[idx]['pair_count'] += 1
        for idx in selected:
            if stats[idx]['pair_count'] > 0:
                stats[idx]['mean_pair_score'] /= stats[idx]['pair_count']

        selected_examples = []
        for rank, idx in enumerate(selected):
            record = records[idx]
            selected_examples.append({
                'rank': rank,
                'index': idx,
                'source_index': record['source_index'],
                'subject': record['subject'],
                'prompt': record['prompt'],
                'target_new': record['target_new'],
                'rephrase_prompt': record.get('rephrase_prompt'),
                'max_pair_score': stats[idx]['max_pair_score'],
                'mean_pair_score': stats[idx]['mean_pair_score'],
                'pair_count': stats[idx]['pair_count'],
            })

        payload = {
            'dataset': args.data_type,
            'dataset_file': str(dataset_file),
            'version': args.version,
            'subset_size': len(selected),
            'candidate_pair_count': len(candidate_pairs),
            'selected_indices': selected,
            'selected_source_indices': [records[idx]['source_index'] for idx in selected],
            'duplicate_prompt_policy': 'allow' if args.allow_duplicate_prompts else 'forbid',
            'unique_prompt_count': len({normalize_prompt_key(records[idx]['prompt']) for idx in selected}),
            'selected_examples': selected_examples,
            'top_pairs': top_pairs,
        }

        if args.output_path:
            output_path = Path(args.output_path)
            if len(subset_sizes) > 1:
                output_path = output_path.with_name(f'{output_path.stem}_{subset_size}{output_path.suffix}')
        else:
            output_dir = Path(args.output_dir or 'outputs/collisionbench')
            output_path = output_dir / f'collisionbench_{args.data_type.lower()}_{subset_size}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        outputs.append(str(output_path))
        print(f'Wrote collision subset to {output_path}')

    print(json.dumps({'outputs': outputs}, indent=2))


if __name__ == '__main__':
    main()
