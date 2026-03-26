import argparse
import json
import re
from pathlib import Path


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


def pair_score(item_a, item_b):
    prompt_overlap = jaccard(tokens(item_a['src']), tokens(item_b['src']))
    subject_overlap = jaccard(tokens(item_a['subject']), tokens(item_b['subject']))
    template_a = make_template(item_a['src'], item_a['subject'])
    template_b = make_template(item_b['src'], item_b['subject'])
    template_match = 1.0 if template_a == template_b else jaccard(tokens(template_a), tokens(template_b))
    score = 0.45 * prompt_overlap + 0.35 * subject_overlap + 0.20 * template_match
    return {
        'score': score,
        'prompt_overlap': prompt_overlap,
        'subject_overlap': subject_overlap,
        'template_overlap': template_match,
        'same_template': template_a == template_b,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--subset_size', type=int, default=16)
    parser.add_argument('--top_pairs', type=int, default=100)
    args = parser.parse_args()

    data = json.loads(Path(args.data_dir, 'ZsRE', 'zsre_mend_edit.json').read_text())
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i]['alt'] == data[j]['alt']:
                continue
            score_info = pair_score(data[i], data[j])
            if score_info['score'] <= 0:
                continue
            pairs.append({
                'i': i,
                'j': j,
                **score_info,
                'subject_i': data[i]['subject'],
                'subject_j': data[j]['subject'],
                'prompt_i': data[i]['src'],
                'prompt_j': data[j]['src'],
            })
    pairs.sort(key=lambda x: x['score'], reverse=True)
    top_pairs = pairs[:args.top_pairs]

    selected = []
    used = set()
    for pair in top_pairs:
        for idx in [pair['i'], pair['j']]:
            if idx not in used:
                used.add(idx)
                selected.append(idx)
            if len(selected) >= args.subset_size:
                break
        if len(selected) >= args.subset_size:
            break

    selected_examples = [
        {
            'index': idx,
            'subject': data[idx]['subject'],
            'prompt': data[idx]['src'],
            'target_new': data[idx]['alt'],
            'rephrase': data[idx].get('rephrase'),
        }
        for idx in selected
    ]

    payload = {
        'dataset': 'ZsRE',
        'subset_size': len(selected),
        'selected_indices': selected,
        'selected_examples': selected_examples,
        'top_pairs': top_pairs,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f'Wrote collision subset to {output_path}')


if __name__ == '__main__':
    main()
