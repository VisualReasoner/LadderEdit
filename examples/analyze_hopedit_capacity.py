import argparse
import json
import math
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text())


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    route = load_json(run_dir / 'hopedit_route_diagnostics.json')
    conflict = load_json(run_dir / 'hopedit_conflict_diagnostics.json')
    memory = load_json(run_dir / 'hopedit_memory_snapshot.json')

    edit_ids = conflict['edit_ids']
    combined = conflict['combined_conflict']
    semantic = conflict.get('semantic_cosine')
    activation = conflict.get('activation_cosine')
    raw_activation = conflict.get('raw_activation_cosine')

    per_case = route['per_case']
    rows = []
    for idx, case in enumerate(per_case):
        offdiag_combined = [combined[idx][j] for j in range(len(edit_ids)) if j != idx] if combined else []
        offdiag_semantic = [semantic[idx][j] for j in range(len(edit_ids)) if j != idx] if semantic else []
        offdiag_activation = [activation[idx][j] for j in range(len(edit_ids)) if j != idx] if activation else []
        offdiag_raw_activation = [raw_activation[idx][j] for j in range(len(edit_ids)) if j != idx] if raw_activation else []
        memory_row = memory[idx] if idx < len(memory) else {}
        rows.append({
            'case_id': case.get('case_id'),
            'edit_id': edit_ids[idx],
            'subject': case.get('subject'),
            'prompt': case.get('prompt'),
            'rewrite_delta': case.get('rewrite_delta'),
            'post_rewrite_acc': case.get('post_rewrite_acc'),
            'post_rephrase_acc': case.get('post_rephrase_acc'),
            'max_combined_conflict': max(offdiag_combined) if offdiag_combined else None,
            'mean_combined_conflict': (sum(offdiag_combined) / len(offdiag_combined)) if offdiag_combined else None,
            'max_semantic_conflict': max(offdiag_semantic) if offdiag_semantic else None,
            'max_activation_conflict': max(offdiag_activation) if offdiag_activation else None,
            'max_raw_activation_conflict': max(offdiag_raw_activation) if offdiag_raw_activation else None,
            'raw_activation_norm': memory_row.get('raw_activation_norm'),
            'activation_norm': memory_row.get('activation_norm'),
        })

    summary = {
        'run_dir': str(run_dir),
        'num_cases': len(rows),
        'correlations': {
            'max_combined_conflict_vs_rewrite_delta': pearson(
                [row['max_combined_conflict'] for row in rows],
                [row['rewrite_delta'] for row in rows],
            ),
            'mean_combined_conflict_vs_rewrite_delta': pearson(
                [row['mean_combined_conflict'] for row in rows],
                [row['rewrite_delta'] for row in rows],
            ),
            'max_combined_conflict_vs_post_rewrite_acc': pearson(
                [row['max_combined_conflict'] for row in rows],
                [row['post_rewrite_acc'] for row in rows],
            ),
            'max_combined_conflict_vs_post_rephrase_acc': pearson(
                [row['max_combined_conflict'] for row in rows],
                [row['post_rephrase_acc'] for row in rows],
            ),
        },
        'per_edit': rows,
    }

    print(json.dumps(summary, indent=2))
