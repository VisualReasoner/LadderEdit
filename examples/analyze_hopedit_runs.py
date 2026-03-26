import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text())


def scalar(dct, *keys):
    cur = dct
    for key in keys:
        if cur is None:
            return None
        cur = cur.get(key)
    return cur


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='name=output_dir')
    args = parser.parse_args()

    rows = []
    for spec in args.run:
        name, run_dir = spec.split('=', 1)
        run_path = Path(run_dir)
        route = load_json(run_path / 'hopedit_route_diagnostics.json')
        conflict = load_json(run_path / 'hopedit_conflict_diagnostics.json')
        rows.append({
            'run': name,
            'rewrite_route_acc': scalar(route, 'routing', 'rewrite', 'route_accuracy'),
            'rephrase_route_acc': scalar(route, 'routing', 'rephrase', 'route_accuracy'),
            'locality_false_activation': scalar(route, 'routing', 'locality', 'false_activation_rate'),
            'locality_no_edit': scalar(route, 'routing', 'locality', 'no_edit_rate'),
            'rewrite_top1_prob': scalar(route, 'routing', 'rewrite', 'top1_prob_mean'),
            'rephrase_top1_prob': scalar(route, 'routing', 'rephrase', 'top1_prob_mean'),
            'locality_top1_prob': scalar(route, 'routing', 'locality', 'top1_prob_mean'),
            'mean_semantic_offdiag': conflict.get('mean_semantic_offdiag'),
            'mean_raw_activation_offdiag': conflict.get('mean_raw_activation_offdiag'),
            'mean_activation_offdiag': conflict.get('mean_activation_offdiag'),
            'mean_combined_offdiag': conflict.get('mean_combined_offdiag'),
            'max_pair_conflict': conflict.get('max_pair_conflict'),
        })

    print(json.dumps(rows, indent=2))
