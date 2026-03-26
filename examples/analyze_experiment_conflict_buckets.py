import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def quantile(values, q):
    if not values:
        return None
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def bucket_index(value, edges):
    idx = 0
    while idx < len(edges) and value > edges[idx]:
        idx += 1
    return idx


def subset_conflict_map(payload):
    mapping = {}
    if payload.get('selected_examples'):
        for row in payload['selected_examples']:
            mapping[int(row['source_index'])] = {
                'max_pair_score': row.get('max_pair_score'),
                'mean_pair_score': row.get('mean_pair_score'),
                'pair_count': row.get('pair_count'),
            }
        return mapping

    top_pairs = payload.get('top_pairs', [])
    stats = defaultdict(lambda: {'max_pair_score': 0.0, 'mean_pair_score': 0.0, 'pair_count': 0})
    selected = set(int(idx) for idx in payload.get('selected_source_indices') or payload.get('selected_indices') or [])
    for pair in top_pairs:
        i = int(pair.get('source_index_i', pair.get('i')))
        j = int(pair.get('source_index_j', pair.get('j')))
        if selected and i not in selected and j not in selected:
            continue
        for idx in [i, j]:
            stats[idx]['max_pair_score'] = max(stats[idx]['max_pair_score'], float(pair['score']))
            stats[idx]['mean_pair_score'] += float(pair['score'])
            stats[idx]['pair_count'] += 1
    for idx in list(stats.keys()):
        if stats[idx]['pair_count'] > 0:
            stats[idx]['mean_pair_score'] /= stats[idx]['pair_count']
    return stats


def load_route_event_map(run_dir: Path):
    route_path = run_dir / 'annotated_route_logs.jsonl'
    if not route_path.exists():
        return {}
    rows = [json.loads(line) for line in route_path.read_text().splitlines() if line.strip()]
    mapping = {}
    for row in rows:
        case_id = row.get('case_id')
        event_type = row.get('event_type')
        if case_id is None or event_type not in {'rewrite', 'rephrase'}:
            continue
        mapping[(int(case_id), event_type)] = row
    return mapping


def load_case_rows(run_name: str, run_dir: Path):
    run_config = load_json(run_dir / 'run_config.json')
    summary = load_json(run_dir / 'summary.json')
    subset_file = run_config.get('index_file')
    if not subset_file:
        raise ValueError(f'Run {run_name} does not reference an index_file; conflict buckets require collision subsets.')
    subset_payload = load_json(Path(subset_file))
    conflict_map = subset_conflict_map(subset_payload)
    route_events = load_route_event_map(run_dir)

    rows = []
    for case in summary['per_case']:
        source_index = int(case.get('source_index', case['case_id']))
        conflict_info = conflict_map.get(source_index, {})
        rewrite_event = route_events.get((int(case['case_id']), 'rewrite'), {})
        rephrase_event = route_events.get((int(case['case_id']), 'rephrase'), {})
        rows.append({
            'run': run_name,
            'method': run_config['editing_method'],
            'backbone': run_config['backbone'],
            'dataset': run_config['data_type'],
            'stream_type': run_config['stream_type'],
            'case_id': int(case['case_id']),
            'source_index': source_index,
            'subject': case.get('subject'),
            'prompt': case.get('prompt'),
            'max_conflict': conflict_info.get('max_pair_score'),
            'mean_conflict': conflict_info.get('mean_pair_score'),
            'rewrite_delta': case.get('rewrite_delta'),
            'post_rewrite_acc': case.get('post_rewrite_acc'),
            'post_rephrase_acc': case.get('post_rephrase_acc'),
            'post_locality_acc': case.get('post_locality_acc'),
            'rewrite_correct': 1.0 if rewrite_event.get('correct_route') else 0.0 if rewrite_event else None,
            'rephrase_correct': 1.0 if rephrase_event.get('correct_route') else 0.0 if rephrase_event else None,
            'rewrite_abstain': 1.0 if rewrite_event.get('chosen_edit_id') is None else 0.0 if rewrite_event else None,
            'rephrase_abstain': 1.0 if rephrase_event.get('chosen_edit_id') is None else 0.0 if rephrase_event else None,
            'rewrite_wrong_route': 1.0 if rewrite_event and rewrite_event.get('chosen_edit_id') is not None and rewrite_event.get('correct_route') is False else 0.0 if rewrite_event else None,
            'rephrase_wrong_route': 1.0 if rephrase_event and rephrase_event.get('chosen_edit_id') is not None and rephrase_event.get('correct_route') is False else 0.0 if rephrase_event else None,
        })
    return rows


def summarize_buckets(rows, num_buckets):
    all_conflicts = [row['max_conflict'] for row in rows if row['max_conflict'] is not None]
    edges = [quantile(all_conflicts, i / num_buckets) for i in range(1, num_buckets)]
    labels = []
    lower = None
    for edge in edges:
        if lower is None:
            labels.append(f'<= {edge:.3f}')
        else:
            labels.append(f'({lower:.3f}, {edge:.3f}]')
        lower = edge
    labels.append('all' if lower is None else f'> {lower:.3f}')

    grouped = {}
    for row in rows:
        idx = bucket_index(row['max_conflict'], edges) if row['max_conflict'] is not None else None
        row['bucket_index'] = idx
        row['bucket_label'] = labels[idx] if idx is not None else 'unknown'
        grouped.setdefault((row['run'], row['bucket_label']), []).append(row)

    bucket_rows = []
    for (run, bucket_label), bucket_rows_raw in grouped.items():
        bucket_rows.append({
            'run': run,
            'bucket_label': bucket_label,
            'count': len(bucket_rows_raw),
            'mean_max_conflict': maybe_mean([r['max_conflict'] for r in bucket_rows_raw]),
            'mean_rewrite_delta': maybe_mean([r['rewrite_delta'] for r in bucket_rows_raw]),
            'mean_post_rewrite_acc': maybe_mean([r['post_rewrite_acc'] for r in bucket_rows_raw]),
            'mean_post_rephrase_acc': maybe_mean([r['post_rephrase_acc'] for r in bucket_rows_raw]),
            'mean_post_locality_acc': maybe_mean([r['post_locality_acc'] for r in bucket_rows_raw]),
            'rewrite_route_acc': maybe_mean([r['rewrite_correct'] for r in bucket_rows_raw]),
            'rephrase_route_acc': maybe_mean([r['rephrase_correct'] for r in bucket_rows_raw]),
            'rewrite_abstain_rate': maybe_mean([r['rewrite_abstain'] for r in bucket_rows_raw]),
            'rephrase_abstain_rate': maybe_mean([r['rephrase_abstain'] for r in bucket_rows_raw]),
            'rewrite_wrong_route_rate': maybe_mean([r['rewrite_wrong_route'] for r in bucket_rows_raw]),
            'rephrase_wrong_route_rate': maybe_mean([r['rephrase_wrong_route'] for r in bucket_rows_raw]),
        })
    bucket_rows.sort(key=lambda row: (row['run'], labels.index(row['bucket_label'])))
    return labels, edges, bucket_rows, rows


def build_markdown(labels, bucket_rows, run_order):
    lines = []
    lines.append('# Experiment Conflict-Stratified Summary')
    lines.append('')
    lines.append('| Run | Bucket | Count | Mean Conflict | Rewrite Delta | Post Rewrite | Post Rephrase | Locality | Rewrite Route Acc | Rewrite Abstain | Rephrase Wrong Route |')
    lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for run in run_order:
        for row in [r for r in bucket_rows if r['run'] == run]:
            route_acc = 'n/a' if row['rewrite_route_acc'] is None else f"{row['rewrite_route_acc']:.3f}"
            rewrite_abstain = 'n/a' if row['rewrite_abstain_rate'] is None else f"{row['rewrite_abstain_rate']:.3f}"
            rephrase_wrong = 'n/a' if row['rephrase_wrong_route_rate'] is None else f"{row['rephrase_wrong_route_rate']:.3f}"
            locality = 'n/a' if row['mean_post_locality_acc'] is None else f"{row['mean_post_locality_acc']:.3f}"
            lines.append(
                f"| {row['run']} | {row['bucket_label']} | {row['count']} | {row['mean_max_conflict']:.3f} | {row['mean_rewrite_delta']:.3f} | {row['mean_post_rewrite_acc']:.3f} | {row['mean_post_rephrase_acc']:.3f} | {locality} | {route_acc} | {rewrite_abstain} | {rephrase_wrong} |"
            )
    lines.append('')
    return '\n'.join(lines) + '\n'


def plot_bucket_metric(labels, bucket_rows, run_order, metric_key, ylabel, output_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = list(range(len(labels)))
    for run in run_order:
        run_rows = {row['bucket_label']: row for row in bucket_rows if row['run'] == run}
        ys = [run_rows[label][metric_key] if label in run_rows else None for label in labels]
        ax.plot(xs, ys, marker='o', label=run)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Collision Score Bucket')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='name=run_dir')
    parser.add_argument('--num-buckets', type=int, default=3)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_order = []
    all_rows = []
    for spec in args.run:
        name, run_dir = spec.split('=', 1)
        run_order.append(name)
        all_rows.extend(load_case_rows(name, Path(run_dir)))

    labels, edges, bucket_rows, enriched_rows = summarize_buckets(all_rows, args.num_buckets)
    summary = {
        'bucket_labels': labels,
        'bucket_edges': edges,
        'rows': bucket_rows,
    }
    (output_dir / 'experiment_conflict_buckets.json').write_text(json.dumps(summary, indent=2))
    (output_dir / 'experiment_conflict_bucket_cases.json').write_text(json.dumps(enriched_rows, indent=2))
    (output_dir / 'experiment_conflict_buckets.md').write_text(build_markdown(labels, bucket_rows, run_order))

    plot_bucket_metric(labels, bucket_rows, run_order, 'mean_rewrite_delta', 'Mean Rewrite Delta', output_dir / 'conflict_bucket_rewrite_delta.png')
    plot_bucket_metric(labels, bucket_rows, run_order, 'mean_post_rewrite_acc', 'Mean Post Rewrite Accuracy', output_dir / 'conflict_bucket_post_rewrite.png')
    plot_bucket_metric(labels, bucket_rows, run_order, 'rewrite_abstain_rate', 'Rewrite Abstain Rate', output_dir / 'conflict_bucket_rewrite_abstain.png')
    plot_bucket_metric(labels, bucket_rows, run_order, 'rephrase_wrong_route_rate', 'Rephrase Wrong-Route Rate', output_dir / 'conflict_bucket_rephrase_wrong_route.png')

    print(json.dumps(summary, indent=2))
