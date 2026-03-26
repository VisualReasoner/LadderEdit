import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def quantile(values, q):
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    values = sorted(float(v) for v in values)
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


def load_case_rows(run_name: str, run_dir: Path):
    route = load_json(run_dir / 'hopedit_route_diagnostics.json')
    conflict = load_json(run_dir / 'hopedit_conflict_diagnostics.json')
    annotated = [json.loads(line) for line in (run_dir / 'hopedit_route_logs_annotated.jsonl').read_text().splitlines() if line.strip()]

    by_case_event = {}
    for row in annotated:
        case_id = row.get('case_id')
        event_type = row.get('event_type')
        if case_id is None or event_type not in {'rewrite', 'rephrase'}:
            continue
        by_case_event[(int(case_id), event_type)] = row

    combined = conflict.get('combined_conflict')
    rows = []
    for idx, case in enumerate(route['per_case']):
        case_id = int(case['case_id'])
        offdiag = [combined[idx][j] for j in range(len(combined[idx])) if j != idx] if combined else []
        rewrite_event = by_case_event.get((case_id, 'rewrite'), {})
        rephrase_event = by_case_event.get((case_id, 'rephrase'), {})
        rows.append({
            'run': run_name,
            'case_id': case_id,
            'subject': case.get('subject'),
            'prompt': case.get('prompt'),
            'max_conflict': max(offdiag) if offdiag else None,
            'mean_conflict': maybe_mean(offdiag),
            'rewrite_delta': case.get('rewrite_delta'),
            'post_rewrite_acc': case.get('post_rewrite_acc'),
            'post_rephrase_acc': case.get('post_rephrase_acc'),
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
    if lower is None:
        labels.append('all')
    else:
        labels.append(f'> {lower:.3f}')

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
    lines.append('# HopEdit Conflict-Stratified Summary')
    lines.append('')
    lines.append('## Bucket Table')
    lines.append('')
    lines.append('| Run | Bucket | Count | Mean Conflict | Rewrite Delta | Post Rewrite | Post Rephrase | Rewrite Route Acc | Rewrite Abstain | Rephrase Wrong Route |')
    lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for run in run_order:
        for row in [r for r in bucket_rows if r['run'] == run]:
            lines.append(
                f"| {row['run']} | {row['bucket_label']} | {row['count']} | {row['mean_max_conflict']:.3f} | {row['mean_rewrite_delta']:.3f} | {row['mean_post_rewrite_acc']:.3f} | {row['mean_post_rephrase_acc']:.3f} | {row['rewrite_route_acc']:.3f} | {row['rewrite_abstain_rate']:.3f} | {row['rephrase_wrong_route_rate']:.3f} |"
            )
    lines.append('')
    lines.append('## Read')
    lines.append('')
    for run in run_order:
        run_rows = [r for r in bucket_rows if r['run'] == run]
        if not run_rows:
            continue
        hardest = max(run_rows, key=lambda r: r['mean_max_conflict'])
        easiest = min(run_rows, key=lambda r: r['mean_max_conflict'])
        lines.append(f"- {run}: easiest bucket rewrite delta {easiest['mean_rewrite_delta']:.3f}, hardest bucket rewrite delta {hardest['mean_rewrite_delta']:.3f}, hardest bucket post rewrite {hardest['mean_post_rewrite_acc']:.3f}.")
    return '\n'.join(lines) + '\n'


def plot_bucket_metric(labels, bucket_rows, run_order, metric_key, ylabel, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = list(range(len(labels)))
    for run in run_order:
        run_rows = {row['bucket_label']: row for row in bucket_rows if row['run'] == run}
        ys = [run_rows[label][metric_key] if label in run_rows else None for label in labels]
        ax.plot(xs, ys, marker='o', label=run)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Max Combined Conflict Bucket')
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
    (output_dir / 'hopedit_conflict_stratified.json').write_text(json.dumps(summary, indent=2))
    (output_dir / 'hopedit_conflict_stratified_cases.json').write_text(json.dumps(enriched_rows, indent=2))
    (output_dir / 'hopedit_conflict_stratified.md').write_text(build_markdown(labels, bucket_rows, run_order))

    plot_bucket_metric(labels, bucket_rows, run_order, 'mean_rewrite_delta', 'Mean Rewrite Delta', output_dir / 'hopedit_conflict_bucket_rewrite_delta.png')
    plot_bucket_metric(labels, bucket_rows, run_order, 'mean_post_rewrite_acc', 'Mean Post Rewrite Accuracy', output_dir / 'hopedit_conflict_bucket_post_rewrite.png')
    plot_bucket_metric(labels, bucket_rows, run_order, 'rewrite_abstain_rate', 'Rewrite Abstain Rate', output_dir / 'hopedit_conflict_bucket_rewrite_abstain.png')
    plot_bucket_metric(labels, bucket_rows, run_order, 'rephrase_wrong_route_rate', 'Rephrase Wrong-Route Rate', output_dir / 'hopedit_conflict_bucket_rephrase_wrong_route.png')

    print(json.dumps(summary, indent=2))
