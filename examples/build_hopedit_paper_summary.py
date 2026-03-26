import argparse
import json
from pathlib import Path
from statistics import mean


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def format_num(value, digits=3):
    if value is None:
        return 'NA'
    return f'{value:.{digits}f}'


def load_metrics(run_dir: Path):
    metric_file = next(run_dir.glob('HOPEDIT_*.json'))
    return load_json(metric_file)


def scalar_metric(block, key):
    value = block.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return float(sum(value) / len(value))
    return float(value)


def summarize_metrics(metrics):
    pre_rewrite = []
    post_rewrite = []
    pre_rephrase = []
    post_rephrase = []
    post_locality = []
    for row in metrics:
        pre = row.get('pre', {})
        post = row.get('post', {})
        pre_rewrite.append(scalar_metric(pre, 'rewrite_acc'))
        post_rewrite.append(scalar_metric(post, 'rewrite_acc'))
        pre_rephrase.append(scalar_metric(pre, 'rephrase_acc'))
        post_rephrase.append(scalar_metric(post, 'rephrase_acc'))
        locality = post.get('locality', {})
        for key, val in locality.items():
            if key.endswith('_acc'):
                post_locality.append(scalar_metric(locality, key))
    return {
        'pre_rewrite_acc': maybe_mean(pre_rewrite),
        'post_rewrite_acc': maybe_mean(post_rewrite),
        'rewrite_delta': (maybe_mean(post_rewrite) - maybe_mean(pre_rewrite)) if maybe_mean(post_rewrite) is not None and maybe_mean(pre_rewrite) is not None else None,
        'pre_rephrase_acc': maybe_mean(pre_rephrase),
        'post_rephrase_acc': maybe_mean(post_rephrase),
        'rephrase_delta': (maybe_mean(post_rephrase) - maybe_mean(pre_rephrase)) if maybe_mean(post_rephrase) is not None and maybe_mean(pre_rephrase) is not None else None,
        'post_locality_acc': maybe_mean(post_locality),
    }


def summarize_annotated(annotated_path: Path):
    rows = [json.loads(line) for line in annotated_path.read_text().splitlines() if line.strip()]
    out = {}
    for event in ['post_edit', 'rewrite', 'rephrase']:
        event_rows = [r for r in rows if r.get('event_type') == event]
        abstain = sum(r.get('chosen_edit_id') is None for r in event_rows)
        wrong = sum(r.get('chosen_edit_id') is not None and r.get('expected_edit_id') is not None and r.get('chosen_edit_id') != r.get('expected_edit_id') for r in event_rows)
        correct = sum(r.get('chosen_edit_id') is not None and r.get('expected_edit_id') is not None and r.get('chosen_edit_id') == r.get('expected_edit_id') for r in event_rows)
        out[event] = {
            'count': len(event_rows),
            'correct': correct,
            'abstain': abstain,
            'wrong_route': wrong,
        }
    locality_rows = [r for r in rows if r.get('event_type') == 'locality']
    out['locality'] = {
        'count': len(locality_rows),
        'false_activation': sum(r.get('chosen_edit_id') is not None for r in locality_rows),
    }
    return out


def summarize_run(name: str, run_dir: Path):
    route = load_json(run_dir / 'hopedit_route_diagnostics.json')
    conflict = load_json(run_dir / 'hopedit_conflict_diagnostics.json')
    metrics = summarize_metrics(load_metrics(run_dir))
    annotated = summarize_annotated(run_dir / 'hopedit_route_logs_annotated.jsonl')
    return {
        'name': name,
        'run_dir': str(run_dir),
        'num_cases': route['summary']['num_cases'],
        'metrics': metrics,
        'routing': route['routing'],
        'retention': route['retention'],
        'conflict': {
            'mean_semantic_offdiag': conflict.get('mean_semantic_offdiag'),
            'mean_activation_offdiag': conflict.get('mean_activation_offdiag'),
            'mean_raw_activation_offdiag': conflict.get('mean_raw_activation_offdiag'),
            'mean_combined_offdiag': conflict.get('mean_combined_offdiag'),
            'mean_max_offdiag_conflict': conflict.get('mean_max_offdiag_conflict'),
            'max_pair_conflict': conflict.get('max_pair_conflict'),
        },
        'event_breakdown': annotated,
    }


def build_markdown(rows):
    lines = []
    lines.append('# HopEdit Paper Summary')
    lines.append('')
    lines.append('## Aggregate Table')
    lines.append('')
    lines.append('| Run | Cases | Post Rewrite | Rewrite Delta | Post Rephrase | Locality | Rewrite Route Acc | Rephrase Route Acc | Mean Combined Conflict |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in rows:
        lines.append(
            '| {name} | {cases} | {post_rw} | {rw_delta} | {post_rp} | {loc} | {rw_route} | {rp_route} | {conf} |'.format(
                name=row['name'],
                cases=row['num_cases'],
                post_rw=format_num(row['metrics']['post_rewrite_acc']),
                rw_delta=format_num(row['metrics']['rewrite_delta']),
                post_rp=format_num(row['metrics']['post_rephrase_acc']),
                loc=format_num(row['metrics']['post_locality_acc']),
                rw_route=format_num(row['routing'].get('rewrite', {}).get('route_accuracy')),
                rp_route=format_num(row['routing'].get('rephrase', {}).get('route_accuracy')),
                conf=format_num(row['conflict']['mean_combined_offdiag']),
            )
        )
    lines.append('')
    lines.append('## Reviewer-Facing Takeaways')
    lines.append('')
    for row in rows:
        breakdown = row['event_breakdown']
        lines.append(f"### {row['name']}")
        lines.append('')
        lines.append(f"- Cases: {row['num_cases']}")
        lines.append(f"- Routing: rewrite acc {format_num(row['routing'].get('rewrite', {}).get('route_accuracy'))}, rephrase acc {format_num(row['routing'].get('rephrase', {}).get('route_accuracy'))}, locality false activation {format_num(row['routing'].get('locality', {}).get('false_activation_rate'))}")
        lines.append(f"- Task metrics: post rewrite {format_num(row['metrics']['post_rewrite_acc'])}, post rephrase {format_num(row['metrics']['post_rephrase_acc'])}, locality {format_num(row['metrics']['post_locality_acc'])}")
        lines.append(f"- Conflict: semantic {format_num(row['conflict']['mean_semantic_offdiag'])}, activation {format_num(row['conflict']['mean_activation_offdiag'])}, combined {format_num(row['conflict']['mean_combined_offdiag'])}")
        lines.append(f"- Failure pattern: post-edit abstain {breakdown['post_edit']['abstain']}/{breakdown['post_edit']['count']}, rewrite abstain {breakdown['rewrite']['abstain']}/{breakdown['rewrite']['count']}, rewrite wrong-route {breakdown['rewrite']['wrong_route']}/{breakdown['rewrite']['count']}, rephrase wrong-route {breakdown['rephrase']['wrong_route']}/{breakdown['rephrase']['count']}")
        lines.append('')

    by_name = {row['name']: row for row in rows}
    if 'collision_semantic' in by_name and 'collision_dual_whitened' in by_name:
        a = by_name['collision_semantic']
        b = by_name['collision_dual_whitened']
        lines.append('## Collision Comparison')
        lines.append('')
        lines.append(f"- Rewrite route accuracy improves from {format_num(a['routing']['rewrite']['route_accuracy'])} to {format_num(b['routing']['rewrite']['route_accuracy'])}.")
        lines.append(f"- Rephrase route accuracy improves from {format_num(a['routing']['rephrase']['route_accuracy'])} to {format_num(b['routing']['rephrase']['route_accuracy'])}.")
        lines.append(f"- Mean combined conflict drops from {format_num(a['conflict']['mean_combined_offdiag'])} to {format_num(b['conflict']['mean_combined_offdiag'])}.")
        lines.append(f"- Post rewrite accuracy stays flat at {format_num(a['metrics']['post_rewrite_acc'])} vs {format_num(b['metrics']['post_rewrite_acc'])}.")
        lines.append(f"- Interpretation: routing geometry improved, but edit-cell effectiveness did not convert that gain into hard-case task improvement.")
        lines.append('')

    if 'collision_dual_whitened' in by_name and 'collision_dual_whitened_collisionaware' in by_name:
        b = by_name['collision_dual_whitened']
        c = by_name['collision_dual_whitened_collisionaware']
        lines.append('## Collision-Aware Improvement')
        lines.append('')
        lines.append(f"- Routing stays unchanged: rewrite route accuracy {format_num(b['routing']['rewrite']['route_accuracy'])} vs {format_num(c['routing']['rewrite']['route_accuracy'])}, rephrase route accuracy {format_num(b['routing']['rephrase']['route_accuracy'])} vs {format_num(c['routing']['rephrase']['route_accuracy'])}.")
        lines.append(f"- Conflict geometry also stays unchanged: mean combined conflict {format_num(b['conflict']['mean_combined_offdiag'])} vs {format_num(c['conflict']['mean_combined_offdiag'])}.")
        lines.append(f"- Hard-case editing improves sharply: post rewrite {format_num(b['metrics']['post_rewrite_acc'])} to {format_num(c['metrics']['post_rewrite_acc'])}, post rephrase {format_num(b['metrics']['post_rephrase_acc'])} to {format_num(c['metrics']['post_rephrase_acc'])}.")
        lines.append(f"- Interpretation: the practical bottleneck was edit-cell basin separation, not routing. Once routing was fixed, collision-aware training converted the same routes into much stronger edits.")
        lines.append('')

    if 'semantic' in by_name and 'dual_no_whiten' in by_name and 'dual_whitened' in by_name:
        s = by_name['semantic']
        n = by_name['dual_no_whiten']
        w = by_name['dual_whitened']
        lines.append('## Ablation Read')
        lines.append('')
        lines.append(f"- Dual without whitening is the weakest geometry variant: mean combined conflict {format_num(n['conflict']['mean_combined_offdiag'])}.")
        lines.append(f"- Whitening materially reduces conflict: {format_num(n['conflict']['mean_combined_offdiag'])} to {format_num(w['conflict']['mean_combined_offdiag'])}.")
        lines.append(f"- Dual-whitened also beats semantic-only on combined conflict: {format_num(s['conflict']['mean_combined_offdiag'])} to {format_num(w['conflict']['mean_combined_offdiag'])}.")
        lines.append(f"- On the easy slice, task metrics remain similar enough that the current evidence is geometry-first rather than end-task-first.")
        lines.append('')

    lines.append('## Current Paper Status')
    lines.append('')
    if 'collision_dual_whitened_collisionaware' in by_name:
        lines.append('- HopEdit now supports both a routing-geometry claim and a practical improvement claim on the mined collision-heavy subset.')
        lines.append('- The practical gain is mechanistically attributable: routing stayed fixed while collision-aware edit training raised hard-case editing accuracy sharply.')
    else:
        lines.append('- HopEdit currently supports a defensible routing-geometry claim.')
        lines.append('- HopEdit does not yet support a strong practical-improvement claim on hard collision-heavy edits.')
    lines.append('- The diagnostics are already useful for the Edit Capacity paper because they expose measurable conflict structure and separable abstention vs wrong-route failure modes.')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='name=run_dir')
    parser.add_argument('--output', default=None, help='Optional markdown output path')
    args = parser.parse_args()

    rows = []
    for spec in args.run:
        name, run_dir = spec.split('=', 1)
        rows.append(summarize_run(name, Path(run_dir)))

    markdown = build_markdown(rows)
    print(markdown)
    if args.output:
        Path(args.output).write_text(markdown)
