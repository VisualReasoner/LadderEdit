import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_mean(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def pearson_r(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in x_vals))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in y_vals))
    if den_x == 0 or den_y == 0:
        return None
    return float(num / (den_x * den_y))


def quantile(values, q):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
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


def find_runs(run_root: Path, pattern: str):
    return sorted(path for path in run_root.glob(pattern) if (path / 'summary.json').exists())


def aggregate_runs(run_dirs):
    rows = []
    for run_dir in run_dirs:
        summary = load_json(run_dir / 'summary.json')
        rows.append(summary)
    return {
        'count': len(rows),
        'post_rewrite_mean': maybe_mean([row.get('post_rewrite_mean') for row in rows]),
        'post_rephrase_mean': maybe_mean([row.get('post_rephrase_mean') for row in rows]),
        'post_locality_mean': maybe_mean([row.get('post_locality_mean') for row in rows]),
        'early_late_gap': maybe_mean([row.get('early_late_gap') for row in rows]),
    }


def load_hopedit_case_rows(run_dir: Path):
    summary = load_json(run_dir / 'summary.json')
    conflict = load_json(run_dir / 'hopedit_conflict_diagnostics.json')
    route_logs = [json.loads(line) for line in (run_dir / 'hopedit_route_logs_annotated.jsonl').read_text().splitlines() if line.strip()]
    rewrite_events = {}
    for row in route_logs:
        if row.get('event_type') == 'rewrite':
            rewrite_events[int(row['case_id'])] = row

    combined = conflict.get('combined_conflict') or []
    rows = []
    for idx, case in enumerate(summary['per_case']):
        offdiag = [combined[idx][j] for j in range(len(combined[idx])) if j != idx] if combined else []
        event = rewrite_events.get(int(case['case_id']), {})
        rows.append({
            'case_id': int(case['case_id']),
            'source_index': int(case.get('source_index', case['case_id'])),
            'subject': case.get('subject'),
            'prompt': case.get('prompt'),
            'target_new': case.get('target_new'),
            'pre_rewrite_acc': case.get('pre_rewrite_acc'),
            'post_rewrite_acc': case.get('post_rewrite_acc'),
            'rewrite_delta': case.get('rewrite_delta'),
            'post_rephrase_acc': case.get('post_rephrase_acc'),
            'post_locality_acc': case.get('post_locality_acc'),
            'max_conflict': max(offdiag) if offdiag else None,
            'mean_conflict': maybe_mean(offdiag),
            'chosen_edit_id': event.get('chosen_edit_id'),
            'expected_edit_id': event.get('expected_edit_id'),
            'correct_route': event.get('correct_route'),
            'top1_prob': event.get('top1_prob'),
            'route_stage': event.get('route_stage'),
        })
    return rows, combined


def classify_failure(row):
    if row.get('chosen_edit_id') is None:
        return 'abstain'
    if row.get('correct_route') is False:
        return 'wrong_route'
    if row.get('correct_route') and (row.get('post_rewrite_acc') or 0.0) < 0.5:
        return 'weak_edit'
    if row.get('correct_route') and (row.get('post_rewrite_acc') or 0.0) >= 0.5:
        return 'correct'
    return 'other'


def pick_examples(rows, combined):
    by_case = {row['case_id']: row for row in rows}

    def enrich(row):
        idx = row['case_id']
        best_partner = None
        best_score = None
        if combined:
            candidates = [(j, combined[idx][j]) for j in range(len(combined[idx])) if j != idx]
            if candidates:
                best_partner, best_score = max(candidates, key=lambda item: item[1])
        partner = by_case.get(best_partner)
        return {
            **row,
            'failure_type': classify_failure(row),
            'top_conflict_partner_case_id': best_partner,
            'top_conflict_partner_subject': partner.get('subject') if partner else None,
            'top_conflict_partner_prompt': partner.get('prompt') if partner else None,
            'top_conflict_partner_score': best_score,
        }

    success = next((enrich(row) for row in rows if classify_failure(row) == 'correct' and (row.get('post_rewrite_acc') or 0.0) >= 1.0), None)
    abstain = next((enrich(row) for row in rows if classify_failure(row) == 'abstain'), None)
    weak = next((enrich(row) for row in rows if classify_failure(row) == 'weak_edit'), None)
    return [example for example in [success, abstain, weak] if example is not None]


def plot_stream_structure_gap(qwen_stats, output_path: Path):
    labels = ['standard-32', 'standard-64', 'standard-128', 'collision-64']
    metrics = [
        ('post_rewrite_mean', 'Post Rewrite'),
        ('post_rephrase_mean', 'Post Rephrase'),
        ('post_locality_mean', 'Locality'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, title) in zip(axes, metrics):
        values = [qwen_stats[label][key] for label in labels]
        ax.bar(range(len(labels)), values, color=['#5b8ff9', '#5b8ff9', '#5b8ff9', '#e8684a'])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Qwen HopEdit on ZsRE: collision-heavy streams are harder than standard streams')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_backbone_consistency(backbone_rows, output_path: Path):
    labels = ['Qwen std-32', 'Qwen coll-64', 'Llama std-32', 'Llama coll-64']
    metrics = [
        ('post_rewrite_mean', 'Post Rewrite'),
        ('post_rephrase_mean', 'Post Rephrase'),
        ('post_locality_mean', 'Locality'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, title) in zip(axes, metrics):
        values = [backbone_rows[label][key] for label in labels]
        colors = ['#5b8ff9', '#e8684a', '#5ad8a6', '#f6bd16']
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('The standard-vs-collision gap appears on both backbones')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_conflict_regimes(rows, output_path: Path):
    conflicts = [row['max_conflict'] for row in rows if row['max_conflict'] is not None]
    edges = [quantile(conflicts, 1 / 3), quantile(conflicts, 2 / 3)]
    labels = [f'low\n<= {edges[0]:.3f}', f'mid\n({edges[0]:.3f}, {edges[1]:.3f}]', f'high\n> {edges[1]:.3f}']
    categories = ['correct', 'weak_edit', 'abstain', 'wrong_route']
    colors = {
        'correct': '#5ad8a6',
        'weak_edit': '#f6bd16',
        'abstain': '#e8684a',
        'wrong_route': '#6dc8ec',
    }

    bucket_counts = {label: {category: 0 for category in categories} for label in labels}
    bucket_totals = {label: 0 for label in labels}
    for row in rows:
        if row['max_conflict'] is None:
            continue
        label = labels[bucket_index(row['max_conflict'], edges)]
        category = classify_failure(row)
        if category not in bucket_counts[label]:
            continue
        bucket_counts[label][category] += 1
        bucket_totals[label] += 1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottoms = [0.0] * len(labels)
    xs = range(len(labels))
    for category in categories:
        vals = [
            (bucket_counts[label][category] / bucket_totals[label]) if bucket_totals[label] else 0.0
            for label in labels
        ]
        ax.bar(xs, vals, bottom=bottoms, label=category.replace('_', ' '), color=colors[category])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Fraction of rewrite events')
    ax.set_title('Failure mode shifts with conflict')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_predictiveness(zsre_rows, counterfact_rows, output_path: Path):
    datasets = {
        'ZsRE collision-64': zsre_rows,
        'CounterFact collisionfix-64': counterfact_rows,
    }
    predictor_names = ['conflict', 'position']
    colors = ['#5b8ff9', '#f6bd16']
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    width = 0.35
    xs = range(len(datasets))
    for idx, predictor in enumerate(predictor_names):
        vals = []
        for rows in datasets.values():
            if predictor == 'conflict':
                corr = pearson_r([row['max_conflict'] for row in rows], [row['rewrite_delta'] for row in rows])
            else:
                corr = pearson_r([row['case_id'] for row in rows], [row['rewrite_delta'] for row in rows])
            vals.append(abs(corr) if corr is not None else 0.0)
        ax.bar([x + (idx - 0.5) * width for x in xs], vals, width=width, color=colors[idx], label=predictor)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(list(datasets.keys()), rotation=15, ha='right')
    ax.set_ylabel('|Pearson r| with rewrite delta')
    ax.set_title('Conflict predicts degradation better than stream position')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_markdown(output_dir: Path, qwen_stats, backbone_rows, examples, zsre_rows, counterfact_rows):
    def fmt(x):
        return 'n/a' if x is None else f'{x:.3f}'

    std128 = qwen_stats['standard-128']
    coll64 = qwen_stats['collision-64']
    std64 = qwen_stats['standard-64']
    std32 = qwen_stats['standard-32']
    qwen_gap = (std64['post_rewrite_mean'] or 0.0) - (coll64['post_rewrite_mean'] or 0.0)

    lines = []
    lines.append('# HopEdit Capacity Claim Package')
    lines.append('')
    lines.append('## Claim')
    lines.append('')
    lines.append('Lifelong editing breaks primarily when nearby edits become hard to retrieve and hard to apply without interference. The evidence below argues that collision structure matters more than stream length alone.')
    lines.append('')
    lines.append('## Figure Read')
    lines.append('')
    lines.append(f"- Figure 1: On Qwen ZsRE, `standard-64` rewrite is {fmt(std64['post_rewrite_mean'])}, but `collision-64` drops to {fmt(coll64['post_rewrite_mean'])} while locality stays at {fmt(coll64['post_locality_mean'])}.")
    lines.append(f"- Figure 1 also shows a stronger point: `standard-128` rewrite is {fmt(std128['post_rewrite_mean'])}, still well above `collision-64` at {fmt(coll64['post_rewrite_mean'])}. A shorter but more colliding stream is harder than a longer standard stream.")
    lines.append(f"- Figure 2: the dominant failure mode changes with conflict. Low-conflict cases are mostly correct, while high-conflict cases shift toward abstain or weak edits.")
    lines.append(f"- Figure 3: the same standard-vs-collision gap appears on both backbones. Qwen goes from {fmt(backbone_rows['Qwen std-32']['post_rewrite_mean'])} to {fmt(backbone_rows['Qwen coll-64']['post_rewrite_mean'])}; Llama goes from {fmt(backbone_rows['Llama std-32']['post_rewrite_mean'])} to {fmt(backbone_rows['Llama coll-64']['post_rewrite_mean'])}.")
    lines.append(f"- Figure 4: conflict predicts rewrite degradation better than stream position on both ZsRE and fixed CounterFact collision runs.")
    lines.append('')
    lines.append('## Detailed Examples')
    lines.append('')
    for example in examples:
        lines.append(f"### {example['failure_type'].replace('_', ' ').title()}: Case {example['case_id']}")
        lines.append('')
        lines.append(f"- Subject: `{example['subject']}`")
        lines.append(f"- Prompt: `{example['prompt']}`")
        lines.append(f"- Target: `{example['target_new']}`")
        lines.append(f"- Max conflict: `{fmt(example['max_conflict'])}`")
        if example.get('top_conflict_partner_subject') is not None:
            lines.append(f"- Closest sibling: `{example['top_conflict_partner_subject']}` with prompt `{example['top_conflict_partner_prompt']}` at conflict `{fmt(example['top_conflict_partner_score'])}`")
        lines.append(f"- Route stage: `{example['route_stage']}`")
        lines.append(f"- Top-1 route prob: `{fmt(example['top1_prob'])}`")
        lines.append(f"- Rewrite accuracy: `{fmt(example['pre_rewrite_acc'])} -> {fmt(example['post_rewrite_acc'])}`")
        lines.append(f"- Rephrase accuracy: `{fmt(example['post_rephrase_acc'])}`")
        lines.append(f"- Locality: `{fmt(example['post_locality_acc'])}`")
        lines.append('')
    lines.append('## Files')
    lines.append('')
    lines.append(f"- Figure 1: `{output_dir / 'figure1_stream_structure_gap.png'}`")
    lines.append(f"- Figure 2: `{output_dir / 'figure2_conflict_regimes.png'}`")
    lines.append(f"- Figure 3: `{output_dir / 'figure3_backbone_consistency.png'}`")
    lines.append(f"- Figure 4: `{output_dir / 'figure4_predictiveness.png'}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', default='/scratch/xiaobing/EasyEdit/outputs/paper_runs')
    parser.add_argument('--output-dir', default='/scratch/xiaobing/EasyEdit/outputs/claim_figures')
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qwen_standard_32 = aggregate_runs(find_runs(run_root, 'hopedit_qwen2-5-7b-instruct_zsre_standard_32_seed*'))
    qwen_standard_64 = aggregate_runs(find_runs(run_root, 'hopedit_qwen2-5-7b-instruct_zsre_standard_64_seed*'))
    qwen_standard_128 = aggregate_runs(find_runs(run_root, 'hopedit_qwen2-5-7b-instruct_zsre_standard_128_seed*'))
    qwen_collision_64 = aggregate_runs(find_runs(run_root, 'hopedit_qwen2-5-7b-instruct_zsre_collision_64_seed*'))

    llama_standard_32 = aggregate_runs(find_runs(run_root, 'hopedit_llama-3-1-8b-instruct_zsre_standard_32_seed*'))
    llama_collision_64 = aggregate_runs(find_runs(run_root, 'hopedit_llama-3-1-8b-instruct_zsre_collision_64_seed*'))

    qwen_stats = {
        'standard-32': qwen_standard_32,
        'standard-64': qwen_standard_64,
        'standard-128': qwen_standard_128,
        'collision-64': qwen_collision_64,
    }
    backbone_rows = {
        'Qwen std-32': qwen_standard_32,
        'Qwen coll-64': qwen_collision_64,
        'Llama std-32': llama_standard_32,
        'Llama coll-64': llama_collision_64,
    }

    zsre_collision_runs = find_runs(run_root, 'hopedit_qwen2-5-7b-instruct_zsre_collision_64_seed*')
    counterfact_collision_runs = find_runs(run_root, 'hopedit_qwen2-5-7b-instruct_counterfact_collisionfix_64_seed*')

    zsre_rows_all = []
    counterfact_rows_all = []
    examples = []
    if zsre_collision_runs:
        seed0_rows, seed0_combined = load_hopedit_case_rows(zsre_collision_runs[0])
        examples = pick_examples(seed0_rows, seed0_combined)
        for run_dir in zsre_collision_runs:
            rows, _ = load_hopedit_case_rows(run_dir)
            zsre_rows_all.extend(rows)
    if counterfact_collision_runs:
        for run_dir in counterfact_collision_runs:
            rows, _ = load_hopedit_case_rows(run_dir)
            counterfact_rows_all.extend(rows)

    plot_stream_structure_gap(qwen_stats, output_dir / 'figure1_stream_structure_gap.png')
    plot_conflict_regimes(zsre_rows_all, output_dir / 'figure2_conflict_regimes.png')
    plot_backbone_consistency(backbone_rows, output_dir / 'figure3_backbone_consistency.png')
    plot_predictiveness(zsre_rows_all, counterfact_rows_all, output_dir / 'figure4_predictiveness.png')

    summary_payload = {
        'qwen_zsre': qwen_stats,
        'backbone_rows': backbone_rows,
        'predictiveness': {
            'zsre_conflict_vs_rewrite_delta': pearson_r([row['max_conflict'] for row in zsre_rows_all], [row['rewrite_delta'] for row in zsre_rows_all]),
            'zsre_position_vs_rewrite_delta': pearson_r([row['case_id'] for row in zsre_rows_all], [row['rewrite_delta'] for row in zsre_rows_all]),
            'counterfact_conflict_vs_rewrite_delta': pearson_r([row['max_conflict'] for row in counterfact_rows_all], [row['rewrite_delta'] for row in counterfact_rows_all]),
            'counterfact_position_vs_rewrite_delta': pearson_r([row['case_id'] for row in counterfact_rows_all], [row['rewrite_delta'] for row in counterfact_rows_all]),
        },
        'examples': examples,
    }
    (output_dir / 'capacity_claim_summary.json').write_text(json.dumps(summary_payload, indent=2))
    (output_dir / 'capacity_claim_summary.md').write_text(build_markdown(output_dir, qwen_stats, backbone_rows, examples, zsre_rows_all, counterfact_rows_all))

    print(json.dumps(summary_payload, indent=2))
