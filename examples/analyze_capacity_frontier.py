import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def build_markdown(rows):
    lines = []
    lines.append('# Capacity Frontier Summary')
    lines.append('')
    lines.append('| Method | Backbone | Dataset | Stream Type | Threshold Rewrite | Threshold Locality | Short Rewrite | Frontier Length | Lengths Seen |')
    lines.append('| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |')
    for row in rows:
        lengths_seen = ', '.join(str(v) for v in row['lengths_seen'])
        lines.append(
            f"| {row['method']} | {row['backbone']} | {row['dataset']} | {row['stream_type']} | {row['rewrite_threshold']:.2f} | {row['locality_threshold']:.2f} | {row['short_post_rewrite']:.3f} | {row['frontier_length']} | {lengths_seen} |"
        )
    lines.append('')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='name=run_dir')
    parser.add_argument('--rewrite-threshold', type=float, default=0.90)
    parser.add_argument('--locality-threshold', type=float, default=0.95)
    parser.add_argument('--stream-type', default=None)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    raw_rows = []
    for spec in args.run:
        _, run_dir = spec.split('=', 1)
        run_path = Path(run_dir)
        run_config = load_json(run_path / 'run_config.json')
        summary = load_json(run_path / 'summary.json')
        if args.stream_type and run_config.get('stream_type') != args.stream_type:
            continue
        row = {
            'method': run_config['editing_method'],
            'backbone': run_config['backbone'],
            'dataset': run_config['data_type'],
            'stream_type': run_config['stream_type'],
            'seed': run_config['seed'],
            'stream_length': summary['stream_length'],
            'post_rewrite_mean': summary['post_rewrite_mean'],
            'post_locality_mean': summary['post_locality_mean'],
            'run_dir': str(run_path),
        }
        raw_rows.append(row)
        grouped[(row['method'], row['backbone'], row['dataset'], row['stream_type'])].append(row)

    frontier_rows = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: (row['stream_length'], row['seed']))
        by_length = defaultdict(list)
        for row in rows:
            by_length[row['stream_length']].append(row)
        lengths = sorted(by_length)
        short_length = lengths[0]
        short_post_rewrite = maybe_mean([row['post_rewrite_mean'] for row in by_length[short_length]])
        rewrite_cutoff = None if short_post_rewrite is None else short_post_rewrite * args.rewrite_threshold
        frontier_length = None
        for length in lengths:
            length_rewrite = maybe_mean([row['post_rewrite_mean'] for row in by_length[length]])
            length_locality = maybe_mean([row['post_locality_mean'] for row in by_length[length]])
            if rewrite_cutoff is None or length_rewrite is None:
                continue
            if length_rewrite >= rewrite_cutoff and (length_locality is None or length_locality >= args.locality_threshold):
                frontier_length = length
        frontier_rows.append({
            'method': key[0],
            'backbone': key[1],
            'dataset': key[2],
            'stream_type': key[3],
            'rewrite_threshold': args.rewrite_threshold,
            'locality_threshold': args.locality_threshold,
            'short_post_rewrite': short_post_rewrite,
            'rewrite_cutoff': rewrite_cutoff,
            'frontier_length': frontier_length,
            'lengths_seen': lengths,
        })

    frontier_rows.sort(key=lambda row: (row['method'], row['backbone'], row['dataset'], row['stream_type']))
    (output_dir / 'capacity_frontier.json').write_text(json.dumps({'rows': frontier_rows, 'raw_rows': raw_rows}, indent=2))
    (output_dir / 'capacity_frontier.md').write_text(build_markdown(frontier_rows))
    print(json.dumps({'rows': frontier_rows}, indent=2))
