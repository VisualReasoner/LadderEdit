import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


OFFICIAL_INCREMENT_FILES = [
    "wiki_big_edit_20240201_20240220.json",
    "wiki_big_edit_20240220_20240301.json",
    "wiki_big_edit_20240301_20240320.json",
    "wiki_big_edit_20240320_20240401.json",
    "wiki_big_edit_20240401_20240501.json",
    "wiki_big_edit_20240501_20240601.json",
    "wiki_big_edit_20240601_20240620.json",
    "wiki_big_edit_20240620_20240701.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', default='lukasthede/WikiBigEdit', type=str)
    parser.add_argument('--arrow_file', default=None, type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--max_records', default=17000, type=int)
    parser.add_argument('--shuffle_seed', default=None, type=int)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--official_increment_dir', default=None, type=str)
    args = parser.parse_args()

    if args.official_increment_dir:
        output_dir = Path(args.official_increment_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in OFFICIAL_INCREMENT_FILES:
            path = hf_hub_download(
                repo_id=args.repo_id,
                repo_type='dataset',
                filename=filename,
                local_dir=str(output_dir),
            )
            print(f'Downloaded {path}')
        return

    if not args.output_file:
        raise ValueError('Either --output_file or --official_increment_dir must be provided')

    if args.arrow_file:
        dataset = Dataset.from_file(args.arrow_file)
    else:
        dataset = load_dataset(args.repo_id, split=args.split)
    if args.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=args.shuffle_seed)
    if args.max_records is not None and args.max_records > 0:
        dataset = dataset.select(range(min(args.max_records, len(dataset))))

    rows = []
    for row in dataset:
        rows.append(
            {
                'tag': row.get('tag'),
                'subject': row.get('subject'),
                'subject_id': row.get('subject_id'),
                'relation': row.get('relation'),
                'relation_id': row.get('relation_id'),
                'object': row.get('object'),
                'object_id': row.get('object_id'),
                'rephrase': row.get('rephrase'),
                'loc': row.get('loc'),
                'loc_ans': row.get('loc_ans'),
                'mhop': row.get('mhop'),
                'mhop_ans': row.get('mhop_ans'),
                'update': row.get('update'),
                'personas': row.get('personas'),
                'ans': row.get('ans'),
                'ground_truth': row.get('object'),
            }
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f'Wrote {len(rows)} WikiBigEdit rows to {output_path}')


if __name__ == '__main__':
    main()
