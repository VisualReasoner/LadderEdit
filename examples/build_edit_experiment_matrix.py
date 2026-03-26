import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from examples.edit_experiment_utils import canonical_run_name


DEFAULT_BACKBONES = {
    'qwen2.5-7b-instruct': {
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
        'hopedit': 'hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware-staged.yaml',
        'GRACE': 'hparams/GRACE/qwen2.5-7b-instruct.yaml',
        'ROME': 'hparams/ROME/qwen2.5-7b-instruct.yaml',
        'MEMIT': 'hparams/MEMIT/qwen2.5-7b-instruct.yaml',
        'SIMIE': 'hparams/SimIE/qwen2.5-7b-instruct.yaml',
        'WISE': 'hparams/WISE/qwen2.5-7b-instruct.yaml',
        'LORA': 'hparams/LoRA/qwen2.5-7b-instruct.yaml',
    },
    'llama-3.1-8b-instruct': {
        'model_name': 'meta-llama/Llama-3.1-8B-Instruct',
        'hopedit': 'hparams/HOPEDIT/llama-3.1-8b-instruct-dual-whitened-collisionaware-staged.yaml',
        'GRACE': 'hparams/GRACE/llama3.1-8b-instruct.yaml',
        'ROME': 'hparams/ROME/llama3.1-8b-instruct.yaml',
        'MEMIT': 'hparams/MEMIT/llama3.1-8b-instruct.yaml',
        'SIMIE': 'hparams/SimIE/llama3.1-8b-instruct.yaml',
        'WISE': 'hparams/WISE/llama3.1-8b-instruct.yaml',
        'LORA': 'hparams/LoRA/llama3.1-8b-instruct.yaml',
    },
}


DATASET_CANDIDATES = {
    'ZsRE': ['ZsRE/zsre_mend_edit.json'],
    'CounterFact': ['counterfact/counterfact-edit.json', 'CounterFact/counterfact-edit.json', 'counterfact.json'],
}


HOPEDIT_ABLATIONS_QWEN = {
    'semantic': 'hparams/HOPEDIT/qwen2.5-7b-instruct-semantic-only.yaml',
    'dual_no_whiten': 'hparams/HOPEDIT/qwen2.5-7b-instruct-dual-no-whiten.yaml',
    'dual_whitened': 'hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened.yaml',
    'collisionaware': 'hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware.yaml',
    'staged': 'hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware-staged.yaml',
}


def resolve_dataset_file(data_dir: Path, data_type: str):
    for rel in DATASET_CANDIDATES.get(data_type, []):
        path = data_dir / rel
        if path.exists():
            return str(path)
    return None


def build_run_spec(repo_root: Path, output_root: Path, data_dir: Path, method: str, backbone_key: str, data_type: str, stream_type: str, stream_length: int, seed: int, index_file: str | None = None):
    backbone = DEFAULT_BACKBONES[backbone_key]
    hparams_rel = backbone['hopedit'] if method == 'HOPEDIT' else backbone[method]
    hparams_path = repo_root / hparams_rel
    dataset_file = resolve_dataset_file(data_dir, data_type)
    ready = dataset_file is not None and hparams_path.exists()
    reason = None
    if dataset_file is None:
        reason = f'missing dataset file for {data_type}'
    elif not hparams_path.exists():
        reason = f'missing hparams file {hparams_rel}'

    model_name = backbone['model_name']
    run_name = canonical_run_name(method, model_name, data_type, stream_type, stream_length, seed)
    output_dir = output_root / run_name
    command = (
        f"/home/xiaobing/anaconda3/envs/easyedit-hopedit/bin/python examples/run_edit_experiment.py "
        f"--editing_method {method} --hparams_dir {hparams_rel} --data_dir {data_dir} --data_type {data_type} "
        f"--ds_size {stream_length} --stream_type {stream_type} --seed {seed} --sequential_edit --output_dir {output_dir}"
    )
    if index_file is not None:
        command += f" --index_file {index_file}"
    return {
        'method': method,
        'backbone_key': backbone_key,
        'model_name': model_name,
        'data_type': data_type,
        'stream_type': stream_type,
        'stream_length': stream_length,
        'seed': seed,
        'hparams_path': hparams_rel,
        'dataset_file': dataset_file,
        'index_file': index_file,
        'run_name': run_name,
        'output_dir': str(output_dir),
        'ready': ready,
        'skip_reason': reason,
        'command': command,
    }


def write_shell(path: Path, commands: list[str]):
    header = [
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        'cd /scratch/xiaobing/EasyEdit',
        'export HF_HOME=${HF_HOME:-/scratch/xiaobing/hf_cache}',
        'export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/xiaobing/hf_cache/transformers}',
        'export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}',
        'export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}',
        '',
    ]
    path.write_text('\n'.join(header + commands) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', default='/scratch/xiaobing/EasyEdit')
    parser.add_argument('--data-dir', default='/scratch/xiaobing/EasyEdit/data/data')
    parser.add_argument('--output-root', default='/scratch/xiaobing/EasyEdit/outputs/paper_runs')
    parser.add_argument('--manifest-dir', default='/scratch/xiaobing/EasyEdit/outputs/experiment_manifests')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--collision-sizes', default='16,64,128')
    parser.add_argument('--standard-lengths', default='32,64,128')
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(v.strip()) for v in args.seeds.split(',') if v.strip()]
    collision_sizes = [int(v.strip()) for v in args.collision_sizes.split(',') if v.strip()]
    standard_lengths = [int(v.strip()) for v in args.standard_lengths.split(',') if v.strip()]

    mine_commands = []
    collision_paths = {}
    for data_type in ['ZsRE', 'CounterFact']:
        dataset_file = resolve_dataset_file(data_dir, data_type)
        for size in collision_sizes:
            out_path = output_root / 'benchmarks' / f'collisionbench_{data_type.lower()}_{size}.json'
            collision_paths[(data_type, size)] = str(out_path)
            if dataset_file is None:
                continue
            mine_commands.append(
                f"/home/xiaobing/anaconda3/envs/easyedit-hopedit/bin/python examples/mine_collisionbench.py --data_dir {data_dir} --data_type {data_type} --subset_size {size} --top_pairs 400 --output_path {out_path}"
            )
    write_shell(manifest_dir / 'mine_collisionbench.sh', mine_commands)

    manifest = {
        'defaults': {
            'backbones': list(DEFAULT_BACKBONES.keys()),
            'methods': ['HOPEDIT', 'GRACE', 'ROME', 'MEMIT', 'SIMIE', 'WISE', 'LORA'],
            'datasets': ['ZsRE', 'CounterFact'],
            'seeds': seeds,
            'collision_sizes': collision_sizes,
            'standard_lengths': standard_lengths,
        },
        'waves': {},
    }

    wave1 = []
    for data_type in ['ZsRE', 'CounterFact']:
        for seed in seeds:
            wave1.append(build_run_spec(repo_root, output_root, data_dir, 'HOPEDIT', 'qwen2.5-7b-instruct', data_type, 'collision', 64, seed, index_file=collision_paths[(data_type, 64)]))
        for length in standard_lengths:
            for seed in seeds:
                wave1.append(build_run_spec(repo_root, output_root, data_dir, 'HOPEDIT', 'qwen2.5-7b-instruct', data_type, 'standard', length, seed))
    manifest['waves']['wave1_hopedit'] = wave1

    wave2 = []
    for method in ['GRACE', 'ROME', 'MEMIT', 'SIMIE', 'WISE', 'LORA']:
        for data_type in ['ZsRE', 'CounterFact']:
            for seed in seeds:
                wave2.append(build_run_spec(repo_root, output_root, data_dir, method, 'qwen2.5-7b-instruct', data_type, 'standard', 32, seed))
                wave2.append(build_run_spec(repo_root, output_root, data_dir, method, 'qwen2.5-7b-instruct', data_type, 'collision', 64, seed, index_file=collision_paths[(data_type, 64)]))
    manifest['waves']['wave2_baselines_qwen'] = wave2

    wave3 = []
    for method in ['HOPEDIT', 'GRACE', 'ROME', 'MEMIT', 'SIMIE', 'WISE', 'LORA']:
        for data_type in ['ZsRE', 'CounterFact']:
            for seed in seeds:
                wave3.append(build_run_spec(repo_root, output_root, data_dir, method, 'llama-3.1-8b-instruct', data_type, 'standard', 32, seed))
                wave3.append(build_run_spec(repo_root, output_root, data_dir, method, 'llama-3.1-8b-instruct', data_type, 'collision', 64, seed, index_file=collision_paths[(data_type, 64)]))
    manifest['waves']['wave3_llama_generalization'] = wave3

    ablation_runs = []
    for ablation_name, hparams_rel in HOPEDIT_ABLATIONS_QWEN.items():
        for seed in seeds:
            spec = build_run_spec(repo_root, output_root, data_dir, 'HOPEDIT', 'qwen2.5-7b-instruct', 'ZsRE', 'collision', 64, seed, index_file=collision_paths[('ZsRE', 64)])
            spec['ablation_name'] = ablation_name
            spec['hparams_path'] = hparams_rel
            spec['command'] = spec['command'].replace(DEFAULT_BACKBONES['qwen2.5-7b-instruct']['hopedit'], hparams_rel)
            spec['run_name'] = f"{spec['run_name']}_{ablation_name}"
            spec['output_dir'] = str(output_root / spec['run_name'])
            spec['command'] = spec['command'].replace(str(output_root / canonical_run_name('HOPEDIT', DEFAULT_BACKBONES['qwen2.5-7b-instruct']['model_name'], 'ZsRE', 'collision', 64, seed)), spec['output_dir'])
            ablation_runs.append(spec)
    manifest['waves']['hopedit_ablation_qwen'] = ablation_runs

    write_json = lambda path, payload: path.write_text(json.dumps(payload, indent=2))
    write_json(manifest_dir / 'experiment_matrix.json', manifest)

    for wave_name, specs in manifest['waves'].items():
        ready_commands = [spec['command'] for spec in specs if spec['ready']]
        write_shell(manifest_dir / f'{wave_name}.sh', ready_commands)

    analysis_commands = [
        f"/home/xiaobing/anaconda3/envs/easyedit-hopedit/bin/python examples/analyze_capacity_frontier.py --output-dir {output_root / 'analyses' / 'capacity_frontier'} "
        + ' '.join(
            f"--run {spec['run_name']}={spec['output_dir']}" for wave in manifest['waves'].values() for spec in wave if spec['ready'] and spec['stream_type'] == 'standard'
        ),
        f"/home/xiaobing/anaconda3/envs/easyedit-hopedit/bin/python examples/analyze_experiment_conflict_buckets.py --output-dir {output_root / 'analyses' / 'collision_buckets'} "
        + ' '.join(
            f"--run {spec['run_name']}={spec['output_dir']}" for wave in manifest['waves'].values() for spec in wave if spec['ready'] and spec['stream_type'] == 'collision'
        ),
    ]
    write_shell(manifest_dir / 'wave4_analysis.sh', analysis_commands)

    print(json.dumps({
        'manifest_path': str(manifest_dir / 'experiment_matrix.json'),
        'wave_scripts': [str(manifest_dir / f'{wave}.sh') for wave in manifest['waves']],
        'analysis_script': str(manifest_dir / 'wave4_analysis.sh'),
        'mine_script': str(manifest_dir / 'mine_collisionbench.sh'),
    }, indent=2))
