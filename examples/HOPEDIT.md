# HopEdit

## Contribution Hierarchy

Use the repo and the paper in this order:

1. Theory: lifelong editing capacity is governed by selective separability.
2. Measurement: Access, Isolation, and Scale are the reporting views of that single bottleneck.
3. Method: HopEdit-v1 is the constructive prototype; HopEdit-v2 is the scalable cell-bank realization.

## Environment

```bash
conda create -n easyedit-hopedit python=3.10
conda activate easyedit-hopedit
pip install -r requirements.txt
```

If your cluster requires a CUDA-specific PyTorch build, reinstall `torch` after the requirements step.

## Quick Smoke Run

```bash
python examples/run_hopedit_editing.py \
  --hparams_dir hparams/HOPEDIT/qwen2.5-7b-instruct-smoke.yaml \
  --data_dir /path/to/editing-data \
  --data_type ZsRE \
  --ds_size 3 \
  --sequential_edit
```

## Oral-Target Workflow

Generate the full NeurIPS oral-target experiment plan:

```bash
python examples/build_hopedit_oral_manifest.py \
  --data_dir /path/to/data \
  --output_root ./outputs/hopedit_oral
```

That command writes:

- `run_controlled_diagnosis.sh`
- `run_wikibigedit_main.sh`
- `run_wikibigedit_wild.sh`
- `run_mechanism_calibration.sh`
- `run_mechanism_composition.sh`
- `build_oral_package.sh`

The package builder enforces the main paper policy:

- controlled diagnosis first
- official WikiBigEdit first increment only
- 10k stream cap with 1k checkpoints
- matched-checkpoint headline claims only
- seeded external baselines when local artifacts are not mounted
- explicit fairness notes for LoRA, AlphaEdit, SIMIE, and SERAC

## Checkpoint-First Suite

Generate the MEMOIR-style checkpoint-first suite:

```bash
python examples/build_hopedit_checkpoint_suite.py \
  --data_dir /path/to/data \
  --output_root ./outputs/hopedit_checkpoint_suite
```

That command writes train/eval scripts for:

- `ZsRE`
- `Hallucination`
- `MQuAKE`
- `WikiBigEdit`

The suite is checkpoint-first by design:

- training jobs save resumable checkpoints only
- offline evaluation jobs score the same checkpoints in `teacher_forcing` and `free_generation` lanes
- the aggregate package is built from saved checkpoint artifacts, not inline training metrics

Aggregate suite outputs with:

```bash
python examples/aggregate_checkpoint_suite.py \
  --suite_root ./outputs/hopedit_checkpoint_suite \
  --output_dir ./outputs/hopedit_checkpoint_suite/package
```

## HopEdit-v2 1k Study

Generate the selective-separability study matrix:

```bash
python examples/build_hopedit_v2_1k_study.py \
  --base_hparams hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware-staged-losslog.yaml \
  --data_dir /path/to/data \
  --output_root ./outputs/hopedit_v2_1k_study
```

That writes:

- generated per-policy/per-budget hparams
- `run_hopedit_v2_1k_study.sh`
- `study_manifest.json`

Aggregate finished runs with:

```bash
python examples/aggregate_hopedit_v2_1k_study.py \
  --study_root ./outputs/hopedit_v2_1k_study \
  --output_path ./outputs/hopedit_v2_1k_study/study_rows.json
```

## Calibration Ablations

The current HopEdit code supports the calibration suite directly through hparams:

- `qwen2.5-7b-instruct-dual-calibration-none.yaml`
- `qwen2.5-7b-instruct-dual-calibration-mean-only.yaml`
- `qwen2.5-7b-instruct-dual-calibration-variance-only.yaml`
- `qwen2.5-7b-instruct-dual-calibration-full.yaml`

These correspond to:

- no calibration
- mean-only centering
- variance-only scaling
- full centering plus scaling

## Composition Proxy Suite

The repo also includes an executable composition proxy suite for the current implementation:

- `semantic_only`
- `activation_only`
- `concat_hidden_activation`
- `concat_plus_calibration`

This keeps the reporting contract stable while the underlying constructive implementation continues to evolve.
