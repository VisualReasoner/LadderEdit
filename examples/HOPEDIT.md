# HopEdit

## Environment

```bash
conda create -n easyedit-hopedit python=3.10
conda activate easyedit-hopedit
pip install -r requirements.txt
```

If your cluster requires a CUDA-specific PyTorch build, reinstall `torch` after the requirements step.

## Example Run

```bash
cd /scratch/xiaobing/EasyEdit
/home/xiaobing/anaconda3/envs/easyedit-hopedit/bin/python examples/run_hopedit_editing.py \
  --hparams_dir hparams/HOPEDIT/qwen3-8b-base.yaml \
  --data_dir /path/to/editing-data \
  --data_type ZsRE \
  --ds_size 3 \
  --sequential_edit
```
