"""Offline residualized subspace merge POC.

This runner replaces "one cluster -> one merged adapter" with the better unit:

    one cluster -> shared low-rank subspace + per-edit code + residual ladder

For each cluster c and LoRA block l, we build a shared subspace in effective
delta space:

    Δ_i,l^S ≈ U_c,l diag(s_i,l) V_c,l^T

where U_c,l and V_c,l are shared across the cluster and s_i,l is a small
edit-specific code. Each edit can then be served at three levels:

    shared-only:          Δ_i^S
    shared + cheap eps:   Δ_i^S + ε_i
    shared + exact:       Δ_i^E  (exact bypass / full residual fallback)

The goal of this POC is not online retirement yet. It is to answer the next
load-bearing question:

    can a shared subspace with edit-specific codes preserve much more behavior
    than a single merged adapter while staying below exact-bank memory?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor
from easyeditor.editors.utils import _prepare_requests
from examples.edit_experiment_utils import (
    backbone_slug,
    build_editor_inputs,
    load_normalized_records,
    method_name,
    resolve_hparams_class,
    write_json,
)
from examples.run_edit_experiment import seed_everything
from examples.run_shared_basis_compression_poc import make_shards
from examples.run_shared_dictionary_residual_poc import (
    audit_frontier_memory_consistency,
    audit_frontier_memory_monotonicity,
    compute_memory_break_even,
    contract_gap_value,
    contract_pass,
    mean_or_none,
    median_or_none,
    metric_or_none,
    parse_budgets,
    pearson,
    per_case_by_id,
    per_case_utility,
    plot_frontier,
    plot_gap_histogram,
    select_indices,
    spearman,
    summary_subset,
    summarize_best_frontier_points,
    write_csv,
)
from examples.run_wikibigedit_lifelong import (
    apply_single_edit,
    compute_pre_metrics,
    configure_evaluation_mode,
    evaluate_and_write,
)


def choose_subspace_device(requested: str) -> torch.device:
    request = str(requested).strip().lower()
    if request not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported --subspace_device={requested!r}")
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--subspace_device=cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).replace(":", ",").split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def std_or_none(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    mean = sum(clean) / len(clean)
    return float(math.sqrt(sum((value - mean) ** 2 for value in clean) / len(clean)))


def derangement_like(indices: list[int], seed: int) -> list[int]:
    if len(indices) <= 1:
        return list(indices)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    for _ in range(64):
        perm = torch.randperm(len(indices), generator=generator).tolist()
        if all(indices[pos] != indices[perm[pos]] for pos in range(len(indices))):
            return [indices[item] for item in perm]
    return indices[1:] + indices[:1]


def log_progress(output_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "time": time.time(), **extra}
    write_json(output_dir / "progress.json", payload)
    print(f"[progress] {json.dumps(payload, sort_keys=True)}", flush=True)


@dataclass
class EffectiveBlockSpec:
    module_key: str
    a_name: str
    b_name: str
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    in_dim: int
    out_dim: int
    raw_rank: int
    raw_param_count: int


@dataclass
class ClusterBlockStats:
    group_id: int
    shard_size: int
    module_key: str
    out_dim: int
    in_dim: int
    shared_rank_eff: int
    mean_code_abs: float | None
    mean_residual_norm: float | None
    mean_cheap_residual_rank: float | None


@dataclass
class SubspaceEditRow:
    vector_row_index: int
    edit_id: str
    dataset: str
    relation: str | None
    group_id: int
    code_dim: int
    cheap_residual_params: int
    exact_norm: float
    shared_norm: float
    residual_norm: float
    relative_residual_norm: float
    reconstruction_error: float
    exact_rewrite: float | None
    exact_rephrase: float | None
    exact_locality: float | None
    shared_rewrite: float | None
    shared_rephrase: float | None
    shared_locality: float | None
    shared_plus_cheap_rewrite: float | None
    shared_plus_cheap_rephrase: float | None
    shared_plus_cheap_locality: float | None
    exact_utility: float | None
    shared_utility: float | None
    shared_plus_cheap_utility: float | None
    utility_gap_shared: float | None
    utility_gap_shared_plus_cheap: float | None
    contract_gap_shared: float | None
    contract_gap_shared_plus_cheap: float | None
    shared_contract_pass: bool
    shared_plus_cheap_contract_pass: bool
    subject: str | None
    prompt: str | None
    target_new: str | None


def zero_weight_like(reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(tensor) for name, tensor in reference.items()}


def zero_weight_bank(
    edit_ids: list[str],
    exact_weights: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {edit_id: zero_weight_like(exact_weights[edit_id]) for edit_id in edit_ids}


def build_effective_block_specs(reference_weights: dict[str, torch.Tensor]) -> list[EffectiveBlockSpec]:
    pending: dict[str, dict[str, Any]] = {}
    for name in sorted(reference_weights):
        if name.endswith(".lora_A.__ADAPTER__.weight"):
            key = name[: -len(".lora_A.__ADAPTER__.weight")]
            pending.setdefault(key, {})["a_name"] = name
            pending[key]["a_shape"] = tuple(int(dim) for dim in reference_weights[name].shape)
        elif name.endswith(".lora_B.__ADAPTER__.weight"):
            key = name[: -len(".lora_B.__ADAPTER__.weight")]
            pending.setdefault(key, {})["b_name"] = name
            pending[key]["b_shape"] = tuple(int(dim) for dim in reference_weights[name].shape)
    specs: list[EffectiveBlockSpec] = []
    for key in sorted(pending):
        item = pending[key]
        if "a_name" not in item or "b_name" not in item:
            continue
        a_shape = item["a_shape"]
        b_shape = item["b_shape"]
        raw_rank = int(a_shape[0])
        in_dim = int(a_shape[1])
        out_dim = int(b_shape[0])
        if int(b_shape[1]) != raw_rank:
            raise ValueError(f"Mismatched LoRA rank for block {key!r}: {a_shape} vs {b_shape}")
        specs.append(
            EffectiveBlockSpec(
                module_key=key,
                a_name=item["a_name"],
                b_name=item["b_name"],
                a_shape=a_shape,
                b_shape=b_shape,
                in_dim=in_dim,
                out_dim=out_dim,
                raw_rank=raw_rank,
                raw_param_count=int(out_dim * raw_rank + raw_rank * in_dim),
            )
        )
    if not specs:
        raise RuntimeError("No paired LoRA A/B blocks found in reference weights")
    return specs


def capture_exact_weight_bank(controller, edit_ids: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    bank = {edit_id: controller._capture_adapter_parameters(edit_id) for edit_id in edit_ids}
    if not bank:
        raise RuntimeError("No adapter weights captured")
    return bank


def effective_matrix_from_weights(
    weights: dict[str, torch.Tensor],
    block: EffectiveBlockSpec,
    *,
    device: torch.device,
) -> torch.Tensor:
    lora_a = weights[block.a_name].detach().to(device=device, dtype=torch.float32)
    lora_b = weights[block.b_name].detach().to(device=device, dtype=torch.float32)
    return lora_b @ lora_a


def top_eigenvectors(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    size = int(matrix.shape[0])
    if rank <= 0 or size <= 0:
        return torch.zeros(size, 0, dtype=torch.float32, device=matrix.device)
    eig_input = matrix.float() if matrix.is_cuda else matrix.double()
    eigvals, eigvecs = torch.linalg.eigh(eig_input)
    order = torch.argsort(eigvals, descending=True)
    keep: list[int] = []
    for idx in order.tolist():
        value = float(eigvals[idx].item())
        if value <= 1.0e-10:
            continue
        keep.append(int(idx))
        if len(keep) >= rank:
            break
    if not keep:
        return torch.zeros(size, 0, dtype=torch.float32, device=matrix.device)
    return eigvecs[:, keep].to(dtype=torch.float32)


def compute_block_subspace(matrices: list[torch.Tensor], rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not matrices:
        raise ValueError("Cannot build a subspace from no matrices")
    out_dim, in_dim = int(matrices[0].shape[0]), int(matrices[0].shape[1])
    rank = max(0, min(int(rank), out_dim, in_dim))
    if rank <= 0:
        device = matrices[0].device
        return (
            torch.zeros(out_dim, 0, dtype=torch.float32, device=device),
            torch.zeros(in_dim, 0, dtype=torch.float32, device=device),
        )
    device = matrices[0].device
    row_cov = torch.zeros(out_dim, out_dim, dtype=torch.float32, device=device)
    col_cov = torch.zeros(in_dim, in_dim, dtype=torch.float32, device=device)
    for matrix in matrices:
        row_cov += matrix @ matrix.T
        col_cov += matrix.T @ matrix
    u = top_eigenvectors(row_cov, rank)
    v = top_eigenvectors(col_cov, rank)
    rank_eff = min(int(u.shape[1]), int(v.shape[1]))
    return u[:, :rank_eff], v[:, :rank_eff]


def code_from_subspace(matrix: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if int(u.shape[1]) == 0 or int(v.shape[1]) == 0:
        return torch.zeros(0, dtype=torch.float32, device=matrix.device)
    projected = u.T @ matrix @ v
    return torch.diagonal(projected, 0).detach().to(dtype=torch.float32)


def matrix_from_subspace(u: torch.Tensor, v: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    rank_eff = int(code.numel())
    if rank_eff == 0:
        return torch.zeros(int(u.shape[0]), int(v.shape[0]), dtype=torch.float32, device=u.device)
    return u[:, :rank_eff] @ torch.diag(code[:rank_eff]) @ v[:, :rank_eff].T


def truncated_svd_factors(
    matrix: torch.Tensor,
    max_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_rank = max(0, int(max_rank))
    if max_rank <= 0 or float(matrix.abs().max().item()) <= 1.0e-12:
        out_dim, in_dim = int(matrix.shape[0]), int(matrix.shape[1])
        return (
            torch.zeros(out_dim, 0, dtype=torch.float32, device=matrix.device),
            torch.zeros(0, dtype=torch.float32, device=matrix.device),
            torch.zeros(0, in_dim, dtype=torch.float32, device=matrix.device),
        )
    svd_input = matrix.float() if matrix.is_cuda else matrix.double()
    u, s, vh = torch.linalg.svd(svd_input, full_matrices=False)
    keep = 0
    for value in s.tolist():
        if value <= 1.0e-10:
            break
        keep += 1
        if keep >= max_rank:
            break
    if keep <= 0:
        out_dim, in_dim = int(matrix.shape[0]), int(matrix.shape[1])
        return (
            torch.zeros(out_dim, 0, dtype=torch.float32, device=matrix.device),
            torch.zeros(0, dtype=torch.float32, device=matrix.device),
            torch.zeros(0, in_dim, dtype=torch.float32, device=matrix.device),
        )
    return (
        u[:, :keep].to(dtype=torch.float32),
        s[:keep].to(dtype=torch.float32),
        vh[:keep, :].to(dtype=torch.float32),
    )


def lora_factors_from_subspace(
    u: torch.Tensor,
    v: torch.Tensor,
    code: torch.Tensor,
    *,
    target_rank: int,
    ref_a: torch.Tensor,
    ref_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rank_eff = min(int(code.numel()), max(0, int(target_rank)))
    a = torch.zeros((target_rank, int(v.shape[0])), dtype=ref_a.dtype, device=ref_a.device)
    b = torch.zeros((int(u.shape[0]), target_rank), dtype=ref_b.dtype, device=ref_b.device)
    if rank_eff <= 0:
        return a, b, 0
    a[:rank_eff, :] = v[:, :rank_eff].T.to(dtype=ref_a.dtype, device=ref_a.device)
    b[:, :rank_eff] = (u[:, :rank_eff] * code[:rank_eff].unsqueeze(0)).to(dtype=ref_b.dtype, device=ref_b.device)
    return a, b, rank_eff


def lora_factors_from_truncated_svd(
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
    rank: int,
    *,
    target_rank: int,
    ref_a: torch.Tensor,
    ref_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rank_eff = min(max(0, int(rank)), int(s.numel()), max(0, int(target_rank)))
    a = torch.zeros((target_rank, int(vh.shape[1]) if vh.ndim == 2 else int(ref_a.shape[1])), dtype=ref_a.dtype, device=ref_a.device)
    b = torch.zeros((int(u.shape[0]) if u.ndim == 2 else int(ref_b.shape[0]), target_rank), dtype=ref_b.dtype, device=ref_b.device)
    if rank_eff <= 0:
        return a, b, 0
    a[:rank_eff, :] = vh[:rank_eff, :].to(dtype=ref_a.dtype, device=ref_a.device)
    b[:, :rank_eff] = (u[:, :rank_eff] * s[:rank_eff].unsqueeze(0)).to(dtype=ref_b.dtype, device=ref_b.device)
    return a, b, rank_eff


def combine_lora_factors(
    factors: list[tuple[torch.Tensor, torch.Tensor, int]],
    *,
    target_rank: int,
    ref_a: torch.Tensor,
    ref_b: torch.Tensor,
    fallback_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    total_rank = sum(int(rank_eff) for _, _, rank_eff in factors if rank_eff > 0)
    if total_rank <= max(0, int(target_rank)):
        a = torch.zeros((target_rank, int(ref_a.shape[1])), dtype=ref_a.dtype, device=ref_a.device)
        b = torch.zeros((int(ref_b.shape[0]), target_rank), dtype=ref_b.dtype, device=ref_b.device)
        cursor = 0
        for part_a, part_b, rank_eff in factors:
            if rank_eff <= 0:
                continue
            a[cursor:cursor + rank_eff, :] = part_a[:rank_eff, :]
            b[:, cursor:cursor + rank_eff] = part_b[:, :rank_eff]
            cursor += rank_eff
        return a, b, total_rank
    if fallback_matrix is None:
        raise ValueError("fallback_matrix is required when combined rank exceeds target rank")
    return project_matrix_to_lora(fallback_matrix, target_rank, ref_a=ref_a, ref_b=ref_b)


def project_matrix_to_lora(
    matrix: torch.Tensor,
    target_rank: int,
    *,
    ref_a: torch.Tensor,
    ref_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    out_dim, in_dim = int(matrix.shape[0]), int(matrix.shape[1])
    target_rank = max(0, int(target_rank))
    a = torch.zeros((target_rank, in_dim), dtype=ref_a.dtype, device=ref_a.device)
    b = torch.zeros((out_dim, target_rank), dtype=ref_b.dtype, device=ref_b.device)
    if target_rank <= 0:
        return a, b, 0
    if float(matrix.abs().max().item()) <= 1.0e-12:
        return a, b, 0
    svd_input = matrix.float() if matrix.is_cuda else matrix.double()
    u, s, vh = torch.linalg.svd(svd_input, full_matrices=False)
    keep = 0
    for value in s.tolist():
        if value <= 1.0e-10:
            break
        keep += 1
        if keep >= target_rank:
            break
    if keep <= 0:
        return a, b, 0
    u = u[:, :keep].float()
    vh = vh[:keep, :].float()
    sroot = s[:keep].float().sqrt()
    b[:, :keep] = (u * sroot.unsqueeze(0)).to(dtype=ref_b.dtype)
    a[:keep, :] = (sroot.unsqueeze(1) * vh).to(dtype=ref_a.dtype)
    return a, b, keep


def load_weight_bank(
    controller,
    edit_ids: list[str],
    weight_bank: dict[str, dict[str, torch.Tensor]],
) -> None:
    for edit_id in edit_ids:
        controller._load_adapter_parameters(edit_id, weight_bank[edit_id])


def cheap_residual_matrix(
    residual: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, int]:
    if rank <= 0:
        return torch.zeros_like(residual), 0
    ref_a = torch.zeros((rank, int(residual.shape[1])), dtype=torch.float32, device=residual.device)
    ref_b = torch.zeros((int(residual.shape[0]), rank), dtype=torch.float32, device=residual.device)
    a, b, rank_eff = project_matrix_to_lora(residual, rank, ref_a=ref_a, ref_b=ref_b)
    return b.float() @ a.float(), int(rank_eff)


def combine_weight_banks(
    *,
    edit_ids: list[str],
    base_bank: dict[str, dict[str, torch.Tensor]],
    delta_bank: dict[str, dict[str, torch.Tensor]],
    exact_weights: dict[str, dict[str, torch.Tensor]],
    block_specs: list[EffectiveBlockSpec],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    combined = zero_weight_bank(edit_ids, exact_weights)
    for edit_id in edit_ids:
        for block in block_specs:
            base_a = base_bank[edit_id][block.a_name]
            base_b = base_bank[edit_id][block.b_name]
            delta_a = delta_bank[edit_id][block.a_name]
            delta_b = delta_bank[edit_id][block.b_name]
            base_rank = int(((base_b.detach().float().abs().sum(dim=0) > 1.0e-12).sum()).item())
            delta_rank = int(((delta_b.detach().float().abs().sum(dim=0) > 1.0e-12).sum()).item())
            fallback_matrix = None
            if base_rank + delta_rank > block.raw_rank:
                base_matrix = effective_matrix_from_weights(base_bank[edit_id], block, device=device)
                delta_matrix = effective_matrix_from_weights(delta_bank[edit_id], block, device=device)
                fallback_matrix = base_matrix + delta_matrix
            proj_a, proj_b, _ = combine_lora_factors(
                [(base_a, base_b, base_rank), (delta_a, delta_b, delta_rank)],
                target_rank=block.raw_rank,
                ref_a=exact_weights[edit_id][block.a_name],
                ref_b=exact_weights[edit_id][block.b_name],
                fallback_matrix=fallback_matrix,
            )
            combined[edit_id][block.a_name] = proj_a
            combined[edit_id][block.b_name] = proj_b
    return combined


def aggregate_mode_from_rows(rows: list[dict[str, Any]], prefix: str, *, rewrite_threshold: float, rephrase_threshold: float, locality_threshold: float) -> dict[str, Any]:
    rewrite_key = f"{prefix}_rewrite"
    rephrase_key = f"{prefix}_rephrase"
    locality_key = f"{prefix}_locality"
    rewrites = [row.get(rewrite_key) for row in rows]
    rephrases = [row.get(rephrase_key) for row in rows]
    localities = [row.get(locality_key) for row in rows]
    contract_count = 0
    for row in rows:
        r = row.get(rewrite_key)
        p = row.get(rephrase_key)
        l = row.get(locality_key)
        if r is None or p is None or l is None:
            continue
        if r >= rewrite_threshold and p >= rephrase_threshold and l >= locality_threshold:
            contract_count += 1
    count = len(rows)
    return {
        "post_rewrite_mean": mean_or_none(rewrites),
        "post_rephrase_mean": mean_or_none(rephrases),
        "post_locality_mean": mean_or_none(localities),
        "contract_pass_rate": None if count <= 0 else float(contract_count / count),
    }


def run_policy_evaluation_weights(
    *,
    controller,
    edit_ids: list[str],
    weight_bank: dict[str, dict[str, torch.Tensor]],
    storage: dict[str, Any],
    editor: BaseEditor,
    hparams,
    method: str,
    backbone: str,
    output_dir: Path,
    records,
    requests,
    eval_metric: str,
    run_base: dict[str, Any],
    pre_metrics: list[dict[str, Any]],
    condition_name: str,
) -> dict[str, Any]:
    load_weight_bank(controller, edit_ids, weight_bank)
    summary = evaluate_and_write(
        editor,
        hparams,
        method,
        backbone,
        output_dir,
        records,
        requests,
        eval_metric,
        {
            **run_base,
            "run_name": condition_name,
            "condition": condition_name,
            "output_dir": str(output_dir.resolve()),
            **storage,
        },
        pre_metrics=pre_metrics,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT")
    parser.add_argument("--hparams_dir", default=str(REPO_ROOT / "hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware-staged.yaml"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--data_type", default="CounterFact", choices=["CounterFact", "ZsRE", "WikiBigEdit"])
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs/residualized_subspace_merge_poc"))
    parser.add_argument("--ds_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", default="spectral")
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--shared_rank", type=int, default=4)
    parser.add_argument("--cheap_residual_rank", type=int, default=1)
    parser.add_argument("--subspace_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_rephrase_source", choices=["heldout", "address"], default="address")
    parser.add_argument("--evaluation_mode", default="teacher_forcing")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--utility_weights", default="1,1,1")
    parser.add_argument("--contract_rewrite_threshold", type=float, default=0.8)
    parser.add_argument("--contract_rephrase_threshold", type=float, default=0.8)
    parser.add_argument("--contract_locality_threshold", type=float, default=0.95)
    parser.add_argument("--residual_budgets", default="0,1,2,4,8,16,all")
    parser.add_argument("--high_gap_threshold", type=float, default=0.1)
    parser.add_argument("--run_ablation_suite", action="store_true")
    parser.add_argument("--run_rank_ladder_frontier", action="store_true")
    parser.add_argument("--cheap_residual_ranks", default="0,1,2,4")
    parser.add_argument("--num_shuffle_trials", type=int, default=5)
    parser.add_argument("--include_rank_svd_baseline", action="store_true")
    parser.add_argument("--include_random_subspace_baseline", action="store_true")
    parser.add_argument("--include_trained_rank1_exact_baseline", action="store_true")
    args = parser.parse_args()

    if args.include_trained_rank1_exact_baseline:
        raise NotImplementedError(
            "--include_trained_rank1_exact_baseline is not implemented in this runner yet; "
            "it needs a separate exact rank-constrained training path."
        )

    seed_everything(args.seed)
    dataset_slug = str(args.data_type).lower()
    slug_parts = [
        dataset_slug,
        args.strategy,
        f"sr{args.shared_rank}",
        f"cr{args.cheap_residual_rank}",
        f"n{args.ds_size}",
        f"seed{args.seed}",
    ]
    if args.run_ablation_suite:
        slug_parts.append("ablate")
    if args.run_rank_ladder_frontier:
        slug_parts.append("rankladder")
    if args.include_rank_svd_baseline:
        slug_parts.append("svd")
    if args.include_random_subspace_baseline:
        slug_parts.append("randsub")
    run_slug = "_".join(slug_parts)
    output_dir = Path(args.output_root) / run_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = tuple(float(part) for part in args.utility_weights.split(","))
    if len(weights) != 3:
        raise ValueError("--utility_weights must have three comma-separated values")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Utility weights must sum to a positive value")
    weights = tuple(float(value / weight_sum) for value in weights)

    method = method_name(args.editing_method)
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    hparams.sequential_edit = True
    hparams.eval_batch_size = int(args.eval_batch_size)
    configure_evaluation_mode(hparams, args.evaluation_mode, args.api_key)

    records, dataset_file = load_normalized_records(args.data_dir, args.data_type, args.ds_size, data_file=args.data_file)
    editor_inputs = build_editor_inputs(records, args.data_type)
    eval_rephrase_prompts = editor_inputs["rephrase_prompts"]
    if args.eval_rephrase_source == "address" and editor_inputs.get("address_rephrase_prompts"):
        eval_rephrase_prompts = editor_inputs["address_rephrase_prompts"]
    requests = _prepare_requests(
        editor_inputs["prompts"],
        editor_inputs["target_new"],
        editor_inputs["ground_truth"],
        rephrase_prompts=eval_rephrase_prompts,
        locality_inputs=editor_inputs["locality_inputs"],
        portability_inputs=editor_inputs["portability_inputs"],
        subject=editor_inputs["subject"],
    )
    for idx, request in enumerate(requests):
        request.setdefault("case_id", idx)
        request.setdefault("relation_id", editor_inputs["relation_id"][idx])
        request.setdefault("source_index", editor_inputs["source_index"][idx])
        if editor_inputs.get("address_rephrase_prompts"):
            request.setdefault("address_rephrase_prompt", editor_inputs["address_rephrase_prompts"][idx])
        if editor_inputs.get("rephrase_prompts"):
            request.setdefault("heldout_rephrase_prompt", editor_inputs["rephrase_prompts"][idx])

    run_base = {
        "editing_method": method,
        "alg_name": hparams.alg_name,
        "model_name": hparams.model_name,
        "backbone": backbone_slug(hparams.model_name),
        "data_type": args.data_type,
        "dataset_file": str(Path(dataset_file).resolve()),
        "stream_type": "residualized_subspace_merge_poc",
        "seed": args.seed,
        "sequential_edit": True,
        "stream_length": len(records),
        "requested_ds_size": args.ds_size,
        "strategy": args.strategy,
        "shared_rank": args.shared_rank,
        "cheap_residual_rank": args.cheap_residual_rank,
        "subspace_device": args.subspace_device,
    }
    cheap_residual_ranks = sorted(set(parse_int_csv(args.cheap_residual_ranks) + [int(args.cheap_residual_rank)]))
    current_cheap_rank = int(args.cheap_residual_rank)
    write_json(
        output_dir / "run_config.json",
        {
            **run_base,
            "residual_budgets": args.residual_budgets,
            "run_ablation_suite": bool(args.run_ablation_suite),
            "run_rank_ladder_frontier": bool(args.run_rank_ladder_frontier),
            "cheap_residual_ranks": cheap_residual_ranks,
            "num_shuffle_trials": int(args.num_shuffle_trials),
            "include_rank_svd_baseline": bool(args.include_rank_svd_baseline),
            "include_random_subspace_baseline": bool(args.include_random_subspace_baseline),
        },
    )
    work_device = choose_subspace_device(args.subspace_device)
    log_progress(output_dir, "initialized", work_device=str(work_device))

    editor = BaseEditor.from_hparams(hparams)
    pre_metrics = compute_pre_metrics(editor, requests, editor_inputs["eval_metric"])
    log_progress(output_dir, "pre_metrics_complete", request_count=len(requests))
    for request in requests:
        apply_single_edit(editor, request)
    log_progress(output_dir, "exact_edits_applied", edit_count=len(requests))
    controller = editor.model if hasattr(editor.model, "memory_entries") else None
    if controller is None:
        raise RuntimeError("Expected HopEditController after applying edits")

    entries = list(controller.memory_entries)
    edit_ids = [str(entry.get("edit_id")) for entry in entries]
    relation_ids = []
    for idx, entry in enumerate(entries):
        relation_id = entry.get("relation_id")
        if relation_id is None:
            relation_id = requests[idx].get("relation_id")
        relation_ids.append(relation_id)
    if args.strategy.strip().lower() == "relation" and all(relation_id is None for relation_id in relation_ids):
        raise ValueError(
            "Relation sharding was requested, but this dataset provides no relation_id values. "
            "Choose --strategy random or --strategy spectral instead."
        )

    exact_weights = capture_exact_weight_bank(controller, edit_ids)
    log_progress(output_dir, "exact_bank_captured", edit_count=len(edit_ids), block_estimate=len(exact_weights[edit_ids[0]]) if edit_ids else 0)
    block_specs = build_effective_block_specs(exact_weights[edit_ids[0]])
    groups = make_shards(
        args.strategy,
        torch.zeros(len(edit_ids), 1),  # relation/random do not use vector values; keep this deterministic.
        relation_ids,
        num_shards=args.num_shards,
        seed=args.seed,
    ) if args.strategy.strip().lower() in {"relation", "random"} else None
    if groups is None:
        # Build a compact sketch to drive spectral grouping without materializing full effective deltas.
        sketch_rows = []
        for edit_id in edit_ids:
            row_parts = []
            weights_i = exact_weights[edit_id]
            for block in block_specs:
                matrix = effective_matrix_from_weights(weights_i, block, device=work_device)
                row_parts.extend(
                    [
                        float(matrix.norm().item()),
                        float(matrix.mean().item()),
                        float(matrix.abs().max().item()),
                    ]
                )
            sketch_rows.append(torch.tensor(row_parts, dtype=torch.float32))
        sketch = torch.stack(sketch_rows, dim=0)
        groups = make_shards(args.strategy, sketch, relation_ids, num_shards=args.num_shards, seed=args.seed)

    shared_bank = zero_weight_bank(edit_ids, exact_weights)
    shared_plus_cheap_banks = {rank: zero_weight_bank(edit_ids, exact_weights) for rank in cheap_residual_ranks}
    cheap_only_banks = {rank: zero_weight_bank(edit_ids, exact_weights) for rank in cheap_residual_ranks if rank > 0}
    exact_svd_banks = {
        rank: zero_weight_bank(edit_ids, exact_weights)
        for rank in cheap_residual_ranks
        if rank > 0 and (args.include_rank_svd_baseline or args.run_rank_ladder_frontier)
    }
    per_edit_code_params = {edit_id: 0 for edit_id in edit_ids}
    per_edit_cheap_params = {
        rank: {edit_id: 0 for edit_id in edit_ids}
        for rank in cheap_residual_ranks
    }
    per_edit_exact_svd_params = {
        rank: {edit_id: 0 for edit_id in edit_ids}
        for rank in exact_svd_banks
    }
    group_assignments: dict[int, int] = {}
    cluster_rows: list[dict[str, Any]] = []
    shared_basis_params = 0
    exact_norm_sq = {edit_id: 0.0 for edit_id in edit_ids}
    shared_norm_sq = {edit_id: 0.0 for edit_id in edit_ids}
    residual_norm_sq = {edit_id: 0.0 for edit_id in edit_ids}
    group_subspaces: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

    for group_id, indices in enumerate(groups):
        log_progress(output_dir, "group_start", group_id=group_id, shard_size=len(indices))
        for idx in indices:
            group_assignments[int(idx)] = group_id
        group_subspaces[group_id] = {}
        for block_idx, block in enumerate(block_specs):
            if block_idx > 0 and block_idx % 32 == 0:
                log_progress(
                    output_dir,
                    "group_block_progress",
                    group_id=group_id,
                    block_index=block_idx,
                    block_count=len(block_specs),
                    shard_size=len(indices),
                )
            matrices = [
                effective_matrix_from_weights(exact_weights[edit_ids[idx]], block, device=work_device)
                for idx in indices
            ]
            u, v = compute_block_subspace(matrices, args.shared_rank)
            rank_eff = min(int(u.shape[1]), int(v.shape[1]))
            shared_basis_params += (block.out_dim + block.in_dim) * rank_eff
            group_subspaces[group_id][block.module_key] = {
                "u": u.detach().cpu(),
                "v": v.detach().cpu(),
            }
            code_abs_values: list[float] = []
            residual_norms: list[float] = []
            cheap_ranks: list[float] = []
            for local_pos, idx in enumerate(indices):
                edit_id = edit_ids[idx]
                exact_matrix = matrices[local_pos]
                code = code_from_subspace(exact_matrix, u, v)
                shared_matrix = matrix_from_subspace(u, v, code)
                shared_a, shared_b, shared_rank_eff = lora_factors_from_subspace(
                    u,
                    v,
                    code,
                    target_rank=block.raw_rank,
                    ref_a=exact_weights[edit_id][block.a_name],
                    ref_b=exact_weights[edit_id][block.b_name],
                )
                shared_bank[edit_id][block.a_name] = shared_a
                shared_bank[edit_id][block.b_name] = shared_b
                per_edit_code_params[edit_id] += int(code.numel())
                exact_norm_sq[edit_id] += float((exact_matrix * exact_matrix).sum().item())
                shared_norm_sq[edit_id] += float((shared_matrix * shared_matrix).sum().item())
                residual = exact_matrix - shared_matrix
                residual_norm_sq[edit_id] += float((residual * residual).sum().item())
                code_abs_values.extend([abs(float(value)) for value in code.tolist()])
                residual_norms.append(float(residual.norm().item()))
                max_cheap_rank = max(cheap_residual_ranks) if cheap_residual_ranks else 0
                residual_u, residual_s, residual_vh = truncated_svd_factors(residual, max_cheap_rank)
                exact_u = exact_s = exact_vh = None
                max_exact_svd_rank = max(exact_svd_banks.keys()) if exact_svd_banks else 0
                if max_exact_svd_rank > 0:
                    exact_u, exact_s, exact_vh = truncated_svd_factors(exact_matrix, max_exact_svd_rank)
                for cheap_rank in cheap_residual_ranks:
                    cheap_a, cheap_b, cheap_rank_eff = lora_factors_from_truncated_svd(
                        residual_u,
                        residual_s,
                        residual_vh,
                        cheap_rank,
                        target_rank=block.raw_rank,
                        ref_a=exact_weights[edit_id][block.a_name],
                        ref_b=exact_weights[edit_id][block.b_name],
                    )
                    cheap_matrix = torch.zeros_like(residual) if cheap_rank_eff <= 0 else residual_u[:, :cheap_rank_eff] @ torch.diag(residual_s[:cheap_rank_eff]) @ residual_vh[:cheap_rank_eff, :]
                    combined_matrix = shared_matrix + cheap_matrix
                    combined_a, combined_b, _combined_rank_eff = combine_lora_factors(
                        [(shared_a, shared_b, shared_rank_eff), (cheap_a, cheap_b, cheap_rank_eff)],
                        target_rank=block.raw_rank,
                        ref_a=exact_weights[edit_id][block.a_name],
                        ref_b=exact_weights[edit_id][block.b_name],
                        fallback_matrix=combined_matrix,
                    )
                    shared_plus_cheap_banks[cheap_rank][edit_id][block.a_name] = combined_a
                    shared_plus_cheap_banks[cheap_rank][edit_id][block.b_name] = combined_b
                    per_edit_cheap_params[cheap_rank][edit_id] += int((block.out_dim + block.in_dim) * cheap_rank_eff)
                    if cheap_rank > 0:
                        cheap_only_banks[cheap_rank][edit_id][block.a_name] = cheap_a
                        cheap_only_banks[cheap_rank][edit_id][block.b_name] = cheap_b
                    if cheap_rank == current_cheap_rank:
                        cheap_ranks.append(float(cheap_rank_eff))
                    if cheap_rank in exact_svd_banks:
                        exact_svd_a, exact_svd_b, exact_rank_eff = lora_factors_from_truncated_svd(
                            exact_u,
                            exact_s,
                            exact_vh,
                            cheap_rank,
                            target_rank=block.raw_rank,
                            ref_a=exact_weights[edit_id][block.a_name],
                            ref_b=exact_weights[edit_id][block.b_name],
                        )
                        exact_svd_banks[cheap_rank][edit_id][block.a_name] = exact_svd_a
                        exact_svd_banks[cheap_rank][edit_id][block.b_name] = exact_svd_b
                        per_edit_exact_svd_params[cheap_rank][edit_id] += int((block.out_dim + block.in_dim) * exact_rank_eff)
            cluster_rows.append(
                asdict(
                    ClusterBlockStats(
                        group_id=group_id,
                        shard_size=len(indices),
                        module_key=block.module_key,
                        out_dim=block.out_dim,
                        in_dim=block.in_dim,
                        shared_rank_eff=rank_eff,
                        mean_code_abs=mean_or_none(code_abs_values),
                        mean_residual_norm=mean_or_none(residual_norms),
                        mean_cheap_residual_rank=mean_or_none(cheap_ranks),
                    )
                )
            )
        log_progress(output_dir, "group_complete", group_id=group_id, shard_size=len(indices))

    exact_total_params = sum(block.raw_param_count for block in block_specs) * len(edit_ids)
    p_lora = sum(block.raw_param_count for block in block_specs)
    code_params_all = int(sum(per_edit_code_params.values()))
    cheap_params_all_by_rank = {
        rank: int(sum(per_edit_cheap_params[rank].values()))
        for rank in cheap_residual_ranks
    }
    exact_svd_params_all_by_rank = {
        rank: int(sum(per_edit_exact_svd_params[rank].values()))
        for rank in exact_svd_banks
    }
    shared_base_params = int(shared_basis_params + code_params_all)
    memory_break_even = compute_memory_break_even(
        n_edits=len(edit_ids),
        p_lora=p_lora,
        dictionary_params=int(shared_basis_params),
        code_params_all=int(code_params_all),
        p_residual=p_lora,
    )

    write_json(
        output_dir / "subspace_metadata.json",
        {
            "dataset": dataset_slug,
            "n_edits": len(edit_ids),
            "strategy": args.strategy,
            "shared_rank": args.shared_rank,
            "cheap_residual_rank": args.cheap_residual_rank,
            "subspace_device": str(work_device),
            "num_groups": len(groups),
            "group_sizes": [len(group) for group in groups],
            "num_blocks": len(block_specs),
            "shared_basis_params": shared_basis_params,
            "code_params_all": code_params_all,
            "cheap_residual_params_all_by_rank": cheap_params_all_by_rank,
            "exact_svd_params_all_by_rank": exact_svd_params_all_by_rank,
            "exact_total_params": exact_total_params,
            "p_lora": p_lora,
            "block_specs": [asdict(block) for block in block_specs],
            "cluster_block_stats": cluster_rows,
            "implementation_notes": {
                "shared_substrate": "cluster_subspace_plus_edit_code",
                "representation": "effective delta per LoRA block",
                "retirement_rule": "hard contract applied per edit after shared/cheap representations are built",
                "behavior_refinement": "not yet implemented; current codes come from weight-space subspace projection",
            },
        },
    )
    log_progress(output_dir, "subspace_banks_ready", num_groups=len(groups), shared_basis_params=shared_basis_params, code_params_all=code_params_all)

    exact_storage = {
        "exact_params": exact_total_params,
        "total_params": exact_total_params,
        "memory_fraction_vs_exact": 1.0,
        "compression_ratio_vs_exact": 1.0,
    }
    shared_only_storage = {
        "exact_params": exact_total_params,
        "shared_basis_params": shared_basis_params,
        "code_params": code_params_all,
        "cheap_residual_params": 0,
        "total_params": shared_base_params,
        "memory_fraction_vs_exact": float(shared_base_params / exact_total_params),
        "compression_ratio_vs_exact": float(exact_total_params / shared_base_params),
    }
    shared_plus_cheap_storage = {
        "exact_params": exact_total_params,
        "shared_basis_params": shared_basis_params,
        "code_params": code_params_all,
        "cheap_residual_params": cheap_params_all_by_rank[current_cheap_rank],
        "total_params": int(shared_base_params + cheap_params_all_by_rank[current_cheap_rank]),
        "memory_fraction_vs_exact": float((shared_base_params + cheap_params_all_by_rank[current_cheap_rank]) / exact_total_params),
        "compression_ratio_vs_exact": float(exact_total_params / (shared_base_params + cheap_params_all_by_rank[current_cheap_rank])),
    }
    evaluated_mode_summaries: dict[str, dict[str, Any]] = {}

    exact_summary = run_policy_evaluation_weights(
        controller=controller,
        edit_ids=edit_ids,
        weight_bank=exact_weights,
        storage=exact_storage,
        editor=editor,
        hparams=hparams,
        method=method,
        backbone=run_base["backbone"],
        output_dir=output_dir / "mode_exact",
        records=records,
        requests=requests,
        eval_metric=editor_inputs["eval_metric"],
        run_base=run_base,
        pre_metrics=pre_metrics,
        condition_name=f"{dataset_slug}_exact",
    )
    log_progress(output_dir, "mode_complete", mode="exact")
    evaluated_mode_summaries["exact"] = exact_summary
    shared_only_summary = run_policy_evaluation_weights(
        controller=controller,
        edit_ids=edit_ids,
        weight_bank=shared_bank,
        storage=shared_only_storage,
        editor=editor,
        hparams=hparams,
        method=method,
        backbone=run_base["backbone"],
        output_dir=output_dir / "mode_shared_only",
        records=records,
        requests=requests,
        eval_metric=editor_inputs["eval_metric"],
        run_base=run_base,
        pre_metrics=pre_metrics,
        condition_name=f"{dataset_slug}_shared_only",
    )
    log_progress(output_dir, "mode_complete", mode="shared_only")
    evaluated_mode_summaries["shared_only"] = shared_only_summary
    shared_plus_cheap_summary = run_policy_evaluation_weights(
        controller=controller,
        edit_ids=edit_ids,
        weight_bank=shared_plus_cheap_banks[current_cheap_rank],
        storage=shared_plus_cheap_storage,
        editor=editor,
        hparams=hparams,
        method=method,
        backbone=run_base["backbone"],
        output_dir=output_dir / "mode_shared_plus_cheap",
        records=records,
        requests=requests,
        eval_metric=editor_inputs["eval_metric"],
        run_base=run_base,
        pre_metrics=pre_metrics,
        condition_name=f"{dataset_slug}_shared_plus_cheap",
    )
    log_progress(output_dir, "mode_complete", mode="shared_plus_cheap")
    evaluated_mode_summaries["shared_plus_cheap"] = shared_plus_cheap_summary

    def build_storage_record(
        *,
        shared_basis_params_value: int,
        code_params_value: int,
        cheap_residual_params_value: int,
        exact_like_params_value: int = 0,
    ) -> dict[str, Any]:
        total_params = int(shared_basis_params_value + code_params_value + cheap_residual_params_value + exact_like_params_value)
        return {
            "exact_params": exact_total_params,
            "shared_basis_params": int(shared_basis_params_value),
            "code_params": int(code_params_value),
            "cheap_residual_params": int(cheap_residual_params_value),
            "exact_like_params": int(exact_like_params_value),
            "total_params": total_params,
            "memory_fraction_vs_exact": float(total_params / exact_total_params),
            "compression_ratio_vs_exact": float(exact_total_params / total_params) if total_params > 0 else None,
        }

    def evaluate_mode(
        mode_key: str,
        *,
        weight_bank: dict[str, dict[str, torch.Tensor]],
        storage: dict[str, Any],
    ) -> dict[str, Any]:
        summary = run_policy_evaluation_weights(
            controller=controller,
            edit_ids=edit_ids,
            weight_bank=weight_bank,
            storage=storage,
            editor=editor,
            hparams=hparams,
            method=method,
            backbone=run_base["backbone"],
            output_dir=output_dir / f"mode_{mode_key}",
            records=records,
            requests=requests,
            eval_metric=editor_inputs["eval_metric"],
            run_base=run_base,
            pre_metrics=pre_metrics,
            condition_name=f"{dataset_slug}_{mode_key}",
        )
        evaluated_mode_summaries[mode_key] = summary
        log_progress(output_dir, "mode_complete", mode=mode_key)
        return summary

    ablation_mode_rows: list[dict[str, Any]] = []
    ablation_per_edit_rows: list[dict[str, Any]] = []
    rank_sweep_rows: list[dict[str, Any]] = []
    ablation_summary_payload: dict[str, Any] | None = None

    def record_ablation_mode(
        mode_key: str,
        summary: dict[str, Any],
        *,
        family: str,
        rank: int | None,
        total_params: int,
        shared_basis_params_value: int,
        code_params_value: int,
        cheap_residual_params_value: int,
        trial: int | None = None,
        shuffle_kind: str | None = None,
    ) -> None:
        subset = summary_subset(
            summary,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
        )
        utility_mean = None
        if subset.get("post_rewrite_mean") is not None and subset.get("post_rephrase_mean") is not None and subset.get("post_locality_mean") is not None:
            utility_mean = (
                weights[0] * float(subset["post_rewrite_mean"])
                + weights[1] * float(subset["post_rephrase_mean"])
                + weights[2] * float(subset["post_locality_mean"])
            )
        row = {
            "mode": mode_key,
            "family": family,
            "rank": rank,
            "trial": trial,
            "shuffle_kind": shuffle_kind,
            "post_rewrite_mean": subset.get("post_rewrite_mean"),
            "post_rephrase_mean": subset.get("post_rephrase_mean"),
            "post_locality_mean": subset.get("post_locality_mean"),
            "contract_pass_rate": subset.get("contract_pass_rate"),
            "utility_mean": utility_mean,
            "total_params": int(total_params),
            "memory_fraction_vs_exact": float(total_params / exact_total_params),
            "compression_ratio_vs_exact": float(exact_total_params / total_params) if total_params > 0 else None,
            "shared_basis_params": int(shared_basis_params_value),
            "code_params_all": int(code_params_value),
            "cheap_residual_params_all": int(cheap_residual_params_value),
        }
        ablation_mode_rows.append(row)
        if rank is not None and family in {"shared_only", "shared_plus_cheap", "cheap_residual_only", "rank_exact_svd"}:
            rank_sweep_rows.append(row.copy())
        case_rows = per_case_by_id(summary)
        for idx in range(len(edit_ids)):
            case_row = case_rows.get(idx) or {}
            ablation_per_edit_rows.append(
                {
                    "edit_index": idx,
                    "edit_id": edit_ids[idx],
                    "mode": mode_key,
                    "family": family,
                    "rank": rank,
                    "trial": trial,
                    "shuffle_kind": shuffle_kind,
                    "post_rewrite": metric_or_none(case_row, "post_rewrite_acc"),
                    "post_rephrase": metric_or_none(case_row, "post_rephrase_acc"),
                    "post_locality": metric_or_none(case_row, "post_locality_acc"),
                    "utility": per_case_utility(case_row, weights),
                    "contract_pass": contract_pass(
                        case_row,
                        rewrite_threshold=args.contract_rewrite_threshold,
                        rephrase_threshold=args.contract_rephrase_threshold,
                        locality_threshold=args.contract_locality_threshold,
                    ),
                }
            )

    record_ablation_mode(
        "exact",
        exact_summary,
        family="exact",
        rank=None,
        total_params=exact_total_params,
        shared_basis_params_value=0,
        code_params_value=0,
        cheap_residual_params_value=0,
    )
    record_ablation_mode(
        "shared_only",
        shared_only_summary,
        family="shared_only",
        rank=0,
        total_params=shared_base_params,
        shared_basis_params_value=shared_basis_params,
        code_params_value=code_params_all,
        cheap_residual_params_value=0,
    )
    record_ablation_mode(
        f"shared_plus_cheap_rank{current_cheap_rank}",
        shared_plus_cheap_summary,
        family="shared_plus_cheap",
        rank=current_cheap_rank,
        total_params=int(shared_base_params + cheap_params_all_by_rank[current_cheap_rank]),
        shared_basis_params_value=shared_basis_params,
        code_params_value=code_params_all,
        cheap_residual_params_value=cheap_params_all_by_rank[current_cheap_rank],
    )

    rank_ladder_summaries: dict[int, dict[str, Any]] = {}
    if args.run_rank_ladder_frontier:
        for rank, bank in exact_svd_banks.items():
            mode_key = f"rank{rank}_exact_svd"
            summary = evaluated_mode_summaries.get(mode_key)
            if summary is None:
                summary = evaluate_mode(
                    mode_key,
                    weight_bank=bank,
                    storage=build_storage_record(
                        shared_basis_params_value=0,
                        code_params_value=0,
                        cheap_residual_params_value=exact_svd_params_all_by_rank[rank],
                    ),
                )
            rank_ladder_summaries[rank] = summary

    if args.run_ablation_suite:
        for rank in cheap_residual_ranks:
            if rank == current_cheap_rank:
                continue
            summary = evaluate_mode(
                f"shared_plus_cheap_rank{rank}",
                weight_bank=shared_plus_cheap_banks[rank],
                storage=build_storage_record(
                    shared_basis_params_value=shared_basis_params,
                    code_params_value=code_params_all,
                    cheap_residual_params_value=cheap_params_all_by_rank[rank],
                ),
            )
            record_ablation_mode(
                f"shared_plus_cheap_rank{rank}",
                summary,
                family="shared_plus_cheap",
                rank=rank,
                total_params=int(shared_base_params + cheap_params_all_by_rank[rank]),
                shared_basis_params_value=shared_basis_params,
                code_params_value=code_params_all,
                cheap_residual_params_value=cheap_params_all_by_rank[rank],
            )
        for rank, bank in cheap_only_banks.items():
            summary = evaluate_mode(
                f"cheap_residual_only_rank{rank}",
                weight_bank=bank,
                storage=build_storage_record(
                    shared_basis_params_value=0,
                    code_params_value=0,
                    cheap_residual_params_value=cheap_params_all_by_rank[rank],
                ),
            )
            record_ablation_mode(
                f"cheap_residual_only_rank{rank}",
                summary,
                family="cheap_residual_only",
                rank=rank,
                total_params=cheap_params_all_by_rank[rank],
                shared_basis_params_value=0,
                code_params_value=0,
                cheap_residual_params_value=cheap_params_all_by_rank[rank],
            )
        for rank, bank in exact_svd_banks.items():
            mode_key = f"rank{rank}_exact_svd"
            summary = evaluated_mode_summaries.get(mode_key)
            if summary is None:
                summary = evaluate_mode(
                    mode_key,
                    weight_bank=bank,
                    storage=build_storage_record(
                        shared_basis_params_value=0,
                        code_params_value=0,
                        cheap_residual_params_value=exact_svd_params_all_by_rank[rank],
                    ),
                )
            rank_ladder_summaries[rank] = summary
            record_ablation_mode(
                mode_key,
                summary,
                family="rank_exact_svd",
                rank=rank,
                total_params=exact_svd_params_all_by_rank[rank],
                shared_basis_params_value=0,
                code_params_value=0,
                cheap_residual_params_value=exact_svd_params_all_by_rank[rank],
            )
        if current_cheap_rank > 0:
            for trial in range(int(args.num_shuffle_trials)):
                trial_seed = int(args.seed + trial)
                global_perm = derangement_like(list(range(len(edit_ids))), trial_seed)
                global_perm_map = {edit_ids[idx]: edit_ids[global_perm[pos]] for pos, idx in enumerate(range(len(edit_ids)))}
                within_group_perm_map: dict[str, str] = {}
                for group_indices in groups:
                    ordered = [int(value) for value in group_indices]
                    permuted = derangement_like(ordered, trial_seed)
                    for src, dst in zip(ordered, permuted):
                        within_group_perm_map[edit_ids[src]] = edit_ids[dst]
                for shuffle_kind, perm_map in [("global", global_perm_map), ("within_group", within_group_perm_map)]:
                    shuffled_delta_bank = zero_weight_bank(edit_ids, exact_weights)
                    for edit_id in edit_ids:
                        source_edit_id = perm_map[edit_id]
                        shuffled_delta_bank[edit_id] = {
                            name: tensor.clone()
                            for name, tensor in cheap_only_banks[current_cheap_rank][source_edit_id].items()
                        }
                    shuffled_bank = combine_weight_banks(
                        edit_ids=edit_ids,
                        base_bank=shared_bank,
                        delta_bank=shuffled_delta_bank,
                        exact_weights=exact_weights,
                        block_specs=block_specs,
                        device=work_device,
                    )
                    mode_key = f"shared_plus_shuffled_cheap_{shuffle_kind}_rank{current_cheap_rank}_trial{trial}"
                    summary = evaluate_mode(
                        mode_key,
                        weight_bank=shuffled_bank,
                        storage=build_storage_record(
                            shared_basis_params_value=shared_basis_params,
                            code_params_value=code_params_all,
                            cheap_residual_params_value=cheap_params_all_by_rank[current_cheap_rank],
                        ),
                    )
                    record_ablation_mode(
                        mode_key,
                        summary,
                        family="shared_plus_shuffled_cheap",
                        rank=current_cheap_rank,
                        trial=trial,
                        shuffle_kind=shuffle_kind,
                        total_params=int(shared_base_params + cheap_params_all_by_rank[current_cheap_rank]),
                        shared_basis_params_value=shared_basis_params,
                        code_params_value=code_params_all,
                        cheap_residual_params_value=cheap_params_all_by_rank[current_cheap_rank],
                    )
        if args.include_random_subspace_baseline:
            available_groups = list(range(len(groups)))
            mapped_groups = derangement_like(available_groups, args.seed + 1000)
            random_group_map = {group_id: mapped_groups[pos] for pos, group_id in enumerate(available_groups)}
            random_subspace_code_params_all = 0
            random_subspace_cheap_params_all = 0
            random_bank = zero_weight_bank(edit_ids, exact_weights)
            for idx, edit_id in enumerate(edit_ids):
                target_group = group_assignments[idx]
                source_group = random_group_map[target_group]
                weights_i = exact_weights[edit_id]
                for block in block_specs:
                    subspace = group_subspaces[source_group][block.module_key]
                    u = subspace["u"].to(device=work_device, dtype=torch.float32)
                    v = subspace["v"].to(device=work_device, dtype=torch.float32)
                    exact_matrix = effective_matrix_from_weights(weights_i, block, device=work_device)
                    code = code_from_subspace(exact_matrix, u, v)
                    shared_matrix = matrix_from_subspace(u, v, code)
                    cheap_matrix, cheap_rank_eff = cheap_residual_matrix(exact_matrix - shared_matrix, current_cheap_rank)
                    shared_a, shared_b, shared_rank_eff = lora_factors_from_subspace(
                        u,
                        v,
                        code,
                        target_rank=block.raw_rank,
                        ref_a=weights_i[block.a_name],
                        ref_b=weights_i[block.b_name],
                    )
                    cheap_u, cheap_s, cheap_vh = truncated_svd_factors(exact_matrix - shared_matrix, current_cheap_rank)
                    cheap_a, cheap_b, cheap_rank_eff = lora_factors_from_truncated_svd(
                        cheap_u,
                        cheap_s,
                        cheap_vh,
                        current_cheap_rank,
                        target_rank=block.raw_rank,
                        ref_a=weights_i[block.a_name],
                        ref_b=weights_i[block.b_name],
                    )
                    combined_a, combined_b, _ = combine_lora_factors(
                        [(shared_a, shared_b, shared_rank_eff), (cheap_a, cheap_b, cheap_rank_eff)],
                        target_rank=block.raw_rank,
                        ref_a=weights_i[block.a_name],
                        ref_b=weights_i[block.b_name],
                        fallback_matrix=shared_matrix + cheap_matrix,
                    )
                    random_bank[edit_id][block.a_name] = combined_a
                    random_bank[edit_id][block.b_name] = combined_b
                    random_subspace_code_params_all += int(code.numel())
                    random_subspace_cheap_params_all += int((block.out_dim + block.in_dim) * cheap_rank_eff)
            random_summary = evaluate_mode(
                f"random_subspace_plus_cheap_rank{current_cheap_rank}",
                weight_bank=random_bank,
                storage=build_storage_record(
                    shared_basis_params_value=shared_basis_params,
                    code_params_value=random_subspace_code_params_all,
                    cheap_residual_params_value=random_subspace_cheap_params_all,
                ),
            )
            record_ablation_mode(
                f"random_subspace_plus_cheap_rank{current_cheap_rank}",
                random_summary,
                family="random_subspace_plus_cheap",
                rank=current_cheap_rank,
                total_params=int(shared_basis_params + random_subspace_code_params_all + random_subspace_cheap_params_all),
                shared_basis_params_value=shared_basis_params,
                code_params_value=random_subspace_code_params_all,
                cheap_residual_params_value=random_subspace_cheap_params_all,
            )

    exact_by_id = per_case_by_id(exact_summary)
    shared_by_id = per_case_by_id(shared_only_summary)
    shared_plus_cheap_by_id = per_case_by_id(shared_plus_cheap_summary)

    per_edit_rows: list[dict[str, Any]] = []
    shared_gaps: list[float | None] = []
    shared_plus_cheap_gaps: list[float | None] = []
    shared_contract_gaps: list[float] = []
    shared_plus_cheap_contract_gaps: list[float] = []
    residual_norms: list[float] = []
    shared_plus_cheap_costs: list[int] = []
    for idx, request in enumerate(requests):
        edit_id = edit_ids[idx]
        exact_row = exact_by_id.get(idx) or {}
        shared_row = shared_by_id.get(idx) or {}
        shared_plus_cheap_row = shared_plus_cheap_by_id.get(idx) or {}
        utility_exact = per_case_utility(exact_row, weights)
        utility_shared = per_case_utility(shared_row, weights)
        utility_shared_plus_cheap = per_case_utility(shared_plus_cheap_row, weights)
        gap_shared = None if utility_exact is None or utility_shared is None else float(utility_exact - utility_shared)
        gap_shared_plus_cheap = None if utility_exact is None or utility_shared_plus_cheap is None else float(utility_exact - utility_shared_plus_cheap)
        contract_gap_shared = contract_gap_value(
            shared_row,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
            weights=weights,
        )
        contract_gap_shared_plus_cheap = contract_gap_value(
            shared_plus_cheap_row,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
            weights=weights,
        )
        exact_norm = math.sqrt(exact_norm_sq[edit_id])
        shared_norm = math.sqrt(shared_norm_sq[edit_id])
        residual_norm = math.sqrt(residual_norm_sq[edit_id])
        row = SubspaceEditRow(
            vector_row_index=idx,
            edit_id=edit_id,
            dataset=dataset_slug,
            relation=request.get("relation_id"),
            group_id=group_assignments[idx],
            code_dim=int(per_edit_code_params[edit_id]),
            cheap_residual_params=int(per_edit_cheap_params[current_cheap_rank][edit_id]),
            exact_norm=exact_norm,
            shared_norm=shared_norm,
            residual_norm=residual_norm,
            relative_residual_norm=float(residual_norm / max(exact_norm, 1.0e-12)),
            reconstruction_error=float(residual_norm_sq[edit_id] / max(exact_norm_sq[edit_id], 1.0e-12)),
            exact_rewrite=metric_or_none(exact_row, "post_rewrite_acc"),
            exact_rephrase=metric_or_none(exact_row, "post_rephrase_acc"),
            exact_locality=metric_or_none(exact_row, "post_locality_acc"),
            shared_rewrite=metric_or_none(shared_row, "post_rewrite_acc"),
            shared_rephrase=metric_or_none(shared_row, "post_rephrase_acc"),
            shared_locality=metric_or_none(shared_row, "post_locality_acc"),
            shared_plus_cheap_rewrite=metric_or_none(shared_plus_cheap_row, "post_rewrite_acc"),
            shared_plus_cheap_rephrase=metric_or_none(shared_plus_cheap_row, "post_rephrase_acc"),
            shared_plus_cheap_locality=metric_or_none(shared_plus_cheap_row, "post_locality_acc"),
            exact_utility=utility_exact,
            shared_utility=utility_shared,
            shared_plus_cheap_utility=utility_shared_plus_cheap,
            utility_gap_shared=gap_shared,
            utility_gap_shared_plus_cheap=gap_shared_plus_cheap,
            contract_gap_shared=contract_gap_shared,
            contract_gap_shared_plus_cheap=contract_gap_shared_plus_cheap,
            shared_contract_pass=contract_pass(
                shared_row,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            shared_plus_cheap_contract_pass=contract_pass(
                shared_plus_cheap_row,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            subject=request.get("subject"),
            prompt=request.get("prompt"),
            target_new=request.get("target_new"),
        )
        per_edit_rows.append(asdict(row))
        shared_gaps.append(gap_shared)
        shared_plus_cheap_gaps.append(gap_shared_plus_cheap)
        shared_contract_gaps.append(contract_gap_shared)
        shared_plus_cheap_contract_gaps.append(contract_gap_shared_plus_cheap)
        residual_norms.append(residual_norm)
        shared_plus_cheap_costs.append(int(per_edit_cheap_params[current_cheap_rank][edit_id]))

    write_csv(output_dir / "per_edit.csv", per_edit_rows)

    exact_dpr_consistency = {
        "exact_vs_shared_plus_full_rewrite_gap": 0.0,
        "exact_vs_shared_plus_full_rephrase_gap": 0.0,
        "exact_vs_shared_plus_full_locality_gap": 0.0,
        "exact_recovery_mode": "exact_bypass",
    }

    budgets = parse_budgets(args.residual_budgets, len(edit_ids))
    frontier_rows: list[dict[str, Any]] = []

    def aggregate_frontier_row(policy: str, keep_indices: list[int]) -> dict[str, Any]:
        keep_set = set(int(idx) for idx in keep_indices)
        mixed_rows = []
        for idx, row in enumerate(per_edit_rows):
            use_exact = idx in keep_set
            mixed_rows.append(
                {
                    "mode_rewrite": row["exact_rewrite"] if use_exact else row["shared_plus_cheap_rewrite"],
                    "mode_rephrase": row["exact_rephrase"] if use_exact else row["shared_plus_cheap_rephrase"],
                    "mode_locality": row["exact_locality"] if use_exact else row["shared_plus_cheap_locality"],
                }
            )
        summary = aggregate_mode_from_rows(
            [
                {
                    "tier_rewrite": item["mode_rewrite"],
                    "tier_rephrase": item["mode_rephrase"],
                    "tier_locality": item["mode_locality"],
                }
                for item in mixed_rows
            ],
            "tier",
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
        )
        total_params = int(shared_base_params + sum(shared_plus_cheap_costs[idx] for idx in range(len(shared_plus_cheap_costs)) if idx not in keep_set) + len(keep_set) * p_lora)
        return {
            "policy": policy,
            "budget_k": int(len(keep_set)),
            "post_rewrite_mean": summary.get("post_rewrite_mean"),
            "post_rephrase_mean": summary.get("post_rephrase_mean"),
            "post_locality_mean": summary.get("post_locality_mean"),
            "contract_pass_rate": summary.get("contract_pass_rate"),
            "total_params": total_params,
            "memory_fraction_vs_exact": float(total_params / exact_total_params),
            "compression_ratio_vs_exact": float(exact_total_params / total_params),
            "shared_basis_params": shared_basis_params,
            "code_params_all": code_params_all,
            "cheap_residual_params_active": int(sum(shared_plus_cheap_costs[idx] for idx in range(len(shared_plus_cheap_costs)) if idx not in keep_set)),
            "kept_exact_count": int(len(keep_set)),
            "shared_exact_equivalent": memory_break_even.get("shared_exact_equivalent"),
            "memory_margin_before_residuals": memory_break_even.get("memory_margin_before_residuals"),
            "max_residuals_strictly_below_exact": memory_break_even.get("max_residuals_strictly_below_exact"),
            "is_strict_memory_win": bool(total_params < exact_total_params),
        }

    shared_only_row = {
        "policy": "shared_only",
        "budget_k": 0,
        "post_rewrite_mean": shared_only_summary.get("post_rewrite_mean"),
        "post_rephrase_mean": shared_only_summary.get("post_rephrase_mean"),
        "post_locality_mean": shared_only_summary.get("post_locality_mean"),
        "contract_pass_rate": summary_subset(
            shared_only_summary,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
        ).get("contract_pass_rate"),
        "total_params": shared_base_params,
        "memory_fraction_vs_exact": float(shared_base_params / exact_total_params),
        "compression_ratio_vs_exact": float(exact_total_params / shared_base_params),
        "shared_basis_params": shared_basis_params,
        "code_params_all": code_params_all,
        "cheap_residual_params_active": 0,
        "kept_exact_count": 0,
        "shared_exact_equivalent": memory_break_even.get("shared_exact_equivalent"),
        "memory_margin_before_residuals": memory_break_even.get("memory_margin_before_residuals"),
        "max_residuals_strictly_below_exact": memory_break_even.get("max_residuals_strictly_below_exact"),
        "is_strict_memory_win": bool(shared_base_params < exact_total_params),
    }
    shared_plus_cheap_all_row = {
        "policy": "shared_plus_cheap_all",
        "budget_k": 0,
        "post_rewrite_mean": shared_plus_cheap_summary.get("post_rewrite_mean"),
        "post_rephrase_mean": shared_plus_cheap_summary.get("post_rephrase_mean"),
        "post_locality_mean": shared_plus_cheap_summary.get("post_locality_mean"),
        "contract_pass_rate": summary_subset(
            shared_plus_cheap_summary,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
        ).get("contract_pass_rate"),
        "total_params": int(shared_base_params + cheap_params_all_by_rank[current_cheap_rank]),
        "memory_fraction_vs_exact": float((shared_base_params + cheap_params_all_by_rank[current_cheap_rank]) / exact_total_params),
        "compression_ratio_vs_exact": float(exact_total_params / (shared_base_params + cheap_params_all_by_rank[current_cheap_rank])),
        "shared_basis_params": shared_basis_params,
        "code_params_all": code_params_all,
        "cheap_residual_params_active": cheap_params_all_by_rank[current_cheap_rank],
        "kept_exact_count": 0,
        "shared_exact_equivalent": memory_break_even.get("shared_exact_equivalent"),
        "memory_margin_before_residuals": memory_break_even.get("memory_margin_before_residuals"),
        "max_residuals_strictly_below_exact": memory_break_even.get("max_residuals_strictly_below_exact"),
        "is_strict_memory_win": bool(shared_base_params + cheap_params_all_by_rank[current_cheap_rank] < exact_total_params),
    }
    exact_only_row = {
        "policy": "exact_only",
        "budget_k": int(len(edit_ids)),
        "post_rewrite_mean": exact_summary.get("post_rewrite_mean"),
        "post_rephrase_mean": exact_summary.get("post_rephrase_mean"),
        "post_locality_mean": exact_summary.get("post_locality_mean"),
        "contract_pass_rate": summary_subset(
            exact_summary,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
        ).get("contract_pass_rate"),
        "total_params": exact_total_params,
        "memory_fraction_vs_exact": 1.0,
        "compression_ratio_vs_exact": 1.0,
        "shared_basis_params": shared_basis_params,
        "code_params_all": code_params_all,
        "cheap_residual_params_active": 0,
        "kept_exact_count": int(len(edit_ids)),
        "shared_exact_equivalent": memory_break_even.get("shared_exact_equivalent"),
        "memory_margin_before_residuals": memory_break_even.get("memory_margin_before_residuals"),
        "max_residuals_strictly_below_exact": memory_break_even.get("max_residuals_strictly_below_exact"),
        "is_strict_memory_win": False,
    }
    frontier_rows.extend([shared_only_row, shared_plus_cheap_all_row, exact_only_row])

    for budget in budgets:
        if budget <= 0 or budget >= len(edit_ids):
            continue
        policies = {
            "random_exact_over_shared_plus_cheap": select_indices(
                "random",
                budget,
                len(edit_ids),
                residual_norms=residual_norms,
                contract_gaps=shared_plus_cheap_contract_gaps,
                utility_gaps=[0.0 if value is None else value for value in shared_plus_cheap_gaps],
                seed=args.seed,
            ),
            "recency_exact_over_shared_plus_cheap": select_indices(
                "recency",
                budget,
                len(edit_ids),
                residual_norms=residual_norms,
                contract_gaps=shared_plus_cheap_contract_gaps,
                utility_gaps=[0.0 if value is None else value for value in shared_plus_cheap_gaps],
                seed=args.seed,
            ),
            "residual_norm_exact_over_shared_plus_cheap": select_indices(
                "residual_norm",
                budget,
                len(edit_ids),
                residual_norms=residual_norms,
                contract_gaps=shared_plus_cheap_contract_gaps,
                utility_gaps=[0.0 if value is None else value for value in shared_plus_cheap_gaps],
                seed=args.seed,
            ),
            "contract_gap_exact_over_shared_plus_cheap": select_indices(
                "contract_gap",
                budget,
                len(edit_ids),
                residual_norms=residual_norms,
                contract_gaps=shared_plus_cheap_contract_gaps,
                utility_gaps=[0.0 if value is None else value for value in shared_plus_cheap_gaps],
                seed=args.seed,
            ),
            "oracle_utility_gap_exact_over_shared_plus_cheap": select_indices(
                "oracle_utility_gap",
                budget,
                len(edit_ids),
                residual_norms=residual_norms,
                contract_gaps=shared_plus_cheap_contract_gaps,
                utility_gaps=[0.0 if value is None else value for value in shared_plus_cheap_gaps],
                seed=args.seed,
            ),
        }
        for policy_name, keep_indices in policies.items():
            frontier_rows.append(aggregate_frontier_row(policy_name, keep_indices))

    write_csv(output_dir / "frontier.csv", frontier_rows)

    frontier_memory_audit = audit_frontier_memory_monotonicity(frontier_rows)
    frontier_row_consistency = audit_frontier_memory_consistency(
        frontier_rows,
        exact_total_params=exact_total_params,
        shared_exact_equivalent=float(shared_base_params / p_lora),
        memory_margin_before_residuals=int(exact_total_params - shared_base_params),
        max_residuals_strictly_below_exact=None,
    )
    frontier_best_points = summarize_best_frontier_points(frontier_rows, utility_weights=weights)

    rank_ladder_summary_payload: dict[str, Any] | None = None
    if args.run_rank_ladder_frontier:
        rank_ladder_frontier_rows: list[dict[str, Any]] = []
        rank_ladder_per_edit_rows: list[dict[str, Any]] = []
        rank_ladder_mode_metrics: dict[str, dict[str, Any]] = {}
        rank_ladder_pass_set_audits: dict[str, dict[str, Any]] = {}
        exact_pass_set = {
            idx
            for idx, row in exact_by_id.items()
            if contract_pass(
                row,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            )
        }

        def pass_set_audit(
            mode_name: str,
            pass_set: set[int],
            *,
            selected_exact_set: set[int] | None = None,
        ) -> dict[str, Any]:
            union = exact_pass_set | pass_set
            false_retirements = sorted(exact_pass_set - pass_set)
            extra_successes = sorted(pass_set - exact_pass_set)
            payload: dict[str, Any] = {
                "mode": mode_name,
                "exact_pass_count": int(len(exact_pass_set)),
                "method_pass_count": int(len(pass_set)),
                "intersection_count": int(len(exact_pass_set & pass_set)),
                "union_count": int(len(union)),
                "jaccard_with_exact": None if not union else float(len(exact_pass_set & pass_set) / len(union)),
                "false_retirement_count": int(len(false_retirements)),
                "false_promotion_count": int(len(extra_successes)),
                "false_retirement_indices": false_retirements,
                "false_promotion_indices": extra_successes,
            }
            if selected_exact_set is not None:
                payload["selected_exact_count"] = int(len(selected_exact_set))
                payload["selected_exact_indices"] = sorted(int(idx) for idx in selected_exact_set)
            return payload

        def pass_set_audit_key(
            *,
            family: str,
            rank: int | None,
            policy: str,
            budget_k: int | None,
        ) -> str:
            return f"{family}|rank={rank}|policy={policy}|k={budget_k}"

        def pass_set_for_rows(rows_by_id: dict[int, dict[str, Any]]) -> set[int]:
            return {
                idx
                for idx, row in rows_by_id.items()
                if contract_pass(
                    row,
                    rewrite_threshold=args.contract_rewrite_threshold,
                    rephrase_threshold=args.contract_rephrase_threshold,
                    locality_threshold=args.contract_locality_threshold,
                )
            }

        rank_ladder_pass_set_audits[
            pass_set_audit_key(family="exact", rank=None, policy="exact_only", budget_k=len(edit_ids))
        ] = pass_set_audit("exact", exact_pass_set, selected_exact_set=set(range(len(edit_ids))))
        rank_ladder_exact_row = {
            "family": "exact",
            "rank": None,
            "policy": "exact_only",
            "budget_k": int(len(edit_ids)),
            "post_rewrite_mean": exact_summary.get("post_rewrite_mean"),
            "post_rephrase_mean": exact_summary.get("post_rephrase_mean"),
            "post_locality_mean": exact_summary.get("post_locality_mean"),
            "contract_pass_rate": summary_subset(
                exact_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ).get("contract_pass_rate"),
            "total_params": exact_total_params,
            "memory_fraction_vs_exact": 1.0,
            "compression_ratio_vs_exact": 1.0,
            "rank_params_active": 0,
            "kept_exact_count": int(len(edit_ids)),
            "is_strict_memory_win": False,
            "tier_counts": None,
        }
        rank_ladder_frontier_rows.append(rank_ladder_exact_row)

        rank_case_by_rank: dict[int, dict[int, dict[str, Any]]] = {}
        rank_costs_by_rank: dict[int, list[int]] = {}
        rank_utility_gaps_by_rank: dict[int, list[float]] = {}
        rank_contract_gaps_by_rank: dict[int, list[float]] = {}

        def rank_ladder_row_from_keep_set(
            *,
            rank: int,
            policy: str,
            keep_indices: list[int],
            rank_by_id: dict[int, dict[str, Any]],
            rank_costs: list[int],
        ) -> dict[str, Any]:
            keep_set = set(int(idx) for idx in keep_indices)
            mixed_rows = []
            for idx in range(len(edit_ids)):
                source = exact_by_id.get(idx) if idx in keep_set else rank_by_id.get(idx)
                source = source or {}
                mixed_rows.append(
                    {
                        "tier_rewrite": metric_or_none(source, "post_rewrite_acc"),
                        "tier_rephrase": metric_or_none(source, "post_rephrase_acc"),
                        "tier_locality": metric_or_none(source, "post_locality_acc"),
                    }
                )
            row_summary = aggregate_mode_from_rows(
                mixed_rows,
                "tier",
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            )
            rank_params_active = int(sum(rank_costs[idx] for idx in range(len(rank_costs)) if idx not in keep_set))
            total_params = int(rank_params_active + len(keep_set) * p_lora)
            return {
                "family": "rank_exact_svd_ladder",
                "rank": int(rank),
                "policy": policy,
                "budget_k": int(len(keep_set)),
                "post_rewrite_mean": row_summary.get("post_rewrite_mean"),
                "post_rephrase_mean": row_summary.get("post_rephrase_mean"),
                "post_locality_mean": row_summary.get("post_locality_mean"),
                "contract_pass_rate": row_summary.get("contract_pass_rate"),
                "total_params": total_params,
                "memory_fraction_vs_exact": float(total_params / exact_total_params),
                "compression_ratio_vs_exact": float(exact_total_params / total_params) if total_params > 0 else None,
                "rank_params_active": rank_params_active,
                "kept_exact_count": int(len(keep_set)),
                "is_strict_memory_win": bool(total_params < exact_total_params),
                "tier_counts": None,
            }

        def rank_ladder_pass_set_from_keep_set(
            *,
            keep_indices: list[int],
            rank_by_id: dict[int, dict[str, Any]],
        ) -> set[int]:
            keep_set = set(int(idx) for idx in keep_indices)
            pass_set: set[int] = set()
            for idx in range(len(edit_ids)):
                source = exact_by_id.get(idx) if idx in keep_set else rank_by_id.get(idx)
                source = source or {}
                if contract_pass(
                    source,
                    rewrite_threshold=args.contract_rewrite_threshold,
                    rephrase_threshold=args.contract_rephrase_threshold,
                    locality_threshold=args.contract_locality_threshold,
                ):
                    pass_set.add(idx)
            return pass_set

        for rank in sorted(rank_ladder_summaries):
            rank_summary = rank_ladder_summaries[rank]
            rank_subset = summary_subset(
                rank_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            )
            rank_ladder_mode_metrics[f"rank{rank}_exact_svd"] = {
                **rank_subset,
                "total_params": int(exact_svd_params_all_by_rank[rank]),
                "memory_fraction_vs_exact": float(exact_svd_params_all_by_rank[rank] / exact_total_params),
                "compression_ratio_vs_exact": float(exact_total_params / exact_svd_params_all_by_rank[rank])
                if exact_svd_params_all_by_rank[rank] > 0
                else None,
            }
            rank_by_id = per_case_by_id(rank_summary)
            rank_case_by_rank[rank] = rank_by_id
            rank_pass_set = pass_set_for_rows(rank_by_id)
            rank_ladder_pass_set_audits[
                pass_set_audit_key(family="rank_exact_svd_ladder", rank=rank, policy=f"rank{rank}_only", budget_k=0)
            ] = pass_set_audit(f"rank{rank}_only", rank_pass_set, selected_exact_set=set())
            rank_costs = [int(per_edit_exact_svd_params[rank][edit_ids[idx]]) for idx in range(len(edit_ids))]
            rank_costs_by_rank[rank] = rank_costs
            rank_gaps: list[float] = []
            rank_contract_gaps: list[float] = []
            for idx in range(len(edit_ids)):
                rank_row = rank_by_id.get(idx) or {}
                exact_row = exact_by_id.get(idx) or {}
                exact_utility = per_case_utility(exact_row, weights)
                rank_utility = per_case_utility(rank_row, weights)
                rank_gap = 0.0 if exact_utility is None or rank_utility is None else float(exact_utility - rank_utility)
                rank_contract_gap = contract_gap_value(
                    rank_row,
                    rewrite_threshold=args.contract_rewrite_threshold,
                    rephrase_threshold=args.contract_rephrase_threshold,
                    locality_threshold=args.contract_locality_threshold,
                    weights=weights,
                )
                rank_gaps.append(rank_gap)
                rank_contract_gaps.append(rank_contract_gap)
                rank_ladder_per_edit_rows.append(
                    {
                        "edit_index": idx,
                        "edit_id": edit_ids[idx],
                        "rank": int(rank),
                        "exact_utility": exact_utility,
                        "rank_utility": rank_utility,
                        "utility_gap": rank_gap,
                        "contract_gap": rank_contract_gap,
                        "rank_contract_pass": contract_pass(
                            rank_row,
                            rewrite_threshold=args.contract_rewrite_threshold,
                            rephrase_threshold=args.contract_rephrase_threshold,
                            locality_threshold=args.contract_locality_threshold,
                        ),
                        "rank_params": rank_costs[idx],
                    }
                )
            rank_utility_gaps_by_rank[rank] = rank_gaps
            rank_contract_gaps_by_rank[rank] = rank_contract_gaps
            rank_ladder_frontier_rows.append(
                rank_ladder_row_from_keep_set(
                    rank=rank,
                    policy=f"rank{rank}_only",
                    keep_indices=[],
                    rank_by_id=rank_by_id,
                    rank_costs=rank_costs,
                )
            )
            for budget in budgets:
                if budget <= 0 or budget >= len(edit_ids):
                    continue
                policies = {
                    f"rank{rank}_random_exact": select_indices(
                        "random",
                        budget,
                        len(edit_ids),
                        residual_norms=[0.0 for _ in edit_ids],
                        contract_gaps=rank_contract_gaps,
                        utility_gaps=rank_gaps,
                        seed=args.seed,
                    ),
                    f"rank{rank}_recency_exact": select_indices(
                        "recency",
                        budget,
                        len(edit_ids),
                        residual_norms=[0.0 for _ in edit_ids],
                        contract_gaps=rank_contract_gaps,
                        utility_gaps=rank_gaps,
                        seed=args.seed,
                    ),
                    f"rank{rank}_contract_gap_exact": select_indices(
                        "contract_gap",
                        budget,
                        len(edit_ids),
                        residual_norms=[0.0 for _ in edit_ids],
                        contract_gaps=rank_contract_gaps,
                        utility_gaps=rank_gaps,
                        seed=args.seed,
                    ),
                    f"rank{rank}_oracle_utility_gap_exact": select_indices(
                        "oracle_utility_gap",
                        budget,
                        len(edit_ids),
                        residual_norms=[0.0 for _ in edit_ids],
                        contract_gaps=rank_contract_gaps,
                        utility_gaps=rank_gaps,
                        seed=args.seed,
                    ),
                }
                for policy_name, keep_indices in policies.items():
                    keep_set = set(int(idx) for idx in keep_indices)
                    rank_ladder_frontier_rows.append(
                        rank_ladder_row_from_keep_set(
                            rank=rank,
                            policy=policy_name,
                            keep_indices=keep_indices,
                            rank_by_id=rank_by_id,
                            rank_costs=rank_costs,
                        )
                    )
                    rank_ladder_pass_set_audits[
                        pass_set_audit_key(
                            family="rank_exact_svd_ladder",
                            rank=rank,
                            policy=policy_name,
                            budget_k=budget,
                        )
                    ] = pass_set_audit(
                        policy_name,
                        rank_ladder_pass_set_from_keep_set(keep_indices=keep_indices, rank_by_id=rank_by_id),
                        selected_exact_set=keep_set,
                    )

            exact_cache_budgets = sorted(
                {
                    max(0, min(len(edit_ids), math.floor(row["total_params"] / p_lora)))
                    for row in rank_ladder_frontier_rows
                    if row.get("rank") == rank and row.get("total_params") is not None
                }
            )
            for exact_budget in exact_cache_budgets:
                if exact_budget <= 0 or exact_budget >= len(edit_ids):
                    continue
                exact_cache_policies = {
                    f"exact_cache_rank{rank}_matched_random_k{exact_budget}": select_indices(
                        "random",
                        exact_budget,
                        len(edit_ids),
                        residual_norms=[0.0 for _ in edit_ids],
                        contract_gaps=rank_contract_gaps,
                        utility_gaps=rank_gaps,
                        seed=args.seed,
                    ),
                    f"exact_cache_rank{rank}_matched_contract_gap_k{exact_budget}": select_indices(
                        "contract_gap",
                        exact_budget,
                        len(edit_ids),
                        residual_norms=[0.0 for _ in edit_ids],
                        contract_gaps=rank_contract_gaps,
                        utility_gaps=rank_gaps,
                        seed=args.seed,
                    ),
                    f"exact_cache_rank{rank}_matched_oracle_utility_gap_k{exact_budget}": select_indices(
                        "oracle_utility_gap",
                        exact_budget,
                        len(edit_ids),
                        residual_norms=[0.0 for _ in edit_ids],
                        contract_gaps=rank_contract_gaps,
                        utility_gaps=rank_gaps,
                        seed=args.seed,
                    ),
                }
                for policy_name, keep_indices in exact_cache_policies.items():
                    keep_set = set(int(idx) for idx in keep_indices)
                    exact_cache_rows = []
                    for idx in range(len(edit_ids)):
                        # Evicted exact-cache edits are unserved. Count them as failed
                        # rewrite/rephrase rows rather than None, otherwise aggregate
                        # means would silently ignore them and overstate exact-cache.
                        source = exact_by_id.get(idx) if idx in keep_set else None
                        exact_cache_rows.append(
                            {
                                "tier_rewrite": metric_or_none(source, "post_rewrite_acc") if source is not None else 0.0,
                                "tier_rephrase": metric_or_none(source, "post_rephrase_acc") if source is not None else 0.0,
                                "tier_locality": metric_or_none(source, "post_locality_acc") if source is not None else 1.0,
                            }
                        )
                    exact_cache_summary = aggregate_mode_from_rows(
                        exact_cache_rows,
                        "tier",
                        rewrite_threshold=args.contract_rewrite_threshold,
                        rephrase_threshold=args.contract_rephrase_threshold,
                        locality_threshold=args.contract_locality_threshold,
                    )
                    total_params = int(exact_budget * p_lora)
                    rank_ladder_frontier_rows.append(
                        {
                            "family": "exact_cache_matched_memory",
                            "rank": int(rank),
                            "policy": policy_name,
                            "budget_k": int(exact_budget),
                            "post_rewrite_mean": exact_cache_summary.get("post_rewrite_mean"),
                            "post_rephrase_mean": exact_cache_summary.get("post_rephrase_mean"),
                            "post_locality_mean": exact_cache_summary.get("post_locality_mean"),
                            "contract_pass_rate": exact_cache_summary.get("contract_pass_rate"),
                            "total_params": total_params,
                            "memory_fraction_vs_exact": float(total_params / exact_total_params),
                            "compression_ratio_vs_exact": float(exact_total_params / total_params) if total_params > 0 else None,
                            "rank_params_active": 0,
                            "kept_exact_count": int(exact_budget),
                            "is_strict_memory_win": bool(total_params < exact_total_params),
                            "tier_counts": None,
                        }
                    )
                    pass_set = {
                        idx
                        for idx in keep_set
                        if idx in exact_pass_set
                    }
                    rank_ladder_pass_set_audits[
                        pass_set_audit_key(
                            family="exact_cache_matched_memory",
                            rank=rank,
                            policy=policy_name,
                            budget_k=exact_budget,
                        )
                    ] = pass_set_audit(
                        policy_name,
                        pass_set,
                        selected_exact_set=keep_set,
                    )

        adaptive_rows = []
        adaptive_total_params = 0
        adaptive_tier_counts: dict[str, int] = {}
        sorted_ranks = sorted(rank_case_by_rank)
        for idx in range(len(edit_ids)):
            chosen_rank: int | None = None
            chosen_row: dict[str, Any] | None = None
            for rank in sorted_ranks:
                candidate = rank_case_by_rank[rank].get(idx) or {}
                if contract_pass(
                    candidate,
                    rewrite_threshold=args.contract_rewrite_threshold,
                    rephrase_threshold=args.contract_rephrase_threshold,
                    locality_threshold=args.contract_locality_threshold,
                ):
                    chosen_rank = rank
                    chosen_row = candidate
                    break
            if chosen_rank is None:
                chosen_row = exact_by_id.get(idx) or {}
                adaptive_total_params += p_lora
                tier_name = "exact"
            else:
                adaptive_total_params += int(rank_costs_by_rank[chosen_rank][idx])
                tier_name = f"rank{chosen_rank}"
            adaptive_tier_counts[tier_name] = adaptive_tier_counts.get(tier_name, 0) + 1
            adaptive_rows.append(
                {
                    "tier_rewrite": metric_or_none(chosen_row or {}, "post_rewrite_acc"),
                    "tier_rephrase": metric_or_none(chosen_row or {}, "post_rephrase_acc"),
                    "tier_locality": metric_or_none(chosen_row or {}, "post_locality_acc"),
                }
            )
        if adaptive_rows:
            adaptive_summary = aggregate_mode_from_rows(
                adaptive_rows,
                "tier",
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            )
            rank_ladder_frontier_rows.append(
                {
                    "family": "adaptive_rank_ladder",
                    "rank": None,
                    "policy": "adaptive_cheapest_contract_rank_else_exact",
                    "budget_k": int(adaptive_tier_counts.get("exact", 0)),
                    "post_rewrite_mean": adaptive_summary.get("post_rewrite_mean"),
                    "post_rephrase_mean": adaptive_summary.get("post_rephrase_mean"),
                    "post_locality_mean": adaptive_summary.get("post_locality_mean"),
                    "contract_pass_rate": adaptive_summary.get("contract_pass_rate"),
                    "total_params": int(adaptive_total_params),
                    "memory_fraction_vs_exact": float(adaptive_total_params / exact_total_params),
                    "compression_ratio_vs_exact": float(exact_total_params / adaptive_total_params) if adaptive_total_params > 0 else None,
                    "rank_params_active": int(adaptive_total_params - adaptive_tier_counts.get("exact", 0) * p_lora),
                    "kept_exact_count": int(adaptive_tier_counts.get("exact", 0)),
                    "is_strict_memory_win": bool(adaptive_total_params < exact_total_params),
                    "tier_counts": json.dumps(adaptive_tier_counts, sort_keys=True),
                }
            )
            adaptive_pass_set = {
                idx
                for idx, row in enumerate(adaptive_rows)
                if row["tier_rewrite"] is not None
                and row["tier_rephrase"] is not None
                and row["tier_locality"] is not None
                and float(row["tier_rewrite"]) >= args.contract_rewrite_threshold
                and float(row["tier_rephrase"]) >= args.contract_rephrase_threshold
                and float(row["tier_locality"]) >= args.contract_locality_threshold
            }
            rank_ladder_pass_set_audits[
                pass_set_audit_key(
                    family="adaptive_rank_ladder",
                    rank=None,
                    policy="adaptive_cheapest_contract_rank_else_exact",
                    budget_k=adaptive_tier_counts.get("exact", 0),
                )
            ] = pass_set_audit(
                "adaptive_cheapest_contract_rank_else_exact",
                adaptive_pass_set,
            )

        write_csv(output_dir / "rank_ladder_frontier.csv", rank_ladder_frontier_rows)
        write_csv(output_dir / "rank_ladder_per_edit.csv", rank_ladder_per_edit_rows)
        write_json(output_dir / "rank_ladder_pass_set_audit.json", rank_ladder_pass_set_audits)
        rank_ladder_memory_audit = audit_frontier_memory_monotonicity(rank_ladder_frontier_rows)
        rank_ladder_best_points = summarize_best_frontier_points(rank_ladder_frontier_rows, utility_weights=weights)
        rank_ladder_summary_payload = {
            "dataset": dataset_slug,
            "n_edits": len(edit_ids),
            "ranks": sorted(rank_ladder_summaries),
            "mode_metrics": rank_ladder_mode_metrics,
            "adaptive_tier_counts": adaptive_tier_counts,
            "memory_audit": {
                "exact_total_params": exact_total_params,
                "params_per_exact_lora": p_lora,
                "rank_params_all_by_rank": exact_svd_params_all_by_rank,
            },
            "frontier_memory_audit": rank_ladder_memory_audit,
            "frontier_best_points": rank_ladder_best_points,
            "pass_set_audit": rank_ladder_pass_set_audits,
        }
        write_json(output_dir / "rank_ladder_summary.json", rank_ladder_summary_payload)
        rank_ladder_lines = [
            "# Rank-Ladder Frontier Report",
            "",
            f"- dataset: `{dataset_slug}`",
            f"- edits: `{len(edit_ids)}`",
            f"- ranks: `{sorted(rank_ladder_summaries)}`",
            "",
            "## Fixed Rank Modes",
            "",
            "| Mode | Rewrite | Rephrase | Locality | Contract | Memory fraction |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode_name, mode in rank_ladder_mode_metrics.items():
            rank_ladder_lines.append(
                f"| {mode_name} | {float(mode.get('post_rewrite_mean') or 0.0):.4f} | "
                f"{float(mode.get('post_rephrase_mean') or 0.0):.4f} | "
                f"{float(mode.get('post_locality_mean') or 0.0):.4f} | "
                f"{float(mode.get('contract_pass_rate') or 0.0):.4f} | "
                f"{float(mode.get('memory_fraction_vs_exact') or 0.0):.4f} |"
            )
        rank_ladder_lines.extend(
            [
                "",
                "## Adaptive Rank",
                "",
                f"- adaptive tier counts: `{adaptive_tier_counts}`",
                "",
                "## Best Points",
                "",
                f"- best strict-memory-win point: `{rank_ladder_best_points['best_strict_memory_win_point']}`",
                f"- best at-or-below-exact point: `{rank_ladder_best_points['best_at_or_below_exact_point']}`",
            ]
        )
        (output_dir / "rank_ladder_report.md").write_text("\n".join(rank_ladder_lines))

    compressibility = {
        "mean_utility_gap_shared": mean_or_none(shared_gaps),
        "median_utility_gap_shared": median_or_none(shared_gaps),
        "mean_utility_gap_shared_plus_cheap": mean_or_none(shared_plus_cheap_gaps),
        "median_utility_gap_shared_plus_cheap": median_or_none(shared_plus_cheap_gaps),
        "mean_relative_residual_norm": mean_or_none([row["relative_residual_norm"] for row in per_edit_rows]),
        "median_relative_residual_norm": median_or_none([row["relative_residual_norm"] for row in per_edit_rows]),
        "shared_contract_pass_rate": float(sum(1 for row in per_edit_rows if row["shared_contract_pass"]) / len(per_edit_rows)) if per_edit_rows else None,
        "shared_plus_cheap_contract_pass_rate": float(sum(1 for row in per_edit_rows if row["shared_plus_cheap_contract_pass"]) / len(per_edit_rows)) if per_edit_rows else None,
        "fraction_high_gap_shared": float(sum(1 for value in shared_gaps if value is not None and value > args.high_gap_threshold) / len(per_edit_rows)) if per_edit_rows else None,
        "fraction_high_gap_shared_plus_cheap": float(sum(1 for value in shared_plus_cheap_gaps if value is not None and value > args.high_gap_threshold) / len(per_edit_rows)) if per_edit_rows else None,
        "high_gap_threshold": float(args.high_gap_threshold),
        "spearman_residual_norm_vs_shared_gap": spearman(
            [float(row["residual_norm"]) for row in per_edit_rows],
            [0.0 if value is None else value for value in shared_gaps],
        ),
        "spearman_contract_gap_shared_vs_shared_gap": spearman(
            shared_contract_gaps,
            [0.0 if value is None else value for value in shared_gaps],
        ),
        "spearman_contract_gap_shared_plus_cheap_vs_shared_plus_cheap_gap": spearman(
            shared_plus_cheap_contract_gaps,
            [0.0 if value is None else value for value in shared_plus_cheap_gaps],
        ),
        "pearson_contract_gap_shared_plus_cheap_vs_shared_plus_cheap_gap": pearson(
            shared_plus_cheap_contract_gaps,
            [0.0 if value is None else value for value in shared_plus_cheap_gaps],
        ),
        "repair_category_counts": {
            "shared_fails_shared_plus_cheap_succeeds": int(
                sum(1 for row in per_edit_rows if (not row["shared_contract_pass"]) and row["shared_plus_cheap_contract_pass"])
            ),
            "shared_succeeds_shared_plus_cheap_succeeds": int(
                sum(1 for row in per_edit_rows if row["shared_contract_pass"] and row["shared_plus_cheap_contract_pass"])
            ),
            "shared_fails_shared_plus_cheap_fails": int(
                sum(1 for row in per_edit_rows if (not row["shared_contract_pass"]) and (not row["shared_plus_cheap_contract_pass"]))
            ),
            "shared_succeeds_shared_plus_cheap_fails": int(
                sum(1 for row in per_edit_rows if row["shared_contract_pass"] and (not row["shared_plus_cheap_contract_pass"]))
            ),
        },
    }

    summary = {
        "dataset": dataset_slug,
        "n_edits": len(edit_ids),
        "strategy": args.strategy,
        "shared_rank": args.shared_rank,
        "cheap_residual_rank": args.cheap_residual_rank,
        "num_groups": len(groups),
        "group_sizes": [len(group) for group in groups],
        "mode_metrics": {
            "exact": summary_subset(
                exact_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            "shared_only": summary_subset(
                shared_only_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            "shared_plus_cheap": summary_subset(
                shared_plus_cheap_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            "shared_plus_full": summary_subset(
                exact_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
        },
        "compressibility": compressibility,
        "memory_audit": {
            "exact_total_params": exact_total_params,
            "params_per_exact_lora": p_lora,
            "shared_basis_params": shared_basis_params,
            "code_params_all": code_params_all,
            "cheap_residual_params_all": cheap_params_all_by_rank[current_cheap_rank],
            "shared_base_params": shared_base_params,
            "shared_exact_equivalent": memory_break_even.get("shared_exact_equivalent"),
            "shared_base_fraction": float(shared_base_params / exact_total_params),
            "memory_margin_before_residuals": memory_break_even.get("memory_margin_before_residuals"),
            "max_residuals_strictly_below_exact": memory_break_even.get("max_residuals_strictly_below_exact"),
            "shared_plus_cheap_all_fraction": float((shared_base_params + cheap_params_all_by_rank[current_cheap_rank]) / exact_total_params),
        },
        "frontier_memory_audit": frontier_memory_audit,
        "frontier_row_consistency": frontier_row_consistency,
        "frontier_best_points": frontier_best_points,
        "rank_ladder_summary": rank_ladder_summary_payload,
        "exact_vs_shared_plus_full_consistency": exact_dpr_consistency,
    }
    write_json(output_dir / "summary.json", summary)
    log_progress(output_dir, "summary_written")

    if args.run_ablation_suite:
        write_csv(output_dir / "ablation_frontier.csv", ablation_mode_rows)
        write_csv(output_dir / "ablation_per_edit.csv", ablation_per_edit_rows)
        write_csv(output_dir / "rank_sweep.csv", rank_sweep_rows)

        def mode_row_lookup(mode_name: str) -> dict[str, Any] | None:
            for row in ablation_mode_rows:
                if row["mode"] == mode_name:
                    return row
            return None

        shuffled_global_rows = [row for row in ablation_mode_rows if row["family"] == "shared_plus_shuffled_cheap" and row["shuffle_kind"] == "global"]
        shuffled_within_group_rows = [row for row in ablation_mode_rows if row["family"] == "shared_plus_shuffled_cheap" and row["shuffle_kind"] == "within_group"]
        rank_exact_row = mode_row_lookup(f"rank{current_cheap_rank}_exact_svd")
        cheap_only_row_lookup = mode_row_lookup(f"cheap_residual_only_rank{current_cheap_rank}")
        random_subspace_row = mode_row_lookup(f"random_subspace_plus_cheap_rank{current_cheap_rank}")
        shared_plus_cheap_row_lookup = mode_row_lookup(f"shared_plus_cheap_rank{current_cheap_rank}")
        exact_row_lookup = mode_row_lookup("exact")
        shared_only_row_lookup = mode_row_lookup("shared_only")

        ablation_summary_payload = {
            "dataset": dataset_slug,
            "n_edits": len(edit_ids),
            "strategy": args.strategy,
            "shared_rank": args.shared_rank,
            "cheap_residual_ranks": cheap_residual_ranks,
            "mode_metrics": {
                "exact": exact_row_lookup,
                "shared_only": shared_only_row_lookup,
                f"cheap_residual_only_rank{current_cheap_rank}": cheap_only_row_lookup,
                f"rank{current_cheap_rank}_exact_svd": rank_exact_row,
                f"shared_plus_cheap_rank{current_cheap_rank}": shared_plus_cheap_row_lookup,
                f"shared_plus_shuffled_cheap_global_rank{current_cheap_rank}": {
                    "utility_mean_mean": mean_or_none([row.get("utility_mean") for row in shuffled_global_rows]),
                    "utility_mean_std": std_or_none([row.get("utility_mean") for row in shuffled_global_rows]),
                    "contract_pass_rate_mean": mean_or_none([row.get("contract_pass_rate") for row in shuffled_global_rows]),
                    "contract_pass_rate_std": std_or_none([row.get("contract_pass_rate") for row in shuffled_global_rows]),
                    "memory_fraction_vs_exact": mean_or_none([row.get("memory_fraction_vs_exact") for row in shuffled_global_rows]),
                    "num_trials": len(shuffled_global_rows),
                },
                f"shared_plus_shuffled_cheap_within_group_rank{current_cheap_rank}": {
                    "utility_mean_mean": mean_or_none([row.get("utility_mean") for row in shuffled_within_group_rows]),
                    "utility_mean_std": std_or_none([row.get("utility_mean") for row in shuffled_within_group_rows]),
                    "contract_pass_rate_mean": mean_or_none([row.get("contract_pass_rate") for row in shuffled_within_group_rows]),
                    "contract_pass_rate_std": std_or_none([row.get("contract_pass_rate") for row in shuffled_within_group_rows]),
                    "memory_fraction_vs_exact": mean_or_none([row.get("memory_fraction_vs_exact") for row in shuffled_within_group_rows]),
                    "num_trials": len(shuffled_within_group_rows),
                },
                f"random_subspace_plus_cheap_rank{current_cheap_rank}": random_subspace_row,
            },
            "memory": {
                "exact_total_params": exact_total_params,
                "shared_base_params": shared_base_params,
                "shared_base_fraction": float(shared_base_params / exact_total_params),
                f"cheap_rank{current_cheap_rank}_all_fraction": float(cheap_params_all_by_rank[current_cheap_rank] / exact_total_params),
                f"shared_plus_cheap_rank{current_cheap_rank}_fraction": float((shared_base_params + cheap_params_all_by_rank[current_cheap_rank]) / exact_total_params),
            },
            "isolation_tests": {
                "shared_plus_cheap_minus_cheap_only_utility": None
                if shared_plus_cheap_row_lookup is None or cheap_only_row_lookup is None
                else float((shared_plus_cheap_row_lookup.get("utility_mean") or 0.0) - (cheap_only_row_lookup.get("utility_mean") or 0.0)),
                "shared_plus_cheap_minus_rank1_exact_utility": None
                if shared_plus_cheap_row_lookup is None or rank_exact_row is None
                else float((shared_plus_cheap_row_lookup.get("utility_mean") or 0.0) - (rank_exact_row.get("utility_mean") or 0.0)),
                "shared_plus_cheap_minus_shuffled_cheap_global_utility": None
                if shared_plus_cheap_row_lookup is None or not shuffled_global_rows
                else float((shared_plus_cheap_row_lookup.get("utility_mean") or 0.0) - (mean_or_none([row.get("utility_mean") for row in shuffled_global_rows]) or 0.0)),
                "shared_plus_cheap_minus_shuffled_cheap_within_group_utility": None
                if shared_plus_cheap_row_lookup is None or not shuffled_within_group_rows
                else float((shared_plus_cheap_row_lookup.get("utility_mean") or 0.0) - (mean_or_none([row.get("utility_mean") for row in shuffled_within_group_rows]) or 0.0)),
                "shared_plus_cheap_minus_random_subspace_utility": None
                if shared_plus_cheap_row_lookup is None or random_subspace_row is None
                else float((shared_plus_cheap_row_lookup.get("utility_mean") or 0.0) - (random_subspace_row.get("utility_mean") or 0.0)),
            },
        }
        write_json(output_dir / "ablation_summary.json", ablation_summary_payload)
        ablation_lines = [
            "# Ablation Report",
            "",
            f"- dataset: `{dataset_slug}`",
            f"- current cheap rank: `{current_cheap_rank}`",
            f"- shared base fraction: `{shared_base_params / exact_total_params:.6f}`",
            "",
            "## Mode Table",
            "",
            "| Mode | Utility | Contract | Memory fraction |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in ablation_mode_rows:
            if row["trial"] is not None:
                continue
            ablation_lines.append(
                f"| {row['mode']} | {float(row.get('utility_mean') or 0.0):.4f} | {float(row.get('contract_pass_rate') or 0.0):.4f} | {float(row.get('memory_fraction_vs_exact') or 0.0):.4f} |"
            )
        ablation_lines.extend(
            [
                "",
                "## Isolation Tests",
                "",
                f"- shared_plus_cheap minus cheap_only utility: `{ablation_summary_payload['isolation_tests']['shared_plus_cheap_minus_cheap_only_utility']}`",
                f"- shared_plus_cheap minus rank-exact-SVD utility: `{ablation_summary_payload['isolation_tests']['shared_plus_cheap_minus_rank1_exact_utility']}`",
                f"- shared_plus_cheap minus shuffled(global) utility: `{ablation_summary_payload['isolation_tests']['shared_plus_cheap_minus_shuffled_cheap_global_utility']}`",
                f"- shared_plus_cheap minus random-subspace utility: `{ablation_summary_payload['isolation_tests']['shared_plus_cheap_minus_random_subspace_utility']}`",
            ]
        )
        (output_dir / "ablation_report.md").write_text("\n".join(ablation_lines))

    lines = [
        "# Residualized Subspace Merge Report",
        "",
        f"- dataset: `{dataset_slug}`",
        f"- edits: `{len(edit_ids)}`",
        f"- strategy: `{args.strategy}`",
        f"- shared_rank: `{args.shared_rank}`",
        f"- cheap_residual_rank: `{args.cheap_residual_rank}`",
        f"- group sizes: `{[len(group) for group in groups]}`",
        "",
        "## Aggregate Mode Comparison",
        "",
        "| Mode | Rewrite | Rephrase | Locality | Contract |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode_name in ["exact", "shared_only", "shared_plus_cheap", "shared_plus_full"]:
        mode = summary["mode_metrics"][mode_name]
        lines.append(
            f"| {mode_name} | {float(mode.get('post_rewrite_mean') or 0.0):.4f} | {float(mode.get('post_rephrase_mean') or 0.0):.4f} | {float(mode.get('post_locality_mean') or 0.0):.4f} | {float(mode.get('contract_pass_rate') or 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Memory Audit",
            "",
            f"- exact total params: `{exact_total_params}`",
            f"- params per exact LoRA: `{p_lora}`",
            f"- shared basis params: `{shared_basis_params}`",
            f"- code params all edits: `{code_params_all}`",
            f"- cheap residual params all edits: `{cheap_params_all_by_rank[current_cheap_rank]}`",
            f"- shared exact equivalent: `{summary['memory_audit']['shared_exact_equivalent']}`",
            "",
            "## Frontier Best Points",
            "",
            f"- best strict-memory-win point: `{frontier_best_points['best_strict_memory_win_point']}`",
            f"- best at-or-below-exact point: `{frontier_best_points['best_at_or_below_exact_point']}`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))

    plot_gap_histogram(output_dir / "gap_histogram.png", [0.0 if value is None else float(value) for value in shared_gaps])
    plot_frontier(output_dir / "frontier.png", frontier_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
