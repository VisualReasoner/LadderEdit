"""Fast adaptive low-rank residual-ladder frontier runner.

This is the LoRA-native method runner for the current NeurIPS story:

    exact edit:     Δ_i^E
    rank-r tier:    Δ_i^(r) = SVD_r(Δ_i^E)
    residual:       ρ_i^(r) = Δ_i^E - Δ_i^(r)
    served update:  Δ_i = Δ_i^(r_i) + z_i ρ_i^(r_i)

Unlike ``run_residualized_subspace_merge_poc.py``, this runner does not build
shared subspaces. It captures exact edit LoRAs, constructs independent low-rank
SVD tiers, evaluates fixed-rank and residual-fallback policies, and writes
strict logical-memory and pass-set audits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
from examples.run_residualized_subspace_merge_poc import (
    EffectiveBlockSpec,
    aggregate_mode_from_rows,
    build_effective_block_specs,
    capture_exact_weight_bank,
    choose_subspace_device,
    lora_factors_from_truncated_svd,
    parse_int_csv,
    run_policy_evaluation_weights,
    zero_weight_bank,
)
from examples.run_shared_dictionary_residual_poc import (
    audit_frontier_memory_consistency,
    audit_frontier_memory_monotonicity,
    contract_gap_value,
    contract_pass,
    metric_or_none,
    parse_budgets,
    per_case_by_id,
    per_case_utility,
    select_indices,
    summary_subset,
    summarize_best_frontier_points,
    write_csv,
)
from examples.run_wikibigedit_lifelong import (
    apply_single_edit,
    compute_pre_metrics,
    configure_evaluation_mode,
)


def log_progress(output_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "time": time.time(), **extra}
    write_json(output_dir / "progress.json", payload)
    print(f"[progress] {json.dumps(payload, sort_keys=True)}", flush=True)


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_weights(raw: str) -> tuple[float, float, float]:
    weights = tuple(float(part) for part in str(raw).split(","))
    if len(weights) != 3:
        raise ValueError("--utility_weights must have three comma-separated values")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Utility weights must sum to a positive value")
    return tuple(float(value / total) for value in weights)


def utility_from_means(row: dict[str, Any], weights: tuple[float, float, float]) -> float | None:
    rewrite = row.get("post_rewrite_mean")
    rephrase = row.get("post_rephrase_mean")
    locality = row.get("post_locality_mean")
    if rewrite is None or rephrase is None or locality is None:
        return None
    return float(weights[0] * float(rewrite) + weights[1] * float(rephrase) + weights[2] * float(locality))


def select_dual_budget(
    *,
    k: int,
    contract_gaps: list[float],
    utility_gaps: list[float],
    residual_norms: list[float],
    marginal_costs: list[int],
) -> list[int]:
    """Non-oracle budget proxy: high behavioral deficit per extra exact param.

    This intentionally avoids exact utility when contract gap is available, but
    uses residual norm as a deterministic tie-breaker. ``utility_gaps`` is only
    used as a final tie-break to make diagnostics stable.
    """
    k = max(0, min(int(k), len(contract_gaps)))
    if k <= 0:
        return []
    order = sorted(
        range(len(contract_gaps)),
        key=lambda idx: (
            float(contract_gaps[idx]) / max(float(marginal_costs[idx]), 1.0),
            float(residual_norms[idx]),
            float(utility_gaps[idx]),
        ),
        reverse=True,
    )
    return sorted(order[:k])


def truncated_svd_from_lora_product(
    weights: dict[str, torch.Tensor],
    block: EffectiveBlockSpec,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the exact SVD of ``B @ A`` through a tiny core SVD.

    The previous implementation materialized the full effective update matrix
    and ran ``torch.linalg.svd`` on it for every edit/layer block. That is
    exact, but brutally slow at N=512. Since every captured edit is already a
    rank-R LoRA update, we can factor

        B @ A = Q_B @ (R_B @ R_A.T) @ Q_A.T

    and SVD only the small R x R core. This preserves the exact effective
    update while avoiding dense out_dim x in_dim SVDs.
    """
    lora_a = weights[block.a_name].detach().to(device=device, dtype=torch.float32)
    lora_b = weights[block.b_name].detach().to(device=device, dtype=torch.float32)
    out_dim = int(lora_b.shape[0])
    in_dim = int(lora_a.shape[1])
    raw_rank = int(lora_a.shape[0])
    if raw_rank <= 0 or float(lora_a.abs().max().item()) <= 1.0e-12 or float(lora_b.abs().max().item()) <= 1.0e-12:
        return (
            torch.zeros(out_dim, 0, dtype=torch.float32, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            torch.zeros(0, in_dim, dtype=torch.float32, device=device),
        )

    q_b, r_b = torch.linalg.qr(lora_b, mode="reduced")
    q_a, r_a = torch.linalg.qr(lora_a.T, mode="reduced")
    core = r_b @ r_a.T
    u_core, s, vh_core = torch.linalg.svd(core, full_matrices=False)
    keep = int((s > 1.0e-10).sum().item())
    if keep <= 0:
        return (
            torch.zeros(out_dim, 0, dtype=torch.float32, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            torch.zeros(0, in_dim, dtype=torch.float32, device=device),
        )
    u = (q_b @ u_core[:, :keep]).to(dtype=torch.float32)
    vh = (vh_core[:keep, :] @ q_a.T).to(dtype=torch.float32)
    return u, s[:keep].to(dtype=torch.float32), vh


def build_rank_svd_banks(
    *,
    edit_ids: list[str],
    exact_weights: dict[str, dict[str, torch.Tensor]],
    block_specs: list[EffectiveBlockSpec],
    ranks: list[int],
    device: torch.device,
) -> tuple[
    dict[int, dict[str, dict[str, torch.Tensor]]],
    dict[int, dict[str, int]],
    dict[int, list[float]],
    dict[int, list[float]],
]:
    rank_banks = {rank: zero_weight_bank(edit_ids, exact_weights) for rank in ranks}
    per_edit_params = {rank: {edit_id: 0 for edit_id in edit_ids} for rank in ranks}
    residual_norm_sq = {rank: [0.0 for _ in edit_ids] for rank in ranks}
    relative_residual_den = [0.0 for _ in edit_ids]

    for idx, edit_id in enumerate(edit_ids):
        weights_i = exact_weights[edit_id]
        for block in block_specs:
            u, s, vh = truncated_svd_from_lora_product(weights_i, block, device=device)
            exact_norm_sq = float((s * s).sum().item())
            relative_residual_den[idx] += exact_norm_sq
            for rank in ranks:
                rank_a, rank_b, rank_eff = lora_factors_from_truncated_svd(
                    u,
                    s,
                    vh,
                    rank,
                    target_rank=block.raw_rank,
                    ref_a=weights_i[block.a_name],
                    ref_b=weights_i[block.b_name],
                )
                rank_banks[rank][edit_id][block.a_name] = rank_a
                rank_banks[rank][edit_id][block.b_name] = rank_b
                per_edit_params[rank][edit_id] += int((block.out_dim + block.in_dim) * rank_eff)
                residual_tail = s[rank_eff:]
                residual_norm_sq[rank][idx] += float((residual_tail * residual_tail).sum().item())

    residual_norms = {
        rank: [float(math.sqrt(value)) for value in values]
        for rank, values in residual_norm_sq.items()
    }
    relative_residual_norms = {
        rank: [
            float(math.sqrt(residual_norm_sq[rank][idx] / max(relative_residual_den[idx], 1.0e-12)))
            for idx in range(len(edit_ids))
        ]
        for rank in ranks
    }
    return rank_banks, per_edit_params, residual_norms, relative_residual_norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT")
    parser.add_argument("--hparams_dir", default=str(REPO_ROOT / "hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware-staged.yaml"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--data_type", default="ZsRE", choices=["CounterFact", "ZsRE", "WikiBigEdit"])
    parser.add_argument("--data_file", default=None)
    parser.add_argument(
        "--indices",
        default=None,
        help="Optional comma/colon-separated source dataset indices to evaluate instead of the first ds_size records.",
    )
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs/rank_ladder_frontier"))
    parser.add_argument("--ds_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranks", default="0:1:2:4")
    parser.add_argument("--residual_budgets", default="0,1,2,4,8,16,all")
    parser.add_argument("--selectors", default="random,recency,residual_norm,contract_gap,dual_budget,oracle_utility_gap")
    parser.add_argument("--subspace_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_rephrase_source", choices=["heldout", "address"], default="address")
    parser.add_argument("--evaluation_mode", default="teacher_forcing")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--utility_weights", default="1,1,1")
    parser.add_argument("--contract_rewrite_threshold", type=float, default=0.8)
    parser.add_argument("--contract_rephrase_threshold", type=float, default=0.8)
    parser.add_argument("--contract_locality_threshold", type=float, default=0.95)
    parser.add_argument(
        "--reuse_existing_modes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing mode_*/summary.json files in the output directory when rerunning after a timeout.",
    )
    parser.add_argument("--include_trained_low_rank_baselines", action="store_true")
    args = parser.parse_args()

    if args.include_trained_low_rank_baselines:
        raise NotImplementedError(
            "Trained low-rank LoRA baselines need a separate rank-constrained "
            "training path. This runner implements the scalable SVD ladder and "
            "paper-grade allocation audits."
        )

    requested_eval_batch_size = int(args.eval_batch_size)
    method = method_name(args.editing_method)
    eval_batch_warning = None
    if method == "HOPEDIT" and requested_eval_batch_size != 1:
        eval_batch_warning = (
            "Forcing eval_batch_size=1 for HOPEDIT rank-ladder evaluation. "
            "The keyed per-edit serving path is not batch-safe; batched "
            "evaluation can collapse exact-mode rewrite/rephrase scores."
        )
        print(f"[eval-batch-warning] {eval_batch_warning}", flush=True)
        args.eval_batch_size = 1

    seed_everything(args.seed)
    requested_ranks = sorted(set(parse_int_csv(args.ranks)))
    if not requested_ranks:
        raise ValueError("--ranks must contain at least one integer rank")
    ranks = list(requested_ranks)
    selectors = [item.strip() for item in str(args.selectors).split(",") if item.strip()]
    supported_selectors = {"random", "recency", "residual_norm", "contract_gap", "dual_budget", "oracle_utility_gap"}
    unknown = sorted(set(selectors) - supported_selectors)
    if unknown:
        raise ValueError(f"Unsupported selectors: {unknown}")
    weights = normalized_weights(args.utility_weights)
    requested_indices = parse_int_csv(args.indices) if args.indices else None

    dataset_slug = str(args.data_type).lower()
    if requested_indices:
        index_slug = "idx" + "-".join(str(index) for index in requested_indices)
        run_slug = f"{dataset_slug}_{index_slug}_seed{args.seed}_ranks{'-'.join(str(rank) for rank in ranks)}"
    else:
        run_slug = f"{dataset_slug}_n{args.ds_size}_seed{args.seed}_ranks{'-'.join(str(rank) for rank in ranks)}"
    output_dir = Path(args.output_root) / run_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid_marker = output_dir / "INVALID_RESULT.json"
    if invalid_marker.exists():
        invalid_marker.unlink()
    write_json(
        output_dir / "run_config.json",
        {
            "data_type": args.data_type,
            "ds_size": args.ds_size,
            "requested_indices": requested_indices,
            "seed": args.seed,
            "requested_ranks": requested_ranks,
            "selectors": selectors,
            "residual_budgets": args.residual_budgets,
            "subspace_device": args.subspace_device,
            "requested_eval_batch_size": requested_eval_batch_size,
            "effective_eval_batch_size": int(args.eval_batch_size),
            "eval_batch_warning": eval_batch_warning,
            "evaluation_mode": args.evaluation_mode,
            "eval_rephrase_source": args.eval_rephrase_source,
            "utility_weights": weights,
            "logical_memory_note": (
                "Memory is inference extra deployed-representation accounting over the frozen base model; "
                "optimizer state, training activations, temporary SVD/workspace tensors, and transient evaluator "
                "object residency are excluded."
            ),
            "frontier_evaluation_note": (
                "Fallback frontiers are assembled from per-edit fixed-rank and exact evaluations. "
                "This is valid for keyed per-edit serving/offline allocation diagnostics; live mixed-bank "
                "runs should be used for final deployment latency/interference checks."
            ),
        },
    )
    work_device = choose_subspace_device(args.subspace_device)
    log_progress(output_dir, "initialized", work_device=str(work_device))

    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    hparams.sequential_edit = True
    hparams.eval_batch_size = int(args.eval_batch_size)
    configure_evaluation_mode(hparams, args.evaluation_mode, args.api_key)

    records, dataset_file = load_normalized_records(
        args.data_dir,
        args.data_type,
        args.ds_size,
        indices=requested_indices,
        data_file=args.data_file,
    )
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
        "stream_type": "adaptive_rank_ladder_frontier",
        "seed": args.seed,
        "sequential_edit": True,
        "stream_length": len(records),
        "requested_ds_size": args.ds_size,
        "requested_indices": requested_indices,
        "requested_ranks": requested_ranks,
    }

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
    if not edit_ids:
        raise RuntimeError("No edit ids found after applying edits")

    exact_weights = capture_exact_weight_bank(controller, edit_ids)
    block_specs = build_effective_block_specs(exact_weights[edit_ids[0]])
    exact_total_params = sum(block.raw_param_count for block in block_specs) * len(edit_ids)
    p_lora = sum(block.raw_param_count for block in block_specs)
    max_exact_rank = max(block.raw_rank for block in block_specs)
    ranks = sorted(set(min(rank, max_exact_rank) for rank in requested_ranks))
    rank_cap_warning = None
    if ranks != requested_ranks:
        rank_cap_warning = (
            f"Requested ranks {requested_ranks} were capped/deduplicated to {ranks} "
            f"because captured exact LoRA raw rank is {max_exact_rank}."
        )
        print(f"[rank-warning] {rank_cap_warning}", flush=True)
    log_progress(output_dir, "exact_bank_captured", edit_count=len(edit_ids), block_count=len(block_specs))

    rank_banks, rank_params_by_rank, residual_norms_by_rank, relative_residual_norms_by_rank = build_rank_svd_banks(
        edit_ids=edit_ids,
        exact_weights=exact_weights,
        block_specs=block_specs,
        ranks=ranks,
        device=work_device,
    )
    rank_params_all_by_rank = {
        rank: int(sum(rank_params_by_rank[rank].values()))
        for rank in ranks
    }
    write_json(
        output_dir / "rank_ladder_metadata.json",
        {
            "dataset": dataset_slug,
            "n_edits": len(edit_ids),
            "requested_ranks": requested_ranks,
            "ranks": ranks,
            "max_exact_raw_rank": max_exact_rank,
            "rank_cap_warning": rank_cap_warning,
            "exact_total_params": exact_total_params,
            "params_per_exact_lora": p_lora,
            "rank_params_all_by_rank": rank_params_all_by_rank,
            "block_specs": [
                {
                    "module_key": block.module_key,
                    "in_dim": block.in_dim,
                    "out_dim": block.out_dim,
                    "raw_rank": block.raw_rank,
                    "raw_param_count": block.raw_param_count,
                }
                for block in block_specs
            ],
        },
    )
    log_progress(output_dir, "rank_banks_ready", ranks=ranks)

    def storage_record(total_params: int, *, rank: int | None = None, kept_exact_count: int = 0) -> dict[str, Any]:
        return {
            "exact_params": exact_total_params,
            "rank": rank,
            "total_params": int(total_params),
            "kept_exact_count": int(kept_exact_count),
            "memory_fraction_vs_exact": float(total_params / exact_total_params),
            "compression_ratio_vs_exact": float(exact_total_params / total_params) if total_params > 0 else None,
            "logical_memory_note": "Inference storage: SVD rank tier stores low-rank factors; exact fallback stores exact LoRA bypass/residual.",
        }

    def run_or_reuse_mode(mode_dir: Path, mode_name: str, evaluator) -> dict[str, Any]:
        summary_path = mode_dir / "summary.json"
        if args.reuse_existing_modes and summary_path.exists():
            summary = read_json_file(summary_path)
            log_progress(output_dir, "mode_reused", mode=mode_name, summary_path=str(summary_path))
            return summary
        summary = evaluator()
        log_progress(output_dir, "mode_complete", mode=mode_name)
        return summary

    exact_summary = run_or_reuse_mode(
        output_dir / "mode_exact",
        "exact",
        lambda: run_policy_evaluation_weights(
            controller=controller,
            edit_ids=edit_ids,
            weight_bank=exact_weights,
            storage=storage_record(exact_total_params, kept_exact_count=len(edit_ids)),
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
        ),
    )

    rank_summaries: dict[int, dict[str, Any]] = {}
    for rank in ranks:
        mode_dir = output_dir / f"mode_rank{rank}_exact_svd"
        rank_summaries[rank] = run_or_reuse_mode(
            mode_dir,
            f"rank{rank}_exact_svd",
            lambda rank=rank, mode_dir=mode_dir: run_policy_evaluation_weights(
                controller=controller,
                edit_ids=edit_ids,
                weight_bank=rank_banks[rank],
                storage=storage_record(rank_params_all_by_rank[rank], rank=rank),
                editor=editor,
                hparams=hparams,
                method=method,
                backbone=run_base["backbone"],
                output_dir=mode_dir,
                records=records,
                requests=requests,
                eval_metric=editor_inputs["eval_metric"],
                run_base=run_base,
                pre_metrics=pre_metrics,
                condition_name=f"{dataset_slug}_rank{rank}_exact_svd",
            ),
        )

    exact_by_id = per_case_by_id(exact_summary)
    exact_summary_subset = summary_subset(
        exact_summary,
        rewrite_threshold=args.contract_rewrite_threshold,
        rephrase_threshold=args.contract_rephrase_threshold,
        locality_threshold=args.contract_locality_threshold,
    )
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
        *,
        family: str,
        mode: str,
        rank: int | None,
        policy: str,
        budget_k: int | None,
        pass_set: set[int],
        selected_exact_set: set[int] | None = None,
    ) -> dict[str, Any]:
        union = exact_pass_set | pass_set
        false_retirements = sorted(exact_pass_set - pass_set)
        false_promotions = sorted(pass_set - exact_pass_set)
        return {
            "family": family,
            "mode": mode,
            "rank": rank,
            "policy": policy,
            "budget_k": budget_k,
            "exact_pass_count": int(len(exact_pass_set)),
            "method_pass_count": int(len(pass_set)),
            "intersection_count": int(len(exact_pass_set & pass_set)),
            "union_count": int(len(union)),
            "jaccard_with_exact": None if not union else float(len(exact_pass_set & pass_set) / len(union)),
            "false_retirement_count": int(len(false_retirements)),
            "false_promotion_count": int(len(false_promotions)),
            "false_retirement_indices": false_retirements,
            "false_promotion_indices": false_promotions,
            "selected_exact_count": None if selected_exact_set is None else int(len(selected_exact_set)),
            "selected_exact_indices": None if selected_exact_set is None else sorted(int(idx) for idx in selected_exact_set),
        }

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

    def row_summary_from_tier_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return aggregate_mode_from_rows(
            rows,
            "tier",
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
        )

    frontier_rows: list[dict[str, Any]] = []
    pass_audit_rows: list[dict[str, Any]] = []
    per_edit_rows: list[dict[str, Any]] = []
    mode_metrics: dict[str, dict[str, Any]] = {}

    exact_frontier_row = {
        "family": "exact",
        "rank": None,
        "policy": "exact_only",
        "budget_k": len(edit_ids),
        "post_rewrite_mean": exact_summary_subset.get("post_rewrite_mean"),
        "post_rephrase_mean": exact_summary_subset.get("post_rephrase_mean"),
        "post_locality_mean": exact_summary_subset.get("post_locality_mean"),
        "contract_pass_rate": exact_summary_subset.get("contract_pass_rate"),
        "utility_mean": utility_from_means(exact_summary_subset, weights),
        "total_params": exact_total_params,
        "memory_fraction_vs_exact": 1.0,
        "compression_ratio_vs_exact": 1.0,
        "rank_params_active": 0,
        "exact_fallback_params_active": exact_total_params,
        "kept_exact_count": len(edit_ids),
        "is_strict_memory_win": False,
        "tier_counts": json.dumps({"exact": len(edit_ids)}, sort_keys=True),
    }
    frontier_rows.append(exact_frontier_row)
    pass_audit_rows.append(
        pass_set_audit(
            family="exact",
            mode="exact",
            rank=None,
            policy="exact_only",
            budget_k=len(edit_ids),
            pass_set=exact_pass_set,
            selected_exact_set=set(range(len(edit_ids))),
        )
    )

    rank_case_by_rank: dict[int, dict[int, dict[str, Any]]] = {}
    rank_costs_by_rank: dict[int, list[int]] = {}
    rank_gaps_by_rank: dict[int, list[float]] = {}
    rank_contract_gaps_by_rank: dict[int, list[float]] = {}

    def mixed_ladder_row(
        *,
        rank: int,
        policy: str,
        keep_indices: list[int],
        rank_by_id: dict[int, dict[str, Any]],
        rank_costs: list[int],
    ) -> tuple[dict[str, Any], set[int]]:
        keep_set = set(int(idx) for idx in keep_indices)
        mixed_rows = []
        pass_set: set[int] = set()
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
            if contract_pass(
                source,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ):
                pass_set.add(idx)
        row_summary = row_summary_from_tier_rows(mixed_rows)
        rank_params_active = int(sum(rank_costs[idx] for idx in range(len(rank_costs)) if idx not in keep_set))
        exact_fallback_params_active = int(len(keep_set) * p_lora)
        total_params = int(rank_params_active + exact_fallback_params_active)
        row = {
            "family": "rank_exact_svd_ladder",
            "rank": int(rank),
            "policy": policy,
            "budget_k": int(len(keep_set)),
            "post_rewrite_mean": row_summary.get("post_rewrite_mean"),
            "post_rephrase_mean": row_summary.get("post_rephrase_mean"),
            "post_locality_mean": row_summary.get("post_locality_mean"),
            "contract_pass_rate": row_summary.get("contract_pass_rate"),
            "utility_mean": utility_from_means(row_summary, weights),
            "total_params": total_params,
            "memory_fraction_vs_exact": float(total_params / exact_total_params),
                "compression_ratio_vs_exact": float(exact_total_params / total_params) if total_params > 0 else None,
                "rank_params_active": rank_params_active,
                "exact_fallback_params_active": exact_fallback_params_active,
                "kept_exact_count": int(len(keep_set)),
                "is_strict_memory_win": bool(total_params < exact_total_params),
                "tier_counts": json.dumps({f"rank{rank}": len(edit_ids) - len(keep_set), "exact": len(keep_set)}, sort_keys=True),
        }
        return row, pass_set

    def exact_cache_row(
        *,
        rank: int,
        policy: str,
        keep_indices: list[int],
    ) -> tuple[dict[str, Any], set[int]]:
        keep_set = set(int(idx) for idx in keep_indices)
        rows = []
        for idx in range(len(edit_ids)):
            source = exact_by_id.get(idx) if idx in keep_set else None
            rows.append(
                {
                    "tier_rewrite": metric_or_none(source, "post_rewrite_acc") if source is not None else 0.0,
                    "tier_rephrase": metric_or_none(source, "post_rephrase_acc") if source is not None else 0.0,
                    "tier_locality": metric_or_none(source, "post_locality_acc") if source is not None else 1.0,
                }
            )
        row_summary = row_summary_from_tier_rows(rows)
        total_params = int(len(keep_set) * p_lora)
        row = {
            "family": "exact_cache_matched_memory",
            "rank": int(rank),
            "policy": policy,
            "budget_k": int(len(keep_set)),
            "post_rewrite_mean": row_summary.get("post_rewrite_mean"),
            "post_rephrase_mean": row_summary.get("post_rephrase_mean"),
            "post_locality_mean": row_summary.get("post_locality_mean"),
            "contract_pass_rate": row_summary.get("contract_pass_rate"),
            "utility_mean": utility_from_means(row_summary, weights),
            "total_params": total_params,
            "memory_fraction_vs_exact": float(total_params / exact_total_params),
            "compression_ratio_vs_exact": float(exact_total_params / total_params) if total_params > 0 else None,
            "rank_params_active": 0,
            "exact_fallback_params_active": total_params,
            "kept_exact_count": int(len(keep_set)),
            "is_strict_memory_win": bool(total_params < exact_total_params),
            "tier_counts": json.dumps({"exact": len(keep_set), "unserved": len(edit_ids) - len(keep_set)}, sort_keys=True),
        }
        return row, set(idx for idx in keep_set if idx in exact_pass_set)

    budgets = parse_budgets(args.residual_budgets, len(edit_ids))
    for rank in ranks:
        rank_summary = rank_summaries[rank]
        rank_subset = summary_subset(
            rank_summary,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
        )
        rank_by_id = per_case_by_id(rank_summary)
        rank_case_by_rank[rank] = rank_by_id
        rank_costs = [int(rank_params_by_rank[rank][edit_ids[idx]]) for idx in range(len(edit_ids))]
        rank_costs_by_rank[rank] = rank_costs
        rank_pass_set = pass_set_for_rows(rank_by_id)
        mode_key = f"rank{rank}_exact_svd"
        mode_metrics[mode_key] = {
            **rank_subset,
            "utility_mean": utility_from_means(rank_subset, weights),
            "total_params": int(rank_params_all_by_rank[rank]),
            "memory_fraction_vs_exact": float(rank_params_all_by_rank[rank] / exact_total_params),
            "compression_ratio_vs_exact": float(exact_total_params / rank_params_all_by_rank[rank])
            if rank_params_all_by_rank[rank] > 0
            else None,
        }
        fixed_row, fixed_pass_set = mixed_ladder_row(
            rank=rank,
            policy=f"rank{rank}_only",
            keep_indices=[],
            rank_by_id=rank_by_id,
            rank_costs=rank_costs,
        )
        frontier_rows.append(fixed_row)
        pass_audit_rows.append(
            pass_set_audit(
                family="rank_exact_svd_ladder",
                mode=f"rank{rank}_only",
                rank=rank,
                policy=f"rank{rank}_only",
                budget_k=0,
                pass_set=fixed_pass_set,
                selected_exact_set=set(),
            )
        )

        utility_gaps: list[float] = []
        contract_gaps: list[float] = []
        for idx in range(len(edit_ids)):
            rank_row = rank_by_id.get(idx) or {}
            exact_row = exact_by_id.get(idx) or {}
            exact_utility = per_case_utility(exact_row, weights)
            rank_utility = per_case_utility(rank_row, weights)
            utility_gap = 0.0 if exact_utility is None or rank_utility is None else float(exact_utility - rank_utility)
            contract_gap = contract_gap_value(
                rank_row,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
                weights=weights,
            )
            utility_gaps.append(utility_gap)
            contract_gaps.append(contract_gap)
            per_edit_rows.append(
                {
                    "edit_index": idx,
                    "edit_id": edit_ids[idx],
                    "rank": int(rank),
                    "exact_utility": exact_utility,
                    "rank_utility": rank_utility,
                    "utility_gap": utility_gap,
                    "contract_gap": contract_gap,
                    "residual_norm": residual_norms_by_rank[rank][idx],
                    "relative_residual_norm": relative_residual_norms_by_rank[rank][idx],
                    "rank_contract_pass": contract_pass(
                        rank_row,
                        rewrite_threshold=args.contract_rewrite_threshold,
                        rephrase_threshold=args.contract_rephrase_threshold,
                        locality_threshold=args.contract_locality_threshold,
                    ),
                    "exact_contract_pass": idx in exact_pass_set,
                    "rank_params": rank_costs[idx],
                    "marginal_exact_fallback_params": int(max(p_lora - rank_costs[idx], 0)),
                }
            )
        rank_gaps_by_rank[rank] = utility_gaps
        rank_contract_gaps_by_rank[rank] = contract_gaps

        for budget in budgets:
            if budget <= 0 or budget >= len(edit_ids):
                continue
            marginal_costs = [int(max(p_lora - value, 0)) for value in rank_costs]
            selector_indices: dict[str, list[int]] = {}
            for selector in selectors:
                policy_name = f"rank{rank}_{selector}_exact"
                if selector == "dual_budget":
                    selector_indices[policy_name] = select_dual_budget(
                        k=budget,
                        contract_gaps=contract_gaps,
                        utility_gaps=utility_gaps,
                        residual_norms=residual_norms_by_rank[rank],
                        marginal_costs=marginal_costs,
                    )
                else:
                    selector_indices[policy_name] = select_indices(
                        selector,
                        budget,
                        len(edit_ids),
                        residual_norms=residual_norms_by_rank[rank],
                        contract_gaps=contract_gaps,
                        utility_gaps=utility_gaps,
                        seed=args.seed,
                    )
            for policy_name, keep_indices in selector_indices.items():
                keep_set = set(int(idx) for idx in keep_indices)
                row, pass_set = mixed_ladder_row(
                    rank=rank,
                    policy=policy_name,
                    keep_indices=keep_indices,
                    rank_by_id=rank_by_id,
                    rank_costs=rank_costs,
                )
                frontier_rows.append(row)
                pass_audit_rows.append(
                    pass_set_audit(
                        family="rank_exact_svd_ladder",
                        mode=policy_name,
                        rank=rank,
                        policy=policy_name,
                        budget_k=budget,
                        pass_set=pass_set,
                        selected_exact_set=keep_set,
                    )
                )

        exact_cache_budgets = sorted(
            {
                max(0, min(len(edit_ids), math.floor(float(row["total_params"]) / p_lora)))
                for row in frontier_rows
                if row.get("rank") == rank and row.get("total_params") is not None
            }
        )
        for exact_budget in exact_cache_budgets:
            if exact_budget <= 0 or exact_budget >= len(edit_ids):
                continue
            cache_policies: dict[str, list[int]] = {}
            for selector in selectors:
                policy_name = f"exact_cache_rank{rank}_matched_{selector}_k{exact_budget}"
                if selector == "dual_budget":
                    cache_policies[policy_name] = select_dual_budget(
                        k=exact_budget,
                        contract_gaps=contract_gaps,
                        utility_gaps=utility_gaps,
                        residual_norms=residual_norms_by_rank[rank],
                        marginal_costs=[p_lora for _ in edit_ids],
                    )
                else:
                    cache_policies[policy_name] = select_indices(
                        selector,
                        exact_budget,
                        len(edit_ids),
                        residual_norms=residual_norms_by_rank[rank],
                        contract_gaps=contract_gaps,
                        utility_gaps=utility_gaps,
                        seed=args.seed,
                    )
            for policy_name, keep_indices in cache_policies.items():
                keep_set = set(int(idx) for idx in keep_indices)
                row, pass_set = exact_cache_row(rank=rank, policy=policy_name, keep_indices=keep_indices)
                frontier_rows.append(row)
                pass_audit_rows.append(
                    pass_set_audit(
                        family="exact_cache_matched_memory",
                        mode=policy_name,
                        rank=rank,
                        policy=policy_name,
                        budget_k=exact_budget,
                        pass_set=pass_set,
                        selected_exact_set=keep_set,
                    )
                )

    adaptive_rows = []
    adaptive_total_params = 0
    adaptive_tier_counts: dict[str, int] = {}
    adaptive_pass_set: set[int] = set()
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
        if contract_pass(
            chosen_row or {},
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
        ):
            adaptive_pass_set.add(idx)
    adaptive_summary = row_summary_from_tier_rows(adaptive_rows)
    adaptive_row = {
        "family": "adaptive_rank_ladder",
        "rank": None,
        "policy": "adaptive_cheapest_contract_rank_else_exact",
        "budget_k": int(adaptive_tier_counts.get("exact", 0)),
        "post_rewrite_mean": adaptive_summary.get("post_rewrite_mean"),
        "post_rephrase_mean": adaptive_summary.get("post_rephrase_mean"),
        "post_locality_mean": adaptive_summary.get("post_locality_mean"),
        "contract_pass_rate": adaptive_summary.get("contract_pass_rate"),
        "utility_mean": utility_from_means(adaptive_summary, weights),
        "total_params": int(adaptive_total_params),
        "memory_fraction_vs_exact": float(adaptive_total_params / exact_total_params),
        "compression_ratio_vs_exact": float(exact_total_params / adaptive_total_params) if adaptive_total_params > 0 else None,
        "rank_params_active": int(adaptive_total_params - adaptive_tier_counts.get("exact", 0) * p_lora),
        "exact_fallback_params_active": int(adaptive_tier_counts.get("exact", 0) * p_lora),
        "kept_exact_count": int(adaptive_tier_counts.get("exact", 0)),
        "is_strict_memory_win": bool(adaptive_total_params < exact_total_params),
        "tier_counts": json.dumps(adaptive_tier_counts, sort_keys=True),
    }
    frontier_rows.append(adaptive_row)
    pass_audit_rows.append(
        pass_set_audit(
            family="adaptive_rank_ladder",
            mode="adaptive_cheapest_contract_rank_else_exact",
            rank=None,
            policy="adaptive_cheapest_contract_rank_else_exact",
            budget_k=int(adaptive_tier_counts.get("exact", 0)),
            pass_set=adaptive_pass_set,
        )
    )

    pass_audit_json = {
        f"{row['family']}|rank={row['rank']}|policy={row['policy']}|k={row['budget_k']}": row
        for row in pass_audit_rows
    }
    frontier_consistency = audit_frontier_memory_consistency(
        frontier_rows,
        exact_total_params=exact_total_params,
        shared_exact_equivalent=None,
        memory_margin_before_residuals=0,
        max_residuals_strictly_below_exact=None,
    )
    frontier_memory_audit = audit_frontier_memory_monotonicity(frontier_rows)
    frontier_best_points = summarize_best_frontier_points(frontier_rows, utility_weights=weights)
    summary = {
        "dataset": dataset_slug,
        "n_edits": len(edit_ids),
        "requested_ranks": requested_ranks,
        "ranks": ranks,
        "selectors": selectors,
        "mode_metrics": mode_metrics,
        "adaptive_tier_counts": adaptive_tier_counts,
        "memory_audit": {
            "exact_total_params": exact_total_params,
            "params_per_exact_lora": p_lora,
            "rank_params_all_by_rank": rank_params_all_by_rank,
            "max_exact_raw_rank": max_exact_rank,
            "rank_cap_warning": rank_cap_warning,
        },
        "frontier_memory_audit": frontier_memory_audit,
        "frontier_row_consistency": frontier_consistency,
        "frontier_best_points": frontier_best_points,
        "pass_set_audit": pass_audit_json,
        "pass_set_audit_rows": pass_audit_rows,
        "implementation_notes": {
            "main_method": "adaptive_low_rank_residual_ladder",
            "rank_tier": "effective LoRA delta truncated SVD per edit and block",
            "fallback": "exact LoRA bypass/residual for selected hard-tail edits",
            "exact_cache_eviction": "evicted edits are counted as rewrite/rephrase failures with locality 1.0",
            "trained_low_rank_baselines": "not implemented in this fast SVD runner",
            "rank_cap_warning": rank_cap_warning,
            "frontier_evaluation": (
                "Mixed fallback/frontier rows are assembled from exact and fixed-rank per-edit metrics. "
                "Use live mixed-bank evaluation for final deployment/interference confirmation."
            ),
            "fallback_accounting": "Exact fallback is accounted as exact bypass for retained hard-tail edits.",
        },
    }

    write_csv(output_dir / "rank_ladder_frontier.csv", frontier_rows)
    write_csv(output_dir / "rank_ladder_per_edit.csv", per_edit_rows)
    write_csv(output_dir / "rank_ladder_pass_set_audit.csv", pass_audit_rows)
    write_json(output_dir / "rank_ladder_pass_set_audit.json", pass_audit_json)
    write_json(output_dir / "rank_ladder_summary.json", summary)

    lines = [
        "# Adaptive Low-Rank Residual-Ladder Report",
        "",
        f"- dataset: `{dataset_slug}`",
        f"- edits: `{len(edit_ids)}`",
        f"- requested ranks: `{requested_ranks}`",
        f"- ranks: `{ranks}`",
        f"- selectors: `{selectors}`",
        f"- exact params: `{exact_total_params}`",
        "",
        "## Fixed Rank Modes",
        "",
        "| Mode | Rewrite | Rephrase | Locality | Contract | Memory fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode_name, mode in mode_metrics.items():
        lines.append(
            f"| {mode_name} | {float(mode.get('post_rewrite_mean') or 0.0):.4f} | "
            f"{float(mode.get('post_rephrase_mean') or 0.0):.4f} | "
            f"{float(mode.get('post_locality_mean') or 0.0):.4f} | "
            f"{float(mode.get('contract_pass_rate') or 0.0):.4f} | "
            f"{float(mode.get('memory_fraction_vs_exact') or 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive Rank",
            "",
            f"- adaptive tier counts: `{adaptive_tier_counts}`",
            "",
            "## Best Points",
            "",
            f"- best strict-memory-win point: `{frontier_best_points['best_strict_memory_win_point']}`",
            f"- best at-or-below-exact point: `{frontier_best_points['best_at_or_below_exact_point']}`",
            "",
            "## Audit Note",
            "",
            "- Pass-set audits are keyed by family, rank, policy, and budget K to avoid per-budget overwrites.",
            "- Memory is inference extra logical representation accounting over the frozen base model, not transient evaluator object residency.",
            "- Mixed fallback rows are offline keyed-serving frontiers assembled from per-edit exact/rank metrics.",
        ]
    )
    if rank_cap_warning:
        lines.append(f"- rank cap warning: `{rank_cap_warning}`")
    (output_dir / "rank_ladder_report.md").write_text("\n".join(lines))
    log_progress(output_dir, "complete", result_dir=str(output_dir.resolve()))


if __name__ == "__main__":
    main()
