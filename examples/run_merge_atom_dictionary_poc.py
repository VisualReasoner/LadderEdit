"""Offline merge-atom shared dictionary POC.

This variant replaces the PCA/SVD-style shared basis with behavior-validated
merged edit atoms. Each edit starts from an exact LoRA adapter vector Δ_i^E.
We greedily merge compatible edits into shared atoms Φ_c, then represent each
edit as one shared merged atom plus an optional exact residual:

    Δ_i = Φ_{cluster(i)} + z_i r_i
    r_i = Δ_i^E - Φ_{cluster(i)}

The critical difference from the earlier shared dictionary runner is that the
shared substrate is built from contract-validated merges of exact edits rather
than weight-space reconstruction.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

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
    summarize_run,
    write_json,
)
from examples.run_edit_experiment import seed_everything
from examples.run_shared_basis_compression_poc import (
    capture_adapter_bank,
    load_vectors_into_adapters,
    make_shards,
)
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
    attach_pre_metrics,
    clear_controller_logs,
    compute_pre_metrics,
    configure_evaluation_mode,
    evaluate_and_write,
    evaluate_requests,
    finalize_locality,
)


@dataclass
class MergeAtom:
    atom_id: int
    group_id: int
    member_indices: list[int]
    vector: torch.Tensor
    validation_contract_rate: float | None
    validation_mean_utility: float | None
    validation_mean_utility_drop: float | None
    merge_operator: str


@dataclass
class MergeEditRow:
    vector_row_index: int
    edit_id: str
    dataset: str
    relation: str | None
    atom_id: int
    alpha_json: str
    exact_norm: float
    dict_norm: float
    residual_norm: float
    relative_residual_norm: float
    reconstruction_error: float
    exact_rewrite: float | None
    exact_rephrase: float | None
    exact_locality: float | None
    dict_rewrite: float | None
    dict_rephrase: float | None
    dict_locality: float | None
    dict_plus_residual_rewrite: float | None
    dict_plus_residual_rephrase: float | None
    dict_plus_residual_locality: float | None
    utility_exact: float | None
    utility_dict: float | None
    utility_dict_plus_residual: float | None
    utility_gap: float | None
    contract_gap: float | None
    dict_contract_pass: bool
    subject: str | None
    prompt: str | None
    target_new: str | None


def merge_average(vectors: torch.Tensor, member_indices: list[int]) -> torch.Tensor:
    return vectors[member_indices].mean(dim=0)


def merge_ties(vectors: torch.Tensor, member_indices: list[int], *, trim_quantile: float) -> torch.Tensor:
    stack = vectors[member_indices].detach().float().cpu().clone()
    if stack.numel() == 0:
        return torch.zeros(vectors.shape[1], dtype=vectors.dtype)
    if trim_quantile > 0.0:
        flat = stack.abs().flatten()
        threshold = torch.quantile(flat, min(max(float(trim_quantile), 0.0), 1.0))
        stack[stack.abs() < threshold] = 0.0
    sign_votes = torch.sign(stack).sum(dim=0)
    elected = torch.sign(sign_votes)
    keep_mask = (torch.sign(stack) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
    numer = (stack * keep_mask).sum(dim=0)
    denom = keep_mask.sum(dim=0).clamp_min(1)
    merged = numer / denom
    zero_mask = elected == 0
    if bool(zero_mask.any()):
        merged[zero_mask] = stack.mean(dim=0)[zero_mask]
    return merged.to(vectors.dtype)


def merge_vectors(
    vectors: torch.Tensor,
    member_indices: list[int],
    *,
    merge_operator: str,
    ties_trim_quantile: float,
) -> torch.Tensor:
    if merge_operator == "average":
        return merge_average(vectors, member_indices)
    if merge_operator == "ties":
        return merge_ties(vectors, member_indices, trim_quantile=ties_trim_quantile)
    raise ValueError(f"Unsupported merge operator: {merge_operator}")


def merge_dictionary_storage_stats(
    *,
    num_vectors: int,
    vector_dim: int,
    atoms: list[MergeAtom],
    keep_indices: list[int],
    charge_dictionary_for_all_edits: bool,
    code_width: int = 1,
) -> dict[str, Any]:
    keep_set = set(int(idx) for idx in keep_indices)
    exact_params = int(num_vectors * vector_dim)
    dictionary_params = int(len(atoms) * vector_dim)
    supported_count = int(num_vectors if charge_dictionary_for_all_edits else num_vectors - len(keep_set))
    code_params = int(max(0, supported_count) * code_width)
    residual_params = int(len(keep_set) * vector_dim)
    total_params = int(dictionary_params + code_params + residual_params)
    return {
        "exact_params": exact_params,
        "dictionary_params": dictionary_params,
        "code_params": code_params,
        "residual_params": residual_params,
        "total_params": total_params,
        "memory_fraction_vs_exact": None if exact_params <= 0 else float(total_params / exact_params),
        "compression_ratio_vs_exact": None if total_params <= 0 else float(exact_params / total_params),
        "kept_residual_count": int(len(keep_set)),
        "num_atoms": int(len(atoms)),
        "code_width": int(code_width),
    }


def evaluate_subset_contract(
    *,
    controller,
    edit_ids: list[str],
    exact_weights: dict[str, dict[str, torch.Tensor]],
    schema: list[tuple[str, tuple[int, ...], int]],
    trial_vectors: torch.Tensor,
    subset_indices: list[int],
    editor: BaseEditor,
    hparams,
    records: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    pre_metrics: list[dict[str, Any]],
    eval_metric: str,
    run_base: dict[str, Any],
    utility_weights: tuple[float, float, float],
    rewrite_threshold: float,
    rephrase_threshold: float,
    locality_threshold: float,
) -> dict[str, Any]:
    load_vectors_into_adapters(controller, edit_ids, trial_vectors, schema, exact_weights)
    clear_controller_logs(editor)
    subset_requests = [requests[idx] for idx in subset_indices]
    subset_records = [records[idx] for idx in subset_indices]
    subset_pre = [pre_metrics[idx] for idx in subset_indices]
    metrics = evaluate_requests(editor, subset_requests, eval_metric)
    metrics = attach_pre_metrics(metrics, subset_pre)
    metrics = finalize_locality(metrics, hparams)
    summary = summarize_run(
        metrics,
        subset_records,
        {
            **run_base,
            "data_type": run_base["data_type"],
            "stream_type": "merge_atom_subset_eval",
            "seed": run_base["seed"],
        },
        memory_snapshot=[],
    )
    case_rows = summary.get("per_case") or []
    contract_flags = [
        contract_pass(
            row,
            rewrite_threshold=rewrite_threshold,
            rephrase_threshold=rephrase_threshold,
            locality_threshold=locality_threshold,
        )
        for row in case_rows
    ]
    utilities = [per_case_utility(row, utility_weights) for row in case_rows]
    return {
        "summary": summary,
        "case_rows": case_rows,
        "all_pass": all(contract_flags),
        "contract_rate": None if not case_rows else float(sum(contract_flags) / len(case_rows)),
        "mean_utility": mean_or_none(utilities),
    }


def build_merge_atoms(
    *,
    controller,
    edit_ids: list[str],
    exact_weights: dict[str, dict[str, torch.Tensor]],
    schema: list[tuple[str, tuple[int, ...], int]],
    exact_vectors: torch.Tensor,
    groups: list[list[int]],
    editor: BaseEditor,
    hparams,
    records: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    pre_metrics: list[dict[str, Any]],
    eval_metric: str,
    run_base: dict[str, Any],
    utility_weights: tuple[float, float, float],
    exact_case_rows: dict[int, dict[str, Any]],
    merge_operator: str,
    ties_trim_quantile: float,
    max_cluster_size: int,
    max_mean_utility_drop: float,
    max_contract_drop_count: int,
    candidate_limit_per_edit: int,
    merge_accept_mode: str,
    rewrite_threshold: float,
    rephrase_threshold: float,
    locality_threshold: float,
) -> tuple[torch.Tensor, dict[int, int], list[MergeAtom], list[dict[str, Any]]]:
    atoms: list[MergeAtom] = []
    assignment: dict[int, int] = {}
    validation_rows: list[dict[str, Any]] = []

    for group_id, group_indices in enumerate(groups):
        local_atom_ids: list[int] = []
        for idx in group_indices:
            best_candidate: dict[str, Any] | None = None
            candidate_atoms = []
            for atom_id in local_atom_ids:
                atom = atoms[atom_id]
                if max_cluster_size > 0 and len(atom.member_indices) >= max_cluster_size:
                    continue
                similarity = float(
                    torch.dot(
                        exact_vectors[idx].detach().float().cpu(),
                        atom.vector.detach().float().cpu(),
                    ).item()
                )
                candidate_atoms.append((similarity, atom_id))
            candidate_atoms.sort(key=lambda item: item[0], reverse=True)
            if candidate_limit_per_edit > 0:
                candidate_atoms = candidate_atoms[:candidate_limit_per_edit]

            for _similarity, atom_id in candidate_atoms:
                atom = atoms[atom_id]
                proposed_members = list(atom.member_indices) + [int(idx)]
                merged_vector = merge_vectors(
                    exact_vectors,
                    proposed_members,
                    merge_operator=merge_operator,
                    ties_trim_quantile=ties_trim_quantile,
                )
                trial_vectors = exact_vectors.clone()
                trial_vectors[proposed_members] = merged_vector
                validation = evaluate_subset_contract(
                    controller=controller,
                    edit_ids=edit_ids,
                    exact_weights=exact_weights,
                    schema=schema,
                    trial_vectors=trial_vectors,
                    subset_indices=proposed_members,
                    editor=editor,
                    hparams=hparams,
                    records=records,
                    requests=requests,
                    pre_metrics=pre_metrics,
                    eval_metric=eval_metric,
                    run_base=run_base,
                    utility_weights=utility_weights,
                    rewrite_threshold=rewrite_threshold,
                    rephrase_threshold=rephrase_threshold,
                    locality_threshold=locality_threshold,
                )

                merged_rows = validation["case_rows"]
                exact_rows = [exact_case_rows[member_idx] for member_idx in proposed_members]
                merged_contract = [
                    contract_pass(
                        row,
                        rewrite_threshold=rewrite_threshold,
                        rephrase_threshold=rephrase_threshold,
                        locality_threshold=locality_threshold,
                    )
                    for row in merged_rows
                ]
                exact_contract = [
                    contract_pass(
                        row,
                        rewrite_threshold=rewrite_threshold,
                        rephrase_threshold=rephrase_threshold,
                        locality_threshold=locality_threshold,
                    )
                    for row in exact_rows
                ]
                merged_utilities = [per_case_utility(row, utility_weights) for row in merged_rows]
                exact_utilities = [per_case_utility(row, utility_weights) for row in exact_rows]
                exact_contract_count = sum(1 for flag in exact_contract if flag)
                merged_contract_count = sum(1 for flag in merged_contract if flag)
                contract_drop = int(exact_contract_count - merged_contract_count)
                exact_mean_utility = mean_or_none(exact_utilities)
                merged_mean_utility = mean_or_none(merged_utilities)
                mean_drop = None
                if exact_mean_utility is not None and merged_mean_utility is not None:
                    mean_drop = float(exact_mean_utility - merged_mean_utility)
                if merge_accept_mode == "absolute_contract":
                    accept = bool(validation["all_pass"])
                elif merge_accept_mode == "relative_drop":
                    accept = (
                        contract_drop <= max_contract_drop_count
                        and (mean_drop is None or mean_drop <= max_mean_utility_drop)
                    )
                elif merge_accept_mode == "hybrid":
                    accept = bool(validation["all_pass"]) and (
                        contract_drop <= max_contract_drop_count
                        and (mean_drop is None or mean_drop <= max_mean_utility_drop)
                    )
                else:
                    raise ValueError(f"Unsupported merge_accept_mode: {merge_accept_mode}")
                candidate = {
                    "atom_id": atom_id,
                    "merged_vector": merged_vector,
                    "members": proposed_members,
                    "contract_drop": contract_drop,
                    "mean_drop": mean_drop,
                    "merged_contract_rate": validation["contract_rate"],
                    "merged_mean_utility": validation["mean_utility"],
                    "accept": accept,
                }
                validation_rows.append(
                    {
                        "group_id": group_id,
                        "candidate_atom_id": atom_id,
                        "incoming_index": int(idx),
                        "proposed_size": len(proposed_members),
                        "merge_accept_mode": merge_accept_mode,
                        "contract_drop": contract_drop,
                        "mean_utility_drop": mean_drop,
                        "accept": accept,
                        "merged_contract_rate": validation["contract_rate"],
                        "merged_mean_utility": validation["mean_utility"],
                    }
                )
                if accept:
                    score = (
                        len(proposed_members),
                        float(validation["contract_rate"] or 0.0),
                        float(validation["mean_utility"] or 0.0),
                    )
                    if best_candidate is None or score > best_candidate["score"]:
                        best_candidate = {**candidate, "score": score}

            if best_candidate is None:
                atom_id = len(atoms)
                vector = exact_vectors[idx].detach().float().cpu().clone()
                atoms.append(
                    MergeAtom(
                        atom_id=atom_id,
                        group_id=group_id,
                        member_indices=[int(idx)],
                        vector=vector,
                        validation_contract_rate=1.0,
                        validation_mean_utility=per_case_utility(exact_case_rows[idx], utility_weights),
                        validation_mean_utility_drop=0.0,
                        merge_operator=merge_operator,
                    )
                )
                local_atom_ids.append(atom_id)
                assignment[int(idx)] = atom_id
            else:
                atom_id = int(best_candidate["atom_id"])
                atom = atoms[atom_id]
                atom.member_indices = list(best_candidate["members"])
                atom.vector = best_candidate["merged_vector"].detach().float().cpu().clone()
                atom.validation_contract_rate = best_candidate["merged_contract_rate"]
                atom.validation_mean_utility = best_candidate["merged_mean_utility"]
                atom.validation_mean_utility_drop = best_candidate["mean_drop"]
                assignment[int(idx)] = atom_id

    dict_vectors = torch.zeros_like(exact_vectors)
    for idx in range(len(edit_ids)):
        atom_id = assignment[int(idx)]
        dict_vectors[idx] = atoms[atom_id].vector.to(exact_vectors.dtype)
    return dict_vectors, assignment, atoms, validation_rows


def run_policy_evaluation_merge(
    *,
    controller,
    edit_ids: list[str],
    exact_weights: dict[str, dict[str, torch.Tensor]],
    schema: list[tuple[str, tuple[int, ...], int]],
    recon_vectors: torch.Tensor,
    exact_vectors: torch.Tensor,
    keep_indices: list[int],
    atoms: list[MergeAtom],
    vector_dim: int,
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
    charge_dictionary_for_all_edits: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mixed_vectors = recon_vectors.clone()
    if keep_indices:
        mixed_vectors[keep_indices] = exact_vectors[keep_indices]
    load_vectors_into_adapters(controller, edit_ids, mixed_vectors, schema, exact_weights)
    storage = merge_dictionary_storage_stats(
        num_vectors=int(exact_vectors.shape[0]),
        vector_dim=vector_dim,
        atoms=atoms,
        keep_indices=keep_indices,
        charge_dictionary_for_all_edits=charge_dictionary_for_all_edits,
    )
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
    return summary, storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT")
    parser.add_argument("--hparams_dir", default=str(REPO_ROOT / "hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware-staged.yaml"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--data_type", default="CounterFact", choices=["CounterFact", "ZsRE", "WikiBigEdit"])
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs/merge_atom_dictionary_poc"))
    parser.add_argument("--ds_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", default="relation")
    parser.add_argument("--num_shards", type=int, default=4)
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
    parser.add_argument("--merge_operator", choices=["average", "ties"], default="average")
    parser.add_argument("--ties_trim_quantile", type=float, default=0.8)
    parser.add_argument("--max_cluster_size", type=int, default=8)
    parser.add_argument("--max_mean_utility_drop", type=float, default=0.05)
    parser.add_argument("--max_contract_drop_count", type=int, default=0)
    parser.add_argument("--candidate_limit_per_edit", type=int, default=4)
    parser.add_argument(
        "--merge_accept_mode",
        choices=["absolute_contract", "relative_drop", "hybrid"],
        default="absolute_contract",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    dataset_slug = str(args.data_type).lower()
    run_slug = f"{dataset_slug}_{args.strategy}_{args.merge_operator}_n{args.ds_size}_seed{args.seed}"
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
        "stream_type": "merge_atom_dictionary_poc",
        "seed": args.seed,
        "sequential_edit": True,
        "stream_length": len(records),
        "requested_ds_size": args.ds_size,
        "strategy": args.strategy,
        "merge_operator": args.merge_operator,
        "merge_accept_mode": args.merge_accept_mode,
    }
    write_json(output_dir / "run_config.json", {**run_base, "residual_budgets": args.residual_budgets})

    editor = BaseEditor.from_hparams(hparams)
    pre_metrics = compute_pre_metrics(editor, requests, editor_inputs["eval_metric"])
    for request in requests:
        apply_single_edit(editor, request)
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
    exact_weights, schema, exact_vectors = capture_adapter_bank(controller, edit_ids)
    vector_dim = int(exact_vectors.shape[1])
    groups = make_shards(args.strategy, exact_vectors, relation_ids, num_shards=args.num_shards, seed=args.seed)
    if args.strategy.strip().lower() == "relation" and all(relation_id is None for relation_id in relation_ids):
        raise ValueError(
            "Relation sharding was requested, but this dataset provides no relation_id values. "
            "Choose --strategy random or --strategy spectral instead."
        )

    exact_summary, exact_storage = run_policy_evaluation_merge(
        controller=controller,
        edit_ids=edit_ids,
        exact_weights=exact_weights,
        schema=schema,
        recon_vectors=exact_vectors,
        exact_vectors=exact_vectors,
        keep_indices=list(range(len(edit_ids))),
        atoms=[],
        vector_dim=vector_dim,
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
        charge_dictionary_for_all_edits=False,
    )
    exact_case_rows = per_case_by_id(exact_summary)

    dict_vectors, assignment, atoms, merge_validation_rows = build_merge_atoms(
        controller=controller,
        edit_ids=edit_ids,
        exact_weights=exact_weights,
        schema=schema,
        exact_vectors=exact_vectors,
        groups=groups,
        editor=editor,
        hparams=hparams,
        records=records,
        requests=requests,
        pre_metrics=pre_metrics,
        eval_metric=editor_inputs["eval_metric"],
        run_base=run_base,
        utility_weights=weights,
        exact_case_rows=exact_case_rows,
        merge_operator=args.merge_operator,
        ties_trim_quantile=args.ties_trim_quantile,
        max_cluster_size=args.max_cluster_size,
        max_mean_utility_drop=args.max_mean_utility_drop,
        max_contract_drop_count=args.max_contract_drop_count,
        candidate_limit_per_edit=args.candidate_limit_per_edit,
        merge_accept_mode=args.merge_accept_mode,
        rewrite_threshold=args.contract_rewrite_threshold,
        rephrase_threshold=args.contract_rephrase_threshold,
        locality_threshold=args.contract_locality_threshold,
    )

    initial_storage = merge_dictionary_storage_stats(
        num_vectors=int(exact_vectors.shape[0]),
        vector_dim=vector_dim,
        atoms=atoms,
        keep_indices=[],
        charge_dictionary_for_all_edits=True,
    )
    write_json(
        output_dir / "dictionary_metadata.json",
        {
            "dataset": dataset_slug,
            "n_edits": len(edit_ids),
            "vector_dim": vector_dim,
            "strategy": args.strategy,
            "merge_operator": args.merge_operator,
            "merge_accept_mode": args.merge_accept_mode,
            "num_atoms": len(atoms),
            "groups": groups,
            "atoms": [
                {
                    "atom_id": atom.atom_id,
                    "group_id": atom.group_id,
                    "size": len(atom.member_indices),
                    "member_indices": atom.member_indices,
                    "validation_contract_rate": atom.validation_contract_rate,
                    "validation_mean_utility": atom.validation_mean_utility,
                    "validation_mean_utility_drop": atom.validation_mean_utility_drop,
                    "merge_operator": atom.merge_operator,
                }
                for atom in atoms
            ],
            "dictionary_params": initial_storage.get("dictionary_params"),
            "code_params_all": initial_storage.get("code_params"),
            "exact_params": exact_vectors.shape[0] * vector_dim,
            "p_lora": int(vector_dim),
            "p_residual": int(vector_dim),
            "contract_thresholds": {
                "rewrite": args.contract_rewrite_threshold,
                "rephrase": args.contract_rephrase_threshold,
                "locality": args.contract_locality_threshold,
            },
            "utility_weights": list(weights),
            "implementation_notes": {
                "shared_substrate": "behavior_validated_merge_atoms",
                "merge_operator": args.merge_operator,
                "merge_accept_mode": args.merge_accept_mode,
                "raw_vector_warning": "Merged atoms currently operate in raw adapter vector space, not canonicalized effective ΔW space.",
            },
        },
    )

    residual_vectors = exact_vectors - dict_vectors
    exact_norms = torch.linalg.norm(exact_vectors, dim=1)
    dict_norms = torch.linalg.norm(dict_vectors, dim=1)
    residual_norms = torch.linalg.norm(residual_vectors, dim=1)
    relative_residual_norms = residual_norms / exact_norms.clamp_min(1.0e-12)
    reconstruction_errors = residual_vectors.pow(2).sum(dim=1) / exact_vectors.pow(2).sum(dim=1).clamp_min(1.0e-12)
    atom_vectors = (
        torch.stack([atom.vector.to(exact_vectors.dtype) for atom in atoms], dim=0)
        if atoms
        else torch.empty((0, vector_dim), dtype=exact_vectors.dtype)
    )
    torch.save(
        {
            "edit_ids": edit_ids,
            "assignment": assignment,
            "exact_vectors": exact_vectors,
            "dict_vectors": dict_vectors,
            "residual_vectors": residual_vectors,
            "atom_vectors": atom_vectors,
        },
        output_dir / "vector_bank.pt",
    )

    dict_summary, dict_storage = run_policy_evaluation_merge(
        controller=controller,
        edit_ids=edit_ids,
        exact_weights=exact_weights,
        schema=schema,
        recon_vectors=dict_vectors,
        exact_vectors=exact_vectors,
        keep_indices=[],
        atoms=atoms,
        vector_dim=vector_dim,
        editor=editor,
        hparams=hparams,
        method=method,
        backbone=run_base["backbone"],
        output_dir=output_dir / "mode_dictionary",
        records=records,
        requests=requests,
        eval_metric=editor_inputs["eval_metric"],
        run_base=run_base,
        pre_metrics=pre_metrics,
        condition_name=f"{dataset_slug}_dictionary",
        charge_dictionary_for_all_edits=True,
    )
    dict_plus_residual_summary, dict_plus_residual_storage = run_policy_evaluation_merge(
        controller=controller,
        edit_ids=edit_ids,
        exact_weights=exact_weights,
        schema=schema,
        recon_vectors=dict_vectors,
        exact_vectors=exact_vectors,
        keep_indices=list(range(len(edit_ids))),
        atoms=atoms,
        vector_dim=vector_dim,
        editor=editor,
        hparams=hparams,
        method=method,
        backbone=run_base["backbone"],
        output_dir=output_dir / "mode_dictionary_plus_residual",
        records=records,
        requests=requests,
        eval_metric=editor_inputs["eval_metric"],
        run_base=run_base,
        pre_metrics=pre_metrics,
        condition_name=f"{dataset_slug}_dictionary_plus_residual",
        charge_dictionary_for_all_edits=True,
    )

    exact_by_id = per_case_by_id(exact_summary)
    dict_by_id = per_case_by_id(dict_summary)
    dpr_by_id = per_case_by_id(dict_plus_residual_summary)

    per_edit_rows: list[dict[str, Any]] = []
    utility_gaps = []
    contract_gaps = []
    residual_norm_list = []
    atom_sizes = {atom.atom_id: len(atom.member_indices) for atom in atoms}
    for idx, request in enumerate(requests):
        exact_row = exact_by_id.get(idx) or {}
        dict_row = dict_by_id.get(idx) or {}
        dpr_row = dpr_by_id.get(idx) or {}
        utility_exact = per_case_utility(exact_row, weights)
        utility_dict = per_case_utility(dict_row, weights)
        utility_dpr = per_case_utility(dpr_row, weights)
        utility_gap = None if utility_exact is None or utility_dict is None else float(utility_exact - utility_dict)
        contract_gap = contract_gap_value(
            dict_row,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
            weights=weights,
        )
        atom_id = assignment[idx]
        row = MergeEditRow(
            vector_row_index=idx,
            edit_id=edit_ids[idx],
            dataset=dataset_slug,
            relation=request.get("relation_id"),
            atom_id=atom_id,
            alpha_json=json.dumps([1.0]),
            exact_norm=float(exact_norms[idx].item()),
            dict_norm=float(dict_norms[idx].item()),
            residual_norm=float(residual_norms[idx].item()),
            relative_residual_norm=float(relative_residual_norms[idx].item()),
            reconstruction_error=float(reconstruction_errors[idx].item()),
            exact_rewrite=metric_or_none(exact_row, "post_rewrite_acc"),
            exact_rephrase=metric_or_none(exact_row, "post_rephrase_acc"),
            exact_locality=metric_or_none(exact_row, "post_locality_acc"),
            dict_rewrite=metric_or_none(dict_row, "post_rewrite_acc"),
            dict_rephrase=metric_or_none(dict_row, "post_rephrase_acc"),
            dict_locality=metric_or_none(dict_row, "post_locality_acc"),
            dict_plus_residual_rewrite=metric_or_none(dpr_row, "post_rewrite_acc"),
            dict_plus_residual_rephrase=metric_or_none(dpr_row, "post_rephrase_acc"),
            dict_plus_residual_locality=metric_or_none(dpr_row, "post_locality_acc"),
            utility_exact=utility_exact,
            utility_dict=utility_dict,
            utility_dict_plus_residual=utility_dpr,
            utility_gap=utility_gap,
            contract_gap=contract_gap,
            dict_contract_pass=contract_pass(
                dict_row,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            subject=request.get("subject"),
            prompt=request.get("prompt"),
            target_new=request.get("target_new"),
        )
        row_dict = asdict(row)
        row_dict["atom_size"] = atom_sizes.get(atom_id)
        per_edit_rows.append(row_dict)
        utility_gaps.append(utility_gap)
        contract_gaps.append(contract_gap)
        residual_norm_list.append(float(residual_norms[idx].item()))
    write_csv(output_dir / "per_edit.csv", per_edit_rows)
    write_csv(output_dir / "merge_validation.csv", merge_validation_rows)

    exact_dpr_consistency = {
        "mode_rewrite_gap": None
        if exact_summary.get("post_rewrite_mean") is None or dict_plus_residual_summary.get("post_rewrite_mean") is None
        else float(abs(float(exact_summary["post_rewrite_mean"]) - float(dict_plus_residual_summary["post_rewrite_mean"]))),
        "mode_rephrase_gap": None
        if exact_summary.get("post_rephrase_mean") is None or dict_plus_residual_summary.get("post_rephrase_mean") is None
        else float(abs(float(exact_summary["post_rephrase_mean"]) - float(dict_plus_residual_summary["post_rephrase_mean"]))),
        "mode_locality_gap": None
        if exact_summary.get("post_locality_mean") is None or dict_plus_residual_summary.get("post_locality_mean") is None
        else float(abs(float(exact_summary["post_locality_mean"]) - float(dict_plus_residual_summary["post_locality_mean"]))),
        "max_per_case_metric_gap": max(
            [
                abs(float((row.get("exact_rewrite") or 0.0) - (row.get("dict_plus_residual_rewrite") or 0.0)))
                for row in per_edit_rows
            ]
            + [
                abs(float((row.get("exact_rephrase") or 0.0) - (row.get("dict_plus_residual_rephrase") or 0.0)))
                for row in per_edit_rows
            ]
            + [
                abs(float((row.get("exact_locality") or 0.0) - (row.get("dict_plus_residual_locality") or 0.0)))
                for row in per_edit_rows
            ]
        )
        if per_edit_rows
        else None,
    }

    budgets = parse_budgets(args.residual_budgets, len(edit_ids))
    policies = ["dictionary_only", "random", "recency", "residual_norm", "contract_gap", "oracle_utility_gap", "exact_only"]
    memory_audit = compute_memory_break_even(
        n_edits=len(edit_ids),
        p_lora=vector_dim,
        dictionary_params=int(dict_storage.get("dictionary_params") or 0),
        code_params_all=int(dict_storage.get("code_params") or 0),
        p_residual=vector_dim,
    )
    frontier_rows: list[dict[str, Any]] = []
    for budget in budgets:
        for policy in policies:
            if policy == "dictionary_only" and budget != 0:
                continue
            if policy == "exact_only" and budget != len(edit_ids):
                continue
            if policy not in {"dictionary_only", "exact_only"} and budget in {0, len(edit_ids)}:
                continue
            keep_indices = select_indices(
                policy,
                budget,
                len(edit_ids),
                residual_norms=residual_norm_list,
                contract_gaps=contract_gaps,
                utility_gaps=[0.0 if value is None else value for value in utility_gaps],
                seed=args.seed,
            )
            if policy == "dictionary_only":
                summary, storage = dict_summary, dict_storage
            elif policy == "exact_only":
                summary, storage = exact_summary, exact_storage
            else:
                summary, storage = run_policy_evaluation_merge(
                    controller=controller,
                    edit_ids=edit_ids,
                    exact_weights=exact_weights,
                    schema=schema,
                    recon_vectors=dict_vectors,
                    exact_vectors=exact_vectors,
                    keep_indices=keep_indices,
                    atoms=atoms,
                    vector_dim=vector_dim,
                    editor=editor,
                    hparams=hparams,
                    method=method,
                    backbone=run_base["backbone"],
                    output_dir=output_dir / f"policy_{policy}_k{budget}",
                    records=records,
                    requests=requests,
                    eval_metric=editor_inputs["eval_metric"],
                    run_base=run_base,
                    pre_metrics=pre_metrics,
                    condition_name=f"{dataset_slug}_{policy}_k{budget}",
                    charge_dictionary_for_all_edits=True,
                )
            summary_row = summary_subset(
                summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            )
            frontier_rows.append(
                {
                    "budget_k": int(budget),
                    "policy": policy,
                    "post_rewrite_mean": summary_row.get("post_rewrite_mean"),
                    "post_rephrase_mean": summary_row.get("post_rephrase_mean"),
                    "post_locality_mean": summary_row.get("post_locality_mean"),
                    "contract_pass_rate": summary_row.get("contract_pass_rate"),
                    "memory_fraction_vs_exact": storage.get("memory_fraction_vs_exact"),
                    "compression_ratio_vs_exact": storage.get("compression_ratio_vs_exact"),
                    "dictionary_params": storage.get("dictionary_params"),
                    "code_params": storage.get("code_params"),
                    "residual_params": storage.get("residual_params"),
                    "total_params": storage.get("total_params"),
                    "kept_residual_count": storage.get("kept_residual_count"),
                    "num_atoms": storage.get("num_atoms"),
                    "shared_exact_equivalent": memory_audit.get("shared_exact_equivalent"),
                    "shared_base_fraction": memory_audit.get("shared_base_fraction"),
                    "memory_margin_before_residuals": memory_audit.get("memory_margin_before_residuals"),
                    "max_residuals_strictly_below_exact": memory_audit.get("max_residuals_strictly_below_exact"),
                    "is_strict_memory_win": None
                    if storage.get("total_params") is None or exact_storage.get("exact_params") is None
                    else bool(int(storage["total_params"]) < int(exact_storage["exact_params"])),
                }
            )
    write_csv(output_dir / "frontier.csv", frontier_rows)
    frontier_memory_audit = audit_frontier_memory_monotonicity(frontier_rows)
    frontier_row_consistency = audit_frontier_memory_consistency(
        frontier_rows,
        exact_total_params=int(memory_audit["exact_total_params"]),
        shared_exact_equivalent=memory_audit.get("shared_exact_equivalent"),
        memory_margin_before_residuals=int(memory_audit["memory_margin_before_residuals"]),
        max_residuals_strictly_below_exact=memory_audit.get("max_residuals_strictly_below_exact"),
    )
    frontier_best_points = summarize_best_frontier_points(frontier_rows, utility_weights=weights)

    compressibility_summary = {
        "mean_utility_gap": mean_or_none(utility_gaps),
        "median_utility_gap": median_or_none(utility_gaps),
        "mean_relative_residual_norm": mean_or_none([row["relative_residual_norm"] for row in per_edit_rows]),
        "median_relative_residual_norm": median_or_none([row["relative_residual_norm"] for row in per_edit_rows]),
        "dictionary_contract_pass_rate": float(sum(1 for row in per_edit_rows if row["dict_contract_pass"]) / len(per_edit_rows)) if per_edit_rows else None,
        "fraction_high_gap_edits": float(sum(1 for value in utility_gaps if value is not None and value > args.high_gap_threshold) / len(per_edit_rows)) if per_edit_rows else None,
        "high_gap_threshold": float(args.high_gap_threshold),
        "spearman_residual_norm_vs_utility_gap": spearman(
            [float(row["residual_norm"]) for row in per_edit_rows],
            [0.0 if value is None else value for value in utility_gaps],
        ),
        "spearman_contract_gap_vs_utility_gap": spearman(
            [float(row["contract_gap"]) for row in per_edit_rows],
            [0.0 if value is None else value for value in utility_gaps],
        ),
        "contract_gap_vs_oracle_utility_gap_corr": spearman(
            [float(row["contract_gap"]) for row in per_edit_rows],
            [0.0 if value is None else value for value in utility_gaps],
        ),
    }

    frontier_by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in frontier_rows:
        frontier_by_policy.setdefault(str(row["policy"]), []).append(row)
    for policy_rows in frontier_by_policy.values():
        policy_rows.sort(key=lambda row: (int(row["budget_k"]), float(row["memory_fraction_vs_exact"] or 0.0)))

    summary = {
        "dataset": dataset_slug,
        "n_edits": len(edit_ids),
        "strategy": args.strategy,
        "merge_operator": args.merge_operator,
        "merge_accept_mode": args.merge_accept_mode,
        "num_atoms": len(atoms),
        "implementation_notes": {
            "shared_substrate": "behavior_validated_merge_atoms",
            "exact_teacher": "current exact HOPEDIT bank",
            "merge_accept_mode": args.merge_accept_mode,
            "raw_vector_warning": "Merged atoms currently operate in raw adapter vector space, not canonicalized effective ΔW space.",
        },
        "mode_metrics": {
            "exact": summary_subset(
                exact_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            "dictionary": summary_subset(
                dict_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            "dictionary_plus_residual": summary_subset(
                dict_plus_residual_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
        },
        "compressibility": compressibility_summary,
        "memory_audit": memory_audit,
        "frontier_memory_audit": frontier_memory_audit,
        "frontier_row_consistency": frontier_row_consistency,
        "frontier_best_points": frontier_best_points,
        "exact_vs_dictionary_plus_residual_consistency": exact_dpr_consistency,
        "merge_atoms": [
            {
                "atom_id": atom.atom_id,
                "group_id": atom.group_id,
                "size": len(atom.member_indices),
                "member_indices": atom.member_indices,
                "validation_contract_rate": atom.validation_contract_rate,
                "validation_mean_utility": atom.validation_mean_utility,
                "validation_mean_utility_drop": atom.validation_mean_utility_drop,
                "merge_operator": atom.merge_operator,
            }
            for atom in atoms
        ],
        "frontier": frontier_by_policy,
    }
    write_json(output_dir / "summary.json", summary)

    lines = [
        "# Merge-Atom Dictionary Report",
        "",
        f"- dataset: `{dataset_slug}`",
        f"- edits: `{len(edit_ids)}`",
        f"- strategy: `{args.strategy}`",
        f"- merge_operator: `{args.merge_operator}`",
        f"- merge_accept_mode: `{args.merge_accept_mode}`",
        f"- num_atoms: `{len(atoms)}`",
        "",
        "## Memory Audit",
        "",
        f"- exact total params: `{memory_audit['exact_total_params']}`",
        f"- params per exact LoRA: `{memory_audit['p_lora']}`",
        f"- shared base params (atoms + all codes): `{memory_audit['shared_base_params']}`",
        f"- shared exact equivalent: `{memory_audit['shared_exact_equivalent']}`",
        f"- shared base fraction: `{memory_audit['shared_base_fraction']}`",
        f"- max residuals strictly below exact: `{memory_audit['max_residuals_strictly_below_exact']}`",
        "",
        "## Aggregate Mode Comparison",
        "",
        "| Mode | Rewrite | Rephrase | Locality | Contract |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode_name in ["exact", "dictionary", "dictionary_plus_residual"]:
        mode_summary = summary["mode_metrics"][mode_name]
        lines.append(
            f"| {mode_name} | {float(mode_summary.get('post_rewrite_mean') or 0.0):.4f} | {float(mode_summary.get('post_rephrase_mean') or 0.0):.4f} | {float(mode_summary.get('post_locality_mean') or 0.0):.4f} | {float(mode_summary.get('contract_pass_rate') or 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Merge Atom Summary",
            "",
            f"- atom sizes: `{[len(atom.member_indices) for atom in atoms]}`",
            f"- mean atom size: `{mean_or_none([len(atom.member_indices) for atom in atoms])}`",
            f"- merge validations recorded: `{len(merge_validation_rows)}`",
            "",
            "## Frontier Best Points",
            "",
            f"- best strict-memory-win point: `{frontier_best_points['best_strict_memory_win_point']}`",
            f"- best at-or-below-exact point: `{frontier_best_points['best_at_or_below_exact_point']}`",
            "",
            "## Residual-Budget Frontier",
            "",
            "| Policy | Budget | Rewrite | Rephrase | Locality | Contract | Memory Fraction | Strict Win |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in frontier_rows:
        lines.append(
            f"| {row['policy']} | {row['budget_k']} | {float(row['post_rewrite_mean'] or 0.0):.4f} | {float(row['post_rephrase_mean'] or 0.0):.4f} | {float(row['post_locality_mean'] or 0.0):.4f} | {float(row['contract_pass_rate'] or 0.0):.4f} | {float(row['memory_fraction_vs_exact'] or 0.0):.4f} | {row['is_strict_memory_win']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines))

    plot_gap_histogram(output_dir / "gap_histogram.png", [0.0 if value is None else float(value) for value in utility_gaps])
    plot_frontier(output_dir / "frontier.png", frontier_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
