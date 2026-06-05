"""Proof-of-concept for capacity-controlled shared-basis edit memory.

This runner trains ordinary HopEdit per-edit LoRA adapters, then treats those
adapters as the "ground truth" edit deltas.  It tests whether groups of edits
can be represented by a shard-level low-rank basis plus per-edit coefficients:

    exact adapters -> vectorize -> shard -> SVD basis -> reconstruct -> eval

The goal is not to introduce a new routing rule yet.  The first question is
whether the value side is compressible at all under meaningful shard layouts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
from examples.run_wikibigedit_lifelong import (
    apply_single_edit,
    compute_pre_metrics,
    configure_evaluation_mode,
    evaluate_and_write,
)


def parse_conditions(raw: str) -> list[tuple[str, int]]:
    conditions = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Condition must be strategy:rank, got {token!r}")
        strategy, rank_raw = token.split(":", 1)
        conditions.append((strategy.strip().lower(), int(rank_raw)))
    if not conditions:
        raise ValueError("At least one compression condition is required")
    return conditions


def tensor_vector_schema(weights: dict[str, torch.Tensor]) -> list[tuple[str, tuple[int, ...], int]]:
    schema = []
    for name in sorted(weights):
        tensor = weights[name].detach().cpu()
        schema.append((name, tuple(int(dim) for dim in tensor.shape), int(tensor.numel())))
    return schema


def flatten_weights(weights: dict[str, torch.Tensor], schema: list[tuple[str, tuple[int, ...], int]]) -> torch.Tensor:
    parts = []
    for name, _shape, _numel in schema:
        if name not in weights:
            raise KeyError(f"Adapter weights missing parameter {name!r}")
        parts.append(weights[name].detach().float().cpu().reshape(-1))
    return torch.cat(parts, dim=0)


def unflatten_weights(
    vector: torch.Tensor,
    schema: list[tuple[str, tuple[int, ...], int]],
    reference: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    vector = vector.detach().cpu()
    offset = 0
    restored = {}
    for name, shape, numel in schema:
        chunk = vector[offset : offset + numel].reshape(shape)
        ref = reference[name]
        restored[name] = chunk.to(dtype=ref.dtype)
        offset += numel
    return restored


def group_lambda_max(similarity: torch.Tensor, indices: list[int]) -> float:
    if len(indices) <= 1:
        return 1.0 if indices else 0.0
    sub = similarity[indices][:, indices]
    try:
        return float(torch.linalg.eigvalsh(sub.double()).max().item())
    except RuntimeError:
        return float("nan")


def assign_random(n: int, num_shards: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    groups = [[] for _ in range(max(1, min(num_shards, n)))]
    for pos, idx in enumerate(order):
        groups[pos % len(groups)].append(idx)
    return [group for group in groups if group]


def assign_relation(relation_ids: list[str | None]) -> list[list[int]]:
    buckets: dict[str, list[int]] = {}
    for idx, relation_id in enumerate(relation_ids):
        key = str(relation_id) if relation_id is not None else "__missing_relation__"
        buckets.setdefault(key, []).append(idx)
    return list(buckets.values())


def assign_spectral(vectors: torch.Tensor, num_shards: int, seed: int) -> list[list[int]]:
    n = int(vectors.shape[0])
    if n == 0:
        return []
    k = max(1, min(int(num_shards), n))
    normalized = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    similarity = normalized @ normalized.T
    centrality = similarity.abs().sum(dim=1)
    order = [int(idx) for idx in torch.argsort(centrality, descending=True).tolist()]
    groups = [[idx] for idx in order[:k]]
    old_lmax = [group_lambda_max(similarity, group) for group in groups]
    rng = random.Random(seed)
    remaining = order[k:]
    # Break ties deterministically but avoid always favoring early relation IDs.
    remaining = sorted(remaining, key=lambda idx: (float(centrality[idx].item()), rng.random()), reverse=True)
    target_size = int(math.ceil(n / k))
    for idx in remaining:
        best_group = 0
        best_cost = float("inf")
        for group_idx, group in enumerate(groups):
            candidate = group + [idx]
            new_lmax = group_lambda_max(similarity, candidate)
            balance_penalty = max(0, len(candidate) - target_size) * 0.05
            cost = (new_lmax - old_lmax[group_idx]) + balance_penalty
            if cost < best_cost:
                best_cost = cost
                best_group = group_idx
        groups[best_group].append(idx)
        old_lmax[best_group] = group_lambda_max(similarity, groups[best_group])
    return [group for group in groups if group]


def make_shards(
    strategy: str,
    vectors: torch.Tensor,
    relation_ids: list[str | None],
    *,
    num_shards: int,
    seed: int,
) -> list[list[int]]:
    strategy = strategy.strip().lower()
    n = int(vectors.shape[0])
    if strategy == "all":
        return [list(range(n))]
    if strategy == "relation":
        return assign_relation(relation_ids)
    if strategy == "random":
        return assign_random(n, num_shards=num_shards, seed=seed)
    if strategy in {"spectral", "spectral_greedy"}:
        return assign_spectral(vectors, num_shards=num_shards, seed=seed)
    raise ValueError(f"Unsupported shard strategy: {strategy!r}")


def reconstruct_shard(matrix: torch.Tensor, rank: int, *, center: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    matrix = matrix.detach().float().cpu()
    rows, dim = int(matrix.shape[0]), int(matrix.shape[1])
    if rows == 0:
        return matrix, {"rank_eff": 0, "tail_energy_ratio": None, "singular_values": []}
    mean = matrix.mean(dim=0, keepdim=True) if center else torch.zeros(1, dim, dtype=matrix.dtype)
    residual = matrix - mean
    total_energy = float((residual * residual).sum().item())
    max_rank = max(0, min(int(rank), rows - 1 if center else rows))
    if max_rank <= 0 or total_energy <= 1.0e-12:
        recon = mean.expand_as(matrix).clone()
        tail = 0.0 if total_energy <= 1.0e-12 else 1.0
        return recon, {"rank_eff": 0, "tail_energy_ratio": tail, "singular_values": []}

    gram = residual @ residual.T
    eigvals, eigvecs = torch.linalg.eigh(gram.double())
    order = torch.argsort(eigvals, descending=True)
    keep = []
    singular_values = []
    for eig_idx in order.tolist():
        value = float(eigvals[eig_idx].item())
        if value <= 1.0e-10:
            continue
        keep.append(int(eig_idx))
        singular_values.append(value ** 0.5)
        if len(keep) >= max_rank:
            break
    if not keep:
        recon = mean.expand_as(matrix).clone()
        return recon, {"rank_eff": 0, "tail_energy_ratio": 1.0, "singular_values": []}
    basis_left = eigvecs[:, keep].float()
    residual_hat = basis_left @ (basis_left.T @ residual)
    recon = mean + residual_hat
    tail_energy = float(((residual - residual_hat) ** 2).sum().item())
    tail_ratio = None if total_energy <= 1.0e-12 else float(tail_energy / total_energy)
    return recon, {
        "rank_eff": len(keep),
        "tail_energy_ratio": tail_ratio,
        "singular_values": singular_values,
    }


def reconstruct_vectors(
    vectors: torch.Tensor,
    groups: list[list[int]],
    rank: int,
    *,
    center: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    recon = torch.zeros_like(vectors)
    shard_rows = []
    normalized = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    similarity = normalized @ normalized.T
    for shard_idx, indices in enumerate(groups):
        shard_matrix = vectors[indices]
        shard_recon, stats = reconstruct_shard(shard_matrix, rank, center=center)
        recon[indices] = shard_recon
        lmax = group_lambda_max(similarity, indices)
        shard_rows.append(
            {
                "shard_id": shard_idx,
                "size": len(indices),
                "indices": indices,
                "lambda_max": lmax,
                "rank_requested": int(rank),
                **stats,
            }
        )
    return recon, shard_rows


def compression_stats(
    *,
    num_vectors: int,
    vector_dim: int,
    groups: list[list[int]],
    shard_rows: list[dict[str, Any]],
    center: bool,
) -> dict[str, Any]:
    exact_params = int(num_vectors * vector_dim)
    compressed = 0
    coeffs = 0
    for group, row in zip(groups, shard_rows):
        rank_eff = int(row.get("rank_eff") or 0)
        compressed += (rank_eff + (1 if center else 0)) * vector_dim
        coeffs += len(group) * rank_eff
    total = int(compressed + coeffs)
    return {
        "exact_params": exact_params,
        "basis_params": int(compressed),
        "coefficient_params": int(coeffs),
        "compressed_params": total,
        "compression_ratio": None if total <= 0 else float(exact_params / total),
    }


def capture_adapter_bank(controller, edit_ids: list[str]) -> tuple[dict[str, dict[str, torch.Tensor]], list[tuple[str, tuple[int, ...], int]], torch.Tensor]:
    exact_weights = {edit_id: controller._capture_adapter_parameters(edit_id) for edit_id in edit_ids}
    if not exact_weights:
        raise RuntimeError("No adapter weights captured")
    first = exact_weights[edit_ids[0]]
    schema = tensor_vector_schema(first)
    vectors = torch.stack([flatten_weights(exact_weights[edit_id], schema) for edit_id in edit_ids], dim=0)
    return exact_weights, schema, vectors


def load_vectors_into_adapters(
    controller,
    edit_ids: list[str],
    vectors: torch.Tensor,
    schema: list[tuple[str, tuple[int, ...], int]],
    reference_weights: dict[str, dict[str, torch.Tensor]],
) -> None:
    for row_idx, edit_id in enumerate(edit_ids):
        weights = unflatten_weights(vectors[row_idx], schema, reference_weights[edit_id])
        controller._load_adapter_parameters(edit_id, weights)


def summary_subset(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "post_rewrite_mean",
        "post_rephrase_mean",
        "post_locality_mean",
        "early_late_gap",
        "memory_entries_final",
    ]
    return {key: summary.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT")
    parser.add_argument("--hparams_dir", default=str(REPO_ROOT / "hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware-staged.yaml"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--data_type", default="CounterFact", choices=["CounterFact", "ZsRE", "WikiBigEdit"])
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs/shared_basis_compression_poc"))
    parser.add_argument("--ds_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", default="all:4,all:8,relation:2,random:4,spectral:4")
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--eval_rephrase_source",
        choices=["heldout", "address"],
        default="heldout",
        help=(
            "Which CounterFact paraphrase view to score as rephrase. "
            "'heldout' uses the second paraphrase when available; 'address' "
            "matches the older v1 protocol by scoring the stored support view."
        ),
    )
    parser.add_argument("--evaluation_mode", default="teacher_forcing")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--skip_exact_eval", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method = method_name(args.editing_method)
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    hparams.sequential_edit = True
    hparams.eval_batch_size = int(args.eval_batch_size)
    if hasattr(hparams, "route_log_dir"):
        hparams.route_log_dir = str(output_dir / "route_logs")
    configure_evaluation_mode(hparams, args.evaluation_mode, args.api_key)

    records, dataset_file = load_normalized_records(
        args.data_dir,
        args.data_type,
        args.ds_size,
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

    backbone = backbone_slug(hparams.model_name)
    run_base = {
        "editing_method": method,
        "alg_name": hparams.alg_name,
        "model_name": hparams.model_name,
        "backbone": backbone,
        "data_type": args.data_type,
        "dataset_file": str(Path(dataset_file).resolve()),
        "stream_type": "shared_basis_compression_poc",
        "eval_rephrase_source": args.eval_rephrase_source,
        "seed": args.seed,
        "sequential_edit": True,
        "stream_length": len(records),
        "requested_ds_size": args.ds_size,
        "hparams_path": str(Path(args.hparams_dir).resolve()),
    }
    write_json(output_dir / "run_config.json", {**run_base, "conditions": args.conditions})

    editor = BaseEditor.from_hparams(hparams)
    pre_start = time.time()
    pre_metrics = compute_pre_metrics(editor, requests, editor_inputs["eval_metric"])
    write_json(output_dir / "pre_eval_time.json", {"seconds": time.time() - pre_start})

    train_start = time.time()
    for request in requests:
        apply_single_edit(editor, request)
    train_seconds = time.time() - train_start
    controller = editor.model if hasattr(editor.model, "memory_entries") else None
    if controller is None:
        raise RuntimeError("Expected HopEditController after applying edits")

    entries = list(controller.memory_entries)
    edit_ids = [str(entry.get("edit_id")) for entry in entries]
    relation_ids = [entry.get("relation_id") or requests[idx].get("relation_id") for idx, entry in enumerate(entries)]
    exact_weights, schema, vectors = capture_adapter_bank(controller, edit_ids)
    vector_dim = int(vectors.shape[1])
    write_json(
        output_dir / "adapter_bank_summary.json",
        {
            "num_edits": len(edit_ids),
            "vector_dim": vector_dim,
            "schema": [{"name": name, "shape": shape, "numel": numel} for name, shape, numel in schema],
            "train_seconds": train_seconds,
            "relation_cluster_count": len({str(item) for item in relation_ids}),
        },
    )

    condition_results = []
    if not args.skip_exact_eval:
        exact_dir = output_dir / "condition_exact"
        exact_summary = evaluate_and_write(
            editor,
            hparams,
            method,
            backbone,
            exact_dir,
            records,
            requests,
            editor_inputs["eval_metric"],
            {
                **run_base,
                "run_name": f"shared_basis_exact_{args.data_type.lower()}_{args.ds_size}_seed{args.seed}",
                "condition": "exact",
                "output_dir": str(exact_dir.resolve()),
                "wall_time_seconds": train_seconds,
            },
            pre_metrics=pre_metrics,
        )
        condition_results.append({"condition": "exact", "summary": summary_subset(exact_summary)})

    for strategy, rank in parse_conditions(args.conditions):
        groups = make_shards(strategy, vectors, relation_ids, num_shards=args.num_shards, seed=args.seed)
        recon_vectors, shard_rows = reconstruct_vectors(vectors, groups, rank, center=bool(args.center))
        load_vectors_into_adapters(controller, edit_ids, recon_vectors, schema, exact_weights)
        stats = compression_stats(
            num_vectors=int(vectors.shape[0]),
            vector_dim=vector_dim,
            groups=groups,
            shard_rows=shard_rows,
            center=bool(args.center),
        )
        condition_name = f"{strategy}_rank{rank}"
        condition_dir = output_dir / f"condition_{condition_name}"
        write_json(
            condition_dir / "compression_metadata.json",
            {
                "condition": condition_name,
                "strategy": strategy,
                "rank": int(rank),
                "center": bool(args.center),
                "num_shards": len(groups),
                "shards": shard_rows,
                **stats,
            },
        )
        summary = evaluate_and_write(
            editor,
            hparams,
            method,
            backbone,
            condition_dir,
            records,
            requests,
            editor_inputs["eval_metric"],
            {
                **run_base,
                "run_name": f"shared_basis_{condition_name}_{args.data_type.lower()}_{args.ds_size}_seed{args.seed}",
                "condition": condition_name,
                "output_dir": str(condition_dir.resolve()),
                "wall_time_seconds": train_seconds,
                **stats,
            },
            pre_metrics=pre_metrics,
        )
        condition_results.append(
            {
                "condition": condition_name,
                "strategy": strategy,
                "rank": int(rank),
                "summary": summary_subset(summary),
                "compression": stats,
                "num_shards": len(groups),
                "lambda_max_mean": None
                if not shard_rows
                else float(sum(float(row.get("lambda_max") or 0.0) for row in shard_rows) / len(shard_rows)),
                "tail_energy_ratio_mean": None
                if not shard_rows
                else float(
                    sum(float(row.get("tail_energy_ratio") or 0.0) for row in shard_rows)
                    / len(shard_rows)
                ),
            }
        )

    load_vectors_into_adapters(controller, edit_ids, vectors, schema, exact_weights)
    write_json(
        output_dir / "shared_basis_compression_summary.json",
        {
            "method": "shared_basis_compression_poc",
            "dataset": args.data_type,
            "dataset_file": str(Path(dataset_file).resolve()),
            "ds_size": int(args.ds_size),
            "seed": int(args.seed),
            "conditions": condition_results,
        },
    )
    print(json.dumps(condition_results, indent=2))
    print(f"Summary written to {output_dir / 'shared_basis_compression_summary.json'}")


if __name__ == "__main__":
    main()
