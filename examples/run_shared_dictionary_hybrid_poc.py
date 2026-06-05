"""Hybrid shared-dictionary POC for exact + retired edit coexistence.

This builds on the stronger adapter-bank compression path rather than the
runtime ``shared_basis_codes`` realization. The workflow is:

    exact per-edit adapters
    -> vectorize adapter bank
    -> fit shard-level low-rank shared dictionary
    -> keep the worst reconstructed edits exact
    -> retire the rest to the shared dictionary
    -> evaluate the mixed bank

The goal is to answer the first load-bearing question for the project:
can a shared adapter dictionary be strong enough to coexist with exact edits?
"""

from __future__ import annotations

import argparse
import json
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
    write_jsonl,
)
from examples.run_edit_experiment import seed_everything
from examples.run_shared_basis_compression_poc import (
    capture_adapter_bank,
    load_vectors_into_adapters,
    make_shards,
    reconstruct_vectors,
)
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
        raise ValueError("At least one dictionary condition is required")
    return conditions


def parse_keep_counts(raw: str, num_vectors: int) -> list[int]:
    values = set()
    for token in str(raw or "").split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "all":
            values.add(int(num_vectors))
            continue
        value = int(token)
        values.add(max(0, min(value, int(num_vectors))))
    if 0 not in values:
        values.add(0)
    return sorted(values)


def choose_keep_indices(residual_norms: torch.Tensor, keep_count: int) -> list[int]:
    keep_count = max(0, min(int(keep_count), int(residual_norms.numel())))
    if keep_count <= 0:
        return []
    order = torch.argsort(residual_norms, descending=True)
    return [int(idx) for idx in order[:keep_count].tolist()]


def hybrid_compression_stats(
    *,
    num_vectors: int,
    vector_dim: int,
    groups: list[list[int]],
    shard_rows: list[dict[str, Any]],
    keep_indices: list[int],
    center: bool,
) -> dict[str, Any]:
    keep_set = set(int(idx) for idx in keep_indices)
    exact_params = int(num_vectors * vector_dim)
    basis_params = 0
    coefficient_params = 0
    retired_count = 0
    active_shards = 0
    for group, row in zip(groups, shard_rows):
        retired_in_group = [idx for idx in group if idx not in keep_set]
        if not retired_in_group:
            continue
        active_shards += 1
        retired_count += len(retired_in_group)
        rank_eff = int(row.get("rank_eff") or 0)
        basis_params += (rank_eff + (1 if center else 0)) * vector_dim
        coefficient_params += len(retired_in_group) * rank_eff
    exact_override_params = int(len(keep_set) * vector_dim)
    total = int(basis_params + coefficient_params + exact_override_params)
    return {
        "exact_params": exact_params,
        "basis_params": int(basis_params),
        "coefficient_params": int(coefficient_params),
        "exact_override_params": exact_override_params,
        "hybrid_params": total,
        "hybrid_compression_ratio": None if total <= 0 else float(exact_params / total),
        "kept_exact_count": int(len(keep_set)),
        "kept_exact_fraction": float(len(keep_set) / num_vectors) if num_vectors else 0.0,
        "retired_count": int(retired_count),
        "retired_fraction": float(retired_count / num_vectors) if num_vectors else 0.0,
        "active_shared_shards": int(active_shards),
    }


def contract_pass_rate(
    summary: dict[str, Any],
    *,
    rewrite_threshold: float,
    rephrase_threshold: float,
    locality_threshold: float,
) -> float | None:
    per_case = summary.get("per_case") or []
    if not per_case:
        return None
    passes = 0
    total = 0
    for row in per_case:
        rewrite = row.get("post_rewrite_acc")
        rephrase = row.get("post_rephrase_acc")
        locality = row.get("post_locality_acc")
        if rewrite is None or rephrase is None or locality is None:
            continue
        total += 1
        if (
            float(rewrite) >= rewrite_threshold
            and float(rephrase) >= rephrase_threshold
            and float(locality) >= locality_threshold
        ):
            passes += 1
    return None if total <= 0 else float(passes / total)


def summary_subset(
    summary: dict[str, Any],
    *,
    rewrite_threshold: float,
    rephrase_threshold: float,
    locality_threshold: float,
) -> dict[str, Any]:
    return {
        "post_rewrite_mean": summary.get("post_rewrite_mean"),
        "post_rephrase_mean": summary.get("post_rephrase_mean"),
        "post_locality_mean": summary.get("post_locality_mean"),
        "early_late_gap": summary.get("early_late_gap"),
        "memory_entries_final": summary.get("memory_entries_final"),
        "contract_pass_rate": contract_pass_rate(
            summary,
            rewrite_threshold=rewrite_threshold,
            rephrase_threshold=rephrase_threshold,
            locality_threshold=locality_threshold,
        ),
    }


def top_residual_rows(
    edit_ids: list[str],
    requests: list[dict[str, Any]],
    residual_norms: torch.Tensor,
    keep_indices: list[int],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    keep_set = set(keep_indices)
    order = torch.argsort(residual_norms, descending=True)
    rows = []
    for idx in order[: min(top_k, int(order.numel()))].tolist():
        idx = int(idx)
        request = requests[idx]
        rows.append(
            {
                "row_index": idx,
                "edit_id": edit_ids[idx],
                "kept_exact": idx in keep_set,
                "residual_norm": float(residual_norms[idx].item()),
                "relation_id": request.get("relation_id"),
                "subject": request.get("subject"),
                "prompt": request.get("prompt"),
                "target_new": request.get("target_new"),
            }
        )
    return rows


def per_case_by_id(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in summary.get("per_case") or []:
        case_id = row.get("case_id")
        if case_id is None:
            continue
        rows[int(case_id)] = row
    return rows


def per_case_utility(row: dict[str, Any]) -> float | None:
    rewrite = row.get("post_rewrite_acc")
    rephrase = row.get("post_rephrase_acc")
    locality = row.get("post_locality_acc")
    if rewrite is None or rephrase is None or locality is None:
        return None
    return float((float(rewrite) + float(rephrase) + float(locality)) / 3.0)


def build_compressibility_rows(
    *,
    edit_ids: list[str],
    requests: list[dict[str, Any]],
    exact_summary: dict[str, Any],
    dict_summary: dict[str, Any],
    residual_norms: torch.Tensor,
    residual_energy: torch.Tensor,
    strategy: str,
    rank: int,
) -> list[dict[str, Any]]:
    exact_rows = per_case_by_id(exact_summary)
    dict_rows = per_case_by_id(dict_summary)
    rows = []
    for idx, (edit_id, request) in enumerate(zip(edit_ids, requests)):
        exact_row = exact_rows.get(idx) or {}
        dict_row = dict_rows.get(idx) or {}
        exact_u = per_case_utility(exact_row)
        dict_u = per_case_utility(dict_row)
        rows.append(
            {
                "row_index": idx,
                "edit_id": edit_id,
                "strategy": strategy,
                "rank": int(rank),
                "relation_id": request.get("relation_id"),
                "subject": request.get("subject"),
                "prompt": request.get("prompt"),
                "target_new": request.get("target_new"),
                "residual_norm": float(residual_norms[idx].item()),
                "residual_energy": float(residual_energy[idx].item()),
                "exact_rewrite": exact_row.get("post_rewrite_acc"),
                "exact_rephrase": exact_row.get("post_rephrase_acc"),
                "exact_locality": exact_row.get("post_locality_acc"),
                "dict_rewrite": dict_row.get("post_rewrite_acc"),
                "dict_rephrase": dict_row.get("post_rephrase_acc"),
                "dict_locality": dict_row.get("post_locality_acc"),
                "exact_utility": exact_u,
                "dict_utility": dict_u,
                "utility_gap": None if exact_u is None or dict_u is None else float(exact_u - dict_u),
            }
        )
    return rows


def write_markdown_report(
    path: Path,
    *,
    dataset: str,
    ds_size: int,
    seed: int,
    exact_summary: dict[str, Any] | None,
    condition_results: list[dict[str, Any]],
    best_by_contract: dict[str, Any] | None,
) -> None:
    lines = [
        f"# Hybrid Shared Dictionary POC",
        "",
        f"- Dataset: `{dataset}`",
        f"- Stream size: `{ds_size}`",
        f"- Seed: `{seed}`",
        "",
    ]
    if exact_summary is not None:
        lines.extend(
            [
                "## Exact Baseline",
                "",
                f"- rewrite: `{exact_summary.get('post_rewrite_mean')}`",
                f"- rephrase: `{exact_summary.get('post_rephrase_mean')}`",
                f"- locality: `{exact_summary.get('post_locality_mean')}`",
                f"- contract: `{exact_summary.get('contract_pass_rate')}`",
                "",
            ]
        )
    if best_by_contract is not None:
        best_summary = best_by_contract.get("summary") or {}
        best_storage = best_by_contract.get("storage") or {}
        lines.extend(
            [
                "## Best Hybrid By Contract",
                "",
                f"- condition: `{best_by_contract.get('condition')}`",
                f"- rewrite: `{best_summary.get('post_rewrite_mean')}`",
                f"- rephrase: `{best_summary.get('post_rephrase_mean')}`",
                f"- locality: `{best_summary.get('post_locality_mean')}`",
                f"- contract: `{best_summary.get('contract_pass_rate')}`",
                f"- kept exact: `{best_storage.get('kept_exact_count')}`",
                f"- compression ratio: `{best_storage.get('hybrid_compression_ratio')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Hybrid Conditions",
            "",
            "| condition | rewrite | rephrase | locality | contract | keep_exact | ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in condition_results:
        summary = row.get("summary") or {}
        storage = row.get("storage") or {}
        lines.append(
            "| {condition} | {rewrite:.4f} | {rephrase:.4f} | {locality:.4f} | {contract:.4f} | {keep_exact} | {ratio:.4f} |".format(
                condition=row.get("condition"),
                rewrite=float(summary.get("post_rewrite_mean") or 0.0),
                rephrase=float(summary.get("post_rephrase_mean") or 0.0),
                locality=float(summary.get("post_locality_mean") or 0.0),
                contract=float(summary.get("contract_pass_rate") or 0.0),
                keep_exact=int(storage.get("kept_exact_count") or 0),
                ratio=float(storage.get("hybrid_compression_ratio") or 0.0),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT")
    parser.add_argument("--hparams_dir", default=str(REPO_ROOT / "hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware-staged.yaml"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--data_type", default="CounterFact", choices=["CounterFact", "ZsRE", "WikiBigEdit"])
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs/shared_dictionary_hybrid_poc"))
    parser.add_argument("--ds_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", default="relation:2,random:4,spectral:4")
    parser.add_argument("--keep_exact_counts", default="0,4,8,16,all")
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--eval_rephrase_source",
        choices=["heldout", "address"],
        default="address",
        help=(
            "Which CounterFact paraphrase view to score as rephrase. "
            "'address' matches the older v1 protocol and is the default for this POC."
        ),
    )
    parser.add_argument("--evaluation_mode", default="teacher_forcing")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--skip_exact_eval", action="store_true")
    parser.add_argument("--contract_rewrite_threshold", type=float, default=0.8)
    parser.add_argument("--contract_rephrase_threshold", type=float, default=0.8)
    parser.add_argument("--contract_locality_threshold", type=float, default=0.95)
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
        "stream_type": "shared_dictionary_hybrid_poc",
        "eval_rephrase_source": args.eval_rephrase_source,
        "seed": args.seed,
        "sequential_edit": True,
        "stream_length": len(records),
        "requested_ds_size": args.ds_size,
        "hparams_path": str(Path(args.hparams_dir).resolve()),
    }
    write_json(
        output_dir / "run_config.json",
        {
            **run_base,
            "conditions": args.conditions,
            "keep_exact_counts": args.keep_exact_counts,
            "contract_thresholds": {
                "rewrite": args.contract_rewrite_threshold,
                "rephrase": args.contract_rephrase_threshold,
                "locality": args.contract_locality_threshold,
            },
        },
    )

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

    keep_counts = parse_keep_counts(args.keep_exact_counts, int(vectors.shape[0]))
    condition_results = []
    exact_summary_subset = None
    exact_summary_full = None
    dict_only_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
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
                "run_name": f"shared_dictionary_exact_{args.data_type.lower()}_{args.ds_size}_seed{args.seed}",
                "condition": "exact",
                "output_dir": str(exact_dir.resolve()),
                "wall_time_seconds": train_seconds,
            },
            pre_metrics=pre_metrics,
        )
        exact_summary_full = exact_summary
        exact_summary_subset = summary_subset(
            exact_summary,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
        )
        condition_results.append({"condition": "exact", "summary": exact_summary_subset})

    for strategy, rank in parse_conditions(args.conditions):
        groups = make_shards(strategy, vectors, relation_ids, num_shards=args.num_shards, seed=args.seed)
        recon_vectors, shard_rows = reconstruct_vectors(vectors, groups, rank, center=bool(args.center))
        residual_vectors = vectors - recon_vectors
        residual_norms = torch.linalg.norm(residual_vectors, dim=1)
        residual_energy = residual_vectors.pow(2).sum(dim=1)
        base_condition = f"{strategy}_rank{rank}"
        for keep_count in keep_counts:
            keep_indices = choose_keep_indices(residual_norms, keep_count)
            mixed_vectors = recon_vectors.clone()
            if keep_indices:
                mixed_vectors[keep_indices] = vectors[keep_indices]
            load_vectors_into_adapters(controller, edit_ids, mixed_vectors, schema, exact_weights)
            stats = hybrid_compression_stats(
                num_vectors=int(vectors.shape[0]),
                vector_dim=vector_dim,
                groups=groups,
                shard_rows=shard_rows,
                keep_indices=keep_indices,
                center=bool(args.center),
            )
            condition_name = f"{base_condition}_keep{keep_count}"
            condition_dir = output_dir / f"condition_{condition_name}"
            write_json(
                condition_dir / "compression_metadata.json",
                {
                    "condition": condition_name,
                    "strategy": strategy,
                    "rank": int(rank),
                    "center": bool(args.center),
                    "keep_exact_count": int(keep_count),
                    "num_shards": len(groups),
                    "shards": shard_rows,
                    "residual_norm_mean": float(residual_norms.mean().item()),
                    "residual_norm_max": float(residual_norms.max().item()),
                    "top_residual_rows": top_residual_rows(
                        edit_ids,
                        requests,
                        residual_norms,
                        keep_indices,
                    ),
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
                    "run_name": f"shared_dictionary_{condition_name}_{args.data_type.lower()}_{args.ds_size}_seed{args.seed}",
                    "condition": condition_name,
                    "output_dir": str(condition_dir.resolve()),
                    "wall_time_seconds": train_seconds,
                    **stats,
                },
                pre_metrics=pre_metrics,
            )
            if keep_count == 0 and exact_summary_full is not None:
                dict_only_rows[(strategy, int(rank))] = build_compressibility_rows(
                    edit_ids=edit_ids,
                    requests=requests,
                    exact_summary=exact_summary_full,
                    dict_summary=summary,
                    residual_norms=residual_norms,
                    residual_energy=residual_energy,
                    strategy=strategy,
                    rank=int(rank),
                )
                write_jsonl(condition_dir / "compressibility_rows.jsonl", dict_only_rows[(strategy, int(rank))])
            condition_results.append(
                {
                    "condition": condition_name,
                    "strategy": strategy,
                    "rank": int(rank),
                    "keep_exact_count": int(keep_count),
                    "summary": summary_subset(
                        summary,
                        rewrite_threshold=args.contract_rewrite_threshold,
                        rephrase_threshold=args.contract_rephrase_threshold,
                        locality_threshold=args.contract_locality_threshold,
                    ),
                    "storage": stats,
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
                    "residual_norm_mean": float(residual_norms.mean().item()),
                    "residual_norm_max": float(residual_norms.max().item()),
                }
            )

    load_vectors_into_adapters(controller, edit_ids, vectors, schema, exact_weights)
    best_by_contract = None
    for row in condition_results:
        if row.get("condition") == "exact":
            continue
        summary = row.get("summary") or {}
        score = summary.get("contract_pass_rate")
        if score is None:
            continue
        if best_by_contract is None or float(score) > float((best_by_contract.get("summary") or {}).get("contract_pass_rate") or -1.0):
            best_by_contract = row

    summary_path = output_dir / "shared_dictionary_hybrid_summary.json"
    write_json(
        summary_path,
        {
            "method": "shared_dictionary_hybrid_poc",
            "dataset": args.data_type,
            "dataset_file": str(Path(dataset_file).resolve()),
            "ds_size": int(args.ds_size),
            "seed": int(args.seed),
            "best_by_contract": None
            if best_by_contract is None
            else {
                "condition": best_by_contract.get("condition"),
                "contract_pass_rate": (best_by_contract.get("summary") or {}).get("contract_pass_rate"),
                "kept_exact_count": (best_by_contract.get("storage") or {}).get("kept_exact_count"),
                "hybrid_compression_ratio": (best_by_contract.get("storage") or {}).get("hybrid_compression_ratio"),
            },
            "conditions": condition_results,
        },
    )
    write_markdown_report(
        output_dir / "report.md",
        dataset=args.data_type,
        ds_size=int(args.ds_size),
        seed=int(args.seed),
        exact_summary=exact_summary_subset,
        condition_results=condition_results,
        best_by_contract=best_by_contract,
    )
    print(json.dumps(condition_results, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
