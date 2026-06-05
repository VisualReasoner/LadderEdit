import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.analyze_hopedit_orthogonalized_factors import apply_relation_whitener, build_relation_whitener
from examples.edit_experiment_utils import load_normalized_records, resolve_hparams_class, write_json


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in re.split(r"[:,\s]+", str(value)) if part.strip()]


def safe_mean(values: list[float | None]) -> float | None:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def percentile_summary(values: list[float | int | None]) -> dict[str, float | None]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {key: None for key in ("p0", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100")}
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.percentile(array, percentile))
        for name, percentile in (
            ("p0", 0),
            ("p10", 10),
            ("p25", 25),
            ("p50", 50),
            ("p75", 75),
            ("p90", 90),
            ("p95", 95),
            ("p99", 99),
            ("p100", 100),
        )
    }


def extract_factor_rows(
    controller: Any,
    prompts: list[str],
    subjects: list[str | None],
    objects: list[str | None],
    *,
    subject_layer: int,
    relation_layer: int,
    subject_pooling: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        stop = start + batch_size
        rows.extend(
            controller._extract_batched_factored_address_keys(
                prompts[start:stop],
                subjects[start:stop],
                objects[start:stop],
                subject_layer_override=subject_layer,
                relation_layer_override=relation_layer,
                subject_pooling_override=subject_pooling,
            )
        )
    return rows


def stack_factor(rows: list[dict[str, Any]], key: str) -> torch.Tensor | None:
    vectors = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, torch.Tensor):
            return None
        vectors.append(value.detach().float().cpu())
    if not vectors:
        return None
    return torch.stack(vectors, dim=0)


def rank_of_target(scores: torch.Tensor, target_idx: int) -> int:
    order = torch.argsort(scores, descending=True)
    matches = torch.nonzero(order == int(target_idx), as_tuple=False).flatten()
    if matches.numel() == 0:
        return int(scores.numel())
    return int(matches[0].item()) + 1


def analyze_scores(
    subject_scores: torch.Tensor,
    relation_scores: torch.Tensor,
    relation_ids: list[Any],
) -> dict[str, Any]:
    n = int(subject_scores.shape[0])
    subject_top = torch.argmax(subject_scores, dim=1).cpu().tolist()
    relation_top = torch.argmax(relation_scores, dim=1).cpu().tolist()
    subject_ranks = []
    relation_ranks = []
    subject_gaps = []
    relation_gaps = []
    q_s = []
    q_r = []
    target_fire = []
    wrong_fire = []
    abstain = []
    rank_independent = []
    same_relation_wrong = []
    cross_relation_wrong = []
    same_relation_baseline = []
    cross_relation_baseline = []

    for idx in range(n):
        subject_row = subject_scores[idx]
        relation_row = relation_scores[idx]
        subject_ranks.append(rank_of_target(subject_row, idx))
        relation_ranks.append(rank_of_target(relation_row, idx))

        if n > 1:
            subject_impostor_top = torch.max(torch.cat([subject_row[:idx], subject_row[idx + 1 :]])).item()
            relation_impostor_top = torch.max(torch.cat([relation_row[:idx], relation_row[idx + 1 :]])).item()
            subject_gaps.append(float(subject_row[idx].item() - subject_impostor_top))
            relation_gaps.append(float(relation_row[idx].item() - relation_impostor_top))
        else:
            subject_gaps.append(None)
            relation_gaps.append(None)

        s_top = int(subject_top[idx])
        r_top = int(relation_top[idx])
        subject_ok = s_top == idx
        relation_ok = r_top == idx
        both_ok = subject_ok and relation_ok
        wrong = s_top == r_top and s_top != idx
        q_s.append(1.0 if subject_ok else 0.0)
        q_r.append(1.0 if relation_ok else 0.0)
        target_fire.append(1.0 if both_ok else 0.0)
        wrong_fire.append(1.0 if wrong else 0.0)
        abstain.append(0.0 if both_ok or wrong else 1.0)

        subject_wrong = 0.0 if subject_ok else 1.0
        relation_wrong = 0.0 if relation_ok else 1.0
        denom = max(1, n - 1)
        rank_independent.append(float(subject_wrong * relation_wrong / denom))

        target_relation_id = relation_ids[idx] if idx < len(relation_ids) else None
        same_count = 0
        for other_idx, relation_id in enumerate(relation_ids):
            if other_idx != idx and relation_id == target_relation_id:
                same_count += 1
        cross_count = max(0, denom - same_count)
        same_relation_baseline.append(float(subject_wrong * relation_wrong * same_count / max(1, denom * denom)))
        cross_relation_baseline.append(float(subject_wrong * relation_wrong * cross_count / max(1, denom * denom)))
        same_relation_wrong.append(1.0 if wrong and relation_ids[s_top] == target_relation_id else 0.0)
        cross_relation_wrong.append(1.0 if wrong and relation_ids[s_top] != target_relation_id else 0.0)

    rank_independent_mean = safe_mean(rank_independent)
    wrong_mean = safe_mean(wrong_fire)
    same_wrong_mean = safe_mean(same_relation_wrong)
    cross_wrong_mean = safe_mean(cross_relation_wrong)
    same_baseline_mean = safe_mean(same_relation_baseline)
    cross_baseline_mean = safe_mean(cross_relation_baseline)
    return {
        "n": n,
        "q_s": safe_mean(q_s),
        "q_r": safe_mean(q_r),
        "target_fire": safe_mean(target_fire),
        "wrong_fire_kappa": wrong_mean,
        "abstain": safe_mean(abstain),
        "rank_independent_kappa": rank_independent_mean,
        "rank_collision_ratio": None
        if rank_independent_mean is None or rank_independent_mean <= 0.0 or wrong_mean is None
        else float(wrong_mean / rank_independent_mean),
        "same_relation_kappa": same_wrong_mean,
        "cross_relation_kappa": cross_wrong_mean,
        "same_relation_kappa_ind": same_baseline_mean,
        "cross_relation_kappa_ind": cross_baseline_mean,
        "same_relation_ratio": None
        if same_baseline_mean is None or same_baseline_mean <= 0.0 or same_wrong_mean is None
        else float(same_wrong_mean / same_baseline_mean),
        "cross_relation_ratio": None
        if cross_baseline_mean is None or cross_baseline_mean <= 0.0 or cross_wrong_mean is None
        else float(cross_wrong_mean / cross_baseline_mean),
        "subject_rank": percentile_summary(subject_ranks),
        "relation_rank": percentile_summary(relation_ranks),
        "subject_target_gap": percentile_summary(subject_gaps),
        "relation_target_gap": percentile_summary(relation_gaps),
        "relation_rank_le_1": float(np.mean(np.asarray(relation_ranks) <= 1)),
        "relation_rank_le_5": float(np.mean(np.asarray(relation_ranks) <= 5)),
        "relation_rank_le_10": float(np.mean(np.asarray(relation_ranks) <= 10)),
        "relation_rank_le_32": float(np.mean(np.asarray(relation_ranks) <= 32)),
        "relation_rank_le_100": float(np.mean(np.asarray(relation_ranks) <= 100)),
    }


def evaluate_layer(
    controller: Any,
    records: list[dict[str, Any]],
    *,
    subject_layer: int,
    relation_layer: int,
    subject_pooling: str,
    batch_size: int,
    whiten_eps: float,
) -> dict[str, Any]:
    prompts = [str(record.get("prompt") or "") for record in records]
    rephrases = [str(record.get("rephrase_prompt") or record.get("prompt") or "") for record in records]
    subjects = [record.get("subject") for record in records]
    objects = [record.get("target_new") for record in records]
    relation_ids = [record.get("relation_id") for record in records]
    query_objects = [None for _ in records]

    trace_rows = extract_factor_rows(
        controller,
        prompts,
        subjects,
        objects,
        subject_layer=subject_layer,
        relation_layer=relation_layer,
        subject_pooling=subject_pooling,
        batch_size=batch_size,
    )
    rephrase_rows = extract_factor_rows(
        controller,
        rephrases,
        subjects,
        query_objects,
        subject_layer=subject_layer,
        relation_layer=relation_layer,
        subject_pooling=subject_pooling,
        batch_size=batch_size,
    )
    rewrite_rows = extract_factor_rows(
        controller,
        prompts,
        subjects,
        query_objects,
        subject_layer=subject_layer,
        relation_layer=relation_layer,
        subject_pooling=subject_pooling,
        batch_size=batch_size,
    )

    trace_subject = stack_factor(trace_rows, "subject_factor")
    trace_relation = stack_factor(trace_rows, "relation_factor")
    if trace_subject is None or trace_relation is None:
        raise RuntimeError(f"Missing trace factors for relation layer {relation_layer}.")

    result: dict[str, Any] = {
        "relation_layer": int(relation_layer),
        "subject_layer": int(subject_layer),
        "subject_pooling": subject_pooling,
        "relation_agreement_mean": safe_mean(
            [
                None
                if not isinstance(trace.get("relation_factor"), torch.Tensor)
                or not isinstance(rephrase.get("relation_factor"), torch.Tensor)
                else float(trace["relation_factor"].dot(rephrase["relation_factor"]).item())
                for trace, rephrase in zip(trace_rows, rephrase_rows)
            ]
        ),
        "subject_found_rate": safe_mean([1.0 if row.get("subject_found") else 0.0 for row in rephrase_rows]),
        "relation_token_count_mean": safe_mean([row.get("relation_token_count") for row in rephrase_rows]),
    }

    trace_relation_by_id = {f"hopedit_{idx:05d}": trace_relation[idx] for idx in range(int(trace_relation.shape[0]))}
    trace_ids = [f"hopedit_{idx:05d}" for idx in range(int(trace_relation.shape[0]))]
    whitener = build_relation_whitener(trace_relation_by_id, trace_ids, eps=whiten_eps)
    trace_relation_whitened = torch.stack(
        [apply_relation_whitener(trace_relation_by_id[trace_id], whitener) for trace_id in trace_ids],
        dim=0,
    )

    for split_name, query_rows in (("rewrite", rewrite_rows), ("rephrase", rephrase_rows)):
        query_subject = stack_factor(query_rows, "subject_factor")
        query_relation = stack_factor(query_rows, "relation_factor")
        if query_subject is None or query_relation is None:
            raise RuntimeError(f"Missing {split_name} factors for relation layer {relation_layer}.")
        subject_scores = query_subject @ trace_subject.T
        relation_scores = query_relation @ trace_relation.T
        result[split_name] = analyze_scores(subject_scores, relation_scores, relation_ids)

        query_relation_whitened = torch.stack(
            [apply_relation_whitener(query_relation[idx], whitener) for idx in range(int(query_relation.shape[0]))],
            dim=0,
        )
        whitened_relation_scores = query_relation_whitened @ trace_relation_whitened.T
        result[f"{split_name}_global_whiten"] = analyze_scores(subject_scores, whitened_relation_scores, relation_ids)

    result["whitener"] = {
        "rank": whitener["rank"],
        "num_vectors": whitener["num_vectors"],
        "eps": whitener["eps"],
        "singular_min_kept": whitener["singular_min_kept"],
        "singular_max": whitener["singular_max"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT")
    parser.add_argument("--hparams_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_type", default="CounterFact")
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ds_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--relation_layers", type=str, default="6,8,10,12,14,16,18,20,24")
    parser.add_argument("--subject_layer", type=int, default=16)
    parser.add_argument("--subject_pooling", type=str, default="last")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--whiten_eps", type=float, default=1.0e-4)
    args = parser.parse_args()

    from easyeditor import BaseEditor
    from easyeditor.models.hopedit.hopedit_main import HopEditController

    seed_everything(args.seed)
    hparams_class = resolve_hparams_class(args.editing_method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    hparams.factored_relation_encoder_impl = "identity"
    hparams.factored_relation_encoder_checkpoint = None
    hparams.factored_relation_encoder_steps = 0
    editor = BaseEditor.from_hparams(hparams)
    if hasattr(editor, "model") and hasattr(editor.model, "_extract_batched_factored_address_keys"):
        controller = editor.model
    else:
        controller = HopEditController(model=editor.model, tok=editor.tok, hparams=hparams)

    records, dataset_file = load_normalized_records(
        args.data_dir,
        args.data_type,
        args.ds_size,
        data_file=args.data_file,
    )
    relation_layers = parse_layers(args.relation_layers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for relation_layer in relation_layers:
        print(f"[relation-layer-sweep] evaluating relation_layer={relation_layer}", flush=True)
        configs.append(
            evaluate_layer(
                controller,
                records,
                subject_layer=args.subject_layer,
                relation_layer=relation_layer,
                subject_pooling=args.subject_pooling,
                batch_size=args.batch_size,
                whiten_eps=args.whiten_eps,
            )
        )

    summary = {
        "run_name": f"hopedit_relation_layer_sweep_{args.data_type.lower()}_{args.ds_size}_seed{args.seed}",
        "dataset_file": str(dataset_file),
        "data_type": args.data_type,
        "ds_size": int(args.ds_size),
        "seed": int(args.seed),
        "subject_layer": int(args.subject_layer),
        "subject_pooling": args.subject_pooling,
        "relation_layers": relation_layers,
        "whiten_eps": float(args.whiten_eps),
        "configs": configs,
    }
    write_json(output_dir / "relation_layer_sweep_summary.json", summary)

    compact = []
    for row in configs:
        compact.append(
            {
                "relation_layer": row["relation_layer"],
                "relation_agreement_mean": row["relation_agreement_mean"],
                "rephrase_q_r": row["rephrase"]["q_r"],
                "rephrase_target_fire": row["rephrase"]["target_fire"],
                "rephrase_kappa": row["rephrase"]["wrong_fire_kappa"],
                "rephrase_relation_rank_p50": row["rephrase"]["relation_rank"]["p50"],
                "rephrase_relation_rank_le_100": row["rephrase"]["relation_rank_le_100"],
                "whiten_rephrase_q_r": row["rephrase_global_whiten"]["q_r"],
                "whiten_rephrase_target_fire": row["rephrase_global_whiten"]["target_fire"],
                "whiten_rephrase_kappa": row["rephrase_global_whiten"]["wrong_fire_kappa"],
                "whiten_rephrase_relation_rank_p50": row["rephrase_global_whiten"]["relation_rank"]["p50"],
                "whiten_rephrase_relation_rank_le_100": row["rephrase_global_whiten"]["relation_rank_le_100"],
                "rewrite_q_r": row["rewrite"]["q_r"],
                "whiten_rewrite_q_r": row["rewrite_global_whiten"]["q_r"],
            }
        )
    write_json(output_dir / "relation_layer_sweep_compact.json", compact)
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
