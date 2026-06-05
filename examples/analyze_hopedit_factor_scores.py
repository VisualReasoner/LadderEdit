import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_relation_ids_from_memory(run_dir: Path) -> dict[str, object]:
    path = run_dir / "memory_snapshot.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        return {}
    mapping = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        relation_id = row.get("relation_id")
        for key in ("trace_id", "edit_id", "memory_id"):
            value = row.get(key)
            if value is not None and relation_id is not None:
                mapping[str(value)] = relation_id
    return mapping


def backfill_relation_ids(rows: list[dict], relation_by_trace: dict[str, object]) -> list[dict]:
    if not relation_by_trace:
        return rows
    updated = []
    for row in rows:
        copied = dict(row)
        trace_ids = copied.get("factor_score_trace_ids")
        relation_ids = copied.get("factor_score_relation_ids")
        if isinstance(trace_ids, list) and not isinstance(relation_ids, list):
            copied["factor_score_relation_ids"] = [relation_by_trace.get(str(trace_id)) for trace_id in trace_ids]
        updated.append(copied)
    return updated


def safe_mean(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def binary_corr(xs: list[int], ys: list[int]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pass_margin(scores: list[float], index: int, threshold: float) -> tuple[bool, float | None]:
    if index < 0 or index >= len(scores):
        return False, None
    indexed = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not indexed:
        return False, None
    indexed.sort(key=lambda item: item[1], reverse=True)
    top_idx, top_score = indexed[0]
    runner_score = indexed[1][1] if len(indexed) > 1 else 0.0
    margin = float(top_score - runner_score)
    return bool(top_idx == index and margin > threshold), margin


def rank_of_index(scores: list[float], index: int) -> int | None:
    if index < 0 or index >= len(scores):
        return None
    indexed = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not indexed:
        return None
    indexed.sort(key=lambda item: item[1], reverse=True)
    for rank, (idx, _) in enumerate(indexed, start=1):
        if idx == index:
            return rank
    return None


def event_lambda(pairs: list[tuple[int, int]]) -> dict:
    if not pairs:
        return {
            "count": 0,
            "p_s": None,
            "p_r": None,
            "p_both": None,
            "product": None,
            "signed_slack": None,
            "lambda": None,
            "product_plus_lambda": None,
            "effective_capacity_proxy": None,
        }
    p_s = safe_mean([left for left, _ in pairs])
    p_r = safe_mean([right for _, right in pairs])
    p_both = safe_mean([1 if left and right else 0 for left, right in pairs])
    product = None if p_s is None or p_r is None else float(p_s * p_r)
    signed_slack = None if product is None or p_both is None else float(p_both - product)
    lambda_value = None if signed_slack is None else float(abs(signed_slack))
    product_plus_lambda = None if product is None or lambda_value is None else float(product + lambda_value)
    return {
        "count": len(pairs),
        "p_s": p_s,
        "p_r": p_r,
        "p_both": p_both,
        "product": product,
        "signed_slack": signed_slack,
        "lambda": lambda_value,
        "product_plus_lambda": product_plus_lambda,
        "effective_capacity_proxy": None
        if product_plus_lambda is None or product_plus_lambda <= 0.0
        else float(1.0 / product_plus_lambda),
    }


def summarize_by_impostor(by_impostor: dict[str, list[tuple[int, int]]]) -> dict:
    rows = [event_lambda(pairs) | {"trace_id": trace_id} for trace_id, pairs in by_impostor.items()]
    rows = [row for row in rows if row.get("product") is not None]
    if not rows:
        return {
            "impostor_count": 0,
            "sum_product": None,
            "sum_signed_slack": None,
            "sum_lambda": None,
            "sum_product_plus_lambda": None,
            "sum_empirical_both": None,
            "independent_pair_gamma": None,
            "poisson_pair_gamma": None,
            "poisson_from_empirical_both": None,
            "poisson_from_product_plus_lambda": None,
        }
    p_boths = [float(row["p_both"]) for row in rows if row.get("p_both") is not None]
    sum_product = float(sum(row["product"] for row in rows if row.get("product") is not None))
    sum_signed_slack = float(sum(row["signed_slack"] for row in rows if row.get("signed_slack") is not None))
    sum_lambda = float(sum(row["lambda"] for row in rows if row.get("lambda") is not None))
    sum_product_plus_lambda = float(
        sum(row["product_plus_lambda"] for row in rows if row.get("product_plus_lambda") is not None)
    )
    sum_empirical_both = float(sum(p_boths))
    sum_p2 = float(sum(p_value * p_value for p_value in p_boths))
    independent_pair_gamma = float(max(0.0, (sum_empirical_both * sum_empirical_both - sum_p2) / 2.0))
    poisson_pair_gamma = float((sum_empirical_both * sum_empirical_both) / 2.0)
    return {
        "impostor_count": len(rows),
        "sum_product": sum_product,
        "sum_signed_slack": sum_signed_slack,
        "sum_lambda": sum_lambda,
        "sum_product_plus_lambda": sum_product_plus_lambda,
        "sum_empirical_both": sum_empirical_both,
        "independent_pair_gamma": independent_pair_gamma,
        "poisson_pair_gamma": poisson_pair_gamma,
        "poisson_from_empirical_both": float(1.0 - math.exp(-sum_empirical_both)),
        "poisson_from_product_plus_lambda": float(1.0 - math.exp(-sum_product_plus_lambda)),
    }


def same_relation_independent_gamma(
    by_impostor: dict[str, list[tuple[int, int]]],
    relation_by_trace: dict[str, object],
) -> float | None:
    if not relation_by_trace:
        return None
    rows = []
    for trace_id, pairs in by_impostor.items():
        relation_id = relation_by_trace.get(trace_id)
        if relation_id is None:
            continue
        summary = event_lambda(pairs)
        if summary.get("p_both") is None:
            continue
        rows.append((trace_id, relation_id, float(summary["p_both"])))
    if len(rows) < 2:
        return 0.0
    total = 0.0
    for left_pos, (_, left_relation, left_p) in enumerate(rows):
        for _, right_relation, right_p in rows[left_pos + 1 :]:
            if left_relation == right_relation:
                total += left_p * right_p
    return float(total)


def difference_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def gamma_over_mu2(gamma_value: float | None, mu_value: float | None) -> float | None:
    if gamma_value is None or mu_value is None or mu_value <= 0.0:
        return None
    return float(gamma_value / (mu_value * mu_value))


def ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def rank_tail_summary(rank_pairs: list[tuple[int, int]], cutoffs: tuple[int, ...] = (1, 2, 4, 8, 16, 32)) -> dict:
    if not rank_pairs:
        return {
            "count": 0,
            "subject_rank_mean": None,
            "relation_rank_mean": None,
            "subject_rank_median": None,
            "relation_rank_median": None,
            "rank_corr": None,
            "subject_tail": {},
            "relation_tail": {},
            "joint_tail": {},
            "top1_both": None,
            "both_miss_top1": None,
        }
    subject = np.asarray([left for left, _ in rank_pairs], dtype=np.float64)
    relation = np.asarray([right for _, right in rank_pairs], dtype=np.float64)
    rank_corr = None
    if float(subject.std()) > 0.0 and float(relation.std()) > 0.0:
        rank_corr = float(np.corrcoef(subject, relation)[0, 1])
    subject_tail = {}
    relation_tail = {}
    joint_tail = {}
    for cutoff in cutoffs:
        subject_tail[f"gt_{cutoff}"] = float(np.mean(subject > cutoff))
        relation_tail[f"gt_{cutoff}"] = float(np.mean(relation > cutoff))
        joint_tail[f"gt_{cutoff}_gt_{cutoff}"] = float(np.mean((subject > cutoff) & (relation > cutoff)))
    top1_both = float(np.mean((subject == 1) & (relation == 1)))
    both_miss_top1 = float(np.mean((subject > 1) & (relation > 1)))
    return {
        "count": len(rank_pairs),
        "subject_rank_mean": float(subject.mean()),
        "relation_rank_mean": float(relation.mean()),
        "subject_rank_median": float(np.median(subject)),
        "relation_rank_median": float(np.median(relation)),
        "rank_corr": rank_corr,
        "subject_tail": subject_tail,
        "relation_tail": relation_tail,
        "joint_tail": joint_tail,
        "top1_both": top1_both,
        "both_miss_top1": both_miss_top1,
    }


def linear_hsic(rows: list[dict]) -> dict:
    grouped: dict[int, tuple[list[list[float]], list[list[float]]]] = {}
    for row in rows:
        subject_scores = row.get("factor_subject_scores")
        relation_scores = row.get("factor_relation_scores")
        if not isinstance(subject_scores, list) or not isinstance(relation_scores, list):
            continue
        if len(subject_scores) != len(relation_scores) or not subject_scores:
            continue
        if any(score is None for score in subject_scores + relation_scores):
            continue
        subject_vectors, relation_vectors = grouped.setdefault(len(subject_scores), ([], []))
        subject_vectors.append([float(score) for score in subject_scores])
        relation_vectors.append([float(score) for score in relation_scores])
    if not grouped:
        return {"count": 0, "score_vector_length": None, "linear_hsic": None, "linear_cka": None}
    score_length, (subject_vectors, relation_vectors) = max(grouped.items(), key=lambda item: len(item[1][0]))
    if len(subject_vectors) < 2:
        return {"count": len(subject_vectors), "score_vector_length": score_length, "linear_hsic": None, "linear_cka": None}
    x = np.asarray(subject_vectors, dtype=np.float64)
    y = np.asarray(relation_vectors, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    hsic = float(np.linalg.norm(cross, ord="fro") ** 2 / max(1, (x.shape[0] - 1) ** 2))
    xx = float(np.linalg.norm(x.T @ x, ord="fro"))
    yy = float(np.linalg.norm(y.T @ y, ord="fro"))
    cka = None if xx <= 0.0 or yy <= 0.0 else float((np.linalg.norm(cross, ord="fro") ** 2) / (xx * yy))
    return {"count": int(x.shape[0]), "score_vector_length": score_length, "linear_hsic": hsic, "linear_cka": cka}


def residualize_relation_scores_against_subject(rows: list[dict]) -> tuple[list[dict], dict]:
    transformed = []
    transformed_count = 0
    coefficients = []
    for row in rows:
        copied = dict(row)
        subject_scores = row.get("factor_subject_scores")
        relation_scores = row.get("factor_relation_scores")
        if (
            isinstance(subject_scores, list)
            and isinstance(relation_scores, list)
            and len(subject_scores) == len(relation_scores)
            and subject_scores
            and not any(score is None for score in subject_scores + relation_scores)
        ):
            subject = np.asarray([float(score) for score in subject_scores], dtype=np.float64)
            relation = np.asarray([float(score) for score in relation_scores], dtype=np.float64)
            denominator = float(np.dot(subject, subject))
            if denominator > 1.0e-12:
                coefficient = float(np.dot(relation, subject) / denominator)
                copied["factor_relation_scores"] = (relation - coefficient * subject).tolist()
                copied["factor_relation_score_transform"] = "score_space_relation_residualized_against_subject"
                transformed_count += 1
                coefficients.append(coefficient)
        transformed.append(copied)
    return transformed, {
        "name": "score_space_relation_residualized_against_subject",
        "transformed_rows": transformed_count,
        "total_rows": len(rows),
        "projection_coefficient_mean": safe_mean(coefficients),
    }


def infer_ds_size(run_dir: Path, rows: list[dict], explicit_ds_size: int | None) -> int:
    if explicit_ds_size is not None:
        return explicit_ds_size
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = {}
        for key in ("stream_length", "requested_ds_size", "ds_size"):
            value = config.get(key)
            if isinstance(value, int) and value > 0:
                return value
    post_edit = [
        row for row in rows if row.get("event_type") == "post_edit" or row.get("route_event") == "post_edit"
    ]
    return len(post_edit) if post_edit else 32


def split_eval_rows(rows: list[dict], ds_size: int) -> dict[str, list[dict]]:
    post_edit = [row for row in rows if row.get("event_type") == "post_edit" or row.get("route_event") == "post_edit"]
    inference = [row for row in rows if not (row.get("event_type") == "post_edit" or row.get("route_event") == "post_edit")]
    if len(inference) >= ds_size * 3:
        inference = inference[: ds_size * 3]
        return {
            "post_edit": post_edit,
            "rewrite": inference[0::3],
            "rephrase": inference[1::3],
            "locality": inference[2::3],
        }
    return {"post_edit": post_edit, "inference": inference}


def analyze_rows(rows: list[dict], query_kind: str) -> dict:
    target_subject_events = []
    target_relation_events = []
    target_both_events = []
    target_all_miss_events = []
    target_disagreement_events = []
    target_margins_s = []
    target_margins_r = []
    target_rank_pairs = []
    all_relative_pairs = []
    all_top_pairs = []
    relative_any_events = []
    top_any_events = []
    relative_any_and_target_miss_events = []
    top_any_and_target_miss_events = []
    wrong_winner_collision_events = []
    wrong_winner_collision_and_target_miss_events = []
    rank_independent_collision_baselines = []
    wrong_winner_same_target_relation_events = []
    wrong_winner_cross_target_relation_events = []
    rank_independent_same_target_relation_baselines = []
    rank_independent_cross_target_relation_baselines = []
    rank1_sum_covariance_terms = []
    rank1_covariance_terms = []
    relative_pair_gamma_counts = []
    relative_same_relation_gamma_counts = []
    top_pair_gamma_counts = []
    top_same_relation_gamma_counts = []
    by_impostor_relative: dict[str, list[tuple[int, int]]] = {}
    by_impostor_top: dict[str, list[tuple[int, int]]] = {}
    relation_by_trace: dict[str, object] = {}
    wrong_fires = 0
    correct_fires = 0
    abstains = 0
    usable_rows = 0
    relation_event_rules: dict[str, int] = {}

    for fallback_index, row in enumerate(rows):
        trace_ids = row.get("factor_score_trace_ids")
        relation_ids = row.get("factor_score_relation_ids")
        subject_scores = row.get("factor_subject_scores")
        relation_scores = row.get("factor_relation_scores")
        if not isinstance(trace_ids, list) or not isinstance(subject_scores, list) or not isinstance(relation_scores, list):
            continue
        if len(trace_ids) != len(subject_scores) or len(trace_ids) != len(relation_scores):
            continue
        target = row.get("expected_edit_id") or row.get("target_edit_id") or f"hopedit_{fallback_index:05d}"
        if target not in trace_ids:
            continue
        if any(score is None for score in subject_scores + relation_scores):
            continue
        usable_rows += 1
        target_idx = trace_ids.index(target)
        delta_s = float(row.get("factor_subject_margin_threshold") if row.get("factor_subject_margin_threshold") is not None else 0.0)
        delta_r = float(row.get("factor_relation_margin_threshold") if row.get("factor_relation_margin_threshold") is not None else 0.0)
        gamma_r = float(row.get("factor_relation_energy_threshold") if row.get("factor_relation_energy_threshold") is not None else 0.0)
        relation_match_rule = str(row.get("factor_relation_match_rule") or "top1_same_trace")
        relation_event_rules[relation_match_rule] = relation_event_rules.get(relation_match_rule, 0) + 1
        if isinstance(relation_ids, list) and len(relation_ids) == len(trace_ids):
            for trace_id, relation_id in zip(trace_ids, relation_ids):
                if relation_id is not None:
                    relation_by_trace.setdefault(trace_id, relation_id)
        subject_scores = [float(score) for score in subject_scores]
        relation_scores = [float(score) for score in relation_scores]
        subject_pass, subject_margin = pass_margin(subject_scores, target_idx, delta_s)
        subject_rank = rank_of_index(subject_scores, target_idx)
        if (
            relation_match_rule == "subject_candidate"
            and isinstance(relation_ids, list)
            and len(relation_ids) == len(trace_ids)
        ):
            target_relation_id = relation_ids[target_idx]
            outside_scores = [
                score
                for idx, score in enumerate(relation_scores)
                if idx != target_idx and relation_ids[idx] != target_relation_id
            ]
            outside_runner = max(outside_scores) if outside_scores else 0.0
            relation_margin = float(relation_scores[target_idx] - outside_runner)
            relation_pass = bool(-relation_scores[target_idx] < gamma_r and relation_margin > delta_r)
        else:
            relation_pass, relation_margin = pass_margin(relation_scores, target_idx, delta_r)
        relation_rank = rank_of_index(relation_scores, target_idx)
        if subject_rank is not None and relation_rank is not None:
            target_rank_pairs.append((subject_rank, relation_rank))
        target_subject_events.append(1 if subject_pass else 0)
        target_relation_events.append(1 if relation_pass else 0)
        target_both_events.append(1 if subject_pass and relation_pass else 0)
        target_all_miss = int((not subject_pass) and (not relation_pass))
        target_all_miss_events.append(target_all_miss)
        target_margins_s.append(subject_margin)
        target_margins_r.append(relation_margin)
        subject_top_idx = int(np.argmax(subject_scores))
        relation_top_idx = int(np.argmax(relation_scores))
        target_disagreement_events.append(1 if subject_top_idx != relation_top_idx else 0)
        subject_top_margin = sorted(subject_scores, reverse=True)[0] - (sorted(subject_scores, reverse=True)[1] if len(subject_scores) > 1 else 0.0)
        relation_top_margin = sorted(relation_scores, reverse=True)[0] - (sorted(relation_scores, reverse=True)[1] if len(relation_scores) > 1 else 0.0)
        wrong_winner_collision = int(
            subject_top_idx == relation_top_idx
            and subject_top_idx != target_idx
            and subject_top_margin > delta_s
            and relation_top_margin > delta_r
        )
        subject_wrong_winner = int(subject_top_idx != target_idx and subject_top_margin > delta_s)
        relation_wrong_winner = int(relation_top_idx != target_idx and relation_top_margin > delta_r)
        num_impostors = max(1, len(trace_ids) - 1)
        rank_independent_baseline = float(subject_wrong_winner * relation_wrong_winner / num_impostors)
        same_target_relation_count = 0
        cross_target_relation_count = num_impostors
        winner_same_target_relation = 0
        winner_cross_target_relation = 0
        if isinstance(relation_ids, list) and len(relation_ids) == len(trace_ids):
            target_relation_id = relation_ids[target_idx]
            same_target_relation_count = sum(
                1
                for idx, relation_id in enumerate(relation_ids)
                if idx != target_idx and relation_id == target_relation_id
            )
            cross_target_relation_count = max(0, num_impostors - same_target_relation_count)
            if wrong_winner_collision:
                if relation_ids[subject_top_idx] == target_relation_id:
                    winner_same_target_relation = 1
                else:
                    winner_cross_target_relation = 1
        same_target_relation_baseline = float(
            subject_wrong_winner * relation_wrong_winner * same_target_relation_count / max(1, num_impostors * num_impostors)
        )
        cross_target_relation_baseline = float(
            subject_wrong_winner * relation_wrong_winner * cross_target_relation_count / max(1, num_impostors * num_impostors)
        )
        # Sum_j Cov(1{j wins subject}, 1{j wins relation}) for this query.
        rank1_sum_covariance = float(wrong_winner_collision - rank_independent_baseline)
        rank1_covariance = float((wrong_winner_collision / num_impostors) - (subject_wrong_winner / num_impostors) * (relation_wrong_winner / num_impostors))
        wrong_winner_collision_events.append(wrong_winner_collision)
        rank_independent_collision_baselines.append(rank_independent_baseline)
        wrong_winner_same_target_relation_events.append(winner_same_target_relation)
        wrong_winner_cross_target_relation_events.append(winner_cross_target_relation)
        rank_independent_same_target_relation_baselines.append(same_target_relation_baseline)
        rank_independent_cross_target_relation_baselines.append(cross_target_relation_baseline)
        rank1_sum_covariance_terms.append(rank1_sum_covariance)
        rank1_covariance_terms.append(rank1_covariance)

        chosen = row.get("chosen_memory_id")
        if chosen == target:
            correct_fires += 1
        elif chosen is None:
            abstains += 1
        else:
            wrong_fires += 1

        relative_fired_indices = []
        top_fired_indices = []
        for idx, trace_id in enumerate(trace_ids):
            if trace_id == target:
                continue
            # Target-relative event used by the theorem diagnostics.
            rel_s = int(subject_scores[idx] > subject_scores[target_idx] - delta_s)
            rel_r = int(relation_scores[idx] > relation_scores[target_idx] - delta_r)
            all_relative_pairs.append((rel_s, rel_r))
            by_impostor_relative.setdefault(trace_id, []).append((rel_s, rel_r))
            if rel_s and rel_r:
                relative_fired_indices.append(idx)
            # Strict top-1 event matching the implementation's hard-AND rule.
            top_s = int(idx == subject_top_idx and subject_top_margin > delta_s)
            if (
                relation_match_rule == "subject_candidate"
                and isinstance(relation_ids, list)
                and len(relation_ids) == len(trace_ids)
            ):
                candidate_relation_id = relation_ids[idx]
                outside_scores = [
                    score
                    for other_idx, score in enumerate(relation_scores)
                    if other_idx != idx and relation_ids[other_idx] != candidate_relation_id
                ]
                outside_runner = max(outside_scores) if outside_scores else 0.0
                candidate_relation_margin = float(relation_scores[idx] - outside_runner)
                top_r = int(-relation_scores[idx] < gamma_r and candidate_relation_margin > delta_r)
            else:
                top_r = int(idx == relation_top_idx and relation_top_margin > delta_r)
            all_top_pairs.append((top_s, top_r))
            by_impostor_top.setdefault(trace_id, []).append((top_s, top_r))
            if top_s and top_r:
                top_fired_indices.append(idx)

        relative_any = int(bool(relative_fired_indices))
        top_any = int(bool(top_fired_indices))
        relative_any_events.append(relative_any)
        top_any_events.append(top_any)
        relative_any_and_target_miss_events.append(1 if relative_any and target_all_miss else 0)
        top_any_and_target_miss_events.append(1 if top_any and target_all_miss else 0)
        wrong_winner_collision_and_target_miss_events.append(1 if wrong_winner_collision and target_all_miss else 0)
        relative_pair_gamma_counts.append(math.comb(len(relative_fired_indices), 2))
        top_pair_gamma_counts.append(math.comb(len(top_fired_indices), 2))
        if isinstance(relation_ids, list) and len(relation_ids) == len(trace_ids):
            relative_same_relation_pairs = 0
            for left_pos, left_idx in enumerate(relative_fired_indices):
                for right_idx in relative_fired_indices[left_pos + 1 :]:
                    if relation_ids[left_idx] == relation_ids[right_idx]:
                        relative_same_relation_pairs += 1
            top_same_relation_pairs = 0
            for left_pos, left_idx in enumerate(top_fired_indices):
                for right_idx in top_fired_indices[left_pos + 1 :]:
                    if relation_ids[left_idx] == relation_ids[right_idx]:
                        top_same_relation_pairs += 1
            relative_same_relation_gamma_counts.append(relative_same_relation_pairs)
            top_same_relation_gamma_counts.append(top_same_relation_pairs)

    target_qs = safe_mean(target_subject_events)
    target_qr = safe_mean(target_relation_events)
    target_fire_rate = safe_mean(target_both_events)
    target_all_miss_rate = safe_mean(target_all_miss_events)
    target_product = None if target_qs is None or target_qr is None else float(target_qs * target_qr)
    target_delta = None if target_fire_rate is None or target_product is None else float(target_fire_rate - target_product)
    target_miss_product = None if target_qs is None or target_qr is None else float((1.0 - target_qs) * (1.0 - target_qr))
    target_miss_delta = None if target_all_miss_rate is None or target_miss_product is None else float(target_all_miss_rate - target_miss_product)
    target_rho = binary_corr(target_subject_events, target_relation_events)
    recall_formula = None
    if target_qs is not None and target_qr is not None:
        covariance_term = 0.0
        if target_rho is not None:
            covariance_term = target_rho * math.sqrt(max(0.0, target_qs * (1.0 - target_qs) * target_qr * (1.0 - target_qr)))
        recall_formula = float(target_qs * target_qr + covariance_term)

    lambda_relative = event_lambda(all_relative_pairs)
    lambda_top = event_lambda(all_top_pairs)
    by_impostor_relative_summary = summarize_by_impostor(by_impostor_relative)
    by_impostor_top_summary = summarize_by_impostor(by_impostor_top)
    relative_gamma = safe_mean(relative_pair_gamma_counts)
    relative_same_relation_gamma = safe_mean(relative_same_relation_gamma_counts)
    top_gamma = safe_mean(top_pair_gamma_counts)
    top_same_relation_gamma = safe_mean(top_same_relation_gamma_counts)
    relative_any_rate = safe_mean(relative_any_events)
    top_any_rate = safe_mean(top_any_events)
    relative_any_and_target_miss_rate = safe_mean(relative_any_and_target_miss_events)
    top_any_and_target_miss_rate = safe_mean(top_any_and_target_miss_events)
    wrong_winner_collision_rate = safe_mean(wrong_winner_collision_events)
    wrong_winner_collision_and_target_miss_rate = safe_mean(wrong_winner_collision_and_target_miss_events)
    rank_independent_collision_baseline = safe_mean(rank_independent_collision_baselines)
    wrong_winner_same_target_relation_rate = safe_mean(wrong_winner_same_target_relation_events)
    wrong_winner_cross_target_relation_rate = safe_mean(wrong_winner_cross_target_relation_events)
    rank_independent_same_target_relation_baseline = safe_mean(rank_independent_same_target_relation_baselines)
    rank_independent_cross_target_relation_baseline = safe_mean(rank_independent_cross_target_relation_baselines)
    rank1_sum_covariance_mean = safe_mean(rank1_sum_covariance_terms)
    rank1_covariance_mean = safe_mean(rank1_covariance_terms)
    relative_mu = by_impostor_relative_summary.get("sum_empirical_both")
    top_mu = by_impostor_top_summary.get("sum_empirical_both")
    relative_independent_gamma = by_impostor_relative_summary.get("independent_pair_gamma")
    top_independent_gamma = by_impostor_top_summary.get("independent_pair_gamma")
    relative_same_relation_independent_gamma = same_relation_independent_gamma(by_impostor_relative, relation_by_trace)
    top_same_relation_independent_gamma = same_relation_independent_gamma(by_impostor_top, relation_by_trace)
    lambda_by_impostor = [event_lambda(pairs) | {"trace_id": trace_id} for trace_id, pairs in by_impostor_relative.items()]
    lambda_by_impostor = [row for row in lambda_by_impostor if row.get("lambda") is not None]
    lambda_by_impostor.sort(key=lambda row: row["lambda"], reverse=True)

    return {
        "query_kind": query_kind,
        "usable_rows": usable_rows,
        "target": {
            "q_s": target_qs,
            "q_r": target_qr,
            "target_fire_rate": target_fire_rate,
            "target_product": target_product,
            "target_slack_delta": target_delta,
            "target_slack_abs": None if target_delta is None else float(abs(target_delta)),
            "target_all_miss_rate": target_all_miss_rate,
            "target_miss_product": target_miss_product,
            "target_miss_slack_delta": target_miss_delta,
            "target_miss_slack_abs": None if target_miss_delta is None else float(abs(target_miss_delta)),
            "rho": target_rho,
            "recall_formula_prediction": recall_formula,
            "factor_disagreement_rate": safe_mean(target_disagreement_events),
            "subject_margin_mean": safe_mean(target_margins_s),
            "relation_margin_mean": safe_mean(target_margins_r),
            "correct_fire_rate": None if usable_rows == 0 else float(correct_fires / usable_rows),
            "wrong_fire_rate": None if usable_rows == 0 else float(wrong_fires / usable_rows),
            "abstain_rate": None if usable_rows == 0 else float(abstains / usable_rows),
            "relation_event_rule_counts": relation_event_rules,
            "rank_summary": rank_tail_summary(target_rank_pairs),
        },
        "impostor_target_relative": lambda_relative,
        "impostor_top1_rule": lambda_top,
        "capacity_phase": {
            "target_product": target_product,
            "target_slack_delta": target_delta,
            "target_slack_abs": None if target_delta is None else float(abs(target_delta)),
            "target_fire_rate": target_fire_rate,
            "impostor_product": lambda_relative.get("product"),
            "impostor_lambda": lambda_relative.get("lambda"),
            "impostor_signed_slack": lambda_relative.get("signed_slack"),
            "impostor_product_plus_lambda": lambda_relative.get("product_plus_lambda"),
            "effective_capacity_proxy": lambda_relative.get("effective_capacity_proxy"),
            "relative_any_impostor_rate": relative_any_rate,
            "top1_any_impostor_rate": top_any_rate,
            "relative_to_top1_any_suppression": None
            if relative_any_rate is None or top_any_rate is None
            else float(relative_any_rate - top_any_rate),
            "relative_mu": relative_mu,
            "top1_mu": top_mu,
            "top1_mu_over_relative_mu": ratio_or_none(top_mu, relative_mu),
            "relative_any_and_target_miss_rate": relative_any_and_target_miss_rate,
            "relative_basin_target_miss_xi": None
            if relative_any_rate is None or target_all_miss_rate is None or relative_any_and_target_miss_rate is None
            else float(relative_any_and_target_miss_rate - relative_any_rate * target_all_miss_rate),
            "top1_any_and_target_miss_rate": top_any_and_target_miss_rate,
            "top1_basin_target_miss_xi": None
            if top_any_rate is None or target_all_miss_rate is None or top_any_and_target_miss_rate is None
            else float(top_any_and_target_miss_rate - top_any_rate * target_all_miss_rate),
            "wrong_winner_collision_rate": wrong_winner_collision_rate,
            "rank_independent_collision_baseline": rank_independent_collision_baseline,
            "rank_collision_excess": difference_or_none(
                wrong_winner_collision_rate,
                rank_independent_collision_baseline,
            ),
            "rank_collision_ratio_to_independent": ratio_or_none(
                wrong_winner_collision_rate,
                rank_independent_collision_baseline,
            ),
            "wrong_winner_same_target_relation_rate": wrong_winner_same_target_relation_rate,
            "wrong_winner_cross_target_relation_rate": wrong_winner_cross_target_relation_rate,
            "rank_independent_same_target_relation_baseline": rank_independent_same_target_relation_baseline,
            "rank_independent_cross_target_relation_baseline": rank_independent_cross_target_relation_baseline,
            "rank_collision_same_target_relation_excess": difference_or_none(
                wrong_winner_same_target_relation_rate,
                rank_independent_same_target_relation_baseline,
            ),
            "rank_collision_cross_target_relation_excess": difference_or_none(
                wrong_winner_cross_target_relation_rate,
                rank_independent_cross_target_relation_baseline,
            ),
            "rank_collision_same_target_relation_ratio": ratio_or_none(
                wrong_winner_same_target_relation_rate,
                rank_independent_same_target_relation_baseline,
            ),
            "rank_collision_cross_target_relation_ratio": ratio_or_none(
                wrong_winner_cross_target_relation_rate,
                rank_independent_cross_target_relation_baseline,
            ),
            "rank1_sum_covariance_mean": rank1_sum_covariance_mean,
            "rank1_covariance_mean": rank1_covariance_mean,
            "wrong_winner_collision_and_target_miss_rate": wrong_winner_collision_and_target_miss_rate,
            "wrong_winner_collision_given_target_miss": ratio_or_none(
                wrong_winner_collision_and_target_miss_rate,
                target_all_miss_rate,
            ),
            "relative_pair_gamma": relative_gamma,
            "relative_independent_pair_gamma": relative_independent_gamma,
            "relative_excess_pair_gamma": difference_or_none(relative_gamma, relative_independent_gamma),
            "relative_gamma_over_mu2": gamma_over_mu2(relative_gamma, relative_mu),
            "relative_same_relation_pair_gamma": relative_same_relation_gamma,
            "relative_same_relation_independent_pair_gamma": relative_same_relation_independent_gamma,
            "relative_same_relation_excess_pair_gamma": difference_or_none(
                relative_same_relation_gamma,
                relative_same_relation_independent_gamma,
            ),
            "relative_same_relation_gamma_over_mu2": gamma_over_mu2(
                relative_same_relation_gamma,
                relative_mu,
            ),
            "top1_pair_gamma": top_gamma,
            "top1_independent_pair_gamma": top_independent_gamma,
            "top1_excess_pair_gamma": difference_or_none(top_gamma, top_independent_gamma),
            "top1_gamma_over_mu2": gamma_over_mu2(top_gamma, top_mu),
            "top1_same_relation_pair_gamma": top_same_relation_gamma,
            "top1_same_relation_independent_pair_gamma": top_same_relation_independent_gamma,
            "top1_same_relation_excess_pair_gamma": difference_or_none(
                top_same_relation_gamma,
                top_same_relation_independent_gamma,
            ),
            "top1_same_relation_gamma_over_mu2": gamma_over_mu2(
                top_same_relation_gamma,
                top_mu,
            ),
            "relative_union_bound_by_impostor": by_impostor_relative_summary,
            "top1_union_bound_by_impostor": by_impostor_top_summary,
        },
        "lambda_by_impostor_top5": lambda_by_impostor[:5],
        "score_dependence": linear_hsic(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--ds_size", default=None, type=int)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--relation_score_residualize_subject", action="store_true")
    args = parser.parse_args()

    log_path = args.run_dir / "annotated_route_logs.jsonl"
    if not log_path.exists():
        log_path = args.run_dir / "route_logs.jsonl"
    rows = read_jsonl(log_path)
    rows = backfill_relation_ids(rows, load_relation_ids_from_memory(args.run_dir))
    score_transform = None
    if args.relation_score_residualize_subject:
        rows, score_transform = residualize_relation_scores_against_subject(rows)
    ds_size = infer_ds_size(args.run_dir, rows, args.ds_size)
    splits = split_eval_rows(rows, ds_size)
    diagnostics = {
        "run_dir": str(args.run_dir),
        "log_path": str(log_path),
        "ds_size": ds_size,
        "score_transform": score_transform,
        "available_splits": sorted(splits.keys()),
        "splits": {},
    }
    for name, split_rows in splits.items():
        diagnostics["splits"][name] = analyze_rows(split_rows, name)

    output = args.output or (args.run_dir / "factor_score_diagnostics.json")
    output.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
