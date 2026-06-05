import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from easyeditor import BaseEditor
from easyeditor.models.hopedit.hopedit_main import FactoredRelationResidualEncoder, HopEditController
from examples.edit_experiment_utils import load_normalized_records, resolve_hparams_class


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def relation_id(record: dict[str, Any]) -> str | None:
    value = record.get("relation_id")
    return None if value is None else str(value)


def build_same_relation_split(
    records: list[dict[str, Any]],
    *,
    test_size: int,
    dev_size: int,
    train_size: int,
    seed: int,
) -> dict[str, Any]:
    test_records = [record for record in records[:test_size] if relation_id(record) is not None]
    test_indices = {int(record["source_index"]) for record in test_records}
    pool = [
        record
        for record in records[test_size:]
        if relation_id(record) is not None and int(record["source_index"]) not in test_indices
    ]
    rng = random.Random(seed)
    rng.shuffle(pool)
    dev_records = pool[:dev_size]
    dev_indices = {int(record["source_index"]) for record in dev_records}
    required_relations = {relation_id(record) for record in test_records + dev_records if relation_id(record) is not None}
    remaining = [record for record in pool[dev_size:] if int(record["source_index"]) not in dev_indices]
    train_records = []
    used_indices = set()
    for required_relation in sorted(required_relations):
        match = next((record for record in remaining if relation_id(record) == required_relation), None)
        if match is None:
            continue
        train_records.append(match)
        used_indices.add(int(match["source_index"]))
    filler = [record for record in remaining if int(record["source_index"]) not in used_indices]
    for record in filler:
        if len(train_records) >= train_size:
            break
        train_records.append(record)
    train_relations = {relation_id(record) for record in train_records if relation_id(record) is not None}
    missing_required_relations = sorted(rel for rel in required_relations if rel not in train_relations)
    return {
        "train": train_records,
        "dev": dev_records,
        "test": test_records,
        "metadata": {
            "split_name": "same_relation",
            "seed": seed,
            "requested_train_size": train_size,
            "requested_dev_size": dev_size,
            "requested_test_size": test_size,
            "train_count": len(train_records),
            "dev_count": len(dev_records),
            "test_count": len(test_records),
            "train_relation_count": len(train_relations),
            "dev_relation_count": len({relation_id(record) for record in dev_records if relation_id(record) is not None}),
            "test_relation_count": len({relation_id(record) for record in test_records if relation_id(record) is not None}),
            "missing_required_relations": missing_required_relations,
        },
    }


def relation_view_rows(records: list[dict[str, Any]], *, include_eval_rephrase: bool) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rid = relation_id(record)
        if rid is None:
            continue
        base = {
            "source_index": int(record["source_index"]),
            "relation_id": rid,
            "subject": record.get("subject"),
            "target_new": record.get("target_new"),
        }
        rows.append({**base, "view_name": "prompt", "text": record.get("prompt")})
        if record.get("address_rephrase_prompt"):
            rows.append({**base, "view_name": "address_rephrase", "text": record.get("address_rephrase_prompt")})
        if include_eval_rephrase and record.get("rephrase_prompt"):
            rows.append({**base, "view_name": "eval_rephrase", "text": record.get("rephrase_prompt")})
    return [row for row in rows if row.get("text")]


def extract_relation_raw_factors(
    controller: HopEditController,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    extracted = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        factors = controller._extract_batched_factored_address_keys(
            [str(row["text"]) for row in batch],
            [row.get("subject") for row in batch],
            [row.get("target_new") for row in batch],
        )
        for row, factor_row in zip(batch, factors):
            raw_factor = factor_row.get("relation_raw_factor")
            if not isinstance(raw_factor, torch.Tensor):
                continue
            subject_factor = factor_row.get("subject_factor")
            extracted.append(
                {
                    **row,
                    "subject_factor": None
                    if not isinstance(subject_factor, torch.Tensor)
                    else subject_factor.detach().float().cpu(),
                    "relation_raw_factor": raw_factor.detach().float().cpu(),
                }
            )
    return extracted


def supervised_contrastive_loss(embeddings: torch.Tensor, group_ids: list[str], *, temperature: float) -> torch.Tensor | None:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return None
    similarity = embeddings @ embeddings.T / max(1.0e-6, float(temperature))
    self_mask = torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    similarity = similarity.masked_fill(self_mask, -1.0e9)
    losses = []
    for idx, group_id in enumerate(group_ids):
        positive_mask = torch.tensor(
            [other_idx != idx and other_group == group_id for other_idx, other_group in enumerate(group_ids)],
            dtype=torch.bool,
            device=embeddings.device,
        )
        if not bool(positive_mask.any().item()):
            continue
        log_denominator = torch.logsumexp(similarity[idx], dim=0)
        log_numerator = torch.logsumexp(similarity[idx][positive_mask], dim=0)
        losses.append(-(log_numerator - log_denominator))
    if not losses:
        return None
    return torch.stack(losses).mean()


def rank_collision_loss(
    encoded: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    query_view_names: set[str],
    beta: float,
    excess_only: bool,
    normalize_by_candidates: bool,
) -> dict[str, Any] | None:
    trace_indices = [
        idx
        for idx, row in enumerate(rows)
        if row.get("view_name") == "prompt" and isinstance(row.get("subject_factor"), torch.Tensor)
    ]
    source_to_trace = {int(rows[idx]["source_index"]): pos for pos, idx in enumerate(trace_indices)}
    query_indices = [
        idx
        for idx, row in enumerate(rows)
        if str(row.get("view_name")) in query_view_names
        and int(row["source_index"]) in source_to_trace
        and isinstance(row.get("subject_factor"), torch.Tensor)
    ]
    if len(trace_indices) < 2 or not query_indices:
        return None
    num_impostors = max(1, len(trace_indices) - 1)
    device = encoded.device
    trace_subject = F.normalize(torch.stack([rows[idx]["subject_factor"] for idx in trace_indices], dim=0).to(device), p=2, dim=-1)
    query_subject = F.normalize(torch.stack([rows[idx]["subject_factor"] for idx in query_indices], dim=0).to(device), p=2, dim=-1)
    trace_relation = encoded[torch.tensor(trace_indices, dtype=torch.long, device=device)]
    query_relation = encoded[torch.tensor(query_indices, dtype=torch.long, device=device)]
    subject_scores = query_subject @ trace_subject.T
    relation_scores = query_relation @ trace_relation.T
    beta = max(1.0e-6, float(beta))
    subject_rank1 = F.softmax(beta * subject_scores, dim=-1)
    relation_rank1 = F.softmax(beta * relation_scores, dim=-1)
    target_cols = torch.tensor(
        [source_to_trace[int(rows[idx]["source_index"])] for idx in query_indices],
        dtype=torch.long,
        device=device,
    )
    mask = torch.ones_like(subject_rank1)
    mask.scatter_(1, target_cols.view(-1, 1), 0.0)
    joint_collision = (subject_rank1 * relation_rank1 * mask).sum(dim=-1)
    if not excess_only:
        raw_loss = joint_collision.mean()
        independent_baseline = None
    else:
        non_target_subject_mass = (subject_rank1 * mask).sum(dim=-1)
        non_target_relation_mass = (relation_rank1 * mask).sum(dim=-1)
        independent_baseline = non_target_subject_mass * non_target_relation_mass / num_impostors
        raw_loss = F.relu(joint_collision - independent_baseline).mean()
    # With diffuse soft ranks, the summed non-excess collision is O(1/N)
    # while the excess-over-independent term is O(1/N^2). Scale each back
    # to O(1) so the rank objective has usable, but not overwhelming,
    # gradient leverage against the contrastive objectives.
    if normalize_by_candidates and excess_only:
        scale = float(num_impostors * num_impostors)
    elif normalize_by_candidates:
        scale = float(num_impostors)
    else:
        scale = 1.0
    return {
        "loss": raw_loss * scale,
        "raw_loss": raw_loss,
        "scale": scale,
        "num_impostors": num_impostors,
        "joint_collision_mean": joint_collision.mean(),
        "independent_baseline_mean": None
        if independent_baseline is None
        else independent_baseline.mean(),
    }


def within_cluster_disambiguation_loss(
    encoded: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    query_view_names: set[str],
    margin: float,
    hard_k: int,
    easy_k: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    prompt_indices = [idx for idx, row in enumerate(rows) if row.get("view_name") == "prompt"]
    source_to_prompt_pos = {int(rows[idx]["source_index"]): pos for pos, idx in enumerate(prompt_indices)}
    relation_to_prompt_pos: dict[str, list[int]] = {}
    for pos, idx in enumerate(prompt_indices):
        rid = relation_id(rows[idx])
        if rid is None:
            continue
        relation_to_prompt_pos.setdefault(rid, []).append(pos)
    query_indices = [
        idx
        for idx, row in enumerate(rows)
        if str(row.get("view_name")) in query_view_names and int(row["source_index"]) in source_to_prompt_pos
    ]
    if len(prompt_indices) < 2 or not query_indices:
        return None

    trace_embeddings = encoded[torch.tensor(prompt_indices, dtype=torch.long, device=encoded.device)].detach()
    losses = []
    pos_distances = []
    hardest_neg_distances = []
    same_counts = []
    cross_counts = []
    skipped_no_same_cluster = 0
    skipped_no_negatives = 0
    hard_k = max(0, int(hard_k))
    easy_k = max(0, int(easy_k))

    for query_idx in query_indices:
        source_index = int(rows[query_idx]["source_index"])
        target_pos = source_to_prompt_pos[source_index]
        target_row = rows[prompt_indices[target_pos]]
        target_relation_id = relation_id(target_row)
        same_pool = [
            pos
            for pos in relation_to_prompt_pos.get(str(target_relation_id), [])
            if pos != target_pos
        ]
        if hard_k > 0 and not same_pool:
            skipped_no_same_cluster += 1
            continue
        if hard_k > 0 and len(same_pool) > hard_k:
            same_positions = rng.sample(same_pool, hard_k)
        else:
            same_positions = list(same_pool)

        cross_pool = [
            pos
            for pos, prompt_idx in enumerate(prompt_indices)
            if pos != target_pos and relation_id(rows[prompt_idx]) != target_relation_id
        ]
        if easy_k > 0 and len(cross_pool) > easy_k:
            cross_positions = rng.sample(cross_pool, easy_k)
        elif easy_k > 0:
            cross_positions = list(cross_pool)
        else:
            cross_positions = []

        negative_positions = same_positions + cross_positions
        if not negative_positions:
            skipped_no_negatives += 1
            continue

        query_embedding = encoded[query_idx]
        positive_embedding = trace_embeddings[target_pos]
        negative_embeddings = trace_embeddings[
            torch.tensor(negative_positions, dtype=torch.long, device=encoded.device)
        ]
        positive_distance = 1.0 - torch.sum(query_embedding * positive_embedding, dim=-1)
        negative_distances = 1.0 - (negative_embeddings @ query_embedding)
        hardest_negative_distance = negative_distances.min()
        losses.append(F.relu(positive_distance - hardest_negative_distance + float(margin)))
        pos_distances.append(positive_distance.detach())
        hardest_neg_distances.append(hardest_negative_distance.detach())
        same_counts.append(float(len(same_positions)))
        cross_counts.append(float(len(cross_positions)))

    if not losses:
        return None
    stacked_losses = torch.stack(losses)
    violation_rate = float((stacked_losses.detach() > 0.0).float().mean().cpu().item())
    return {
        "loss": stacked_losses.mean(),
        "eligible_queries": len(losses),
        "total_queries": len(query_indices),
        "skipped_no_same_cluster": skipped_no_same_cluster,
        "skipped_no_negatives": skipped_no_negatives,
        "violation_rate": violation_rate,
        "positive_distance_mean": float(torch.stack(pos_distances).mean().cpu().item()),
        "hardest_negative_distance_mean": float(torch.stack(hardest_neg_distances).mean().cpu().item()),
        "same_negative_count_mean": float(sum(same_counts) / len(same_counts)),
        "cross_negative_count_mean": float(sum(cross_counts) / len(cross_counts)),
    }


def whiten_trace_query_embeddings(
    trace_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if trace_embeddings.ndim != 2 or query_embeddings.ndim != 2 or trace_embeddings.shape[0] < 2:
        return trace_embeddings, query_embeddings
    eps = max(1.0e-12, float(eps))
    mean = trace_embeddings.mean(dim=0, keepdim=True)
    centered_trace = trace_embeddings - mean
    centered_query = query_embeddings - mean
    # The trace count is much smaller than hidden size in our use case; SVD
    # gives a stable low-rank whitening basis without materializing d x d covariances.
    _, singular_values, components_t = torch.linalg.svd(centered_trace, full_matrices=False)
    valid = singular_values > eps
    if not bool(valid.any().item()):
        return trace_embeddings, query_embeddings
    components = components_t[valid].T
    scale = (max(1, trace_embeddings.shape[0] - 1) ** 0.5) / singular_values[valid].clamp_min(eps)
    trace_whitened = (centered_trace @ components) * scale
    query_whitened = (centered_query @ components) * scale
    return F.normalize(trace_whitened, p=2, dim=-1), F.normalize(query_whitened, p=2, dim=-1)


def mine_hard_winner_triplets(
    encoded: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    query_view_names: set[str],
    bank_size: int,
    whiten_eps: float,
    relation_margin_threshold: float,
    subject_margin_threshold: float,
    topk: int,
    max_triplets_per_query: int,
    same_relation_only: bool,
    rng: random.Random,
) -> dict[str, Any]:
    prompt_indices = [
        idx
        for idx, row in enumerate(rows)
        if row.get("view_name") == "prompt" and isinstance(row.get("subject_factor"), torch.Tensor)
    ]
    if len(prompt_indices) < 2:
        return {"triplets": [], "reason": "too_few_prompts"}
    bank_size = int(bank_size)
    if bank_size > 0 and len(prompt_indices) > bank_size:
        bank_positions = sorted(rng.sample(range(len(prompt_indices)), bank_size))
    else:
        bank_positions = list(range(len(prompt_indices)))
    bank_prompt_indices = [prompt_indices[pos] for pos in bank_positions]
    source_to_bank_col = {int(rows[idx]["source_index"]): col for col, idx in enumerate(bank_prompt_indices)}
    query_indices = [
        idx
        for idx, row in enumerate(rows)
        if str(row.get("view_name")) in query_view_names
        and int(row["source_index"]) in source_to_bank_col
        and isinstance(row.get("subject_factor"), torch.Tensor)
    ]
    if not query_indices:
        return {
            "triplets": [],
            "reason": "no_queries_in_bank",
            "bank_size": len(bank_prompt_indices),
            "query_count": 0,
        }

    device = encoded.device
    with torch.no_grad():
        trace_relation = encoded[torch.tensor(bank_prompt_indices, dtype=torch.long, device=device)].detach()
        query_relation = encoded[torch.tensor(query_indices, dtype=torch.long, device=device)].detach()
        trace_relation, query_relation = whiten_trace_query_embeddings(
            trace_relation,
            query_relation,
            eps=whiten_eps,
        )
        trace_subject = F.normalize(
            torch.stack([rows[idx]["subject_factor"] for idx in bank_prompt_indices], dim=0).to(device),
            p=2,
            dim=-1,
        )
        query_subject = F.normalize(
            torch.stack([rows[idx]["subject_factor"] for idx in query_indices], dim=0).to(device),
            p=2,
            dim=-1,
        )
        subject_scores = query_subject @ trace_subject.T
        relation_scores = query_relation @ trace_relation.T

    triplets: list[tuple[int, int, int]] = []
    exact_triplets = 0
    near_triplets = 0
    same_relation_triplets = 0
    cross_relation_triplets = 0
    skipped_target_top = 0
    skipped_no_collision = 0
    skipped_same_relation_filter = 0
    topk = max(1, int(topk))
    max_triplets_per_query = max(1, int(max_triplets_per_query))
    for query_row_idx, query_idx in enumerate(query_indices):
        target_col = source_to_bank_col[int(rows[query_idx]["source_index"])]
        subject_row = subject_scores[query_row_idx]
        relation_row = relation_scores[query_row_idx]
        subject_sorted, subject_order = torch.sort(subject_row, descending=True)
        relation_sorted, relation_order = torch.sort(relation_row, descending=True)
        subject_top_col = int(subject_order[0].item())
        relation_top_col = int(relation_order[0].item())
        subject_margin = float(subject_sorted[0].item() - (subject_sorted[1].item() if subject_sorted.numel() > 1 else 0.0))
        relation_margin = float(relation_sorted[0].item() - (relation_sorted[1].item() if relation_sorted.numel() > 1 else 0.0))
        if subject_margin <= float(subject_margin_threshold) or relation_margin <= float(relation_margin_threshold):
            skipped_no_collision += 1
            continue
        target_prompt_idx = bank_prompt_indices[target_col]
        candidate_cols: list[int] = []
        if topk <= 1:
            if subject_top_col == target_col or relation_top_col == target_col:
                skipped_target_top += 1
                continue
            if subject_top_col == relation_top_col:
                candidate_cols = [subject_top_col]
        else:
            subject_top_cols = [int(col.item()) for col in subject_order[: min(topk, subject_order.numel())]]
            relation_top_cols = {int(col.item()) for col in relation_order[: min(topk, relation_order.numel())]}
            candidate_cols = [
                col
                for col in subject_top_cols
                if col != target_col and col in relation_top_cols
            ]
            candidate_cols.sort(
                key=lambda col: float(subject_row[col].item() + relation_row[col].item()),
                reverse=True,
            )
        if not candidate_cols:
            skipped_no_collision += 1
            continue
        kept_for_query = 0
        for candidate_col in candidate_cols:
            wrong_prompt_idx = bank_prompt_indices[candidate_col]
            same_relation = relation_id(rows[wrong_prompt_idx]) == relation_id(rows[target_prompt_idx])
            if same_relation_only and not same_relation:
                skipped_same_relation_filter += 1
                continue
            triplets.append((query_idx, target_prompt_idx, wrong_prompt_idx))
            if candidate_col == subject_top_col and candidate_col == relation_top_col:
                exact_triplets += 1
            else:
                near_triplets += 1
            if same_relation:
                same_relation_triplets += 1
            else:
                cross_relation_triplets += 1
            kept_for_query += 1
            if kept_for_query >= max_triplets_per_query:
                break
        if kept_for_query == 0:
            skipped_no_collision += 1
    return {
        "triplets": triplets,
        "bank_size": len(bank_prompt_indices),
        "query_count": len(query_indices),
        "triplet_count": len(triplets),
        "exact_triplet_count": exact_triplets,
        "near_triplet_count": near_triplets,
        "same_relation_triplet_count": same_relation_triplets,
        "cross_relation_triplet_count": cross_relation_triplets,
        "skipped_target_top": skipped_target_top,
        "skipped_no_collision": skipped_no_collision,
        "skipped_same_relation_filter": skipped_same_relation_filter,
        "topk": topk,
        "max_triplets_per_query": max_triplets_per_query,
        "same_relation_only": bool(same_relation_only),
    }


def hard_winner_triplet_loss(
    encoded: torch.Tensor,
    triplets: list[tuple[int, int, int]],
    *,
    margin: float,
) -> dict[str, Any] | None:
    if not triplets:
        return None
    device = encoded.device
    query_indices = torch.tensor([triplet[0] for triplet in triplets], dtype=torch.long, device=device)
    target_indices = torch.tensor([triplet[1] for triplet in triplets], dtype=torch.long, device=device)
    wrong_indices = torch.tensor([triplet[2] for triplet in triplets], dtype=torch.long, device=device)
    query_embeddings = encoded[query_indices]
    target_embeddings = encoded[target_indices].detach()
    wrong_embeddings = encoded[wrong_indices].detach()
    positive_distances = 1.0 - torch.sum(query_embeddings * target_embeddings, dim=-1)
    wrong_distances = 1.0 - torch.sum(query_embeddings * wrong_embeddings, dim=-1)
    losses = F.relu(positive_distances - wrong_distances + float(margin))
    return {
        "loss": losses.mean(),
        "triplet_count": len(triplets),
        "violation_rate": float((losses.detach() > 0.0).float().mean().cpu().item()),
        "positive_distance_mean": float(positive_distances.detach().mean().cpu().item()),
        "wrong_distance_mean": float(wrong_distances.detach().mean().cpu().item()),
    }


def hard_winner_infonce_loss(
    encoded: torch.Tensor,
    triplets: list[tuple[int, int, int]],
    *,
    temperature: float,
) -> dict[str, Any] | None:
    if not triplets:
        return None
    grouped: dict[tuple[int, int], list[int]] = {}
    for query_idx, target_idx, wrong_idx in triplets:
        grouped.setdefault((query_idx, target_idx), []).append(wrong_idx)
    losses = []
    positive_distances = []
    hardest_wrong_distances = []
    negative_counts = []
    temperature = max(1.0e-6, float(temperature))
    for (query_idx, target_idx), wrong_indices in grouped.items():
        if not wrong_indices:
            continue
        # De-duplicate while preserving mining order so repeated near-winner
        # candidates do not overweight one wrong trace in the denominator.
        wrong_indices = list(dict.fromkeys(int(idx) for idx in wrong_indices))
        query_embedding = encoded[int(query_idx)]
        target_embedding = encoded[int(target_idx)].detach()
        wrong_embeddings = encoded[
            torch.tensor(wrong_indices, dtype=torch.long, device=encoded.device)
        ].detach()
        positive_score = torch.sum(query_embedding * target_embedding, dim=-1)
        wrong_scores = wrong_embeddings @ query_embedding
        logits = torch.cat([positive_score.view(1), wrong_scores], dim=0) / temperature
        labels = torch.zeros(1, dtype=torch.long, device=encoded.device)
        losses.append(F.cross_entropy(logits.view(1, -1), labels))
        positive_distances.append((1.0 - positive_score).detach())
        hardest_wrong_distances.append((1.0 - wrong_scores.max()).detach())
        negative_counts.append(float(len(wrong_indices)))
    if not losses:
        return None
    return {
        "loss": torch.stack(losses).mean(),
        "triplet_count": len(triplets),
        "query_count": len(losses),
        "negative_count_mean": float(sum(negative_counts) / len(negative_counts)),
        "positive_distance_mean": float(torch.stack(positive_distances).mean().cpu().item()),
        "wrong_distance_mean": float(torch.stack(hardest_wrong_distances).mean().cpu().item()),
    }


def evaluate_relation_encoder(
    encoder: FactoredRelationResidualEncoder,
    rows: list[dict[str, Any]],
    *,
    relation_margin_threshold: float,
    relation_energy_threshold: float,
    relation_whiten_eps: float | None = None,
) -> dict[str, Any]:
    prompt_rows = [row for row in rows if row.get("view_name") == "prompt"]
    query_rows = [row for row in rows if row.get("view_name") == "eval_rephrase"]
    prompt_by_source = {int(row["source_index"]): row for row in prompt_rows}
    query_rows = [row for row in query_rows if int(row["source_index"]) in prompt_by_source]
    if not prompt_rows or not query_rows:
        return {
            "trace_top1_q_r": None,
            "family_margin_q_r": None,
            "count": 0,
        }
    with torch.no_grad():
        trace_raw = torch.stack([row["relation_raw_factor"] for row in prompt_rows], dim=0)
        query_raw = torch.stack([row["relation_raw_factor"] for row in query_rows], dim=0)
        trace_emb = F.normalize(encoder(trace_raw), p=2, dim=-1)
        query_emb = F.normalize(encoder(query_raw), p=2, dim=-1)
        if relation_whiten_eps is not None:
            trace_emb, query_emb = whiten_trace_query_embeddings(
                trace_emb,
                query_emb,
                eps=relation_whiten_eps,
            )
        scores = query_emb @ trace_emb.T
        can_score_subject = all(isinstance(row.get("subject_factor"), torch.Tensor) for row in prompt_rows + query_rows)
        subject_scores = None
        if can_score_subject:
            trace_subject = F.normalize(torch.stack([row["subject_factor"] for row in prompt_rows], dim=0), p=2, dim=-1)
            query_subject = F.normalize(torch.stack([row["subject_factor"] for row in query_rows], dim=0), p=2, dim=-1)
            subject_scores = query_subject @ trace_subject.T
    source_to_trace_index = {int(row["source_index"]): idx for idx, row in enumerate(prompt_rows)}
    trace_top1_hits = []
    family_margin_hits = []
    margins = []
    candidate_scores = []
    subject_top1_hits = []
    relation_top1_hits = []
    target_both_top1_hits = []
    wrong_winner_collisions = []
    wrong_winner_same_target_relation_events = []
    wrong_winner_cross_target_relation_events = []
    rank_independent_baselines = []
    rank_independent_same_target_relation_baselines = []
    rank_independent_cross_target_relation_baselines = []
    subject_target_ranks = []
    relation_target_ranks = []
    for query_idx, query_row in enumerate(query_rows):
        target_idx = source_to_trace_index[int(query_row["source_index"])]
        score_row = scores[query_idx]
        sorted_scores, sorted_indices = torch.sort(score_row, descending=True)
        top_idx = int(sorted_indices[0].item())
        runner_score = float(sorted_scores[1].item()) if sorted_scores.numel() > 1 else 0.0
        top_margin = float(sorted_scores[0].item() - runner_score)
        trace_top1_hits.append(1.0 if top_idx == target_idx and top_margin > relation_margin_threshold else 0.0)

        target_score = float(score_row[target_idx].item())
        target_relation_id = relation_id(query_row)
        outside_scores = [
            float(score_row[idx].item())
            for idx, prompt_row in enumerate(prompt_rows)
            if relation_id(prompt_row) != target_relation_id
        ]
        outside_runner = max(outside_scores) if outside_scores else 0.0
        family_margin = float(target_score - outside_runner)
        candidate_scores.append(target_score)
        margins.append(family_margin)
        family_margin_hits.append(
            1.0
            if -target_score < relation_energy_threshold and family_margin > relation_margin_threshold
            else 0.0
        )
        relation_top1_hits.append(1.0 if top_idx == target_idx and top_margin > relation_margin_threshold else 0.0)
        relation_rank = int((score_row > score_row[target_idx]).sum().item()) + 1
        relation_target_ranks.append(float(relation_rank))
        if subject_scores is not None:
            subject_row = subject_scores[query_idx]
            subject_sorted, subject_indices = torch.sort(subject_row, descending=True)
            subject_top_idx = int(subject_indices[0].item())
            subject_runner_score = float(subject_sorted[1].item()) if subject_sorted.numel() > 1 else 0.0
            subject_margin = float(subject_sorted[0].item() - subject_runner_score)
            subject_top1 = subject_top_idx == target_idx and subject_margin > 0.0
            relation_top1 = top_idx == target_idx and top_margin > relation_margin_threshold
            subject_top1_hits.append(1.0 if subject_top1 else 0.0)
            target_both_top1_hits.append(1.0 if subject_top1 and relation_top1 else 0.0)
            subject_rank = int((subject_row > subject_row[target_idx]).sum().item()) + 1
            subject_target_ranks.append(float(subject_rank))
            wrong_collision = (
                subject_top_idx == top_idx
                and subject_top_idx != target_idx
                and subject_margin > 0.0
                and top_margin > relation_margin_threshold
            )
            wrong_winner_collisions.append(1.0 if wrong_collision else 0.0)
            num_impostors = max(1, len(prompt_rows) - 1)
            subject_wrong_winner = 1.0 if subject_top_idx != target_idx and subject_margin > 0.0 else 0.0
            relation_wrong_winner = 1.0 if top_idx != target_idx and top_margin > relation_margin_threshold else 0.0
            rank_independent_baselines.append(subject_wrong_winner * relation_wrong_winner / num_impostors)
            same_target_relation_count = sum(
                1
                for prompt_idx, prompt_row in enumerate(prompt_rows)
                if prompt_idx != target_idx and relation_id(prompt_row) == target_relation_id
            )
            cross_target_relation_count = max(0, num_impostors - same_target_relation_count)
            wrong_same_relation = (
                wrong_collision
                and relation_id(prompt_rows[subject_top_idx]) == target_relation_id
            )
            wrong_winner_same_target_relation_events.append(1.0 if wrong_same_relation else 0.0)
            wrong_winner_cross_target_relation_events.append(
                1.0 if wrong_collision and not wrong_same_relation else 0.0
            )
            rank_independent_same_target_relation_baselines.append(
                subject_wrong_winner
                * relation_wrong_winner
                * same_target_relation_count
                / max(1, num_impostors * num_impostors)
            )
            rank_independent_cross_target_relation_baselines.append(
                subject_wrong_winner
                * relation_wrong_winner
                * cross_target_relation_count
                / max(1, num_impostors * num_impostors)
            )
    wrong_winner_collision = None if not wrong_winner_collisions else float(sum(wrong_winner_collisions) / len(wrong_winner_collisions))
    rank_independent_baseline = None if not rank_independent_baselines else float(sum(rank_independent_baselines) / len(rank_independent_baselines))
    same_relation_collision = None if not wrong_winner_same_target_relation_events else float(
        sum(wrong_winner_same_target_relation_events) / len(wrong_winner_same_target_relation_events)
    )
    cross_relation_collision = None if not wrong_winner_cross_target_relation_events else float(
        sum(wrong_winner_cross_target_relation_events) / len(wrong_winner_cross_target_relation_events)
    )
    same_relation_baseline = None if not rank_independent_same_target_relation_baselines else float(
        sum(rank_independent_same_target_relation_baselines) / len(rank_independent_same_target_relation_baselines)
    )
    cross_relation_baseline = None if not rank_independent_cross_target_relation_baselines else float(
        sum(rank_independent_cross_target_relation_baselines) / len(rank_independent_cross_target_relation_baselines)
    )
    return {
        "count": len(query_rows),
        "trace_top1_q_r": float(sum(trace_top1_hits) / len(trace_top1_hits)),
        "family_margin_q_r": float(sum(family_margin_hits) / len(family_margin_hits)),
        "candidate_score_mean": float(sum(candidate_scores) / len(candidate_scores)),
        "family_margin_mean": float(sum(margins) / len(margins)),
        "subject_top1_q_s": None if not subject_top1_hits else float(sum(subject_top1_hits) / len(subject_top1_hits)),
        "relation_top1_q_r": None if not relation_top1_hits else float(sum(relation_top1_hits) / len(relation_top1_hits)),
        "target_both_top1_rate": None if not target_both_top1_hits else float(sum(target_both_top1_hits) / len(target_both_top1_hits)),
        "wrong_winner_collision_kappa": wrong_winner_collision,
        "rank_independent_collision_baseline": rank_independent_baseline,
        "rank_collision_excess": None
        if wrong_winner_collision is None or rank_independent_baseline is None
        else float(wrong_winner_collision - rank_independent_baseline),
        "rank_collision_ratio_to_independent": None
        if wrong_winner_collision is None or rank_independent_baseline is None or rank_independent_baseline <= 0.0
        else float(wrong_winner_collision / rank_independent_baseline),
        "wrong_winner_same_target_relation_rate": same_relation_collision,
        "wrong_winner_cross_target_relation_rate": cross_relation_collision,
        "rank_independent_same_target_relation_baseline": same_relation_baseline,
        "rank_independent_cross_target_relation_baseline": cross_relation_baseline,
        "rank_collision_same_target_relation_excess": None
        if same_relation_collision is None or same_relation_baseline is None
        else float(same_relation_collision - same_relation_baseline),
        "rank_collision_cross_target_relation_excess": None
        if cross_relation_collision is None or cross_relation_baseline is None
        else float(cross_relation_collision - cross_relation_baseline),
        "rank_collision_same_target_relation_ratio": None
        if same_relation_collision is None or same_relation_baseline is None or same_relation_baseline <= 0.0
        else float(same_relation_collision / same_relation_baseline),
        "rank_collision_cross_target_relation_ratio": None
        if cross_relation_collision is None or cross_relation_baseline is None or cross_relation_baseline <= 0.0
        else float(cross_relation_collision / cross_relation_baseline),
        "subject_rank_mean": None if not subject_target_ranks else float(sum(subject_target_ranks) / len(subject_target_ranks)),
        "relation_rank_mean": None if not relation_target_ranks else float(sum(relation_target_ranks) / len(relation_target_ranks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--dev_size", type=int, default=50)
    parser.add_argument("--test_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extract_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--pair_weight", type=float, default=1.0)
    parser.add_argument("--relation_contrastive_weight", type=float, default=0.25)
    parser.add_argument("--relation_classification_weight", type=float, default=0.25)
    parser.add_argument("--rank_collision_weight", type=float, default=0.0)
    parser.add_argument("--rank_collision_beta", type=float, default=20.0)
    parser.add_argument("--rank_collision_excess_only", action="store_true")
    parser.add_argument("--rank_collision_query_views", default="address_rephrase")
    parser.add_argument("--rank_collision_no_candidate_normalization", action="store_true")
    parser.add_argument("--within_cluster_weight", type=float, default=0.0)
    parser.add_argument("--within_cluster_margin", type=float, default=0.2)
    parser.add_argument("--within_cluster_hard_k", type=int, default=4)
    parser.add_argument("--within_cluster_easy_k", type=int, default=2)
    parser.add_argument("--within_cluster_query_views", default="address_rephrase")
    parser.add_argument("--hard_winner_weight", type=float, default=0.0)
    parser.add_argument("--hard_winner_margin", type=float, default=0.3)
    parser.add_argument("--hard_winner_bank_size", type=int, default=512)
    parser.add_argument("--hard_winner_mining_frequency", type=int, default=2)
    parser.add_argument("--hard_winner_query_views", default="address_rephrase")
    parser.add_argument("--hard_winner_whiten_eps", type=float, default=1.0e-4)
    parser.add_argument("--hard_winner_topk", type=int, default=1)
    parser.add_argument("--hard_winner_max_per_query", type=int, default=1)
    parser.add_argument("--hard_winner_same_relation_only", action="store_true")
    parser.add_argument("--hard_winner_loss_impl", choices=["hinge", "infonce"], default="hinge")
    parser.add_argument("--hard_winner_temperature", type=float, default=0.2)
    parser.add_argument("--dev_relation_whiten", action="store_true")
    parser.add_argument("--dev_relation_whiten_eps", type=float, default=1.0e-4)
    parser.add_argument(
        "--selection_metric",
        choices=[
            "family_q_r",
            "trace_q_r",
            "target_fire_minus_kappa",
            "minus_same_relation_kappa",
        ],
        default="family_q_r",
    )
    parser.add_argument("--relation_margin_threshold", type=float, default=0.0)
    parser.add_argument("--relation_energy_threshold", type=float, default=0.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams_class = resolve_hparams_class("HOPEDIT")
    hparams = hparams_class.from_hparams(args.hparams_dir)
    hparams.factored_relation_encoder_impl = "identity"
    hparams.factored_relation_encoder_checkpoint = None
    hparams.factored_relation_encoder_steps = 0

    records, dataset_file = load_normalized_records(
        args.data_dir,
        "CounterFact",
        0,
        data_file=args.data_file,
    )
    split = build_same_relation_split(
        records,
        test_size=args.test_size,
        dev_size=args.dev_size,
        train_size=args.train_size,
        seed=args.seed,
    )
    write_json(
        output_dir / "relation_encoder_split.json",
        {
            **split["metadata"],
            "dataset_file": str(dataset_file),
            "train_indices": [record["source_index"] for record in split["train"]],
            "dev_indices": [record["source_index"] for record in split["dev"]],
            "test_indices": [record["source_index"] for record in split["test"]],
        },
    )

    editor = BaseEditor.from_hparams(hparams)
    controller = HopEditController(model=editor.model, tok=editor.tok, hparams=hparams)
    train_rows = extract_relation_raw_factors(
        controller,
        relation_view_rows(split["train"], include_eval_rephrase=False),
        batch_size=args.extract_batch_size,
    )
    dev_rows = extract_relation_raw_factors(
        controller,
        relation_view_rows(split["dev"], include_eval_rephrase=True),
        batch_size=args.extract_batch_size,
    )
    if not train_rows:
        raise RuntimeError("No relation training rows were extracted.")

    raw_factors = torch.stack([row["relation_raw_factor"] for row in train_rows], dim=0)
    input_dim = int(raw_factors.shape[1])
    relation_ids = sorted({relation_id(row) for row in train_rows if relation_id(row) is not None})
    relation_to_label = {rid: idx for idx, rid in enumerate(relation_ids)}
    relation_labels = torch.tensor([relation_to_label[relation_id(row)] for row in train_rows], dtype=torch.long)
    pair_ids = [str(row["source_index"]) for row in train_rows]
    relation_group_ids = [str(row["relation_id"]) for row in train_rows]
    rank_collision_query_views = {
        view_name.strip()
        for view_name in str(args.rank_collision_query_views).split(",")
        if view_name.strip()
    }
    within_cluster_query_views = {
        view_name.strip()
        for view_name in str(args.within_cluster_query_views).split(",")
        if view_name.strip()
    }
    hard_winner_query_views = {
        view_name.strip()
        for view_name in str(args.hard_winner_query_views).split(",")
        if view_name.strip()
    }

    encoder = FactoredRelationResidualEncoder(input_dim, args.hidden_dim)
    classifier = nn.Linear(input_dim, len(relation_ids))
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=0.0)

    best_payload = None
    history = []
    hard_winner_mining_payload: dict[str, Any] | None = None
    hard_winner_triplets: list[tuple[int, int, int]] = []
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        encoded = F.normalize(encoder(raw_factors), p=2, dim=-1)
        pair_loss = supervised_contrastive_loss(encoded, pair_ids, temperature=args.temperature)
        relation_loss = supervised_contrastive_loss(encoded, relation_group_ids, temperature=args.temperature)
        logits = classifier(encoded)
        classification_loss = F.cross_entropy(logits, relation_labels)
        rank_loss = None
        rank_loss_payload = None
        if float(args.rank_collision_weight) > 0.0:
            rank_loss_payload = rank_collision_loss(
                encoded,
                train_rows,
                query_view_names=rank_collision_query_views,
                beta=args.rank_collision_beta,
                excess_only=bool(args.rank_collision_excess_only),
                normalize_by_candidates=not bool(args.rank_collision_no_candidate_normalization),
            )
            rank_loss = None if rank_loss_payload is None else rank_loss_payload["loss"]
        within_cluster_payload = None
        within_cluster_loss = None
        if float(args.within_cluster_weight) > 0.0:
            within_cluster_payload = within_cluster_disambiguation_loss(
                encoded,
                train_rows,
                query_view_names=within_cluster_query_views,
                margin=args.within_cluster_margin,
                hard_k=args.within_cluster_hard_k,
                easy_k=args.within_cluster_easy_k,
                rng=random.Random((args.seed + 1) * 1_000_003 + epoch),
            )
            within_cluster_loss = None if within_cluster_payload is None else within_cluster_payload["loss"]
        hard_winner_payload = None
        hard_winner_loss = None
        if float(args.hard_winner_weight) > 0.0:
            mining_frequency = max(1, int(args.hard_winner_mining_frequency))
            if hard_winner_mining_payload is None or (epoch - 1) % mining_frequency == 0:
                hard_winner_mining_payload = mine_hard_winner_triplets(
                    encoded,
                    train_rows,
                    query_view_names=hard_winner_query_views,
                    bank_size=args.hard_winner_bank_size,
                    whiten_eps=args.hard_winner_whiten_eps,
                    relation_margin_threshold=args.relation_margin_threshold,
                    subject_margin_threshold=0.0,
                    topk=args.hard_winner_topk,
                    max_triplets_per_query=args.hard_winner_max_per_query,
                    same_relation_only=bool(args.hard_winner_same_relation_only),
                    rng=random.Random((args.seed + 11) * 1_000_003 + epoch),
                )
                hard_winner_triplets = list(hard_winner_mining_payload.get("triplets") or [])
            if args.hard_winner_loss_impl == "infonce":
                hard_winner_payload = hard_winner_infonce_loss(
                    encoded,
                    hard_winner_triplets,
                    temperature=args.hard_winner_temperature,
                )
            else:
                hard_winner_payload = hard_winner_triplet_loss(
                    encoded,
                    hard_winner_triplets,
                    margin=args.hard_winner_margin,
                )
            hard_winner_loss = None if hard_winner_payload is None else hard_winner_payload["loss"]
        loss = torch.zeros((), dtype=encoded.dtype)
        if pair_loss is not None:
            loss = loss + args.pair_weight * pair_loss
        if relation_loss is not None:
            loss = loss + args.relation_contrastive_weight * relation_loss
        loss = loss + args.relation_classification_weight * classification_loss
        if rank_loss is not None:
            loss = loss + args.rank_collision_weight * rank_loss
        if within_cluster_loss is not None:
            loss = loss + args.within_cluster_weight * within_cluster_loss
        if hard_winner_loss is not None:
            loss = loss + args.hard_winner_weight * hard_winner_loss
        loss.backward()
        optimizer.step()

        encoder.eval()
        dev_metrics = evaluate_relation_encoder(
            encoder,
            dev_rows,
            relation_margin_threshold=args.relation_margin_threshold,
            relation_energy_threshold=args.relation_energy_threshold,
            relation_whiten_eps=args.dev_relation_whiten_eps if args.dev_relation_whiten else None,
        )
        if args.selection_metric == "trace_q_r":
            selection_score = dev_metrics.get("trace_top1_q_r")
        elif args.selection_metric == "target_fire_minus_kappa":
            target_fire = dev_metrics.get("target_both_top1_rate")
            kappa = dev_metrics.get("wrong_winner_collision_kappa")
            selection_score = None if target_fire is None or kappa is None else float(target_fire - kappa)
        elif args.selection_metric == "minus_same_relation_kappa":
            same_kappa = dev_metrics.get("wrong_winner_same_target_relation_rate")
            selection_score = None if same_kappa is None else -float(same_kappa)
        else:
            selection_score = dev_metrics.get("family_margin_q_r")
        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu().item()),
            "pair_loss": None if pair_loss is None else float(pair_loss.detach().cpu().item()),
            "relation_contrastive_loss": None if relation_loss is None else float(relation_loss.detach().cpu().item()),
            "classification_loss": float(classification_loss.detach().cpu().item()),
            "rank_collision_loss": None if rank_loss is None else float(rank_loss.detach().cpu().item()),
            "rank_collision_raw_loss": None
            if rank_loss_payload is None
            else float(rank_loss_payload["raw_loss"].detach().cpu().item()),
            "rank_collision_scale": None if rank_loss_payload is None else rank_loss_payload["scale"],
            "rank_collision_num_impostors": None
            if rank_loss_payload is None
            else rank_loss_payload["num_impostors"],
            "rank_collision_joint_mean": None
            if rank_loss_payload is None
            else float(rank_loss_payload["joint_collision_mean"].detach().cpu().item()),
            "rank_collision_independent_baseline_mean": None
            if rank_loss_payload is None or rank_loss_payload["independent_baseline_mean"] is None
            else float(rank_loss_payload["independent_baseline_mean"].detach().cpu().item()),
            "within_cluster_loss": None
            if within_cluster_loss is None
            else float(within_cluster_loss.detach().cpu().item()),
            "within_cluster_eligible_queries": None
            if within_cluster_payload is None
            else within_cluster_payload["eligible_queries"],
            "within_cluster_total_queries": None
            if within_cluster_payload is None
            else within_cluster_payload["total_queries"],
            "within_cluster_skipped_no_same_cluster": None
            if within_cluster_payload is None
            else within_cluster_payload["skipped_no_same_cluster"],
            "within_cluster_skipped_no_negatives": None
            if within_cluster_payload is None
            else within_cluster_payload["skipped_no_negatives"],
            "within_cluster_violation_rate": None
            if within_cluster_payload is None
            else within_cluster_payload["violation_rate"],
            "within_cluster_positive_distance_mean": None
            if within_cluster_payload is None
            else within_cluster_payload["positive_distance_mean"],
            "within_cluster_hardest_negative_distance_mean": None
            if within_cluster_payload is None
            else within_cluster_payload["hardest_negative_distance_mean"],
            "within_cluster_same_negative_count_mean": None
            if within_cluster_payload is None
            else within_cluster_payload["same_negative_count_mean"],
            "within_cluster_cross_negative_count_mean": None
            if within_cluster_payload is None
            else within_cluster_payload["cross_negative_count_mean"],
            "hard_winner_loss": None
            if hard_winner_loss is None
            else float(hard_winner_loss.detach().cpu().item()),
            "hard_winner_loss_impl": args.hard_winner_loss_impl,
            "hard_winner_temperature": args.hard_winner_temperature,
            "hard_winner_triplet_count": None
            if hard_winner_payload is None
            else hard_winner_payload["triplet_count"],
            "hard_winner_query_count": None
            if hard_winner_payload is None
            else hard_winner_payload.get("query_count"),
            "hard_winner_negative_count_mean": None
            if hard_winner_payload is None
            else hard_winner_payload.get("negative_count_mean"),
            "hard_winner_violation_rate": None
            if hard_winner_payload is None
            else hard_winner_payload.get("violation_rate"),
            "hard_winner_positive_distance_mean": None
            if hard_winner_payload is None
            else hard_winner_payload["positive_distance_mean"],
            "hard_winner_wrong_distance_mean": None
            if hard_winner_payload is None
            else hard_winner_payload["wrong_distance_mean"],
            "hard_winner_mining": None
            if hard_winner_mining_payload is None
            else {
                key: value
                for key, value in hard_winner_mining_payload.items()
                if key != "triplets"
            },
            "selection_metric": args.selection_metric,
            "selection_score": None if selection_score is None else float(selection_score),
            "dev": dev_metrics,
        }
        history.append(row)
        if selection_score is not None and (
            best_payload is None or float(selection_score) > float(best_payload["best_selection_score"])
        ):
            best_payload = {
                "epoch": epoch,
                "encoder_state_dict": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
                "classifier_state_dict": {key: value.detach().cpu() for key, value in classifier.state_dict().items()},
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "relation_ids": relation_ids,
                "train_count": len(split["train"]),
                "dev_count": len(split["dev"]),
                "test_count": len(split["test"]),
                "train_row_count": len(train_rows),
                "dev_row_count": len(dev_rows),
                "best_dev_q_r": dev_metrics.get("family_margin_q_r"),
                "best_dev_family_q_r": dev_metrics.get("family_margin_q_r"),
                "best_dev_trace_top1_q_r": dev_metrics.get("trace_top1_q_r"),
                "selection_metric": args.selection_metric,
                "best_selection_score": float(selection_score),
                "dev_metrics": dev_metrics,
                "split_metadata": split["metadata"],
                "relation_match_rule": "subject_candidate",
                "rank_collision_weight": args.rank_collision_weight,
                "rank_collision_beta": args.rank_collision_beta,
                "rank_collision_excess_only": bool(args.rank_collision_excess_only),
                "rank_collision_query_views": sorted(rank_collision_query_views),
                "rank_collision_normalize_by_candidates": not bool(args.rank_collision_no_candidate_normalization),
                "within_cluster_weight": args.within_cluster_weight,
                "within_cluster_margin": args.within_cluster_margin,
                "within_cluster_hard_k": args.within_cluster_hard_k,
                "within_cluster_easy_k": args.within_cluster_easy_k,
                "within_cluster_query_views": sorted(within_cluster_query_views),
                "hard_winner_weight": args.hard_winner_weight,
                "hard_winner_margin": args.hard_winner_margin,
                "hard_winner_bank_size": args.hard_winner_bank_size,
                "hard_winner_mining_frequency": args.hard_winner_mining_frequency,
                "hard_winner_query_views": sorted(hard_winner_query_views),
                "hard_winner_whiten_eps": args.hard_winner_whiten_eps,
                "hard_winner_topk": args.hard_winner_topk,
                "hard_winner_max_per_query": args.hard_winner_max_per_query,
                "hard_winner_same_relation_only": bool(args.hard_winner_same_relation_only),
                "hard_winner_loss_impl": args.hard_winner_loss_impl,
                "hard_winner_temperature": args.hard_winner_temperature,
                "dev_relation_whiten": bool(args.dev_relation_whiten),
                "dev_relation_whiten_eps": args.dev_relation_whiten_eps,
            }
        print(json.dumps(row))

    if best_payload is None:
        raise RuntimeError("No valid dev metric was produced; cannot save a selected relation encoder.")
    checkpoint_path = output_dir / "relation_encoder.pt"
    torch.save(best_payload, checkpoint_path)
    write_json(output_dir / "relation_encoder_train_history.json", history)
    write_json(
        output_dir / "relation_encoder_summary.json",
        {
            "checkpoint_path": str(checkpoint_path),
            "dataset_file": str(dataset_file),
            "best_epoch": best_payload["epoch"],
            "best_dev_family_q_r": best_payload["best_dev_family_q_r"],
            "best_dev_trace_top1_q_r": best_payload["best_dev_trace_top1_q_r"],
            "selection_metric": args.selection_metric,
            "best_selection_score": best_payload["best_selection_score"],
            "split_metadata": split["metadata"],
            "rank_collision_weight": args.rank_collision_weight,
            "rank_collision_beta": args.rank_collision_beta,
            "rank_collision_excess_only": bool(args.rank_collision_excess_only),
            "rank_collision_query_views": sorted(rank_collision_query_views),
            "rank_collision_normalize_by_candidates": not bool(args.rank_collision_no_candidate_normalization),
            "within_cluster_weight": args.within_cluster_weight,
            "within_cluster_margin": args.within_cluster_margin,
            "within_cluster_hard_k": args.within_cluster_hard_k,
            "within_cluster_easy_k": args.within_cluster_easy_k,
            "within_cluster_query_views": sorted(within_cluster_query_views),
            "hard_winner_weight": args.hard_winner_weight,
            "hard_winner_margin": args.hard_winner_margin,
            "hard_winner_bank_size": args.hard_winner_bank_size,
            "hard_winner_mining_frequency": args.hard_winner_mining_frequency,
            "hard_winner_query_views": sorted(hard_winner_query_views),
            "hard_winner_whiten_eps": args.hard_winner_whiten_eps,
            "hard_winner_topk": args.hard_winner_topk,
            "hard_winner_max_per_query": args.hard_winner_max_per_query,
            "hard_winner_same_relation_only": bool(args.hard_winner_same_relation_only),
            "hard_winner_loss_impl": args.hard_winner_loss_impl,
            "hard_winner_temperature": args.hard_winner_temperature,
            "dev_relation_whiten": bool(args.dev_relation_whiten),
            "dev_relation_whiten_eps": args.dev_relation_whiten_eps,
        },
    )


if __name__ == "__main__":
    main()
