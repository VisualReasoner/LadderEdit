"""Score-only proof of concept for proof-carrying edit capsules.

This runner does not mutate the base model.  It reuses HopEdit's subject and
relation factor extraction, calibrates a global guard certificate on disjoint
CounterFact edits, and evaluates strict, whitened, and capsule-calibrated
routing decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor
from easyeditor.models.hopedit.capsule_certificates import (
    assert_disjoint_source_indices,
    build_global_capsule_certificate,
    route_capsules,
)
from easyeditor.models.hopedit.hopedit_main import HopEditController
from examples.analyze_hopedit_orthogonalized_factors import apply_relation_whitener, build_relation_whitener
from examples.edit_experiment_utils import load_normalized_records, resolve_hparams_class, write_json
from examples.probe_hopedit_relation_layer_sweep import analyze_scores, seed_everything


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in re.split(r"[:,\s]+", str(value)) if part.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in re.split(r"[:,\s]+", str(value)) if part.strip()]


def safe_mean(values: list[float | int | bool | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def infer_counterfact_subject_from_prompt(prompt: str | None) -> str | None:
    """Best-effort subject extraction for CounterFact neighborhood prompts."""

    text = str(prompt or "").strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip(" .")
    of_match = re.search(r"\bof\s+(.+?)\s+(?:is|was|are|were|has|had|,|$)", text)
    if of_match:
        candidate = of_match.group(1).strip(" ,.")
        if candidate and not candidate.lower().startswith(("the ", "a ", "an ")):
            return candidate
    if "," in text:
        candidate = text.split(",", 1)[0].strip(" ,.")
        if candidate:
            return candidate
    verb_match = re.match(r"(.+?)\s+(?:is|was|are|were|has|had)\b", text)
    if verb_match:
        candidate = verb_match.group(1).strip(" ,.")
        if candidate and not candidate.lower().startswith(("the ", "where ", "what ")):
            return candidate
    return None


def select_split(records: list[dict[str, Any]], *, start: int, size: int, name: str) -> list[dict[str, Any]]:
    stop = int(start) + int(size)
    if start < 0 or size <= 0 or stop > len(records):
        raise ValueError(f"Invalid {name} split start={start}, size={size}, dataset_len={len(records)}")
    return records[start:stop]


def assert_disjoint_splits(calib_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> None:
    assert_disjoint_source_indices(calib_records, eval_records)


def extract_factor_rows(
    controller: HopEditController,
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


def stack_factor(rows: list[dict[str, Any]], key: str) -> torch.Tensor:
    dim = None
    for row in rows:
        value = row.get(key)
        if isinstance(value, torch.Tensor):
            dim = int(value.numel())
            break
    if dim is None:
        raise RuntimeError(f"All rows are missing factor {key!r}.")
    vectors = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, torch.Tensor):
            vectors.append(value.detach().float().cpu())
        else:
            vectors.append(torch.zeros(dim, dtype=torch.float32))
    if not vectors:
        raise RuntimeError(f"No rows available for factor {key!r}.")
    return torch.stack(vectors, dim=0)


def row_zscores(scores: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(float(eps))
    return (scores - mean) / std


def row_margin_scores(scores: torch.Tensor) -> torch.Tensor:
    """Per-candidate gap to the best competing candidate in the same query row."""

    if int(scores.shape[1]) <= 1:
        return scores
    top2_values, top2_indices = torch.topk(scores, k=2, dim=1)
    top1_values = top2_values[:, 0:1]
    top2_values_only = top2_values[:, 1:2]
    top1_indices = top2_indices[:, 0:1]
    candidate_indices = torch.arange(scores.shape[1], device=scores.device).view(1, -1)
    runner = torch.where(candidate_indices == top1_indices, top2_values_only, top1_values)
    return scores - runner


def rank_feature(scores: torch.Tensor) -> torch.Tensor:
    n = int(scores.shape[1])
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    rank_values = torch.arange(1, n + 1, device=scores.device, dtype=torch.float32).view(1, -1)
    ranks.scatter_(1, order, rank_values.expand_as(order).float())
    if n <= 1:
        return torch.ones_like(ranks)
    return 1.0 - torch.log(ranks) / math.log(n + 1.0)


def pair_feature_tensor(subject_scores: torch.Tensor, relation_scores: torch.Tensor) -> torch.Tensor:
    subject_z = row_zscores(subject_scores)
    relation_z = row_zscores(relation_scores)
    subject_margin = row_margin_scores(subject_z)
    relation_margin = row_margin_scores(relation_z)
    subject_rank = rank_feature(subject_z)
    relation_rank = rank_feature(relation_z)
    subject_norm = subject_z / subject_z.max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
    relation_norm = relation_z / relation_z.max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
    return torch.stack(
        [
            subject_z,
            relation_z,
            subject_margin,
            relation_margin,
            subject_rank,
            relation_rank,
            subject_norm,
            relation_norm,
        ],
        dim=-1,
    )


class CapsuleScoreCombiner(nn.Module):
    def __init__(self, input_dim: int, *, model_type: str, hidden_dim: int):
        super().__init__()
        if model_type == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif model_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported learned model type: {model_type!r}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


CAPSULE_FEATURE_NAMES = [
    "subject_z",
    "relation_z",
    "subject_margin",
    "relation_margin",
    "subject_rank",
    "relation_rank",
    "subject_norm",
    "relation_norm",
]


def learned_score_state(model: CapsuleScoreCombiner | None) -> dict[str, Any] | None:
    if model is None:
        return None
    if isinstance(model.net, nn.Linear):
        return {
            "model_type": "linear",
            "feature_names": CAPSULE_FEATURE_NAMES,
            "feature_weights": [float(value) for value in model.net.weight.detach().view(-1).cpu().tolist()],
            "feature_bias": float(model.net.bias.detach().view(-1)[0].cpu().item()),
        }
    return {
        "model_type": "mlp",
        "feature_names": CAPSULE_FEATURE_NAMES,
        "state_dict": {
            key: value.detach().cpu().tolist()
            for key, value in model.state_dict().items()
        },
    }


def capsule_score_matrix(
    subject_scores: torch.Tensor,
    relation_scores: torch.Tensor,
    *,
    score_family: str,
    learned_model: CapsuleScoreCombiner | None = None,
) -> torch.Tensor:
    family = str(score_family or "min_z").strip().lower()
    subject_z = row_zscores(subject_scores)
    relation_z = row_zscores(relation_scores)
    if family == "min_z":
        return torch.minimum(subject_z, relation_z)
    if family in {"margin_min_z", "min_margin_z"}:
        return torch.minimum(row_margin_scores(subject_z), row_margin_scores(relation_z))
    if family in {"subject_margin", "subject_margin_z"}:
        return row_margin_scores(subject_z)
    if family in {"relation_margin", "relation_margin_z"}:
        return row_margin_scores(relation_z)
    if family in {"learned_linear", "learned_mlp"}:
        if learned_model is None:
            raise ValueError(f"score_family={score_family!r} requires a learned_model")
        features = pair_feature_tensor(subject_scores, relation_scores)
        flat = features.reshape(-1, features.shape[-1])
        with torch.no_grad():
            logits = learned_model.eval()(flat).detach().float().cpu()
        return logits.reshape(features.shape[0], features.shape[1])
    raise ValueError(f"Unsupported capsule score family: {score_family!r}")


def build_trace_bank(
    controller: HopEditController,
    records: list[dict[str, Any]],
    *,
    subject_layer: int,
    relation_layer: int,
    subject_pooling: str,
    batch_size: int,
    whiten_eps: float,
) -> dict[str, Any]:
    prompts = [str(record.get("prompt") or "") for record in records]
    subjects = [record.get("subject") for record in records]
    objects = [record.get("target_new") for record in records]
    rows = extract_factor_rows(
        controller,
        prompts,
        subjects,
        objects,
        subject_layer=subject_layer,
        relation_layer=relation_layer,
        subject_pooling=subject_pooling,
        batch_size=batch_size,
    )
    trace_subject = stack_factor(rows, "subject_factor")
    trace_relation = stack_factor(rows, "relation_factor")
    trace_ids = [f"capsule_{idx:05d}" for idx in range(len(records))]
    relation_by_id = {trace_id: trace_relation[idx] for idx, trace_id in enumerate(trace_ids)}
    whitener = build_relation_whitener(relation_by_id, trace_ids, eps=whiten_eps)
    trace_relation_whitened = torch.stack(
        [apply_relation_whitener(relation_by_id[trace_id], whitener) for trace_id in trace_ids],
        dim=0,
    )
    return {
        "records": records,
        "trace_ids": trace_ids,
        "trace_rows": rows,
        "trace_subject": trace_subject,
        "trace_relation": trace_relation,
        "trace_relation_whitened": trace_relation_whitened,
        "whitener": whitener,
        "relation_ids": [record.get("relation_id") for record in records],
    }


def score_queries(
    controller: HopEditController,
    bank: dict[str, Any],
    prompts: list[str],
    subjects: list[str | None],
    *,
    subject_layer: int,
    relation_layer: int,
    subject_pooling: str,
    batch_size: int,
    score_family: str,
    learned_model: CapsuleScoreCombiner | None = None,
) -> dict[str, Any]:
    rows = extract_factor_rows(
        controller,
        prompts,
        subjects,
        [None for _ in prompts],
        subject_layer=subject_layer,
        relation_layer=relation_layer,
        subject_pooling=subject_pooling,
        batch_size=batch_size,
    )
    query_subject = stack_factor(rows, "subject_factor")
    query_relation = stack_factor(rows, "relation_factor")
    subject_scores = query_subject @ bank["trace_subject"].T
    relation_scores = query_relation @ bank["trace_relation"].T
    query_relation_whitened = torch.stack(
        [apply_relation_whitener(query_relation[idx], bank["whitener"]) for idx in range(int(query_relation.shape[0]))],
        dim=0,
    )
    relation_scores_whitened = query_relation_whitened @ bank["trace_relation_whitened"].T
    capsule_scores = capsule_score_matrix(
        subject_scores,
        relation_scores_whitened,
        score_family=score_family,
        learned_model=learned_model,
    )
    return {
        "rows": rows,
        "subject_scores": subject_scores,
        "relation_scores": relation_scores,
        "relation_scores_whitened": relation_scores_whitened,
        "capsule_scores": capsule_scores,
        "subject_found_rate": safe_mean([row.get("subject_found") for row in rows]),
    }


def support_queries(records: list[dict[str, Any]]) -> tuple[list[str], list[str | None], list[int], list[str]]:
    prompts: list[str] = []
    subjects: list[str | None] = []
    targets: list[int] = []
    view_names: list[str] = []
    for idx, record in enumerate(records):
        for view_name, key in (("rewrite", "prompt"), ("address_rephrase", "address_rephrase_prompt")):
            prompt = record.get(key)
            if prompt:
                prompts.append(str(prompt))
                subjects.append(record.get("subject"))
                targets.append(idx)
                view_names.append(view_name)
        if not record.get("address_rephrase_prompt") and record.get("rephrase_prompt"):
            prompts.append(str(record.get("rephrase_prompt")))
            subjects.append(record.get("subject"))
            targets.append(idx)
            view_names.append("rephrase_support")
    return prompts, subjects, targets, view_names


def rephrase_queries(records: list[dict[str, Any]]) -> tuple[list[str], list[str | None], list[int]]:
    prompts = [str(record.get("rephrase_prompt") or record.get("prompt") or "") for record in records]
    subjects = [record.get("subject") for record in records]
    targets = list(range(len(records)))
    return prompts, subjects, targets


def locality_queries(records: list[dict[str, Any]], *, max_per_record: int) -> tuple[list[str], list[str | None], list[int]]:
    prompts: list[str] = []
    subjects: list[str | None] = []
    owner_indices: list[int] = []
    for idx, record in enumerate(records):
        locality = record.get("locality_prompt") or []
        if not isinstance(locality, list):
            locality = [locality]
        for prompt in locality[: max(0, int(max_per_record))]:
            if not prompt:
                continue
            prompts.append(str(prompt))
            subjects.append(infer_counterfact_subject_from_prompt(str(prompt)) or record.get("subject"))
            owner_indices.append(idx)
    return prompts, subjects, owner_indices


def collect_calibration_scores(
    controller: HopEditController,
    calib_bank: dict[str, Any],
    *,
    subject_layer: int,
    relation_layer: int,
    subject_pooling: str,
    batch_size: int,
    max_locality_per_record: int,
    score_family: str,
    learned_model: CapsuleScoreCombiner | None = None,
    support_row_indices: list[int] | None = None,
    locality_row_indices: list[int] | None = None,
) -> dict[str, Any]:
    support_prompts, support_subjects, support_targets, support_views = support_queries(calib_bank["records"])
    support = score_queries(
        controller,
        calib_bank,
        support_prompts,
        support_subjects,
        subject_layer=subject_layer,
        relation_layer=relation_layer,
        subject_pooling=subject_pooling,
        batch_size=batch_size,
        score_family=score_family,
        learned_model=learned_model,
    )
    support_scores = []
    guard_scores = []
    support_filter = set(range(len(support_targets))) if support_row_indices is None else set(int(idx) for idx in support_row_indices)
    for row_idx, target_idx in enumerate(support_targets):
        if row_idx not in support_filter:
            continue
        row = support["capsule_scores"][row_idx]
        support_scores.append(float(row[target_idx].item()))
        if row.numel() > 1:
            masked = row.clone()
            masked[target_idx] = float("-inf")
            guard_scores.append(float(torch.max(masked).item()))

    loc_prompts, loc_subjects, _ = locality_queries(calib_bank["records"], max_per_record=max_locality_per_record)
    locality_subject_found_rate = None
    if loc_prompts:
        locality = score_queries(
            controller,
            calib_bank,
            loc_prompts,
            loc_subjects,
            subject_layer=subject_layer,
            relation_layer=relation_layer,
            subject_pooling=subject_pooling,
            batch_size=batch_size,
            score_family=score_family,
            learned_model=learned_model,
        )
        locality_subject_found_rate = locality["subject_found_rate"]
        locality_filter = (
            set(range(len(loc_prompts))) if locality_row_indices is None else set(int(idx) for idx in locality_row_indices)
        )
        guard_scores.extend(
            float(torch.max(row).item())
            for row_idx, row in enumerate(locality["capsule_scores"])
            if row_idx in locality_filter
        )

    return {
        "support_scores": support_scores,
        "guard_scores": guard_scores,
        "support_views": support_views,
        "support_subject_found_rate": support["subject_found_rate"],
        "locality_subject_found_rate": locality_subject_found_rate,
        "num_support_queries": len(support_prompts),
        "num_locality_guard_queries": len(loc_prompts),
    }


def split_indices(count: int, *, train_frac: float, val_frac: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(int(seed))
    indices = np.arange(int(count))
    rng.shuffle(indices)
    train_stop = int(round(count * float(train_frac)))
    val_stop = int(round(count * (float(train_frac) + float(val_frac))))
    train = indices[:train_stop].astype(int).tolist()
    val = indices[train_stop:val_stop].astype(int).tolist()
    holdout = indices[val_stop:].astype(int).tolist()
    return train, val, holdout


def binary_auc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    labels = labels.detach().float().cpu()
    scores = scores.detach().float().cpu()
    pos = labels > 0.5
    neg = ~pos
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, int(scores.numel()) + 1, dtype=torch.float32)
    pos_rank_sum = float(ranks[pos].sum().item())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def gather_learning_examples(
    *,
    subject_scores: torch.Tensor,
    relation_scores: torch.Tensor,
    target_indices: list[int] | None,
    row_indices: list[int],
    hard_negatives: int,
    include_positive: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = pair_feature_tensor(subject_scores, relation_scores)
    base_scores = capsule_score_matrix(subject_scores, relation_scores, score_family="min_z")
    feature_rows = []
    labels = []
    for row_idx in row_indices:
        row_idx = int(row_idx)
        if row_idx < 0 or row_idx >= int(features.shape[0]):
            continue
        target_idx = None if target_indices is None else int(target_indices[row_idx])
        if include_positive and target_idx is not None:
            feature_rows.append(features[row_idx, target_idx])
            labels.append(1.0)
        candidate_scores = base_scores[row_idx].clone()
        if target_idx is not None:
            candidate_scores[target_idx] = float("-inf")
        k = min(max(1, int(hard_negatives)), int(candidate_scores.numel()))
        neg_indices = torch.topk(candidate_scores, k=k).indices.tolist()
        for neg_idx in neg_indices:
            if math.isfinite(float(candidate_scores[int(neg_idx)].item())):
                feature_rows.append(features[row_idx, int(neg_idx)])
                labels.append(0.0)
    if not feature_rows:
        raise ValueError("No training examples could be gathered for learned CapsuleEdit scorer.")
    return torch.stack(feature_rows, dim=0).float(), torch.tensor(labels, dtype=torch.float32)


def train_learned_score_combiner(
    controller: HopEditController,
    calib_bank: dict[str, Any],
    *,
    subject_layer: int,
    relation_layer: int,
    subject_pooling: str,
    batch_size: int,
    max_locality_per_record: int,
    model_type: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hard_negatives: int,
    train_frac: float,
    val_frac: float,
    patience: int,
    seed: int,
) -> dict[str, Any]:
    support_prompts, support_subjects, support_targets, _ = support_queries(calib_bank["records"])
    support = score_queries(
        controller,
        calib_bank,
        support_prompts,
        support_subjects,
        subject_layer=subject_layer,
        relation_layer=relation_layer,
        subject_pooling=subject_pooling,
        batch_size=batch_size,
        score_family="min_z",
    )
    loc_prompts, loc_subjects, _ = locality_queries(calib_bank["records"], max_per_record=max_locality_per_record)
    locality = score_queries(
        controller,
        calib_bank,
        loc_prompts,
        loc_subjects,
        subject_layer=subject_layer,
        relation_layer=relation_layer,
        subject_pooling=subject_pooling,
        batch_size=batch_size,
        score_family="min_z",
    )

    support_train, support_val, support_holdout = split_indices(
        len(support_targets),
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )
    loc_train, loc_val, loc_holdout = split_indices(
        len(loc_prompts),
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed + 17,
    )

    train_pos_x, train_pos_y = gather_learning_examples(
        subject_scores=support["subject_scores"],
        relation_scores=support["relation_scores_whitened"],
        target_indices=support_targets,
        row_indices=support_train,
        hard_negatives=hard_negatives,
        include_positive=True,
    )
    train_loc_x, train_loc_y = gather_learning_examples(
        subject_scores=locality["subject_scores"],
        relation_scores=locality["relation_scores_whitened"],
        target_indices=None,
        row_indices=loc_train,
        hard_negatives=hard_negatives,
        include_positive=False,
    )
    val_pos_x, val_pos_y = gather_learning_examples(
        subject_scores=support["subject_scores"],
        relation_scores=support["relation_scores_whitened"],
        target_indices=support_targets,
        row_indices=support_val,
        hard_negatives=hard_negatives,
        include_positive=True,
    )
    val_loc_x, val_loc_y = gather_learning_examples(
        subject_scores=locality["subject_scores"],
        relation_scores=locality["relation_scores_whitened"],
        target_indices=None,
        row_indices=loc_val,
        hard_negatives=hard_negatives,
        include_positive=False,
    )

    train_x = torch.cat([train_pos_x, train_loc_x], dim=0)
    train_y = torch.cat([train_pos_y, train_loc_y], dim=0)
    val_x = torch.cat([val_pos_x, val_loc_x], dim=0)
    val_y = torch.cat([val_pos_y, val_loc_y], dim=0)

    model = CapsuleScoreCombiner(train_x.shape[-1], model_type=model_type, hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    pos = float(train_y.sum().item())
    neg = float(train_y.numel() - train_y.sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_auc = -1.0
    best_epoch = -1
    stale = 0
    history = []
    generator = torch.Generator().manual_seed(int(seed))
    for epoch in range(int(epochs)):
        order = torch.randperm(train_x.shape[0], generator=generator)
        model.train()
        for start in range(0, int(order.numel()), 512):
            batch_idx = order[start : start + 512]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[batch_idx]), train_y[batch_idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())
            val_auc = binary_auc(val_y, val_logits)
        history.append({"epoch": epoch, "val_loss": val_loss, "val_auc": val_auc})
        score = -1.0 if val_auc is None else float(val_auc)
        if score > best_auc + 1.0e-5:
            best_auc = score
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "metadata": {
            "model_type": model_type,
            "hidden_dim": int(hidden_dim),
            "epochs_run": len(history),
            "best_epoch": int(best_epoch),
            "best_val_auc": None if best_auc < 0 else float(best_auc),
            "train_examples": int(train_x.shape[0]),
            "val_examples": int(val_x.shape[0]),
            "support_train_queries": len(support_train),
            "support_val_queries": len(support_val),
            "support_holdout_queries": len(support_holdout),
            "locality_train_queries": len(loc_train),
            "locality_val_queries": len(loc_val),
            "locality_holdout_queries": len(loc_holdout),
            "history": history,
        },
        "support_holdout_indices": support_holdout,
        "locality_holdout_indices": loc_holdout,
    }


def hard_and_decisions(subject_scores: torch.Tensor, relation_scores: torch.Tensor) -> list[int | None]:
    decisions = []
    for idx in range(int(subject_scores.shape[0])):
        subject_top = int(torch.argmax(subject_scores[idx]).item())
        relation_top = int(torch.argmax(relation_scores[idx]).item())
        decisions.append(subject_top if subject_top == relation_top else None)
    return decisions


def evaluate_capsule_decisions(
    *,
    score_matrix: torch.Tensor,
    trace_ids: list[str],
    target_indices: list[int] | None,
    certificate: dict[str, Any],
    guard_scores: list[float],
    conflict_margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    accepted_any = []
    correct = []
    wrong = []
    abstain = []
    conflict = []
    for query_idx in range(int(score_matrix.shape[0])):
        sorted_indices = torch.argsort(score_matrix[query_idx], descending=True)
        top_idx = int(sorted_indices[0].item()) if int(sorted_indices.numel()) > 0 else None
        runner_idx = int(sorted_indices[1].item()) if int(sorted_indices.numel()) > 1 else None
        scores_by_id = {
            trace_id: float(score_matrix[query_idx, trace_idx].item())
            for trace_idx, trace_id in enumerate(trace_ids)
        }
        decision = route_capsules(
            scores_by_id,
            theta_accept=float(certificate["theta_accept"]),
            guard_scores=guard_scores,
            conflict_margin=conflict_margin,
        )
        target_idx = None if target_indices is None else int(target_indices[query_idx])
        target_id = None if target_idx is None else trace_ids[target_idx]
        is_accepted = bool(decision.accepted)
        is_correct = bool(is_accepted and decision.edit_id == target_id) if target_id is not None else False
        is_wrong = bool(is_accepted and (target_id is None or decision.edit_id != target_id))
        accepted_any.append(1.0 if is_accepted else 0.0)
        correct.append(1.0 if is_correct else 0.0)
        wrong.append(1.0 if is_wrong else 0.0)
        abstain.append(0.0 if is_accepted else 1.0)
        conflict.append(1.0 if decision.abstain_reason == "conflict" else 0.0)
        audit_rows.append(
            {
                "query_index": query_idx,
                "target_trace_id": target_id,
                "target_score": None if target_idx is None else float(score_matrix[query_idx, target_idx].item()),
                "top_trace_id": None if top_idx is None else trace_ids[top_idx],
                "top_score": None if top_idx is None else float(score_matrix[query_idx, top_idx].item()),
                "runner_trace_id": None if runner_idx is None else trace_ids[runner_idx],
                "runner_score": None if runner_idx is None else float(score_matrix[query_idx, runner_idx].item()),
                "decision": decision.to_json(),
                "accepted_correct": is_correct,
                "accepted_wrong": is_wrong,
            }
        )
    return (
        {
            "certified_activation_rate": safe_mean(accepted_any),
            "correct_accept_rate": safe_mean(correct),
            "wrong_accept_rate": safe_mean(wrong),
            "abstain": safe_mean(abstain),
            "conflict_abstain": safe_mean(conflict),
        },
        audit_rows,
    )


def evaluate_eval_split(
    controller: HopEditController,
    eval_bank: dict[str, Any],
    certificate: dict[str, Any],
    guard_scores: list[float],
    *,
    subject_layer: int,
    relation_layer: int,
    subject_pooling: str,
    batch_size: int,
    max_locality_per_record: int,
    conflict_margin: float,
    score_family: str,
    learned_model: CapsuleScoreCombiner | None = None,
) -> dict[str, Any]:
    rephrase_prompts, rephrase_subjects, rephrase_targets = rephrase_queries(eval_bank["records"])
    rephrase = score_queries(
        controller,
        eval_bank,
        rephrase_prompts,
        rephrase_subjects,
        subject_layer=subject_layer,
        relation_layer=relation_layer,
        subject_pooling=subject_pooling,
        batch_size=batch_size,
        score_family=score_family,
        learned_model=learned_model,
    )
    strict = analyze_scores(rephrase["subject_scores"], rephrase["relation_scores"], eval_bank["relation_ids"])
    whitened = analyze_scores(rephrase["subject_scores"], rephrase["relation_scores_whitened"], eval_bank["relation_ids"])
    capsule_metrics, rephrase_audit = evaluate_capsule_decisions(
        score_matrix=rephrase["capsule_scores"],
        trace_ids=eval_bank["trace_ids"],
        target_indices=rephrase_targets,
        certificate=certificate,
        guard_scores=guard_scores,
        conflict_margin=conflict_margin,
    )

    loc_prompts, loc_subjects, loc_owners = locality_queries(eval_bank["records"], max_per_record=max_locality_per_record)
    locality_metrics: dict[str, Any] = {
        "num_queries": len(loc_prompts),
        "subject_found_rate": None,
        "capsule_false_accept_rate": None,
        "strict_false_accept_rate": None,
        "whitened_false_accept_rate": None,
    }
    locality_audit: list[dict[str, Any]] = []
    if loc_prompts:
        locality = score_queries(
            controller,
            eval_bank,
            loc_prompts,
            loc_subjects,
            subject_layer=subject_layer,
            relation_layer=relation_layer,
            subject_pooling=subject_pooling,
            batch_size=batch_size,
            score_family=score_family,
            learned_model=learned_model,
        )
        loc_capsule_metrics, locality_audit = evaluate_capsule_decisions(
            score_matrix=locality["capsule_scores"],
            trace_ids=eval_bank["trace_ids"],
            target_indices=None,
            certificate=certificate,
            guard_scores=guard_scores,
            conflict_margin=conflict_margin,
        )
        strict_loc = hard_and_decisions(locality["subject_scores"], locality["relation_scores"])
        whiten_loc = hard_and_decisions(locality["subject_scores"], locality["relation_scores_whitened"])
        locality_metrics.update(
            {
                "subject_found_rate": locality["subject_found_rate"],
                "capsule_false_accept_rate": loc_capsule_metrics["wrong_accept_rate"],
                "capsule_abstain": loc_capsule_metrics["abstain"],
                "strict_false_accept_rate": safe_mean([decision is not None for decision in strict_loc]),
                "whitened_false_accept_rate": safe_mean([decision is not None for decision in whiten_loc]),
            }
        )
        for row, owner_idx in zip(locality_audit, loc_owners):
            row["owner_index"] = owner_idx

    return {
        "strict_rephrase": strict,
        "whitened_rephrase": whitened,
        "capsule_rephrase": capsule_metrics,
        "locality": locality_metrics,
        "rephrase_subject_found_rate": rephrase["subject_found_rate"],
        "audit_rows": {
            "rephrase": rephrase_audit,
            "locality": locality_audit,
        },
    }


def write_audit(path: Path, *, eval_size: int, split_name: str, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row)
            payload["eval_size"] = int(eval_size)
            payload["split"] = split_name
            handle.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT")
    parser.add_argument("--hparams_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_type", default="CounterFact")
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calib_size", type=int, default=1000)
    parser.add_argument("--calib_start", type=int, default=4096)
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--eval_sizes", default="32,128,512")
    parser.add_argument("--subject_layer", type=int, default=16)
    parser.add_argument("--relation_layer", type=int, default=24)
    parser.add_argument("--subject_pooling", default="last")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--whiten_eps", type=float, default=1.0e-4)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--beta_sweep", default="0.01,0.05,0.10")
    parser.add_argument("--conflict_margin", type=float, default=0.0)
    parser.add_argument(
        "--score_family",
        default="min_z",
        choices=["min_z", "margin_min_z", "subject_margin", "relation_margin", "learned_linear", "learned_mlp"],
    )
    parser.add_argument("--learned_hidden_dim", type=int, default=32)
    parser.add_argument("--learned_epochs", type=int, default=50)
    parser.add_argument("--learned_lr", type=float, default=1.0e-3)
    parser.add_argument("--learned_weight_decay", type=float, default=1.0e-3)
    parser.add_argument("--learned_hard_negatives", type=int, default=8)
    parser.add_argument("--learned_train_frac", type=float, default=0.60)
    parser.add_argument("--learned_val_frac", type=float, default=0.20)
    parser.add_argument("--learned_patience", type=int, default=8)
    parser.add_argument("--max_locality_per_record", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.data_type not in {"CounterFact", "ZsRE"}:
        raise NotImplementedError("CapsuleEdit score POC currently supports CounterFact and ZsRE.")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "capsule_audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()

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

    max_eval_size = max(parse_csv_ints(args.eval_sizes))
    records, dataset_file = load_normalized_records(
        args.data_dir,
        args.data_type,
        max(args.calib_start + args.calib_size, args.eval_start + max_eval_size),
        data_file=args.data_file,
    )
    calib_records = select_split(records, start=args.calib_start, size=args.calib_size, name="calibration")
    calib_bank = build_trace_bank(
        controller,
        calib_records,
        subject_layer=args.subject_layer,
        relation_layer=args.relation_layer,
        subject_pooling=args.subject_pooling,
        batch_size=args.batch_size,
        whiten_eps=args.whiten_eps,
    )
    learned_model = None
    learned_metadata = None
    support_holdout_indices = None
    locality_holdout_indices = None
    if str(args.score_family).startswith("learned_"):
        learned_kind = "linear" if args.score_family == "learned_linear" else "mlp"
        learned_payload = train_learned_score_combiner(
            controller,
            calib_bank,
            subject_layer=args.subject_layer,
            relation_layer=args.relation_layer,
            subject_pooling=args.subject_pooling,
            batch_size=args.batch_size,
            max_locality_per_record=args.max_locality_per_record,
            model_type=learned_kind,
            hidden_dim=args.learned_hidden_dim,
            epochs=args.learned_epochs,
            lr=args.learned_lr,
            weight_decay=args.learned_weight_decay,
            hard_negatives=args.learned_hard_negatives,
            train_frac=args.learned_train_frac,
            val_frac=args.learned_val_frac,
            patience=args.learned_patience,
            seed=args.seed,
        )
        learned_model = learned_payload["model"]
        learned_metadata = learned_payload["metadata"]
        support_holdout_indices = learned_payload["support_holdout_indices"]
        locality_holdout_indices = learned_payload["locality_holdout_indices"]
    calibration_scores = collect_calibration_scores(
        controller,
        calib_bank,
        subject_layer=args.subject_layer,
        relation_layer=args.relation_layer,
        subject_pooling=args.subject_pooling,
        batch_size=args.batch_size,
        max_locality_per_record=args.max_locality_per_record,
        score_family=args.score_family,
        learned_model=learned_model,
        support_row_indices=support_holdout_indices,
        locality_row_indices=locality_holdout_indices,
    )
    certificate = build_global_capsule_certificate(
        support_scores=calibration_scores["support_scores"],
        guard_scores=calibration_scores["guard_scores"],
        alpha_reject=args.alpha,
        beta_false_fire=args.beta,
    )
    learned_state = learned_score_state(learned_model)
    capsule_score_config = {
        "method": "CapsuleEdit-score-config",
        "score_family": args.score_family,
        "theta_accept": float(certificate["theta_accept"]),
        "conflict_margin": float(args.conflict_margin),
        "alpha_reject": float(args.alpha),
        "beta_false_fire": float(args.beta),
        "feature_names": CAPSULE_FEATURE_NAMES,
        "learned_score": learned_metadata,
        "learned_score_state": learned_state,
        "layers": {
            "subject_layer": int(args.subject_layer),
            "relation_layer": int(args.relation_layer),
            "subject_pooling": args.subject_pooling,
            "whiten_eps": float(args.whiten_eps),
        },
    }
    if isinstance(learned_state, dict):
        if learned_state.get("model_type") == "linear":
            capsule_score_config["feature_weights"] = learned_state.get("feature_weights")
            capsule_score_config["feature_bias"] = learned_state.get("feature_bias")
        elif learned_state.get("model_type") == "mlp":
            capsule_score_config["mlp_state_dict"] = learned_state.get("state_dict")
    write_json(output_dir / "capsule_score_config.json", capsule_score_config)

    beta_sweep = []
    for beta in parse_csv_floats(args.beta_sweep):
        beta_sweep.append(
            build_global_capsule_certificate(
                support_scores=calibration_scores["support_scores"],
                guard_scores=calibration_scores["guard_scores"],
                alpha_reject=args.alpha,
                beta_false_fire=beta,
            )
        )

    eval_results = []
    for eval_size in parse_csv_ints(args.eval_sizes):
        eval_records = select_split(records, start=args.eval_start, size=eval_size, name=f"eval_{eval_size}")
        assert_disjoint_splits(calib_records, eval_records)
        eval_bank = build_trace_bank(
            controller,
            eval_records,
            subject_layer=args.subject_layer,
            relation_layer=args.relation_layer,
            subject_pooling=args.subject_pooling,
            batch_size=args.batch_size,
            whiten_eps=args.whiten_eps,
        )
        eval_result = evaluate_eval_split(
            controller,
            eval_bank,
            certificate,
            calibration_scores["guard_scores"],
            subject_layer=args.subject_layer,
            relation_layer=args.relation_layer,
            subject_pooling=args.subject_pooling,
            batch_size=args.batch_size,
            max_locality_per_record=args.max_locality_per_record,
            conflict_margin=args.conflict_margin,
            score_family=args.score_family,
            learned_model=learned_model,
        )
        write_audit(audit_path, eval_size=eval_size, split_name="rephrase", rows=eval_result["audit_rows"]["rephrase"])
        write_audit(audit_path, eval_size=eval_size, split_name="locality", rows=eval_result["audit_rows"]["locality"])
        eval_result.pop("audit_rows")
        eval_result["eval_size"] = int(eval_size)
        eval_result["split"] = {
            "eval_start": int(args.eval_start),
            "source_indices": [int(record["source_index"]) for record in eval_records],
        }
        eval_result["whitener"] = {
            key: value
            for key, value in eval_bank["whitener"].items()
            if key not in {"mean", "basis", "scale"}
        }
        eval_results.append(eval_result)

    summary = {
        "method": "CapsuleEdit-score-poc",
        "dataset_file": str(dataset_file),
        "data_type": args.data_type,
        "seed": int(args.seed),
        "layers": {
            "subject_layer": int(args.subject_layer),
            "relation_layer": int(args.relation_layer),
            "subject_pooling": args.subject_pooling,
            "whiten_eps": float(args.whiten_eps),
            "score_family": args.score_family,
        },
        "learned_score": learned_metadata,
        "learned_score_state": learned_state,
        "capsule_score_config_path": str((output_dir / "capsule_score_config.json").resolve()),
        "split": {
            "calib_start": int(args.calib_start),
            "calib_size": int(args.calib_size),
            "calib_source_indices": [int(record["source_index"]) for record in calib_records],
            "eval_start": int(args.eval_start),
            "eval_sizes": parse_csv_ints(args.eval_sizes),
        },
        "calibration": calibration_scores,
        "certificate": certificate,
        "beta_sweep": beta_sweep,
        "eval_results": eval_results,
        "audit_path": str(audit_path),
    }
    write_json(output_dir / "capsuleedit_score_poc_summary.json", summary)

    compact = []
    for row in eval_results:
        compact.append(
            {
                "eval_size": row["eval_size"],
                "strict_rephrase_target_fire": row["strict_rephrase"]["target_fire"],
                "strict_rephrase_kappa": row["strict_rephrase"]["wrong_fire_kappa"],
                "whitened_rephrase_target_fire": row["whitened_rephrase"]["target_fire"],
                "whitened_rephrase_kappa": row["whitened_rephrase"]["wrong_fire_kappa"],
                "capsule_rephrase_correct": row["capsule_rephrase"]["correct_accept_rate"],
                "capsule_rephrase_wrong": row["capsule_rephrase"]["wrong_accept_rate"],
                "capsule_rephrase_abstain": row["capsule_rephrase"]["abstain"],
                "capsule_locality_false_accept": row["locality"]["capsule_false_accept_rate"],
            }
        )
    write_json(output_dir / "capsuleedit_score_poc_compact.json", compact)
    print(json.dumps(compact, indent=2), flush=True)
    print(f"Summary written to {output_dir / 'capsuleedit_score_poc_summary.json'}", flush=True)
    print(f"Audit written to {audit_path}", flush=True)


if __name__ == "__main__":
    main()
