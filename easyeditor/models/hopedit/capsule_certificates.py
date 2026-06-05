"""Calibration utilities for proof-carrying edit capsules.

The score-only CapsuleEdit proof of concept treats an edit as active only when
its address score is unusual relative to calibrated guard queries.  Higher
scores mean stronger evidence that the capsule should fire.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EditCapsule:
    edit_id: str
    subject: str | None
    relation_id: str | None
    target_new: str | None
    theta_accept: float
    alpha_reject: float
    beta_false_fire: float
    support_accept_rate: float
    guard_false_accept_rate: float
    certificate_status: str
    support_count: int
    guard_count: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapsuleDecision:
    accepted: bool
    edit_id: str | None
    p_accept: float | None
    score: float | None
    runner_score: float | None
    margin: float | None
    conflict_set: list[str]
    abstain_reason: str | None
    top_trace_id: str | None = None
    runner_trace_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _finite_scores(scores: Sequence[float]) -> list[float]:
    return [float(score) for score in scores if isfinite(float(score))]


def conformal_upper_threshold(guard_scores: Sequence[float], beta: float) -> float:
    """Return a conservative upper-tail threshold for false activation control.

    Accepting a query when ``score > threshold`` yields an empirical guard
    exceedance of at most approximately ``beta``.  The index follows the
    split-conformal one-sided quantile convention.  If the calibration set is
    too small for the requested beta, the threshold is ``inf`` and no query is
    certified.
    """

    if not 0.0 < float(beta) < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta!r}")
    scores = sorted(_finite_scores(guard_scores))
    if not scores:
        return float("inf")
    rank = int(ceil((len(scores) + 1) * (1.0 - float(beta))))
    if rank > len(scores):
        return float("inf")
    return float(scores[max(0, rank - 1)])


def support_recall_threshold(support_scores: Sequence[float], alpha: float) -> float:
    """Largest-ish threshold that preserves ``1-alpha`` support acceptance.

    This is a diagnostic threshold, not the threshold used for safety.  The
    active threshold is guard-calibrated; support recall tells us whether that
    safety threshold leaves enough positive support.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    scores = sorted(_finite_scores(support_scores))
    if not scores:
        return float("-inf")
    rank = int(ceil((len(scores) + 1) * float(alpha)))
    rank = min(max(1, rank), len(scores))
    return float(scores[rank - 1])


def conformal_guard_p_value(guard_scores: Sequence[float], score: float) -> float:
    """Upper-tail conformal p-value against guard scores."""

    guards = _finite_scores(guard_scores)
    if not guards:
        return 1.0
    score = float(score)
    exceed_or_tie = sum(1 for guard in guards if guard >= score)
    return float((1 + exceed_or_tie) / (len(guards) + 1))


def fraction_above(scores: Sequence[float], threshold: float) -> float:
    values = _finite_scores(scores)
    if not values:
        return 0.0
    return float(sum(1 for value in values if value > float(threshold)) / len(values))


def build_global_capsule_certificate(
    *,
    support_scores: Sequence[float],
    guard_scores: Sequence[float],
    alpha_reject: float,
    beta_false_fire: float,
) -> dict[str, Any]:
    """Build the marginal/global certificate used by the score-only POC."""

    theta_accept = conformal_upper_threshold(guard_scores, beta_false_fire)
    theta_support = support_recall_threshold(support_scores, alpha_reject)
    support_accept_rate = fraction_above(support_scores, theta_accept)
    guard_false_accept_rate = fraction_above(guard_scores, theta_accept)
    if not isfinite(theta_accept):
        status = "uncertified_too_few_guards"
    elif theta_accept <= theta_support:
        status = "valid"
    else:
        status = "infeasible_support_guard_gap"
    return {
        "theta_accept": float(theta_accept),
        "theta_support": float(theta_support),
        "alpha_reject": float(alpha_reject),
        "beta_false_fire": float(beta_false_fire),
        "support_accept_rate": float(support_accept_rate),
        "guard_false_accept_rate": float(guard_false_accept_rate),
        "certificate_status": status,
        "support_count": len(_finite_scores(support_scores)),
        "guard_count": len(_finite_scores(guard_scores)),
    }


def allocate_uniform_risk_budget(global_beta: float, num_capsules: int) -> list[float]:
    if num_capsules <= 0:
        return []
    if not 0.0 <= float(global_beta) <= 1.0:
        raise ValueError(f"global_beta must be in [0, 1], got {global_beta!r}")
    value = float(global_beta) / int(num_capsules)
    return [value for _ in range(int(num_capsules))]


def assert_disjoint_source_indices(
    calibration_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
) -> None:
    """Raise if calibration and evaluation records share dataset source indices."""

    calibration_ids = {int(record["source_index"]) for record in calibration_records}
    evaluation_ids = {int(record["source_index"]) for record in evaluation_records}
    overlap = sorted(calibration_ids & evaluation_ids)
    if overlap:
        raise ValueError(f"Calibration/eval split overlap detected: first overlaps={overlap[:10]}")


def route_capsules(
    scores_by_edit_id: Mapping[str, float],
    *,
    theta_accept: float,
    guard_scores: Sequence[float] = (),
    conflict_margin: float = 0.0,
) -> CapsuleDecision:
    """Route a query through calibrated capsules.

    If multiple capsules pass the threshold, the top capsule is accepted only
    when it clears the runner-up by ``conflict_margin``.  Otherwise the system
    abstains with an explicit conflict set.
    """

    finite_rows = [
        (str(edit_id), float(score))
        for edit_id, score in scores_by_edit_id.items()
        if isfinite(float(score))
    ]
    if not finite_rows:
        return CapsuleDecision(False, None, None, None, None, None, [], "no_scores")
    finite_rows.sort(key=lambda row: row[1], reverse=True)
    top_id, top_score = finite_rows[0]
    runner_id = finite_rows[1][0] if len(finite_rows) > 1 else None
    all_runner_score = finite_rows[1][1] if len(finite_rows) > 1 else None
    accepted_rows = [row for row in finite_rows if row[1] > float(theta_accept)]
    if not accepted_rows:
        return CapsuleDecision(
            False,
            None,
            conformal_guard_p_value(guard_scores, top_score) if guard_scores else None,
            top_score,
            all_runner_score,
            None,
            [],
            "below_threshold",
            top_id,
            runner_id,
        )
    top_id, top_score = accepted_rows[0]
    runner_score = accepted_rows[1][1] if len(accepted_rows) > 1 else None
    accepted_runner_id = accepted_rows[1][0] if len(accepted_rows) > 1 else None
    margin = None if runner_score is None else float(top_score - runner_score)
    if runner_score is not None and margin is not None and margin <= float(conflict_margin):
        return CapsuleDecision(
            False,
            None,
            conformal_guard_p_value(guard_scores, top_score) if guard_scores else None,
            top_score,
            runner_score,
            margin,
            [edit_id for edit_id, _ in accepted_rows],
            "conflict",
            top_id,
            accepted_runner_id,
        )
    return CapsuleDecision(
        True,
        top_id,
        conformal_guard_p_value(guard_scores, top_score) if guard_scores else None,
        top_score,
        runner_score,
        margin,
        [edit_id for edit_id, _ in accepted_rows],
        None,
        top_id,
        accepted_runner_id,
    )
