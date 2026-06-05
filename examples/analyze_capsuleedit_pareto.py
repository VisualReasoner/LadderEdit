"""Build reliability Pareto curves from CapsuleEdit score-only POC outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_mean(values: list[float | bool]) -> float | None:
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_grid(
    *,
    support_scores: list[float],
    guard_scores: list[float],
    audit_rows: list[dict[str, Any]],
    num_points: int,
) -> list[float]:
    scores = []
    scores.extend(support_scores)
    scores.extend(guard_scores)
    for row in audit_rows:
        decision = row.get("decision") or {}
        for value in (row.get("target_score"), decision.get("score"), decision.get("runner_score")):
            number = finite(value)
            if number is not None:
                scores.append(number)
    if not scores:
        return [0.0]
    lo = min(scores)
    hi = max(scores)
    if num_points <= 1 or hi <= lo:
        return [float(lo)]
    step = (hi - lo) / float(num_points - 1)
    return [float(lo + step * idx) for idx in range(num_points)]


def calibration_point(support_scores: list[float], guard_scores: list[float], theta: float) -> dict[str, float | None]:
    support_accept = safe_mean([score > theta for score in support_scores])
    guard_accept = safe_mean([score > theta for score in guard_scores])
    return {
        "calib_support_accept": support_accept,
        "calib_guard_false_accept": guard_accept,
        "calib_abstain": None if support_accept is None else float(1.0 - support_accept),
    }


def eval_point(rows: list[dict[str, Any]], theta: float) -> dict[str, float | None]:
    correct = []
    wrong = []
    accepted = []
    abstain = []
    for row in rows:
        decision = row.get("decision") or {}
        top_score = finite(decision.get("score"))
        target_score = finite(row.get("target_score"))
        target_id = row.get("target_trace_id")
        top_id = decision.get("top_trace_id")
        if top_score is None:
            continue
        is_accepted = bool(top_score > theta)
        accepted.append(is_accepted)
        abstain.append(not is_accepted)
        if target_id is None:
            wrong.append(is_accepted)
            continue
        if top_id is not None:
            is_target_top = top_id == target_id
        else:
            is_target_top = target_score is not None and abs(float(top_score) - float(target_score)) <= 1.0e-6
        correct.append(bool(is_accepted and is_target_top))
        wrong.append(bool(is_accepted and not is_target_top))
    return {
        "correct_accept": safe_mean(correct),
        "wrong_accept": safe_mean(wrong),
        "certified_activation": safe_mean(accepted),
        "abstain": safe_mean(abstain),
    }


def auc_x_y(points: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    xy = []
    for point in points:
        x = finite(point.get(x_key))
        y = finite(point.get(y_key))
        if x is not None and y is not None:
            xy.append((x, y))
    if len(xy) < 2:
        return None
    xy.sort()
    area = 0.0
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return float(area)


def build_pareto(summary: dict[str, Any], audit_rows: list[dict[str, Any]], *, num_points: int) -> dict[str, Any]:
    calibration = summary["calibration"]
    support_scores = [float(score) for score in calibration["support_scores"] if finite(score) is not None]
    guard_scores = [float(score) for score in calibration["guard_scores"] if finite(score) is not None]
    thresholds = score_grid(
        support_scores=support_scores,
        guard_scores=guard_scores,
        audit_rows=audit_rows,
        num_points=num_points,
    )
    by_eval_size: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in audit_rows:
        by_eval_size.setdefault(int(row["eval_size"]), {}).setdefault(str(row["split"]), []).append(row)

    curves = []
    for eval_size, split_rows in sorted(by_eval_size.items()):
        for theta in thresholds:
            base = {"eval_size": eval_size, "theta": float(theta)}
            base.update(calibration_point(support_scores, guard_scores, theta))
            rephrase_point = eval_point(split_rows.get("rephrase", []), theta)
            locality_point = eval_point(split_rows.get("locality", []), theta)
            curves.append(
                {
                    **base,
                    "rephrase_correct_accept": rephrase_point["correct_accept"],
                    "rephrase_wrong_accept": rephrase_point["wrong_accept"],
                    "rephrase_activation": rephrase_point["certified_activation"],
                    "rephrase_abstain": rephrase_point["abstain"],
                    "locality_false_accept": locality_point["wrong_accept"],
                    "locality_activation": locality_point["certified_activation"],
                    "locality_abstain": locality_point["abstain"],
                }
            )

    summaries = []
    for eval_size in sorted(by_eval_size):
        points = [point for point in curves if int(point["eval_size"]) == eval_size]
        feasible_005 = [
            point
            for point in points
            if point.get("locality_false_accept") is not None and float(point["locality_false_accept"]) <= 0.05
        ]
        feasible_007 = [
            point
            for point in points
            if point.get("locality_false_accept") is not None and float(point["locality_false_accept"]) <= 0.07
        ]
        best_005 = max(feasible_005, key=lambda row: row.get("rephrase_correct_accept") or -1.0) if feasible_005 else None
        best_007 = max(feasible_007, key=lambda row: row.get("rephrase_correct_accept") or -1.0) if feasible_007 else None
        summaries.append(
            {
                "eval_size": eval_size,
                "auc_rephrase_vs_locality_false_accept": auc_x_y(points, "locality_false_accept", "rephrase_correct_accept"),
                "best_rephrase_at_locality_fa_le_0_05": best_005,
                "best_rephrase_at_locality_fa_le_0_07": best_007,
            }
        )
    return {
        "num_thresholds": len(thresholds),
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
        "curves": curves,
        "summaries": summaries,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_thresholds", type=int, default=401)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    audit_rows = read_jsonl(Path(args.audit))
    pareto = build_pareto(summary, audit_rows, num_points=args.num_thresholds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "capsuleedit_pareto_summary.json").write_text(json.dumps(pareto, indent=2))
    write_csv(output_dir / "capsuleedit_pareto_curve.csv", pareto["curves"])
    print(json.dumps(pareto["summaries"], indent=2), flush=True)
    print(f"Pareto summary written to {output_dir / 'capsuleedit_pareto_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
