"""Analyze selective retirement for edit-memory consolidation candidates.

This script treats a compressed condition as a candidate semantic memory and
the exact condition as the episodic teacher.  For each edit, it computes a
contract score and a binary silent-failure label:

    exact passes contract, but compressed condition fails contract.

Sweeping a threshold over the compressed contract score gives the selective
prediction curve for retiring exact modules: retire high-scoring edits, keep
the rest episodic.  The goal is not to prove calibration from a single run; it
is to expose the retirement-rate / false-retirement / memory tradeoff that a
calibrated consolidator would later control on held-out calibration edits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _mean_optional(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _metric_mean(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        flat = []
        for item in value:
            inner = _metric_mean(item)
            if inner is not None:
                flat.append(inner)
        return _mean_optional(flat)
    if isinstance(value, dict):
        candidates = []
        for key, item in value.items():
            if key.endswith("_acc"):
                inner = _metric_mean(item)
                if inner is not None:
                    candidates.append(inner)
        return _mean_optional(candidates)
    return None


def _case_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    post = row.get("post", {})
    return {
        "rewrite": _metric_mean(post.get("rewrite_acc")),
        "rephrase": _metric_mean(post.get("rephrase_acc")),
        "locality": _metric_mean(post.get("locality")),
    }


def _contract_score(metrics: dict[str, float | None], *, missing_value: float = 0.0) -> float:
    values = [
        metrics.get("rewrite"),
        metrics.get("rephrase"),
        metrics.get("locality"),
    ]
    clean = [missing_value if value is None else float(value) for value in values]
    return float(min(clean))


def _contract_pass(metrics: dict[str, float | None], *, acc_threshold: float, locality_threshold: float) -> bool:
    rewrite = metrics.get("rewrite")
    rephrase = metrics.get("rephrase")
    locality = metrics.get("locality")
    if rewrite is None or rephrase is None or locality is None:
        return False
    return bool(
        float(rewrite) >= acc_threshold
        and float(rephrase) >= acc_threshold
        and float(locality) >= locality_threshold
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _condition_dirs(root: Path) -> dict[str, Path]:
    dirs = {}
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("condition_"):
            continue
        condition = path.name.removeprefix("condition_")
        dirs[condition] = path
    return dirs


def _compression_stats(condition_dir: Path, exact_count: int) -> dict[str, float | None]:
    metadata_path = condition_dir / "compression_metadata.json"
    if not metadata_path.exists():
        return {
            "exact_params": None,
            "compressed_params": None,
            "compression_ratio": None,
            "per_edit_exact_params": None,
        }
    metadata = _load_json(metadata_path)
    exact_params = metadata.get("exact_params")
    compressed_params = metadata.get("compressed_params")
    per_edit_exact = None
    if exact_params is not None and exact_count:
        per_edit_exact = float(exact_params) / float(exact_count)
    return {
        "exact_params": None if exact_params is None else float(exact_params),
        "compressed_params": None if compressed_params is None else float(compressed_params),
        "compression_ratio": None if metadata.get("compression_ratio") is None else float(metadata["compression_ratio"]),
        "per_edit_exact_params": per_edit_exact,
    }


def _split_indices(n: int, calibration_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    cut = int(round(float(calibration_fraction) * n))
    cut = max(1, min(n - 1, cut)) if n >= 2 else n
    return sorted(indices[:cut]), sorted(indices[cut:])


def _evaluate_threshold(
    examples: list[dict[str, Any]],
    threshold: float,
    *,
    exact_params: float | None,
    compressed_params: float | None,
    per_edit_exact_params: float | None,
) -> dict[str, Any]:
    eligible = [row for row in examples if row["exact_contract_pass"]]
    retired = [row for row in eligible if row["candidate_score"] >= threshold]
    failures = [row for row in retired if row["silent_failure"]]
    retirement_rate = None if not eligible else float(len(retired) / len(eligible))
    false_retirement_rate = None if not retired else float(len(failures) / len(retired))
    memory_params = None
    memory_compression_ratio = None
    if exact_params is not None and compressed_params is not None and per_edit_exact_params is not None:
        kept_exact = max(0, len(eligible) - len(retired))
        memory_params = float(compressed_params + kept_exact * per_edit_exact_params)
        memory_compression_ratio = None if memory_params <= 0 else float(exact_params / memory_params)
    return {
        "threshold": float(threshold),
        "eligible_edits": len(eligible),
        "retired_edits": len(retired),
        "silent_failures_retired": len(failures),
        "retirement_rate": retirement_rate,
        "false_retirement_rate": false_retirement_rate,
        "estimated_memory_params": memory_params,
        "estimated_memory_compression_ratio": memory_compression_ratio,
    }


def _curve(
    examples: list[dict[str, Any]],
    *,
    exact_params: float | None,
    compressed_params: float | None,
    per_edit_exact_params: float | None,
) -> list[dict[str, Any]]:
    thresholds = sorted({float(row["candidate_score"]) for row in examples}, reverse=True)
    thresholds.append(-math.inf)
    return [
        _evaluate_threshold(
            examples,
            threshold,
            exact_params=exact_params,
            compressed_params=compressed_params,
            per_edit_exact_params=per_edit_exact_params,
        )
        for threshold in thresholds
    ]


def _choose_threshold(calibration_curve: list[dict[str, Any]], target_false_retirement: float) -> dict[str, Any] | None:
    feasible = [
        row
        for row in calibration_curve
        if row["retired_edits"] > 0
        and row["false_retirement_rate"] is not None
        and row["false_retirement_rate"] <= target_false_retirement
    ]
    if not feasible:
        return None
    return max(feasible, key=lambda row: (row["retired_edits"], row["estimated_memory_compression_ratio"] or 0.0))


def analyze_condition(
    root: Path,
    condition: str,
    *,
    acc_threshold: float,
    locality_threshold: float,
    calibration_fraction: float,
    seed: int,
    target_false_retirement: float,
) -> dict[str, Any]:
    exact_dir = root / "condition_exact"
    candidate_dir = root / f"condition_{condition}"
    exact_metrics = _load_json(exact_dir / "metrics.json")
    candidate_metrics = _load_json(candidate_dir / "metrics.json")
    n = min(len(exact_metrics), len(candidate_metrics))
    compression = _compression_stats(candidate_dir, n)
    examples = []
    for idx in range(n):
        exact_case = _case_metrics(exact_metrics[idx])
        candidate_case = _case_metrics(candidate_metrics[idx])
        exact_pass = _contract_pass(
            exact_case,
            acc_threshold=acc_threshold,
            locality_threshold=locality_threshold,
        )
        candidate_pass = _contract_pass(
            candidate_case,
            acc_threshold=acc_threshold,
            locality_threshold=locality_threshold,
        )
        request = candidate_metrics[idx].get("requested_rewrite", {})
        examples.append(
            {
                "case_index": idx,
                "source_index": request.get("source_index"),
                "relation_id": request.get("relation_id"),
                "subject": request.get("subject"),
                "exact_contract_pass": exact_pass,
                "candidate_contract_pass": candidate_pass,
                "silent_failure": bool(exact_pass and not candidate_pass),
                "candidate_score": _contract_score(candidate_case),
                "exact_metrics": exact_case,
                "candidate_metrics": candidate_case,
            }
        )
    eligible = [row for row in examples if row["exact_contract_pass"]]
    silent_failures = [row for row in examples if row["silent_failure"]]
    calibration_indices, eval_indices = _split_indices(len(examples), calibration_fraction, seed)
    calibration_examples = [examples[idx] for idx in calibration_indices]
    eval_examples = [examples[idx] for idx in eval_indices]
    calibration_curve = _curve(
        calibration_examples,
        exact_params=compression["exact_params"],
        compressed_params=compression["compressed_params"],
        per_edit_exact_params=compression["per_edit_exact_params"],
    )
    eval_curve = _curve(
        eval_examples,
        exact_params=compression["exact_params"],
        compressed_params=compression["compressed_params"],
        per_edit_exact_params=compression["per_edit_exact_params"],
    )
    selected = _choose_threshold(calibration_curve, target_false_retirement)
    selected_eval = None
    if selected is not None:
        selected_eval = _evaluate_threshold(
            eval_examples,
            selected["threshold"],
            exact_params=compression["exact_params"],
            compressed_params=compression["compressed_params"],
            per_edit_exact_params=compression["per_edit_exact_params"],
        )
    relation_rows = {}
    for row in examples:
        relation_id = str(row.get("relation_id") or "__missing__")
        bucket = relation_rows.setdefault(relation_id, {"count": 0, "eligible": 0, "silent_failures": 0})
        bucket["count"] += 1
        bucket["eligible"] += int(row["exact_contract_pass"])
        bucket["silent_failures"] += int(row["silent_failure"])
    return {
        "condition": condition,
        "n_cases": n,
        "contract": {
            "acc_threshold": acc_threshold,
            "locality_threshold": locality_threshold,
            "score": "min(rewrite_acc, rephrase_acc, locality_acc)",
            "silent_failure": "exact contract passes but compressed candidate contract fails",
        },
        "compression": compression,
        "eligible_exact_pass_edits": len(eligible),
        "silent_merge_failures": len(silent_failures),
        "silent_merge_failure_rate_among_eligible": None if not eligible else float(len(silent_failures) / len(eligible)),
        "calibration": {
            "target_false_retirement": target_false_retirement,
            "calibration_fraction": calibration_fraction,
            "seed": seed,
            "contamination_warning": (
                "POC only: current compressed basis was trained on the same edits being scored. "
                "Use a disjoint consolidation/calibration split for a publishable calibrated guarantee."
            ),
            "selected_on_calibration": selected,
            "selected_evaluated_on_heldout_indices": selected_eval,
        },
        "curve_all": _curve(
            examples,
            exact_params=compression["exact_params"],
            compressed_params=compression["compressed_params"],
            per_edit_exact_params=compression["per_edit_exact_params"],
        ),
        "curve_calibration": calibration_curve,
        "curve_eval": eval_curve,
        "relation_breakdown": relation_rows,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Shared-basis POC output root")
    parser.add_argument("--conditions", default=None, help="Comma-separated condition names without condition_ prefix")
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--acc_threshold", default=0.5, type=float)
    parser.add_argument("--locality_threshold", default=0.85, type=float)
    parser.add_argument("--target_false_retirement", default=0.05, type=float)
    parser.add_argument("--calibration_fraction", default=0.5, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    root = args.root
    dirs = _condition_dirs(root)
    if "exact" not in dirs:
        raise FileNotFoundError(f"Missing {root / 'condition_exact'}")
    if args.conditions:
        conditions = [token.strip() for token in args.conditions.split(",") if token.strip()]
    else:
        conditions = sorted(condition for condition in dirs if condition != "exact")
    results = [
        analyze_condition(
            root,
            condition,
            acc_threshold=args.acc_threshold,
            locality_threshold=args.locality_threshold,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
            target_false_retirement=args.target_false_retirement,
        )
        for condition in conditions
    ]
    payload = {
        "method": "retirement_gate_analysis",
        "root": str(root),
        "conditions": conditions,
        "results": results,
    }
    output = args.output or (root / "retirement_gate_analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    for row in results:
        selected = row["calibration"]["selected_evaluated_on_heldout_indices"]
        print(
            json.dumps(
                {
                    "condition": row["condition"],
                    "eligible": row["eligible_exact_pass_edits"],
                    "silent_merge_failures": row["silent_merge_failures"],
                    "silent_failure_rate": row["silent_merge_failure_rate_among_eligible"],
                    "selected_eval": selected,
                }
            )
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
