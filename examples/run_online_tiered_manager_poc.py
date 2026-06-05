from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a list of metrics")
    return payload


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return _mean([float(v) for v in value])
    return None


def _extract_row(metric: dict[str, Any], fallback_idx: int) -> dict[str, Any]:
    req = metric.get("requested_rewrite", {}) if isinstance(metric, dict) else {}
    post = metric.get("post", {}) if isinstance(metric, dict) else {}
    locality = post.get("locality", {}) if isinstance(post, dict) else {}
    if isinstance(locality, dict):
        locality_value = _scalar(locality.get("neighborhood_acc"))
    else:
        locality_value = _scalar(locality)
    return {
        "case_id": int(metric.get("case_id", fallback_idx)),
        "prompt": req.get("prompt"),
        "subject": req.get("subject"),
        "target_new": req.get("target_new"),
        "ground_truth": req.get("ground_truth"),
        "rephrase_prompt": req.get("rephrase_prompt"),
        "rewrite": _scalar(post.get("rewrite_acc")),
        "rephrase": _scalar(post.get("rephrase_acc")),
        "locality": locality_value,
    }


def _utility(row: dict[str, Any], rewrite_weight: float, rephrase_weight: float, locality_weight: float) -> float:
    pairs = [
        (row.get("rewrite"), rewrite_weight),
        (row.get("rephrase"), rephrase_weight),
        (row.get("locality"), locality_weight),
    ]
    denom = sum(weight for _, weight in pairs if weight > 0)
    if denom <= 0:
        raise ValueError("At least one positive utility weight is required")
    total = 0.0
    for value, weight in pairs:
        if value is None:
            continue
        total += float(value) * weight
    return float(total / denom)


def _contract_pass(row: dict[str, Any], min_rewrite: float, min_rephrase: float, min_locality: float) -> bool:
    return (
        row.get("rewrite") is not None
        and row.get("rewrite") >= min_rewrite
        and row.get("rephrase") is not None
        and row.get("rephrase") >= min_rephrase
        and row.get("locality") is not None
        and row.get("locality") >= min_locality
    )


def _summarize_rows(
    name: str,
    rows: list[dict[str, Any]],
    chosen_sources: list[str] | None = None,
) -> dict[str, Any]:
    rewrite = [row["rewrite"] for row in rows if row.get("rewrite") is not None]
    rephrase = [row["rephrase"] for row in rows if row.get("rephrase") is not None]
    locality = [row["locality"] for row in rows if row.get("locality") is not None]
    summary = {
        "policy": name,
        "mean_rewrite": _mean(rewrite),
        "mean_rephrase": _mean(rephrase),
        "mean_locality": _mean(locality),
        "mean_score": _mean([_utility(row, 1.0, 1.0, 1.0) for row in rows]),
        "contract_pass_rate": _mean(
            [
                1.0 if _contract_pass(row, 0.8, 0.8, 0.95) else 0.0
                for row in rows
            ]
        ),
    }
    if chosen_sources is not None:
        exact_count = sum(1 for source in chosen_sources if source == "exact")
        summary["served_exact_count"] = exact_count
        summary["served_exact_fraction"] = float(exact_count / len(chosen_sources)) if chosen_sources else 0.0
        summary["served_shared_count"] = len(chosen_sources) - exact_count
        summary["served_shared_fraction"] = float((len(chosen_sources) - exact_count) / len(chosen_sources)) if chosen_sources else 0.0
    return summary


@dataclass
class EditObservation:
    stream_index: int
    case_id: int
    prompt: str | None
    subject: str | None
    target_new: str | None
    ground_truth: str | None
    rephrase_prompt: str | None
    exact: dict[str, Any]
    shared: dict[str, Any]
    score_exact: float
    score_shared: float
    contract_exact: bool
    contract_shared: bool
    gap_hat: float
    radius: float

    @property
    def priority(self) -> float:
        return self.gap_hat + self.radius


@dataclass
class EditState:
    observation: EditObservation
    active_exact: bool = True
    retired: bool = False
    retirement_count: int = 0
    promotion_count: int = 0
    gap_samples: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.gap_samples:
            self.gap_samples.append(self.observation.gap_hat)

    @property
    def case_id(self) -> int:
        return self.observation.case_id

    @property
    def gap_hat(self) -> float:
        return float(sum(self.gap_samples) / len(self.gap_samples))

    def radius(self, confidence_scale: float) -> float:
        return float(confidence_scale / math.sqrt(max(1, len(self.gap_samples))))

    def priority(self, confidence_scale: float) -> float:
        return self.gap_hat + self.radius(confidence_scale)


class OnlineTieredManager:
    def __init__(self, *, exact_capacity: int, confidence_scale: float, force_keep_shared_fail: bool = True):
        self.exact_capacity = max(int(exact_capacity), 0)
        self.confidence_scale = float(confidence_scale)
        self.force_keep_shared_fail = force_keep_shared_fail
        self.states: dict[int, EditState] = {}
        self.stream_order: list[int] = []
        self.active_exact_ids: set[int] = set()
        self.event_log: list[dict[str, Any]] = []

    def ingest(self, observation: EditObservation) -> dict[str, Any]:
        state = EditState(observation=observation)
        self.states[observation.case_id] = state
        self.stream_order.append(observation.case_id)
        self.active_exact_ids.add(observation.case_id)
        return self._rebalance(trigger_case_id=observation.case_id, step=observation.stream_index, reason="arrival")

    def _rebalance(self, *, trigger_case_id: int, step: int, reason: str) -> dict[str, Any]:
        previous_exact_ids = set(self.active_exact_ids)
        must_keep: list[EditState] = []
        soft: list[EditState] = []

        for case_id in self.stream_order:
            state = self.states[case_id]
            if self.force_keep_shared_fail and not state.observation.contract_shared:
                must_keep.append(state)
            else:
                soft.append(state)

        chosen: list[EditState]
        if self.exact_capacity <= 0:
            chosen = []
        elif len(must_keep) >= self.exact_capacity:
            chosen = sorted(
                must_keep,
                key=lambda state: (state.priority(self.confidence_scale), state.observation.score_exact),
                reverse=True,
            )[: self.exact_capacity]
        else:
            remaining = self.exact_capacity - len(must_keep)
            chosen = list(must_keep)
            chosen.extend(
                sorted(
                    soft,
                    key=lambda state: (state.priority(self.confidence_scale), state.observation.score_exact),
                    reverse=True,
                )[:remaining]
            )

        self.active_exact_ids = {state.case_id for state in chosen}
        promoted_ids = sorted(self.active_exact_ids - previous_exact_ids)
        retired_ids = sorted(previous_exact_ids - self.active_exact_ids)

        for case_id in promoted_ids:
            state = self.states[case_id]
            state.active_exact = True
            state.retired = False
            state.promotion_count += 1
        for case_id in retired_ids:
            state = self.states[case_id]
            state.active_exact = False
            state.retired = True
            state.retirement_count += 1
        for case_id in self.active_exact_ids & previous_exact_ids:
            self.states[case_id].active_exact = True
        for case_id in set(self.states) - self.active_exact_ids:
            if case_id not in retired_ids:
                self.states[case_id].active_exact = False

        trigger_source = "exact" if trigger_case_id in self.active_exact_ids else "shared"
        trigger_state = self.states[trigger_case_id]
        chosen_row = trigger_state.observation.exact if trigger_source == "exact" else trigger_state.observation.shared
        event = {
            "step": step,
            "case_id": trigger_case_id,
            "reason": reason,
            "exact_capacity": self.exact_capacity,
            "must_keep_count": len(must_keep),
            "active_exact_count": len(self.active_exact_ids),
            "trigger_source_after_step": trigger_source,
            "trigger_contract_exact": trigger_state.observation.contract_exact,
            "trigger_contract_shared": trigger_state.observation.contract_shared,
            "trigger_gap_hat": trigger_state.gap_hat,
            "trigger_priority": trigger_state.priority(self.confidence_scale),
            "promoted_ids": promoted_ids,
            "retired_ids": retired_ids,
            "forced_hard_demotions": max(0, len(must_keep) - self.exact_capacity),
            "chosen_rewrite": chosen_row.get("rewrite"),
            "chosen_rephrase": chosen_row.get("rephrase"),
            "chosen_locality": chosen_row.get("locality"),
            "chosen_score": _utility(chosen_row, 1.0, 1.0, 1.0),
        }
        self.event_log.append(event)
        return event

    def final_active_exact_ids(self) -> list[int]:
        return sorted(self.active_exact_ids)


def _build_observations(
    exact_json: Path,
    shared_json: Path,
    rewrite_weight: float,
    rephrase_weight: float,
    locality_weight: float,
    min_rewrite: float,
    min_rephrase: float,
    min_locality: float,
    confidence_scale: float,
) -> list[EditObservation]:
    exact_metrics = _load_metrics(exact_json)
    shared_metrics = _load_metrics(shared_json)
    exact_rows = {_extract_row(metric, idx)["case_id"]: _extract_row(metric, idx) for idx, metric in enumerate(exact_metrics)}
    shared_rows = {_extract_row(metric, idx)["case_id"]: _extract_row(metric, idx) for idx, metric in enumerate(shared_metrics)}
    case_ids = sorted(set(exact_rows) & set(shared_rows))
    if not case_ids:
        raise ValueError("No overlapping case_ids found between exact and shared runs")

    observations: list[EditObservation] = []
    for stream_index, case_id in enumerate(case_ids):
        exact = exact_rows[case_id]
        shared = shared_rows[case_id]
        observed_dims = sum(
            1
            for field in ("rewrite", "rephrase", "locality")
            if exact.get(field) is not None and shared.get(field) is not None
        )
        observed_dims = max(observed_dims, 1)
        observations.append(
            EditObservation(
                stream_index=stream_index,
                case_id=case_id,
                prompt=exact.get("prompt"),
                subject=exact.get("subject"),
                target_new=exact.get("target_new"),
                ground_truth=exact.get("ground_truth"),
                rephrase_prompt=exact.get("rephrase_prompt"),
                exact=exact,
                shared=shared,
                score_exact=_utility(exact, rewrite_weight, rephrase_weight, locality_weight),
                score_shared=_utility(shared, rewrite_weight, rephrase_weight, locality_weight),
                contract_exact=_contract_pass(exact, min_rewrite, min_rephrase, min_locality),
                contract_shared=_contract_pass(shared, min_rewrite, min_rephrase, min_locality),
                gap_hat=_utility(exact, rewrite_weight, rephrase_weight, locality_weight)
                - _utility(shared, rewrite_weight, rephrase_weight, locality_weight),
                radius=float(confidence_scale / math.sqrt(observed_dims)),
            )
        )
    return observations


def _render_report(
    dataset_name: str,
    exact_json: Path,
    shared_json: Path,
    exact_capacity: int,
    arrival_summary: dict[str, Any],
    pure_exact: dict[str, Any],
    pure_shared: dict[str, Any],
    final_active_exact_fraction: float,
    hard_rows: list[EditObservation],
    event_log: list[dict[str, Any]],
    output_path: Path,
) -> None:
    total_retired = sum(len(event["retired_ids"]) for event in event_log)
    total_promoted = sum(len(event["promoted_ids"]) for event in event_log)
    max_forced = max((event["forced_hard_demotions"] for event in event_log), default=0)
    lines = [
        "# Online Tiered Manager POC",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Exact tier proxy: `{exact_json}`",
        f"- Shared tier: `{shared_json}`",
        f"- Exact capacity: `{exact_capacity}`",
        "",
        "This is the first runtime-style proof of concept for exact-first, shadow-shared, budgeted retirement.",
        "",
        "Service loop:",
        "",
        "1. A new edit is written to the exact tier.",
        "2. A shadow shared copy is evaluated on edit-time probes.",
        "3. The manager keeps only the highest-priority hard edits exact, under a fixed exact-capacity budget.",
        "4. The currently served source for the arriving edit is logged after the rebalance step.",
        "",
        "Important caveat:",
        "",
        "This report summarizes arrival-time service decisions. It does not yet measure future retention of an edit after more edits arrive.",
        "",
        "## Arrival-Time Summary",
        "",
        "| policy | rewrite | rephrase | locality | score | contract_pass | served_exact_fraction |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| pure_exact | {pure_exact['mean_rewrite']:.4f} | {pure_exact['mean_rephrase']:.4f} | {pure_exact['mean_locality']:.4f} | {pure_exact['mean_score']:.4f} | {pure_exact['contract_pass_rate']:.4f} | - |",
        f"| pure_shared | {pure_shared['mean_rewrite']:.4f} | {pure_shared['mean_rephrase']:.4f} | {pure_shared['mean_locality']:.4f} | {pure_shared['mean_score']:.4f} | {pure_shared['contract_pass_rate']:.4f} | - |",
        f"| online_manager | {arrival_summary['mean_rewrite']:.4f} | {arrival_summary['mean_rephrase']:.4f} | {arrival_summary['mean_locality']:.4f} | {arrival_summary['mean_score']:.4f} | {arrival_summary['contract_pass_rate']:.4f} | {arrival_summary['served_exact_fraction']:.4f} |",
        "",
        "## Manager Behavior",
        "",
        f"- Final active exact fraction: `{final_active_exact_fraction:.4f}`",
        f"- Total retirements triggered during the stream: `{total_retired}`",
        f"- Total promotions triggered during the stream: `{total_promoted}`",
        f"- Maximum forced hard demotions at any step: `{max_forced}`",
        "",
        "## Hard-Edit Examples",
        "",
        "These are edits where exact passes the contract and shared fails it.",
        "",
        "| case_id | prompt | target_new | exact (r, rp, l) | shared (r, rp, l) | gap_hat |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for obs in hard_rows:
        lines.append(
            f"| {obs.case_id} | {obs.prompt} | {obs.target_new} | "
            f"({obs.exact['rewrite']:.3f}, {obs.exact['rephrase']:.3f}, {obs.exact['locality']:.3f}) | "
            f"({obs.shared['rewrite']:.3f}, {obs.shared['rephrase']:.3f}, {obs.shared['locality']:.3f}) | "
            f"{obs.gap_hat:.3f} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--exact-json", type=Path, required=True)
    parser.add_argument("--shared-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-capacity", type=int, required=True)
    parser.add_argument("--rewrite-weight", type=float, default=1.0)
    parser.add_argument("--rephrase-weight", type=float, default=1.0)
    parser.add_argument("--locality-weight", type=float, default=1.0)
    parser.add_argument("--min-rewrite", type=float, default=0.8)
    parser.add_argument("--min-rephrase", type=float, default=0.8)
    parser.add_argument("--min-locality", type=float, default=0.95)
    parser.add_argument("--confidence-scale", type=float, default=0.10)
    parser.add_argument("--top-hard-cases", type=int, default=8)
    args = parser.parse_args()

    observations = _build_observations(
        exact_json=args.exact_json,
        shared_json=args.shared_json,
        rewrite_weight=args.rewrite_weight,
        rephrase_weight=args.rephrase_weight,
        locality_weight=args.locality_weight,
        min_rewrite=args.min_rewrite,
        min_rephrase=args.min_rephrase,
        min_locality=args.min_locality,
        confidence_scale=args.confidence_scale,
    )
    manager = OnlineTieredManager(
        exact_capacity=args.exact_capacity,
        confidence_scale=args.confidence_scale,
        force_keep_shared_fail=True,
    )

    arrival_rows: list[dict[str, Any]] = []
    arrival_sources: list[str] = []
    for obs in observations:
        event = manager.ingest(obs)
        source = event["trigger_source_after_step"]
        arrival_sources.append(source)
        arrival_rows.append(obs.exact if source == "exact" else obs.shared)

    pure_exact = _summarize_rows("pure_exact", [obs.exact for obs in observations])
    pure_shared = _summarize_rows("pure_shared", [obs.shared for obs in observations])
    arrival_summary = _summarize_rows("online_manager", arrival_rows, arrival_sources)
    arrival_summary["exact_capacity"] = args.exact_capacity
    arrival_summary["final_active_exact_count"] = len(manager.final_active_exact_ids())
    arrival_summary["final_active_exact_fraction"] = float(len(manager.final_active_exact_ids()) / len(observations)) if observations else 0.0

    hard_rows = [obs for obs in observations if obs.contract_exact and not obs.contract_shared]
    hard_rows.sort(key=lambda obs: obs.gap_hat, reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_name": args.dataset_name,
        "exact_json": str(args.exact_json),
        "shared_json": str(args.shared_json),
        "exact_capacity": args.exact_capacity,
        "num_cases": len(observations),
        "arrival_summary": arrival_summary,
        "pure_exact": pure_exact,
        "pure_shared": pure_shared,
        "final_active_exact_ids": manager.final_active_exact_ids(),
        "hard_edit_count": len(hard_rows),
        "hard_edit_fraction": float(len(hard_rows) / len(observations)) if observations else 0.0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "events.json").write_text(json.dumps(manager.event_log, indent=2))
    with (args.output_dir / "arrival_decisions.csv").open("w", newline="") as handle:
        fieldnames = [
            "step",
            "case_id",
            "reason",
            "exact_capacity",
            "must_keep_count",
            "active_exact_count",
            "trigger_source_after_step",
            "trigger_contract_exact",
            "trigger_contract_shared",
            "trigger_gap_hat",
            "trigger_priority",
            "forced_hard_demotions",
            "chosen_rewrite",
            "chosen_rephrase",
            "chosen_locality",
            "chosen_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in manager.event_log:
            writer.writerow({key: event.get(key) for key in fieldnames})
    _render_report(
        dataset_name=args.dataset_name,
        exact_json=args.exact_json,
        shared_json=args.shared_json,
        exact_capacity=args.exact_capacity,
        arrival_summary=arrival_summary,
        pure_exact=pure_exact,
        pure_shared=pure_shared,
        final_active_exact_fraction=arrival_summary["final_active_exact_fraction"],
        hard_rows=hard_rows[: args.top_hard_cases],
        event_log=manager.event_log,
        output_path=args.output_dir / "report.md",
    )
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'events.json'}")
    print(f"Wrote {args.output_dir / 'arrival_decisions.csv'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
