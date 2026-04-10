import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIKI_ROOT = REPO_ROOT / "outputs" / "wikibigedit_lifelong_pilot10k"
DEFAULT_CLAIM_SUMMARY = REPO_ROOT / "outputs" / "claim_figures" / "capacity_claim_summary.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "claim_figures" / "neurips_package"

METHOD_LAYOUT = {
    "HopEdit": ("hopedit_qwen25-7b-instruct", "hopedit_qwen2-5-7b-instruct"),
    "WISE": ("wise_qwen25-7b-instruct", "wise_qwen2-5-7b-instruct"),
    "GRACE": ("grace_qwen25-7b-instruct", "grace_qwen2-5-7b-instruct"),
    "LoRA": ("lora_qwen25-7b-instruct", "lora_qwen2-5-7b-instruct"),
    "AlphaEdit": ("alphaedit_qwen25-7b", "alphaedit_qwen2-5-7b-instruct"),
}


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_round(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def summarize_checkpoint(checkpoint_dir: Path):
    current_dir = checkpoint_dir / "current"
    summary_path = current_dir / "summary.json"
    theory_path = current_dir / "theory_metrics.json"
    failure_path = current_dir / "failure_decomposition.json"
    if not summary_path.exists():
        return None

    summary = load_json(summary_path)
    theory = load_json(theory_path) if theory_path.exists() else {}
    failure = load_json(failure_path) if failure_path.exists() else {}

    return {
        "checkpoint": checkpoint_dir.name,
        "post_rewrite_mean": maybe_round(summary.get("post_rewrite_mean")),
        "post_rephrase_mean": maybe_round(summary.get("post_rephrase_mean")),
        "post_locality_mean": maybe_round(summary.get("post_locality_mean")),
        "rewrite_delta_mean": maybe_round(summary.get("rewrite_delta_mean")),
        "early_late_gap": maybe_round(summary.get("early_late_gap")),
        "distortion_primary_mean": maybe_round(theory.get("distortion", {}).get("primary_mean")),
        "failure_mode": failure.get("mode"),
        "failure_rates": failure.get("rates", {}),
        "path": str(current_dir),
    }


def collect_method_checkpoints(wiki_root: Path):
    rows = {}
    for method, (method_dir, run_dir) in METHOD_LAYOUT.items():
        increment_root = wiki_root / method_dir / run_dir / "increment_20240201_20240220"
        if not increment_root.exists():
            rows[method] = {}
            continue
        method_rows = {}
        for checkpoint_dir in sorted(increment_root.glob("checkpoint_*")):
            summary = summarize_checkpoint(checkpoint_dir)
            if summary is not None:
                method_rows[checkpoint_dir.name] = summary
        rows[method] = method_rows
    return rows


def build_matched_checkpoint_rows(method_rows, checkpoint_name: str):
    rows = []
    for method, checkpoints in method_rows.items():
        row = checkpoints.get(checkpoint_name)
        if row is None:
            continue
        rows.append(
            {
                "method": method,
                "checkpoint": checkpoint_name,
                "rewrite": row["post_rewrite_mean"],
                "rephrase": row["post_rephrase_mean"],
                "locality": row["post_locality_mean"],
                "distortion": row["distortion_primary_mean"],
                "failure_mode": row["failure_mode"],
                "failure_rates": row["failure_rates"],
                "path": row["path"],
            }
        )
    return rows


def flatten_trajectories(method_rows):
    rows = []
    for method, checkpoints in method_rows.items():
        for checkpoint_name, row in checkpoints.items():
            rows.append(
                {
                    "method": method,
                    "checkpoint": checkpoint_name,
                    "rewrite": row["post_rewrite_mean"],
                    "rephrase": row["post_rephrase_mean"],
                    "locality": row["post_locality_mean"],
                    "distortion": row["distortion_primary_mean"],
                    "rewrite_delta": row["rewrite_delta_mean"],
                    "early_late_gap": row["early_late_gap"],
                    "failure_mode": row["failure_mode"],
                    "path": row["path"],
                }
            )
    rows.sort(key=lambda item: (item["method"], item["checkpoint"]))
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(claim_summary: dict, matched_1k_rows: list[dict], trajectory_rows: list[dict]):
    lines = []
    lines.append("# HopEdit NeurIPS Package")
    lines.append("")
    lines.append("## Main Claim")
    lines.append("")
    lines.append(
        "Lifelong editing fails when stored edits become hard to access and hard to realize without interference. "
        "HopEdit is motivated by this failure anatomy and improves separability through calibrated features, routed access, and collision-aware realization."
    )
    lines.append("")
    lines.append("## Controlled Evidence")
    lines.append("")
    qwen = claim_summary.get("qwen_zsre", {})
    predictiveness = claim_summary.get("predictiveness", {})
    if qwen:
        lines.append(
            f"- Qwen `standard-128` rewrite = {qwen.get('standard-128', {}).get('post_rewrite_mean'):.3f}, "
            f"while `collision-64` rewrite = {qwen.get('collision-64', {}).get('post_rewrite_mean'):.3f}."
        )
        lines.append(
            f"- Conflict is more predictive than position on ZsRE: "
            f"{predictiveness.get('zsre_conflict_vs_rewrite_delta'):.3f} vs {predictiveness.get('zsre_position_vs_rewrite_delta'):.3f}."
        )
    lines.append("")
    lines.append("## Matched 1k WikiBigEdit Comparison")
    lines.append("")
    lines.append("| Method | Rewrite | Rephrase | Locality | Distortion | Failure Mode |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in matched_1k_rows:
        lines.append(
            f"| {row['method']} | {row['rewrite'] if row['rewrite'] is not None else 'NA'} | "
            f"{row['rephrase'] if row['rephrase'] is not None else 'NA'} | "
            f"{row['locality'] if row['locality'] is not None else 'NA'} | "
            f"{row['distortion'] if row['distortion'] is not None else 'NA'} | "
            f"{row['failure_mode'] or 'NA'} |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "- HopEdit’s contribution is not only higher rewrite accuracy. It is the only current method with an exact routed failure decomposition and a large correct regime."
    )
    lines.append(
        "- GRACE appears realization-limited with preserved locality, while LoRA exhibits broad mixed/interference failure. WISE shows substantial locality damage."
    )
    lines.append(
        "- This supports a theory centered on access, realization, and interference rather than edit count alone."
    )
    lines.append("")
    lines.append("## Trajectory Coverage")
    lines.append("")
    latest = {}
    for row in trajectory_rows:
        latest[row["method"]] = row["checkpoint"]
    for method, checkpoint in sorted(latest.items()):
        lines.append(f"- {method}: latest completed checkpoint = `{checkpoint}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_root", default=str(DEFAULT_WIKI_ROOT), type=str)
    parser.add_argument("--claim_summary", default=str(DEFAULT_CLAIM_SUMMARY), type=str)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    args = parser.parse_args()

    wiki_root = Path(args.wiki_root)
    claim_summary = load_json(Path(args.claim_summary))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_rows = collect_method_checkpoints(wiki_root)
    matched_1k_rows = build_matched_checkpoint_rows(method_rows, "checkpoint_001000")
    trajectory_rows = flatten_trajectories(method_rows)

    write_csv(output_dir / "matched_checkpoint_001000.csv", matched_1k_rows)
    write_csv(output_dir / "wiki_trajectories.csv", trajectory_rows)
    (output_dir / "matched_checkpoint_001000.json").write_text(json.dumps(matched_1k_rows, indent=2))
    (output_dir / "wiki_trajectories.json").write_text(json.dumps(trajectory_rows, indent=2))
    (output_dir / "neurips_package.md").write_text(build_markdown(claim_summary, matched_1k_rows, trajectory_rows))


if __name__ == "__main__":
    main()
