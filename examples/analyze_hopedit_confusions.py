import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize_template(prompt: str, subject: str) -> str:
    text = (prompt or "").strip().lower()
    subj = (subject or "").strip().lower()
    if subj:
        text = text.replace(subj, "<subj>")
    text = re.sub(r"\s+", " ", text)
    return text


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def summarize_failures(per_case: list[dict], metric_key: str, threshold: float) -> list[tuple[str, int, list[tuple[str, float]]]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for case in per_case:
        value = case.get(metric_key)
        if value is None or value >= threshold:
            continue
        template = normalize_template(case.get("prompt", ""), case.get("subject", ""))
        grouped[template].append((case.get("subject", ""), float(value)))
    ranked = []
    for template, items in grouped.items():
        ranked.append((template, len(items), sorted(items, key=lambda row: row[1])))
    ranked.sort(key=lambda row: (-row[1], row[0]))
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HopEdit failures by subject-abstracted prompt template.")
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--rewrite-threshold", type=float, default=1.0)
    parser.add_argument("--rephrase-threshold", type=float, default=1.0)
    parser.add_argument("--locality-threshold", type=float, default=1.0)
    args = parser.parse_args()

    summary = load_summary(args.summary_json)
    per_case = summary.get("per_case") or []
    if not per_case:
        raise SystemExit("No per_case entries found in summary.")

    print(f"Loaded {len(per_case)} cases from {args.summary_json}")
    print()

    metrics = [
        ("post_rewrite_acc", args.rewrite_threshold, "rewrite"),
        ("post_rephrase_acc", args.rephrase_threshold, "rephrase"),
        ("post_locality_acc", args.locality_threshold, "locality"),
    ]
    for metric_key, threshold, label in metrics:
        failures = summarize_failures(per_case, metric_key, threshold)
        count = sum(group_size for _, group_size, _ in failures)
        print(f"{label} failures (< {threshold}): {count}")
        for template, group_size, items in failures[:10]:
            values = ", ".join(f"{subject}={value:.3f}" for subject, value in items[:5])
            print(f"  {group_size:>2}  {template} :: {values}")
        print()

    locality_templates = Counter(
        normalize_template(case.get("prompt", ""), case.get("subject", ""))
        for case in per_case
        if case.get("post_locality_acc") is not None and case["post_locality_acc"] < args.locality_threshold
    )
    print("Top locality templates:")
    for template, count in locality_templates.most_common(10):
        print(f"  {count:>2}  {template}")


if __name__ == "__main__":
    main()
