import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.edit_experiment_utils import load_normalized_records, write_json


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mean(values: list[float | None]) -> float | None:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def percentile_summary(values: list[int | float]) -> dict[str, float]:
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


def normalize_spaces(text: str) -> str:
    return " ".join(str(text or "").split())


def mask_literal(text: str, value: Any, token: str) -> str:
    if value is None:
        return text
    value = normalize_spaces(str(value))
    if not value:
        return text
    return re.sub(re.escape(value), token, text, flags=re.IGNORECASE)


def make_text(record: dict[str, Any], field: str, mode: str, prefix: str) -> str:
    text = normalize_spaces(record.get(field) or record.get("prompt") or "")
    if mode in {"mask_subject", "mask_subject_object"}:
        text = mask_literal(text, record.get("subject"), "[SUBJ]")
    if mode == "mask_subject_object":
        text = mask_literal(text, record.get("target_new"), "[OBJ]")
        text = mask_literal(text, record.get("ground_truth"), "[OBJ]")
    if prefix:
        text = f"{prefix}{text}"
    return text


def encode_sentence_transformer(model_name: str, texts: list[str], *, batch_size: int, device: str) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.detach().float().cpu()


def build_whitener(vectors: torch.Tensor, eps: float) -> dict[str, Any]:
    matrix = vectors.detach().double().cpu()
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    keep = singular_values > float(eps)
    if not bool(keep.any().item()):
        raise ValueError("All singular values were below the whitening eps.")
    basis = vh[keep].T.contiguous()
    scale = (float(max(1, matrix.shape[0] - 1)) ** 0.5) / (singular_values[keep] + float(eps))
    return {
        "mean": mean,
        "basis": basis,
        "scale": scale,
        "rank": int(keep.sum().item()),
        "num_vectors": int(matrix.shape[0]),
        "eps": float(eps),
        "singular_min_kept": float(singular_values[keep].min().item()),
        "singular_max": float(singular_values.max().item()),
    }


def apply_whitener(vectors: torch.Tensor, whitener: dict[str, Any]) -> torch.Tensor:
    matrix = vectors.detach().double().cpu()
    whitened = ((matrix - whitener["mean"]) @ whitener["basis"]) * whitener["scale"]
    return F.normalize(whitened.float(), p=2, dim=-1)


def rank_of_target(scores: torch.Tensor, target_idx: int) -> int:
    order = torch.argsort(scores, descending=True)
    match = torch.nonzero(order == int(target_idx), as_tuple=False).flatten()
    return int(match[0].item()) + 1 if match.numel() else int(scores.numel())


def analyze_scores(scores: torch.Tensor) -> dict[str, Any]:
    n = int(scores.shape[0])
    ranks = [rank_of_target(scores[idx], idx) for idx in range(n)]
    target_scores = [float(scores[idx, idx].item()) for idx in range(n)]
    target_gaps = []
    for idx in range(n):
        row = scores[idx]
        if n <= 1:
            target_gaps.append(None)
            continue
        impostor_top = torch.max(torch.cat([row[:idx], row[idx + 1 :]])).item()
        target_gaps.append(float(row[idx].item() - impostor_top))
    top1 = [1.0 if rank == 1 else 0.0 for rank in ranks]
    return {
        "q_r": safe_mean(top1),
        "rank": percentile_summary(ranks),
        "target_score": percentile_summary(target_scores),
        "target_gap": percentile_summary([gap for gap in target_gaps if gap is not None]),
        "rank_le_1": float(np.mean(np.asarray(ranks) <= 1)),
        "rank_le_5": float(np.mean(np.asarray(ranks) <= 5)),
        "rank_le_10": float(np.mean(np.asarray(ranks) <= 10)),
        "rank_le_32": float(np.mean(np.asarray(ranks) <= 32)),
        "rank_le_100": float(np.mean(np.asarray(ranks) <= 100)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_type", default="CounterFact")
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ds_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--text_modes", default="full,mask_subject_object")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whiten_eps", type=float, default=1.0e-4)
    parser.add_argument("--trace_prefix", default="")
    parser.add_argument("--query_prefix", default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    records, dataset_file = load_normalized_records(
        args.data_dir,
        args.data_type,
        args.ds_size,
        data_file=args.data_file,
    )
    text_modes = [part.strip() for part in re.split(r"[:,\s]+", args.text_modes) if part.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for mode in text_modes:
        trace_texts = [make_text(record, "prompt", mode, args.trace_prefix) for record in records]
        query_texts = [make_text(record, "rephrase_prompt", mode, args.query_prefix) for record in records]
        all_texts = trace_texts + query_texts
        print(f"[external-relation-encoder] encoding mode={mode} texts={len(all_texts)}", flush=True)
        encoded = encode_sentence_transformer(
            args.encoder_model,
            all_texts,
            batch_size=args.batch_size,
            device=args.device,
        )
        trace = F.normalize(encoded[: len(records)], p=2, dim=-1)
        query = F.normalize(encoded[len(records) :], p=2, dim=-1)
        scores = query @ trace.T
        whitener = build_whitener(trace, args.whiten_eps)
        trace_whitened = apply_whitener(trace, whitener)
        query_whitened = apply_whitener(query, whitener)
        whitened_scores = query_whitened @ trace_whitened.T
        configs.append(
            {
                "text_mode": mode,
                "trace_prefix": args.trace_prefix,
                "query_prefix": args.query_prefix,
                "raw": analyze_scores(scores),
                "global_whiten": analyze_scores(whitened_scores),
                "whitener": {
                    "rank": whitener["rank"],
                    "num_vectors": whitener["num_vectors"],
                    "eps": whitener["eps"],
                    "singular_min_kept": whitener["singular_min_kept"],
                    "singular_max": whitener["singular_max"],
                },
            }
        )

    summary = {
        "run_name": f"hopedit_external_relation_encoder_{args.data_type.lower()}_{args.ds_size}_seed{args.seed}",
        "dataset_file": str(dataset_file),
        "encoder_model": args.encoder_model,
        "data_type": args.data_type,
        "ds_size": int(args.ds_size),
        "seed": int(args.seed),
        "configs": configs,
    }
    write_json(output_dir / "external_relation_encoder_summary.json", summary)
    compact = [
        {
            "text_mode": row["text_mode"],
            "raw_q_r": row["raw"]["q_r"],
            "raw_rank_p50": row["raw"]["rank"]["p50"],
            "raw_rank_le_100": row["raw"]["rank_le_100"],
            "whiten_q_r": row["global_whiten"]["q_r"],
            "whiten_rank_p50": row["global_whiten"]["rank"]["p50"],
            "whiten_rank_le_100": row["global_whiten"]["rank_le_100"],
        }
        for row in configs
    ]
    write_json(output_dir / "external_relation_encoder_compact.json", compact)
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
