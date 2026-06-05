"""Native shared-dictionary + residual substrate report.

This runner makes the method object explicit:

    exact update Δ_i^E
    shared dictionary code D α_i
    residual r_i = Δ_i^E - D α_i
    served update Δ_i(z_i) = D α_i + z_i r_i

It produces per-edit compressibility diagnostics and a residual-budget frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor
from easyeditor.editors.utils import _prepare_requests
from examples.edit_experiment_utils import (
    backbone_slug,
    build_editor_inputs,
    load_normalized_records,
    method_name,
    resolve_hparams_class,
    write_json,
)
from examples.run_edit_experiment import seed_everything
from examples.run_shared_basis_compression_poc import (
    capture_adapter_bank,
    load_vectors_into_adapters,
    make_shards,
)
from examples.run_wikibigedit_lifelong import (
    apply_single_edit,
    compute_pre_metrics,
    configure_evaluation_mode,
    evaluate_and_write,
)


@dataclass
class EditResidualRecord:
    vector_row_index: int
    edit_id: str
    dataset: str
    relation: str | None
    alpha_json: str
    exact_norm: float
    dict_norm: float
    residual_norm: float
    relative_residual_norm: float
    reconstruction_error: float
    exact_rewrite: float | None
    exact_rephrase: float | None
    exact_locality: float | None
    dict_rewrite: float | None
    dict_rephrase: float | None
    dict_locality: float | None
    dict_plus_residual_rewrite: float | None
    dict_plus_residual_rephrase: float | None
    dict_plus_residual_locality: float | None
    utility_exact: float | None
    utility_dict: float | None
    utility_dict_plus_residual: float | None
    utility_gap: float | None
    contract_gap: float | None
    dict_contract_pass: bool
    subject: str | None
    prompt: str | None
    target_new: str | None


def parse_budgets(raw: str, num_vectors: int) -> list[int]:
    values = set()
    for token in str(raw or "").split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "all":
            values.add(int(num_vectors))
        else:
            values.add(max(0, min(int(token), int(num_vectors))))
    values.add(0)
    values.add(int(num_vectors))
    return sorted(values)


def per_case_by_id(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in summary.get("per_case") or []:
        case_id = row.get("case_id")
        if case_id is not None:
            rows[int(case_id)] = row
    return rows


def metric_or_none(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def per_case_utility(row: dict[str, Any], weights: tuple[float, float, float]) -> float | None:
    rewrite = metric_or_none(row, "post_rewrite_acc")
    rephrase = metric_or_none(row, "post_rephrase_acc")
    locality = metric_or_none(row, "post_locality_acc")
    if rewrite is None or rephrase is None or locality is None:
        return None
    w_r, w_p, w_l = weights
    return float(w_r * rewrite + w_p * rephrase + w_l * locality)


def contract_pass(
    row: dict[str, Any],
    *,
    rewrite_threshold: float,
    rephrase_threshold: float,
    locality_threshold: float,
) -> bool:
    rewrite = metric_or_none(row, "post_rewrite_acc")
    rephrase = metric_or_none(row, "post_rephrase_acc")
    locality = metric_or_none(row, "post_locality_acc")
    if rewrite is None or rephrase is None or locality is None:
        return False
    return (
        rewrite >= rewrite_threshold
        and rephrase >= rephrase_threshold
        and locality >= locality_threshold
    )


def contract_gap_value(
    row: dict[str, Any],
    *,
    rewrite_threshold: float,
    rephrase_threshold: float,
    locality_threshold: float,
    weights: tuple[float, float, float],
) -> float:
    rewrite = metric_or_none(row, "post_rewrite_acc") or 0.0
    rephrase = metric_or_none(row, "post_rephrase_acc") or 0.0
    locality = metric_or_none(row, "post_locality_acc") or 0.0
    w_r, w_p, w_l = weights
    return float(
        w_r * max(0.0, rewrite_threshold - rewrite)
        + w_p * max(0.0, rephrase_threshold - rephrase)
        + w_l * max(0.0, locality_threshold - locality)
    )


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + end - 1) / 2.0 + 1.0
        for idx in order[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 1.0e-12 or den_y <= 1.0e-12:
        return None
    return float(num / (den_x * den_y))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def median_or_none(values: list[float | None]) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2 == 1:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def summary_subset(
    summary: dict[str, Any],
    *,
    rewrite_threshold: float,
    rephrase_threshold: float,
    locality_threshold: float,
) -> dict[str, Any]:
    per_case = summary.get("per_case") or []
    return {
        "post_rewrite_mean": summary.get("post_rewrite_mean"),
        "post_rephrase_mean": summary.get("post_rephrase_mean"),
        "post_locality_mean": summary.get("post_locality_mean"),
        "contract_pass_rate": None
        if not per_case
        else float(
            sum(
                1
                for row in per_case
                if contract_pass(
                    row,
                    rewrite_threshold=rewrite_threshold,
                    rephrase_threshold=rephrase_threshold,
                    locality_threshold=locality_threshold,
                )
            )
            / len(per_case)
        ),
    }


def tensor_vector_schema(weights: dict[str, torch.Tensor]) -> list[tuple[str, tuple[int, ...], int]]:
    schema = []
    for name in sorted(weights):
        tensor = weights[name].detach().cpu()
        schema.append((name, tuple(int(dim) for dim in tensor.shape), int(tensor.numel())))
    return schema


def flatten_weights(weights: dict[str, torch.Tensor], schema: list[tuple[str, tuple[int, ...], int]]) -> torch.Tensor:
    parts = []
    for name, _shape, _numel in schema:
        parts.append(weights[name].detach().float().cpu().reshape(-1))
    return torch.cat(parts, dim=0)


def factorize_shard(matrix: torch.Tensor, rank: int, *, center: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    matrix = matrix.detach().float().cpu()
    rows, dim = int(matrix.shape[0]), int(matrix.shape[1])
    mean = matrix.mean(dim=0, keepdim=True) if center else torch.zeros(1, dim, dtype=matrix.dtype)
    residual = matrix - mean
    total_energy = float((residual * residual).sum().item())
    max_rank = max(0, min(int(rank), rows - 1 if center else rows))
    if rows <= 0 or max_rank <= 0 or total_energy <= 1.0e-12:
        codes = torch.zeros((rows, 0), dtype=matrix.dtype)
        recon = mean.expand_as(matrix).clone()
        return recon, codes, mean, torch.zeros((0, dim), dtype=matrix.dtype), 0
    gram = residual @ residual.T
    eigvals, eigvecs = torch.linalg.eigh(gram.double())
    order = torch.argsort(eigvals, descending=True)
    keep = []
    singular = []
    for eig_idx in order.tolist():
        value = float(eigvals[eig_idx].item())
        if value <= 1.0e-10:
            continue
        keep.append(int(eig_idx))
        singular.append(value ** 0.5)
        if len(keep) >= max_rank:
            break
    if not keep:
        codes = torch.zeros((rows, 0), dtype=matrix.dtype)
        recon = mean.expand_as(matrix).clone()
        return recon, codes, mean, torch.zeros((0, dim), dtype=matrix.dtype), 0
    left = eigvecs[:, keep].float()
    singular_values = torch.tensor(singular, dtype=matrix.dtype)
    codes = left * singular_values.unsqueeze(0)
    basis = (residual.T @ left) / singular_values.unsqueeze(0)
    recon = mean + codes @ basis.T
    return recon, codes, mean, basis.T.contiguous(), len(keep)


def factorize_dictionary(
    vectors: torch.Tensor,
    groups: list[list[int]],
    rank: int,
    *,
    center: bool,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[dict[str, Any]]]:
    recon = torch.zeros_like(vectors)
    alpha_by_index: dict[int, torch.Tensor] = {}
    shard_rows = []
    normalized = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    similarity = normalized @ normalized.T
    for shard_id, indices in enumerate(groups):
        shard_matrix = vectors[indices]
        shard_recon, codes, mean, basis, rank_eff = factorize_shard(shard_matrix, rank, center=center)
        recon[indices] = shard_recon
        for local_idx, global_idx in enumerate(indices):
            alpha_by_index[int(global_idx)] = codes[local_idx].detach().cpu()
        sub = similarity[indices][:, indices]
        try:
            lambda_max = float(torch.linalg.eigvalsh(sub.double()).max().item()) if len(indices) > 1 else 1.0
        except RuntimeError:
            lambda_max = float("nan")
        residual = shard_matrix - shard_recon
        total_energy = float(((shard_matrix - mean) ** 2).sum().item())
        tail_energy = float((residual ** 2).sum().item())
        shard_rows.append(
            {
                "shard_id": shard_id,
                "size": len(indices),
                "rank_eff": int(rank_eff),
                "lambda_max": lambda_max,
                "tail_energy_ratio": None if total_energy <= 1.0e-12 else float(tail_energy / total_energy),
                "indices": indices,
            }
        )
    return recon, alpha_by_index, shard_rows


def hybrid_compression_stats(
    *,
    num_vectors: int,
    vector_dim: int,
    groups: list[list[int]],
    shard_rows: list[dict[str, Any]],
    keep_indices: list[int],
    center: bool,
    charge_dictionary_for_all_edits: bool,
) -> dict[str, Any]:
    keep_set = set(int(idx) for idx in keep_indices)
    exact_params = int(num_vectors * vector_dim)
    basis_params = 0
    coefficient_params = 0
    retired_count = 0
    active_shards = 0
    for group, row in zip(groups, shard_rows):
        supported_in_group = group if charge_dictionary_for_all_edits else [idx for idx in group if idx not in keep_set]
        if not supported_in_group:
            continue
        active_shards += 1
        retired_count += len([idx for idx in group if idx not in keep_set])
        rank_eff = int(row.get("rank_eff") or 0)
        basis_params += (rank_eff + (1 if center else 0)) * vector_dim
        coefficient_params += len(supported_in_group) * rank_eff
    residual_params = int(len(keep_set) * vector_dim)
    total = int(basis_params + coefficient_params + residual_params)
    return {
        "exact_params": exact_params,
        "dictionary_params": int(basis_params),
        "code_params": int(coefficient_params),
        "residual_params": residual_params,
        "total_params": total,
        "memory_fraction_vs_exact": None if exact_params <= 0 else float(total / exact_params),
        "compression_ratio_vs_exact": None if total <= 0 else float(exact_params / total),
        "kept_residual_count": int(len(keep_set)),
    }


def compute_memory_break_even(
    *,
    n_edits: int,
    p_lora: int,
    dictionary_params: int,
    code_params_all: int,
    p_residual: int | None = None,
) -> dict[str, Any]:
    if p_residual is None:
        p_residual = p_lora
    exact_total = int(n_edits * p_lora)
    shared_base = int(dictionary_params + code_params_all)
    margin = int(exact_total - shared_base)
    shared_exact_equivalent = None if p_lora <= 0 else float(shared_base / p_lora)
    if margin < 0:
        max_strict = -1
        max_leq = -1
    else:
        max_strict = int((margin - 1) // p_residual) if p_residual > 0 else -1
        max_leq = int(margin // p_residual) if p_residual > 0 else -1
    max_strict = min(max_strict, int(n_edits))
    max_leq = min(max_leq, int(n_edits))
    return {
        "n_edits": int(n_edits),
        "p_lora": int(p_lora),
        "p_residual": int(p_residual),
        "exact_total_params": int(exact_total),
        "dictionary_params": int(dictionary_params),
        "code_params_all": int(code_params_all),
        "shared_base_params": int(shared_base),
        "shared_exact_equivalent": shared_exact_equivalent,
        "shared_base_fraction": None if exact_total <= 0 else float(shared_base / exact_total),
        "memory_margin_before_residuals": int(margin),
        "max_residuals_strictly_below_exact": int(max_strict),
        "max_residuals_at_or_below_exact": int(max_leq),
        "retired_edits_required_strictly_below_exact": int(n_edits - max_strict) if max_strict >= 0 else int(n_edits + 1),
    }


def audit_frontier_memory_monotonicity(frontier_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in frontier_rows:
        policy = str(row.get("policy") or "")
        rows_by_policy.setdefault(policy, []).append(row)
    checks: list[dict[str, Any]] = []
    all_pass = True
    for policy, rows in rows_by_policy.items():
        ordered = sorted(rows, key=lambda item: int(item.get("budget_k") or 0))
        last_total: int | None = None
        last_ratio: float | None = None
        violations: list[dict[str, Any]] = []
        for row in ordered:
            total = row.get("total_params")
            ratio = row.get("compression_ratio_vs_exact")
            if total is None:
                continue
            total_int = int(total)
            if last_total is not None and total_int < last_total:
                violations.append(
                    {
                        "budget_k": int(row.get("budget_k") or 0),
                        "prev_total_params": int(last_total),
                        "total_params": total_int,
                        "kind": "total_params",
                    }
                )
            last_total = total_int
            if ratio is not None:
                ratio_float = float(ratio)
                if last_ratio is not None and ratio_float > last_ratio + 1.0e-9:
                    violations.append(
                        {
                            "budget_k": int(row.get("budget_k") or 0),
                            "prev_compression_ratio_vs_exact": float(last_ratio),
                            "compression_ratio_vs_exact": ratio_float,
                            "kind": "compression_ratio_vs_exact",
                        }
                    )
                last_ratio = ratio_float
        policy_pass = len(violations) == 0
        all_pass = all_pass and policy_pass
        checks.append(
            {
                "policy": policy,
                "monotone_non_decreasing_total_params": policy_pass,
                "monotone_non_increasing_compression_ratio": policy_pass,
                "violations": violations,
            }
        )
    return {
        "all_policies_monotone_non_decreasing_total_params": all_pass,
        "policy_checks": checks,
    }


def audit_frontier_memory_consistency(
    frontier_rows: list[dict[str, Any]],
    *,
    exact_total_params: int,
    shared_exact_equivalent: float | None,
    memory_margin_before_residuals: int,
    max_residuals_strictly_below_exact: int | None,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for row in frontier_rows:
        total = row.get("total_params")
        compression_ratio = row.get("compression_ratio_vs_exact")
        strict_win = row.get("is_strict_memory_win")
        row_shared_exact_equivalent = row.get("shared_exact_equivalent")
        row_max_strict = row.get("max_residuals_strictly_below_exact")
        row_memory_margin = row.get("memory_margin_before_residuals")
        if total is None:
            continue
        total_int = int(total)
        expected_strict_win = total_int < int(exact_total_params)
        checks = {
            "strict_memory_win_matches_total_params": (strict_win is None) or (bool(strict_win) == expected_strict_win),
            "compression_ratio_matches_total_params": True,
            "shared_exact_equivalent_constant": True,
            "memory_margin_constant": True,
            "max_residuals_strictly_below_exact_constant": True,
        }
        if compression_ratio is not None:
            ratio = float(compression_ratio)
            if total_int < int(exact_total_params):
                checks["compression_ratio_matches_total_params"] = ratio > 1.0
            elif total_int == int(exact_total_params):
                checks["compression_ratio_matches_total_params"] = abs(ratio - 1.0) <= 1.0e-9
            else:
                checks["compression_ratio_matches_total_params"] = ratio < 1.0
        if shared_exact_equivalent is not None and row_shared_exact_equivalent is not None:
            checks["shared_exact_equivalent_constant"] = abs(float(row_shared_exact_equivalent) - float(shared_exact_equivalent)) <= 1.0e-9
        if row_memory_margin is not None:
            checks["memory_margin_constant"] = int(row_memory_margin) == int(memory_margin_before_residuals)
        if max_residuals_strictly_below_exact is not None and row_max_strict is not None:
            checks["max_residuals_strictly_below_exact_constant"] = int(row_max_strict) == int(max_residuals_strictly_below_exact)
        if not all(checks.values()):
            violations.append(
                {
                    "policy": row.get("policy"),
                    "budget_k": row.get("budget_k"),
                    "total_params": total_int,
                    "compression_ratio_vs_exact": compression_ratio,
                    "is_strict_memory_win": strict_win,
                    "checks": checks,
                }
            )
    return {
        "all_rows_consistent": len(violations) == 0,
        "violations": violations,
    }


def summarize_best_frontier_points(
    frontier_rows: list[dict[str, Any]],
    *,
    utility_weights: tuple[float, float, float],
) -> dict[str, Any]:
    wr, wp, wl = utility_weights

    def utility(row: dict[str, Any]) -> float:
        return (
            wr * float(row.get("post_rewrite_mean") or 0.0)
            + wp * float(row.get("post_rephrase_mean") or 0.0)
            + wl * float(row.get("post_locality_mean") or 0.0)
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(row.get("contract_pass_rate") or 0.0),
            utility(row),
            -float(row.get("memory_fraction_vs_exact") or 0.0),
        )

    strict_rows = [row for row in frontier_rows if bool(row.get("is_strict_memory_win"))]
    leq_rows = [row for row in frontier_rows if row.get("memory_fraction_vs_exact") is not None and float(row["memory_fraction_vs_exact"]) <= 1.0 + 1.0e-9]
    best_strict = max(strict_rows, key=score) if strict_rows else None
    best_leq = max(leq_rows, key=score) if leq_rows else None

    def export(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "policy": row.get("policy"),
            "budget_k": row.get("budget_k"),
            "utility_mean": utility(row),
            "contract_pass_rate": row.get("contract_pass_rate"),
            "post_rewrite_mean": row.get("post_rewrite_mean"),
            "post_rephrase_mean": row.get("post_rephrase_mean"),
            "post_locality_mean": row.get("post_locality_mean"),
            "memory_fraction_vs_exact": row.get("memory_fraction_vs_exact"),
            "compression_ratio_vs_exact": row.get("compression_ratio_vs_exact"),
            "is_strict_memory_win": row.get("is_strict_memory_win"),
        }

    return {
        "best_strict_memory_win_point": export(best_strict),
        "best_at_or_below_exact_point": export(best_leq),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_rgb_png(path: Path, width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> None:
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend([r, g, b])
    data = zlib.compress(bytes(raw), level=9)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header) + png_chunk(b"IDAT", data) + png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def blank_canvas(width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)) -> list[list[tuple[int, int, int]]]:
    return [[color for _ in range(width)] for _ in range(height)]


def draw_rect(pixels, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width - 1, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height - 1, y1))
    if x1 < x0 or y1 < y0:
        return
    for y in range(y0, y1 + 1):
        row = pixels[y]
        for x in range(x0, x1 + 1):
            row[x] = color


def draw_line(pixels, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= y0 < len(pixels) and 0 <= x0 < len(pixels[0]):
            pixels[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def plot_gap_histogram(path: Path, values: list[float], *, width: int = 800, height: int = 500, bins: int = 12) -> None:
    pixels = blank_canvas(width, height)
    margin_left, margin_right, margin_top, margin_bottom = 60, 30, 30, 50
    draw_line(pixels, margin_left, height - margin_bottom, width - margin_right, height - margin_bottom, (0, 0, 0))
    draw_line(pixels, margin_left, margin_top, margin_left, height - margin_bottom, (0, 0, 0))
    if not values:
        write_rgb_png(path, width, height, pixels)
        return
    min_v = min(values)
    max_v = max(values)
    if max_v - min_v <= 1.0e-9:
        max_v = min_v + 1.0
    counts = [0] * bins
    for value in values:
        idx = int((value - min_v) / (max_v - min_v) * bins)
        idx = min(bins - 1, max(0, idx))
        counts[idx] += 1
    max_count = max(counts) or 1
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    bar_w = max(1, plot_w // bins - 4)
    for idx, count in enumerate(counts):
        x0 = margin_left + idx * plot_w // bins + 2
        x1 = x0 + bar_w
        bar_h = int(plot_h * (count / max_count))
        y1 = height - margin_bottom - 1
        y0 = y1 - bar_h
        draw_rect(pixels, x0, y0, x1, y1, (70, 130, 180))
    write_rgb_png(path, width, height, pixels)


def plot_frontier(path: Path, rows: list[dict[str, Any]], *, width: int = 900, height: int = 550) -> None:
    pixels = blank_canvas(width, height)
    margin_left, margin_right, margin_top, margin_bottom = 70, 40, 30, 60
    draw_line(pixels, margin_left, height - margin_bottom, width - margin_right, height - margin_bottom, (0, 0, 0))
    draw_line(pixels, margin_left, margin_top, margin_left, height - margin_bottom, (0, 0, 0))
    if not rows:
        write_rgb_png(path, width, height, pixels)
        return
    colors = {
        "random": (220, 20, 60),
        "recency": (255, 140, 0),
        "residual_norm": (65, 105, 225),
        "contract_gap": (34, 139, 34),
        "oracle_utility_gap": (138, 43, 226),
        "dictionary_only": (105, 105, 105),
        "exact_only": (0, 0, 0),
    }
    xs = [float(row["memory_fraction_vs_exact"]) for row in rows if row.get("memory_fraction_vs_exact") is not None]
    ys = [float(row["contract_pass_rate"]) for row in rows if row.get("contract_pass_rate") is not None]
    if not xs or not ys:
        write_rgb_png(path, width, height, pixels)
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(1.0, max(ys))
    if x_max - x_min <= 1.0e-9:
        x_max = x_min + 1.0
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(row)
    for policy, policy_rows in grouped.items():
        policy_rows = sorted(policy_rows, key=lambda row: (float(row["memory_fraction_vs_exact"]), int(row["budget_k"])))
        points = []
        for row in policy_rows:
            x_val = float(row["memory_fraction_vs_exact"])
            y_val = float(row["contract_pass_rate"])
            x = margin_left + int((x_val - x_min) / (x_max - x_min) * (width - margin_left - margin_right))
            y = height - margin_bottom - int((y_val - y_min) / (y_max - y_min) * (height - margin_top - margin_bottom))
            points.append((x, y))
        color = colors.get(policy, (0, 0, 255))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            draw_line(pixels, x0, y0, x1, y1, color)
        for x, y in points:
            draw_rect(pixels, x - 2, y - 2, x + 2, y + 2, color)
    write_rgb_png(path, width, height, pixels)


def select_indices(policy: str, k: int, num_vectors: int, *, residual_norms: list[float], contract_gaps: list[float], utility_gaps: list[float], seed: int) -> list[int]:
    k = max(0, min(int(k), int(num_vectors)))
    if policy == "dictionary_only" or k <= 0:
        return []
    if policy == "exact_only" or k >= num_vectors:
        return list(range(num_vectors))
    if policy == "random":
        rng = random.Random(seed + 1000 * k)
        return sorted(rng.sample(list(range(num_vectors)), k))
    if policy == "recency":
        return list(range(num_vectors - k, num_vectors))
    if policy == "residual_norm":
        order = sorted(range(num_vectors), key=lambda idx: residual_norms[idx], reverse=True)
        return sorted(order[:k])
    if policy == "contract_gap":
        order = sorted(range(num_vectors), key=lambda idx: contract_gaps[idx], reverse=True)
        return sorted(order[:k])
    if policy == "oracle_utility_gap":
        order = sorted(range(num_vectors), key=lambda idx: utility_gaps[idx], reverse=True)
        return sorted(order[:k])
    raise ValueError(f"Unsupported policy: {policy}")


def run_policy_evaluation(
    *,
    controller,
    edit_ids: list[str],
    exact_weights: dict[str, dict[str, torch.Tensor]],
    schema: list[tuple[str, tuple[int, ...], int]],
    recon_vectors: torch.Tensor,
    exact_vectors: torch.Tensor,
    keep_indices: list[int],
    groups: list[list[int]],
    shard_rows: list[dict[str, Any]],
    vector_dim: int,
    center: bool,
    editor,
    hparams,
    method: str,
    backbone: str,
    output_dir: Path,
    records,
    requests,
    eval_metric: str,
    run_base: dict[str, Any],
    pre_metrics: list[dict[str, Any]],
    condition_name: str,
    charge_dictionary_for_all_edits: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mixed_vectors = recon_vectors.clone()
    if keep_indices:
        mixed_vectors[keep_indices] = exact_vectors[keep_indices]
    load_vectors_into_adapters(controller, edit_ids, mixed_vectors, schema, exact_weights)
    storage = hybrid_compression_stats(
        num_vectors=int(exact_vectors.shape[0]),
        vector_dim=vector_dim,
        groups=groups,
        shard_rows=shard_rows,
        keep_indices=keep_indices,
        center=center,
        charge_dictionary_for_all_edits=charge_dictionary_for_all_edits,
    )
    summary = evaluate_and_write(
        editor,
        hparams,
        method,
        backbone,
        output_dir,
        records,
        requests,
        eval_metric,
        {
            **run_base,
            "run_name": condition_name,
            "condition": condition_name,
            "output_dir": str(output_dir.resolve()),
            **storage,
        },
        pre_metrics=pre_metrics,
    )
    return summary, storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", default="HOPEDIT")
    parser.add_argument("--hparams_dir", default=str(REPO_ROOT / "hparams/HOPEDIT/qwen2.5-7b-instruct-dual-whitened-collisionaware-staged.yaml"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--data_type", default="CounterFact", choices=["CounterFact", "ZsRE", "WikiBigEdit"])
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs/shared_dictionary_residual"))
    parser.add_argument("--ds_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", default="relation")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_rephrase_source", choices=["heldout", "address"], default="address")
    parser.add_argument("--evaluation_mode", default="teacher_forcing")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--utility_weights", default="1,1,1")
    parser.add_argument("--contract_rewrite_threshold", type=float, default=0.8)
    parser.add_argument("--contract_rephrase_threshold", type=float, default=0.8)
    parser.add_argument("--contract_locality_threshold", type=float, default=0.95)
    parser.add_argument("--residual_budgets", default="0,1,2,4,8,16,all")
    parser.add_argument("--high_gap_threshold", type=float, default=0.1)
    args = parser.parse_args()

    seed_everything(args.seed)
    dataset_slug = str(args.data_type).lower()
    output_dir = Path(args.output_root) / dataset_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = tuple(float(part) for part in args.utility_weights.split(","))
    if len(weights) != 3:
        raise ValueError("--utility_weights must have three comma-separated values")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Utility weights must sum to a positive value")
    weights = tuple(float(value / weight_sum) for value in weights)

    method = method_name(args.editing_method)
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    hparams.sequential_edit = True
    hparams.eval_batch_size = int(args.eval_batch_size)
    if hasattr(hparams, "route_log_dir"):
        hparams.route_log_dir = str(output_dir / "route_logs")
    configure_evaluation_mode(hparams, args.evaluation_mode, args.api_key)

    records, dataset_file = load_normalized_records(args.data_dir, args.data_type, args.ds_size, data_file=args.data_file)
    editor_inputs = build_editor_inputs(records, args.data_type)
    eval_rephrase_prompts = editor_inputs["rephrase_prompts"]
    if args.eval_rephrase_source == "address" and editor_inputs.get("address_rephrase_prompts"):
        eval_rephrase_prompts = editor_inputs["address_rephrase_prompts"]
    requests = _prepare_requests(
        editor_inputs["prompts"],
        editor_inputs["target_new"],
        editor_inputs["ground_truth"],
        rephrase_prompts=eval_rephrase_prompts,
        locality_inputs=editor_inputs["locality_inputs"],
        portability_inputs=editor_inputs["portability_inputs"],
        subject=editor_inputs["subject"],
    )
    for idx, request in enumerate(requests):
        request.setdefault("case_id", idx)
        request.setdefault("relation_id", editor_inputs["relation_id"][idx])
        request.setdefault("source_index", editor_inputs["source_index"][idx])
        if editor_inputs.get("address_rephrase_prompts"):
            request.setdefault("address_rephrase_prompt", editor_inputs["address_rephrase_prompts"][idx])
        if editor_inputs.get("rephrase_prompts"):
            request.setdefault("heldout_rephrase_prompt", editor_inputs["rephrase_prompts"][idx])

    run_base = {
        "editing_method": method,
        "alg_name": hparams.alg_name,
        "model_name": hparams.model_name,
        "backbone": backbone_slug(hparams.model_name),
        "data_type": args.data_type,
        "dataset_file": str(Path(dataset_file).resolve()),
        "stream_type": "shared_dictionary_residual_poc",
        "seed": args.seed,
        "sequential_edit": True,
        "stream_length": len(records),
        "requested_ds_size": args.ds_size,
        "strategy": args.strategy,
        "rank": int(args.rank),
    }
    write_json(output_dir / "run_config.json", {**run_base, "residual_budgets": args.residual_budgets})

    editor = BaseEditor.from_hparams(hparams)
    pre_metrics = compute_pre_metrics(editor, requests, editor_inputs["eval_metric"])
    for request in requests:
        apply_single_edit(editor, request)
    controller = editor.model if hasattr(editor.model, "memory_entries") else None
    if controller is None:
        raise RuntimeError("Expected HopEditController after applying edits")

    entries = list(controller.memory_entries)
    edit_ids = [str(entry.get("edit_id")) for entry in entries]
    relation_ids = [entry.get("relation_id") or requests[idx].get("relation_id") for idx, entry in enumerate(entries)]
    exact_weights, schema, exact_vectors = capture_adapter_bank(controller, edit_ids)
    vector_dim = int(exact_vectors.shape[1])
    groups = make_shards(args.strategy, exact_vectors, relation_ids, num_shards=args.num_shards, seed=args.seed)
    if args.strategy.strip().lower() == "relation" and all(relation_id is None for relation_id in relation_ids):
        raise ValueError(
            "Relation sharding was requested, but this dataset provides no relation_id values. "
            "Choose --strategy random or --strategy spectral instead of silently collapsing to one shard."
        )
    dict_vectors, alpha_by_index, shard_rows = factorize_dictionary(exact_vectors, groups, args.rank, center=bool(args.center))
    initial_storage = hybrid_compression_stats(
        num_vectors=int(exact_vectors.shape[0]),
        vector_dim=vector_dim,
        groups=groups,
        shard_rows=shard_rows,
        keep_indices=[],
        center=bool(args.center),
        charge_dictionary_for_all_edits=True,
    )
    write_json(
        output_dir / "dictionary_metadata.json",
        {
            "dataset": dataset_slug,
            "n_edits": len(edit_ids),
            "vector_dim": vector_dim,
            "strategy": args.strategy,
            "rank": int(args.rank),
            "center": bool(args.center),
            "groups": groups,
            "shard_rows": shard_rows,
            "dictionary_params": initial_storage.get("dictionary_params"),
            "code_params_all": initial_storage.get("code_params"),
            "exact_params": initial_storage.get("exact_params"),
            "p_lora": int(vector_dim),
            "p_residual": int(vector_dim),
            "contract_thresholds": {
                "rewrite": args.contract_rewrite_threshold,
                "rephrase": args.contract_rephrase_threshold,
                "locality": args.contract_locality_threshold,
            },
            "utility_weights": list(weights),
            "canonicalization_warning": "LoRA factors are not canonicalized; vector banks live in raw adapter parameter space.",
        },
    )
    residual_vectors = exact_vectors - dict_vectors
    exact_norms = torch.linalg.norm(exact_vectors, dim=1)
    dict_norms = torch.linalg.norm(dict_vectors, dim=1)
    residual_norms = torch.linalg.norm(residual_vectors, dim=1)
    relative_residual_norms = residual_norms / exact_norms.clamp_min(1.0e-12)
    reconstruction_errors = residual_vectors.pow(2).sum(dim=1) / exact_vectors.pow(2).sum(dim=1).clamp_min(1.0e-12)

    exact_summary, exact_storage = run_policy_evaluation(
        controller=controller,
        edit_ids=edit_ids,
        exact_weights=exact_weights,
        schema=schema,
        recon_vectors=exact_vectors,
        exact_vectors=exact_vectors,
        keep_indices=list(range(len(edit_ids))),
        groups=groups,
        shard_rows=shard_rows,
        vector_dim=vector_dim,
        center=bool(args.center),
        editor=editor,
        hparams=hparams,
        method=method,
        backbone=run_base["backbone"],
        output_dir=output_dir / "mode_exact",
        records=records,
        requests=requests,
        eval_metric=editor_inputs["eval_metric"],
        run_base=run_base,
        pre_metrics=pre_metrics,
        condition_name=f"{dataset_slug}_exact",
        charge_dictionary_for_all_edits=False,
    )
    dict_summary, dict_storage = run_policy_evaluation(
        controller=controller,
        edit_ids=edit_ids,
        exact_weights=exact_weights,
        schema=schema,
        recon_vectors=dict_vectors,
        exact_vectors=exact_vectors,
        keep_indices=[],
        groups=groups,
        shard_rows=shard_rows,
        vector_dim=vector_dim,
        center=bool(args.center),
        editor=editor,
        hparams=hparams,
        method=method,
        backbone=run_base["backbone"],
        output_dir=output_dir / "mode_dictionary",
        records=records,
        requests=requests,
        eval_metric=editor_inputs["eval_metric"],
        run_base=run_base,
        pre_metrics=pre_metrics,
        condition_name=f"{dataset_slug}_dictionary",
        charge_dictionary_for_all_edits=True,
    )
    dict_plus_residual_summary, dict_plus_residual_storage = run_policy_evaluation(
        controller=controller,
        edit_ids=edit_ids,
        exact_weights=exact_weights,
        schema=schema,
        recon_vectors=dict_vectors,
        exact_vectors=exact_vectors,
        keep_indices=list(range(len(edit_ids))),
        groups=groups,
        shard_rows=shard_rows,
        vector_dim=vector_dim,
        center=bool(args.center),
        editor=editor,
        hparams=hparams,
        method=method,
        backbone=run_base["backbone"],
        output_dir=output_dir / "mode_dictionary_plus_residual",
        records=records,
        requests=requests,
        eval_metric=editor_inputs["eval_metric"],
        run_base=run_base,
        pre_metrics=pre_metrics,
        condition_name=f"{dataset_slug}_dictionary_plus_residual",
        charge_dictionary_for_all_edits=True,
    )

    alpha_bank = []
    max_alpha_dim = max((int(alpha.numel()) for alpha in alpha_by_index.values()), default=0)
    for idx in range(len(edit_ids)):
        alpha = alpha_by_index.get(idx, torch.zeros(0))
        padded = torch.zeros(max_alpha_dim, dtype=torch.float32)
        if alpha.numel() > 0:
            padded[: alpha.numel()] = alpha.detach().float().cpu()
        alpha_bank.append(padded)
    alpha_bank_tensor = torch.stack(alpha_bank, dim=0) if alpha_bank else torch.zeros((0, 0), dtype=torch.float32)
    torch.save(
        {
            "edit_ids": edit_ids,
            "exact_vectors": exact_vectors.detach().cpu(),
            "dict_vectors": dict_vectors.detach().cpu(),
            "residual_vectors": residual_vectors.detach().cpu(),
            "alpha_bank": alpha_bank_tensor,
            "alpha_dims": [int(alpha_by_index.get(idx, torch.zeros(0)).numel()) for idx in range(len(edit_ids))],
            "groups": groups,
            "shard_rows": shard_rows,
            "schema": schema,
            "note": "Raw adapter-space vectors are saved for reproducibility; these are not canonicalized LoRA factors.",
        },
        output_dir / "vector_bank.pt",
    )

    exact_by_id = per_case_by_id(exact_summary)
    dict_by_id = per_case_by_id(dict_summary)
    dpr_by_id = per_case_by_id(dict_plus_residual_summary)

    records_out: list[EditResidualRecord] = []
    utility_gaps = []
    contract_gaps = []
    residual_norm_list = []
    for idx, request in enumerate(requests):
        exact_row = exact_by_id.get(idx) or {}
        dict_row = dict_by_id.get(idx) or {}
        dpr_row = dpr_by_id.get(idx) or {}
        utility_exact = per_case_utility(exact_row, weights)
        utility_dict = per_case_utility(dict_row, weights)
        utility_dpr = per_case_utility(dpr_row, weights)
        utility_gap = None if utility_exact is None or utility_dict is None else float(utility_exact - utility_dict)
        contract_gap = contract_gap_value(
            dict_row,
            rewrite_threshold=args.contract_rewrite_threshold,
            rephrase_threshold=args.contract_rephrase_threshold,
            locality_threshold=args.contract_locality_threshold,
            weights=weights,
        )
        record = EditResidualRecord(
            vector_row_index=idx,
            edit_id=edit_ids[idx],
            dataset=dataset_slug,
            relation=request.get("relation_id"),
            alpha_json=json.dumps([float(x) for x in alpha_by_index.get(idx, torch.zeros(0)).tolist()]),
            exact_norm=float(exact_norms[idx].item()),
            dict_norm=float(dict_norms[idx].item()),
            residual_norm=float(residual_norms[idx].item()),
            relative_residual_norm=float(relative_residual_norms[idx].item()),
            reconstruction_error=float(reconstruction_errors[idx].item()),
            exact_rewrite=metric_or_none(exact_row, "post_rewrite_acc"),
            exact_rephrase=metric_or_none(exact_row, "post_rephrase_acc"),
            exact_locality=metric_or_none(exact_row, "post_locality_acc"),
            dict_rewrite=metric_or_none(dict_row, "post_rewrite_acc"),
            dict_rephrase=metric_or_none(dict_row, "post_rephrase_acc"),
            dict_locality=metric_or_none(dict_row, "post_locality_acc"),
            dict_plus_residual_rewrite=metric_or_none(dpr_row, "post_rewrite_acc"),
            dict_plus_residual_rephrase=metric_or_none(dpr_row, "post_rephrase_acc"),
            dict_plus_residual_locality=metric_or_none(dpr_row, "post_locality_acc"),
            utility_exact=utility_exact,
            utility_dict=utility_dict,
            utility_dict_plus_residual=utility_dpr,
            utility_gap=utility_gap,
            contract_gap=contract_gap,
            dict_contract_pass=contract_pass(
                dict_row,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            subject=request.get("subject"),
            prompt=request.get("prompt"),
            target_new=request.get("target_new"),
        )
        records_out.append(record)
        residual_norm_list.append(record.residual_norm)
        contract_gaps.append(contract_gap)
        utility_gaps.append(0.0 if utility_gap is None else utility_gap)

    per_edit_rows = [asdict(record) for record in records_out]
    write_csv(output_dir / "per_edit.csv", per_edit_rows)

    exact_dpr_case_gaps = []
    for idx in range(len(edit_ids)):
        exact_row = exact_by_id.get(idx) or {}
        dpr_row = dpr_by_id.get(idx) or {}
        for key in ("post_rewrite_acc", "post_rephrase_acc", "post_locality_acc"):
            exact_val = metric_or_none(exact_row, key)
            dpr_val = metric_or_none(dpr_row, key)
            if exact_val is None or dpr_val is None:
                continue
            exact_dpr_case_gaps.append(abs(exact_val - dpr_val))
    exact_dpr_consistency = {
        "mode_rewrite_gap": None
        if exact_summary.get("post_rewrite_mean") is None or dict_plus_residual_summary.get("post_rewrite_mean") is None
        else float(abs(float(exact_summary["post_rewrite_mean"]) - float(dict_plus_residual_summary["post_rewrite_mean"]))),
        "mode_rephrase_gap": None
        if exact_summary.get("post_rephrase_mean") is None or dict_plus_residual_summary.get("post_rephrase_mean") is None
        else float(abs(float(exact_summary["post_rephrase_mean"]) - float(dict_plus_residual_summary["post_rephrase_mean"]))),
        "mode_locality_gap": None
        if exact_summary.get("post_locality_mean") is None or dict_plus_residual_summary.get("post_locality_mean") is None
        else float(abs(float(exact_summary["post_locality_mean"]) - float(dict_plus_residual_summary["post_locality_mean"]))),
        "max_per_case_metric_gap": None if not exact_dpr_case_gaps else float(max(exact_dpr_case_gaps)),
    }

    frontier_rows: list[dict[str, Any]] = []
    budgets = parse_budgets(args.residual_budgets, len(edit_ids))
    policies = ["dictionary_only", "random", "recency", "residual_norm", "contract_gap", "oracle_utility_gap", "exact_only"]
    memory_audit = compute_memory_break_even(
        n_edits=len(edit_ids),
        p_lora=vector_dim,
        dictionary_params=int(dict_storage.get("dictionary_params") or 0),
        code_params_all=int(dict_storage.get("code_params") or 0),
        p_residual=vector_dim,
    )
    for budget in budgets:
        for policy in policies:
            if policy == "dictionary_only" and budget != 0:
                continue
            if policy == "exact_only" and budget != len(edit_ids):
                continue
            if policy not in {"dictionary_only", "exact_only"} and budget == len(edit_ids):
                continue
            if policy not in {"dictionary_only", "exact_only"} and budget == 0:
                continue
            keep_indices = select_indices(
                policy,
                budget,
                len(edit_ids),
                residual_norms=residual_norm_list,
                contract_gaps=contract_gaps,
                utility_gaps=utility_gaps,
                seed=args.seed,
            )
            if policy == "dictionary_only":
                summary, storage = dict_summary, dict_storage
            elif policy == "exact_only":
                summary, storage = exact_summary, exact_storage
            else:
                summary, storage = run_policy_evaluation(
                    controller=controller,
                    edit_ids=edit_ids,
                    exact_weights=exact_weights,
                    schema=schema,
                    recon_vectors=dict_vectors,
                    exact_vectors=exact_vectors,
                    keep_indices=keep_indices,
                    groups=groups,
                    shard_rows=shard_rows,
                    vector_dim=vector_dim,
                    center=bool(args.center),
                    editor=editor,
                    hparams=hparams,
                    method=method,
                    backbone=run_base["backbone"],
                    output_dir=output_dir / f"policy_{policy}_k{budget}",
                    records=records,
                    requests=requests,
                    eval_metric=editor_inputs["eval_metric"],
                    run_base=run_base,
                    pre_metrics=pre_metrics,
                    condition_name=f"{dataset_slug}_{policy}_k{budget}",
                    charge_dictionary_for_all_edits=True,
                )
            summary_row = summary_subset(
                summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            )
            frontier_rows.append(
                {
                    "budget_k": int(budget),
                    "policy": policy,
                    "post_rewrite_mean": summary_row.get("post_rewrite_mean"),
                    "post_rephrase_mean": summary_row.get("post_rephrase_mean"),
                    "post_locality_mean": summary_row.get("post_locality_mean"),
                    "contract_pass_rate": summary_row.get("contract_pass_rate"),
                    "memory_fraction_vs_exact": storage.get("memory_fraction_vs_exact"),
                    "compression_ratio_vs_exact": storage.get("compression_ratio_vs_exact"),
                    "dictionary_params": storage.get("dictionary_params"),
                    "code_params": storage.get("code_params"),
                    "residual_params": storage.get("residual_params"),
                    "total_params": storage.get("total_params"),
                    "kept_residual_count": storage.get("kept_residual_count"),
                    "shared_exact_equivalent": memory_audit.get("shared_exact_equivalent"),
                    "shared_base_fraction": memory_audit.get("shared_base_fraction"),
                    "memory_margin_before_residuals": memory_audit.get("memory_margin_before_residuals"),
                    "max_residuals_strictly_below_exact": memory_audit.get("max_residuals_strictly_below_exact"),
                    "is_strict_memory_win": None
                    if storage.get("total_params") is None or dict_storage.get("exact_params") is None
                    else bool(int(storage["total_params"]) < int(dict_storage["exact_params"])),
                }
            )
    write_csv(output_dir / "frontier.csv", frontier_rows)
    frontier_memory_audit = audit_frontier_memory_monotonicity(frontier_rows)
    frontier_row_consistency = audit_frontier_memory_consistency(
        frontier_rows,
        exact_total_params=int(memory_audit["exact_total_params"]),
        shared_exact_equivalent=memory_audit.get("shared_exact_equivalent"),
        memory_margin_before_residuals=int(memory_audit["memory_margin_before_residuals"]),
        max_residuals_strictly_below_exact=memory_audit.get("max_residuals_strictly_below_exact"),
    )
    frontier_best_points = summarize_best_frontier_points(frontier_rows, utility_weights=weights)

    utility_gap_values = [record.utility_gap for record in records_out]
    relative_residual_values = [record.relative_residual_norm for record in records_out]
    contract_gap_values = [record.contract_gap for record in records_out]
    compressibility_summary = {
        "mean_utility_gap": mean_or_none(utility_gap_values),
        "median_utility_gap": median_or_none(utility_gap_values),
        "mean_relative_residual_norm": mean_or_none(relative_residual_values),
        "median_relative_residual_norm": median_or_none(relative_residual_values),
        "dictionary_contract_pass_rate": float(sum(1 for record in records_out if record.dict_contract_pass) / len(records_out)) if records_out else None,
        "fraction_high_gap_edits": float(sum(1 for value in utility_gap_values if value is not None and value > args.high_gap_threshold) / len(records_out)) if records_out else None,
        "high_gap_threshold": float(args.high_gap_threshold),
        "spearman_residual_norm_vs_utility_gap": spearman(
            [record.residual_norm for record in records_out],
            [0.0 if record.utility_gap is None else record.utility_gap for record in records_out],
        ),
        "spearman_contract_gap_vs_utility_gap": spearman(
            [record.contract_gap or 0.0 for record in records_out],
            [0.0 if record.utility_gap is None else record.utility_gap for record in records_out],
        ),
        "contract_gap_vs_oracle_utility_gap_corr": spearman(
            [record.contract_gap or 0.0 for record in records_out],
            [0.0 if record.utility_gap is None else record.utility_gap for record in records_out],
        ),
    }

    frontier_by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in frontier_rows:
        frontier_by_policy.setdefault(str(row["policy"]), []).append(row)
    for policy_rows in frontier_by_policy.values():
        policy_rows.sort(key=lambda row: (int(row["budget_k"]), float(row["memory_fraction_vs_exact"] or 0.0)))

    summary = {
        "dataset": dataset_slug,
        "n_edits": len(edit_ids),
        "dictionary_rank": int(args.rank),
        "strategy": args.strategy,
        "code_fitting_mode": "weight_only",
        "implementation_notes": {
            "dictionary_representation": "raw_adapter_vector_space",
            "canonicalization_warning": "LoRA factors are not canonicalized; vector banks live in raw adapter parameter space.",
            "behavior_refinement": "not_implemented_yet",
        },
        "mode_metrics": {
            "exact": summary_subset(
                exact_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            "dictionary": summary_subset(
                dict_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
            "dictionary_plus_residual": summary_subset(
                dict_plus_residual_summary,
                rewrite_threshold=args.contract_rewrite_threshold,
                rephrase_threshold=args.contract_rephrase_threshold,
                locality_threshold=args.contract_locality_threshold,
            ),
        },
        "compressibility": compressibility_summary,
        "memory_audit": memory_audit,
        "frontier_memory_audit": frontier_memory_audit,
        "frontier_row_consistency": frontier_row_consistency,
        "frontier_best_points": frontier_best_points,
        "exact_vs_dictionary_plus_residual_consistency": exact_dpr_consistency,
        "frontier": frontier_by_policy,
    }
    write_json(output_dir / "summary.json", summary)

    lines = [
        "# Shared Dictionary Residual Report",
        "",
        "## Dataset and Configuration",
        "",
        f"- dataset: `{dataset_slug}`",
        f"- model: `{hparams.model_name}`",
        f"- edits: `{len(edit_ids)}`",
        f"- strategy: `{args.strategy}`",
        f"- dictionary_rank: `{args.rank}`",
        f"- utility_weights: `{weights}`",
        f"- contract thresholds: `(rewrite={args.contract_rewrite_threshold}, rephrase={args.contract_rephrase_threshold}, locality={args.contract_locality_threshold})`",
            "",
            "## Implementation Notes",
            "",
            "- code fitting mode: `weight_only`",
            "- warning: raw adapter vectors are not canonicalized LoRA factors",
            "- behavioral refinement of dictionary codes is not implemented in this runner yet",
            "",
            "## Memory Audit",
            "",
            f"- exact total params: `{memory_audit['exact_total_params']}`",
            f"- params per exact LoRA: `{memory_audit['p_lora']}`",
            f"- shared base params (dictionary + all codes): `{memory_audit['shared_base_params']}`",
            f"- shared exact equivalent: `{memory_audit['shared_exact_equivalent']}`",
            f"- shared base fraction: `{memory_audit['shared_base_fraction']}`",
            f"- params per retained residual: `{memory_audit['p_residual']}`",
            f"- memory margin before residuals: `{memory_audit['memory_margin_before_residuals']}`",
            f"- max residuals strictly below exact: `{memory_audit['max_residuals_strictly_below_exact']}`",
            f"- max residuals at or below exact: `{memory_audit['max_residuals_at_or_below_exact']}`",
            f"- retired edits required for strict memory win: `{memory_audit['retired_edits_required_strictly_below_exact']}`",
            f"- frontier memory monotonicity passed: `{frontier_memory_audit['all_policies_monotone_non_decreasing_total_params']}`",
            f"- frontier row consistency passed: `{frontier_row_consistency['all_rows_consistent']}`",
            "",
            "## Aggregate Mode Comparison",
            "",
        "| Mode | Rewrite | Rephrase | Locality | Contract |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode_name in ["exact", "dictionary", "dictionary_plus_residual"]:
        mode_summary = summary["mode_metrics"][mode_name]
        lines.append(
            f"| {mode_name} | {float(mode_summary.get('post_rewrite_mean') or 0.0):.4f} | {float(mode_summary.get('post_rephrase_mean') or 0.0):.4f} | {float(mode_summary.get('post_locality_mean') or 0.0):.4f} | {float(mode_summary.get('contract_pass_rate') or 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Compressibility Statistics",
            "",
            f"- mean utility gap: `{compressibility_summary['mean_utility_gap']}`",
            f"- median utility gap: `{compressibility_summary['median_utility_gap']}`",
            f"- mean relative residual norm: `{compressibility_summary['mean_relative_residual_norm']}`",
            f"- median relative residual norm: `{compressibility_summary['median_relative_residual_norm']}`",
            f"- dictionary contract pass rate: `{compressibility_summary['dictionary_contract_pass_rate']}`",
            f"- fraction high-gap edits (`>{args.high_gap_threshold}`): `{compressibility_summary['fraction_high_gap_edits']}`",
            f"- Spearman residual norm vs utility gap: `{compressibility_summary['spearman_residual_norm_vs_utility_gap']}`",
            f"- Spearman contract gap vs utility gap: `{compressibility_summary['spearman_contract_gap_vs_utility_gap']}`",
            f"- Spearman contract gap vs oracle utility gap: `{compressibility_summary['contract_gap_vs_oracle_utility_gap_corr']}`",
            "",
            "## Exact vs Dictionary+Residual Consistency",
            "",
            f"- rewrite gap: `{exact_dpr_consistency['mode_rewrite_gap']}`",
            f"- rephrase gap: `{exact_dpr_consistency['mode_rephrase_gap']}`",
            f"- locality gap: `{exact_dpr_consistency['mode_locality_gap']}`",
            f"- max per-case metric gap: `{exact_dpr_consistency['max_per_case_metric_gap']}`",
            "",
            "## Frontier Memory Audit",
            "",
        ]
    )
    for audit_row in frontier_memory_audit["policy_checks"]:
        lines.append(
            f"- {audit_row['policy']}: monotone_non_decreasing_total_params=`{audit_row['monotone_non_decreasing_total_params']}`, monotone_non_increasing_compression_ratio=`{audit_row['monotone_non_increasing_compression_ratio']}`, violations=`{audit_row['violations']}`"
        )
    lines.extend(
        [
            "",
            "## Frontier Best Points",
            "",
            f"- best strict-memory-win point: `{frontier_best_points['best_strict_memory_win_point']}`",
            f"- best at-or-below-exact point: `{frontier_best_points['best_at_or_below_exact_point']}`",
            f"- frontier row consistency violations: `{frontier_row_consistency['violations']}`",
            "",
            "## Residual-Budget Frontier",
            "",
            "| Policy | Budget | Rewrite | Rephrase | Locality | Contract | Memory Fraction | Strict Win |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in frontier_rows:
        lines.append(
            f"| {row['policy']} | {row['budget_k']} | {float(row['post_rewrite_mean'] or 0.0):.4f} | {float(row['post_rephrase_mean'] or 0.0):.4f} | {float(row['post_locality_mean'] or 0.0):.4f} | {float(row['contract_pass_rate'] or 0.0):.4f} | {float(row['memory_fraction_vs_exact'] or 0.0):.4f} | {row['is_strict_memory_win']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines))

    plot_gap_histogram(output_dir / "gap_histogram.png", [float(value or 0.0) for value in utility_gap_values])
    plot_frontier(output_dir / "frontier.png", frontier_rows)

    print(json.dumps(summary, indent=2))
    print(f"Report written to {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
