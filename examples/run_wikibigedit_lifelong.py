import argparse
import copy
import json
import math
import random
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyeditor import BaseEditor
from easyeditor.editors.utils import _prepare_requests
from easyeditor.evaluate import compute_edit_quality
from easyeditor.evaluate.evaluate import (
    compute_locality_quality,
    compute_portability_quality,
    compute_rewrite_or_rephrase_quality,
)
from easyeditor.models.hopedit.diagnostics import (
    annotate_route_logs,
    export_memory_snapshot,
    summarize_conflicts,
    summarize_route_diagnostics,
    summarize_trace_confusion_audit,
)
from easyeditor.models.hopedit.hopedit_main import load_hopedit_runtime_checkpoint
from easyeditor.evaluate.evaluate_utils import normalize_answer

from examples.edit_experiment_utils import (
    backbone_slug,
    mean_optional,
    metric_mean,
    nested_acc_mean,
    method_name,
    placeholder_artifact,
    resolve_hparams_class,
    summarize_run,
    write_json,
    write_jsonl,
)
from examples.checkpoint_suite_utils import (
    build_checkpoint_manifest,
    infer_memory_semantics,
    update_checkpoint_manifest_status,
    write_checkpoint_manifest,
)


INCREMENTS = [
    "20240201_20240220",
    "20240220_20240301",
    "20240301_20240320",
    "20240320_20240401",
    "20240401_20240501",
    "20240501_20240601",
    "20240601_20240620",
    "20240620_20240701",
]

REWRITE_SUCCESS_THRESHOLD = 0.5
LOCALITY_SUCCESS_THRESHOLD = 0.95
VALID_EVALUATION_MODES = {"teacher_forcing", "wild_em", "wild_llm_judge", "free_generation"}
VALID_PRE_EVAL_MODES = VALID_EVALUATION_MODES | {"match", "skip"}
VALID_EVAL_POLICIES = {"full", "skip"}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GPUKeepalive:
    def __init__(
        self,
        device,
        *,
        interval_seconds: float = 10.0,
        burst_seconds: float = 0.5,
        matrix_size: int = 2048,
        label: str = "checkpoint_phase",
    ):
        self.device = self._normalize_device(device)
        self.interval_seconds = max(float(interval_seconds), 1.0)
        self.burst_seconds = max(float(burst_seconds), 0.05)
        self.matrix_size = max(int(matrix_size), 256)
        self.label = label
        self._stop_event = threading.Event()
        self._thread = None
        self._error = None

    @staticmethod
    def _normalize_device(device):
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            if device.startswith("cuda"):
                return torch.device(device)
            if device.isdigit():
                return torch.device(f"cuda:{device}")
            return torch.device(device)
        return torch.device(f"cuda:{int(device)}")

    def _run(self):
        try:
            dtype = torch.float16
            work_a = torch.randn(
                (self.matrix_size, self.matrix_size),
                device=self.device,
                dtype=dtype,
            )
            work_b = torch.randn(
                (self.matrix_size, self.matrix_size),
                device=self.device,
                dtype=dtype,
            )
            work_out = torch.empty_like(work_a)
            while not self._stop_event.is_set():
                deadline = time.monotonic() + self.burst_seconds
                while time.monotonic() < deadline and not self._stop_event.is_set():
                    torch.mm(work_a, work_b, out=work_out)
                    work_a, work_out = work_out, work_a
                torch.cuda.synchronize(self.device)
                self._stop_event.wait(self.interval_seconds)
        except Exception as exc:  # pragma: no cover - defensive logging only
            self._error = exc

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"gpu-keepalive-{self.label}")
        self._thread.start()
        print(
            json.dumps(
                {
                    "phase": "gpu_keepalive_start",
                    "label": self.label,
                    "device": str(self.device),
                    "interval_seconds": self.interval_seconds,
                    "burst_seconds": self.burst_seconds,
                    "matrix_size": self.matrix_size,
                }
            ),
            flush=True,
        )

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(self.interval_seconds + self.burst_seconds + 5.0, 10.0))
        print(
            json.dumps(
                {
                    "phase": "gpu_keepalive_stop",
                    "label": self.label,
                    "had_error": self._error is not None,
                    "error": str(self._error) if self._error is not None else None,
                }
            ),
            flush=True,
        )
        self._thread = None


@contextmanager
def maybe_gpu_keepalive(args, hparams, label: str):
    if not getattr(args, "gpu_keepalive", True) or not torch.cuda.is_available():
        yield
        return
    keepalive = GPUKeepalive(
        getattr(hparams, "device", 0),
        interval_seconds=args.gpu_keepalive_interval_seconds,
        burst_seconds=args.gpu_keepalive_burst_seconds,
        matrix_size=args.gpu_keepalive_matrix_size,
        label=label,
    )
    keepalive.start()
    try:
        yield
    finally:
        keepalive.stop()


def snapshot_evaluation_state(hparams):
    return {
        "has_evaluation_type": hasattr(hparams, "evaluation_type"),
        "evaluation_type": getattr(hparams, "evaluation_type", None),
        "has_api_key": hasattr(hparams, "api_key"),
        "api_key": getattr(hparams, "api_key", None),
    }


def restore_evaluation_state(hparams, state: dict):
    if state.get("has_evaluation_type"):
        hparams.evaluation_type = state.get("evaluation_type")
    elif hasattr(hparams, "evaluation_type"):
        delattr(hparams, "evaluation_type")

    if state.get("has_api_key"):
        hparams.api_key = state.get("api_key")
    elif hasattr(hparams, "api_key"):
        delattr(hparams, "api_key")


def configure_evaluation_mode(hparams, mode: str, api_key: str | None = None):
    if mode == "teacher_forcing":
        if hasattr(hparams, "evaluation_type"):
            delattr(hparams, "evaluation_type")
        if hasattr(hparams, "api_key"):
            delattr(hparams, "api_key")
    elif mode == "free_generation":
        hparams.evaluation_type = "generate-text"
        if hasattr(hparams, "api_key"):
            delattr(hparams, "api_key")
    elif mode == "wild_em":
        hparams.evaluation_type = "LLM-judge"
        hparams.api_key = None
    elif mode == "wild_llm_judge":
        hparams.evaluation_type = "LLM-judge"
        hparams.api_key = api_key
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")


def generated_answer_score(prediction: str | None, target: str | None) -> float:
    pred_norm = normalize_answer("" if prediction is None else prediction)
    target_norm = normalize_answer("" if target is None else target)
    if not pred_norm or not target_norm:
        return 0.0
    if pred_norm == target_norm:
        return 1.0
    return 1.0 if target_norm in pred_norm else 0.0


def attach_free_generation_scores(metrics: list[dict]):
    for metric in metrics:
        request = metric.get("requested_rewrite", {})
        target_new = request.get("target_new")
        post = metric.get("post", {})
        pre = metric.get("pre", {})

        rewrite_gen = post.get("rewrite_gen_content")
        if rewrite_gen is not None and "rewrite_acc" not in post:
            post["rewrite_acc"] = [generated_answer_score(rewrite_gen, target_new)]
        rephrase_gen = post.get("rephrase_gen_content")
        if rephrase_gen is not None and "rephrase_acc" not in post:
            post["rephrase_acc"] = [generated_answer_score(rephrase_gen, target_new)]

        pre_rewrite_gen = pre.get("rewrite_gen_content")
        if pre_rewrite_gen is not None and "rewrite_acc" not in pre:
            pre["rewrite_acc"] = [generated_answer_score(pre_rewrite_gen, target_new)]
        pre_rephrase_gen = pre.get("rephrase_gen_content")
        if pre_rephrase_gen is not None and "rephrase_acc" not in pre:
            pre["rephrase_acc"] = [generated_answer_score(pre_rephrase_gen, target_new)]
    return metrics


def resolve_pre_metrics_path(pre_metrics_root: Path, method: str, backbone: str, increment: str) -> Path:
    candidates = [
        pre_metrics_root / f"{method.lower()}_{backbone}" / f"increment_{increment}" / "pre_metrics.json",
        pre_metrics_root / f"increment_{increment}" / "pre_metrics.json",
        pre_metrics_root / f"pre_metrics_{increment}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find pre_metrics.json for increment {increment} under {pre_metrics_root}. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def normalize_text_for_match(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def string_match(lhs: str | None, rhs: str | None) -> float:
    return float(normalize_text_for_match(lhs) == normalize_text_for_match(rhs))


def load_increment_records(increment_dir: Path, increment: str) -> tuple[list[dict], Path]:
    path = increment_dir / f"wiki_big_edit_{increment}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing WikiBigEdit increment file: {path}")
    payload = json.loads(path.read_text())
    return payload, path


def sample_records(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(records) <= limit:
        return list(records)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(records)), limit))
    return [records[idx] for idx in chosen]


def _clean_optional_text(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _clean_required_text(value, field_name: str):
    value = _clean_optional_text(value)
    if value is None:
        raise ValueError(f"WikiBigEdit required field {field_name} is missing or NaN")
    return value


def normalize_wikibigedit_records(records: list[dict], base_index: int = 0) -> list[dict]:
    normalized = []
    for offset, item in enumerate(records):
        prompt = _clean_optional_text(item.get("update") or item.get("prompt"))
        if prompt is None:
            raise KeyError(f"WikiBigEdit record {base_index + offset} is missing an update/prompt field")

        target_new = _clean_required_text(item.get("ans") or item.get("target_new"), "ans/target_new")
        ground_truth = _clean_optional_text(item.get("ground_truth")) or "<|endoftext|>"
        locality_prompt = _clean_optional_text(item.get("loc") or item.get("locality"))
        locality_ground_truth = _clean_optional_text(item.get("loc_ans") or item.get("locality_ans"))

        locality = {}
        if locality_prompt is not None and locality_ground_truth is not None:
            locality["locality"] = {
                "prompt": [locality_prompt],
                "ground_truth": [locality_ground_truth],
            }

        portability = {}
        personas_prompt = _clean_optional_text(item.get("personas") or item.get("portability_personas"))
        if personas_prompt is not None and target_new is not None:
            portability["personas"] = {
                "prompt": [personas_prompt],
                "ground_truth": [target_new],
            }
        mhop_prompt = _clean_optional_text(item.get("mhop") or item.get("portability_hop"))
        mhop_answer = _clean_optional_text(item.get("mhop_ans") or item.get("portability_hop_ans"))
        if mhop_prompt is not None and mhop_answer is not None:
            portability["mhop"] = {
                "prompt": [mhop_prompt],
                "ground_truth": [mhop_answer],
            }

        normalized.append(
            {
                "source_index": base_index + offset,
                "prompt": prompt,
                "subject": _clean_optional_text(item.get("subject")),
                "rephrase_prompt": _clean_optional_text(item.get("rephrase")),
                "target_new": target_new,
                "ground_truth": ground_truth,
                "locality": locality,
                "portability": portability,
                "tag": item.get("tag"),
            }
        )
    return normalized


def build_editor_inputs(records: list[dict]):
    prompts = [record["prompt"] for record in records]
    subject = [record["subject"] for record in records]
    target_new = [record["target_new"] for record in records]
    ground_truth = [record["ground_truth"] for record in records]
    rephrase_values = [record.get("rephrase_prompt") for record in records]
    rephrase_prompts = rephrase_values if any(value is not None for value in rephrase_values) else None

    locality_inputs = None
    locality_keys = sorted({key for record in records for key in (record.get("locality") or {}).keys()})
    if locality_keys:
        locality_inputs = {}
        for key in locality_keys:
            locality_inputs[key] = {"prompt": [], "ground_truth": []}
            for record in records:
                bucket = (record.get("locality") or {}).get(key)
                locality_inputs[key]["prompt"].append(bucket.get("prompt") if bucket is not None else None)
                locality_inputs[key]["ground_truth"].append(bucket.get("ground_truth") if bucket is not None else None)

    portability_inputs = None
    portability_keys = sorted({key for record in records for key in (record.get("portability") or {}).keys()})
    if portability_keys:
        portability_inputs = {}
        for key in portability_keys:
            portability_inputs[key] = {"prompt": [], "ground_truth": []}
            for record in records:
                bucket = (record.get("portability") or {}).get(key)
                portability_inputs[key]["prompt"].append(bucket.get("prompt") if bucket is not None else None)
                portability_inputs[key]["ground_truth"].append(bucket.get("ground_truth") if bucket is not None else None)

    requests = _prepare_requests(
        prompts,
        target_new,
        ground_truth,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        subject=subject,
    )
    for idx, request in enumerate(requests):
        request.setdefault("case_id", idx)
    return requests, "token_em"


def compute_pre_metrics(editor: BaseEditor, requests: list[dict], eval_metric: str):
    if should_batch_teacher_forcing_eval(editor, eval_metric):
        return [{"pre": row["post"]} for row in evaluate_requests(editor, requests, eval_metric)]
    rows = []
    for request in requests:
        rows.append(
            {
                "pre": compute_edit_quality(
                    editor.model,
                    editor.model_name,
                    editor.hparams,
                    editor.tok,
                    request,
                    editor.hparams.device,
                    eval_metric=eval_metric,
                    test_generation=False,
                )
            }
        )
    return rows


def evaluate_requests(editor: BaseEditor, requests: list[dict], eval_metric: str):
    if should_batch_teacher_forcing_eval(editor, eval_metric):
        return evaluate_requests_batched(editor, requests)
    rows = []
    for idx, request in enumerate(requests):
        post = compute_edit_quality(
            editor.model,
            editor.model_name,
            editor.hparams,
            editor.tok,
            request,
            editor.hparams.device,
            eval_metric=eval_metric,
            test_generation=False,
        )
        rows.append(
            {
                "case_id": idx,
                "requested_rewrite": request,
                "post": post,
                "time": None,
            }
        )
    return rows


def should_batch_teacher_forcing_eval(editor: BaseEditor, eval_metric: str) -> bool:
    return eval_metric == "token_em" and not hasattr(editor.hparams, "evaluation_type")


def _batched_metric_value(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _single_text(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _batched_locality_value(value):
    if value is None:
        return None
    return [value]


def evaluate_requests_batched(editor: BaseEditor, requests: list[dict]):
    batch_size = max(int(getattr(editor.hparams, "eval_batch_size", 32)), 1)
    rows = [
        {
            "case_id": idx,
            "requested_rewrite": request,
            "post": {"locality": {}, "portability": {}},
            "time": None,
        }
        for idx, request in enumerate(requests)
    ]

    for batch_start in range(0, len(requests), batch_size):
        batch_end = min(batch_start + batch_size, len(requests))
        batch_requests = requests[batch_start:batch_end]
        batch_rows = rows[batch_start:batch_end]

        rewrite_result = compute_rewrite_or_rephrase_quality(
            editor.model,
            editor.model_name,
            editor.hparams,
            editor.tok,
            [request["prompt"] for request in batch_requests],
            [request["target_new"] for request in batch_requests],
            device=editor.hparams.device,
            eval_metric="token_em",
        )
        rewrite_scores = rewrite_result.get("rewrite_acc") or []
        for row, score in zip(batch_rows, rewrite_scores):
            row["post"]["rewrite_acc"] = _batched_metric_value(score)

        rephrase_subset = [
            (idx, request["rephrase_prompt"], request["target_new"])
            for idx, request in enumerate(batch_requests)
            if request.get("rephrase_prompt") is not None
        ]
        if rephrase_subset:
            rephrase_result = compute_rewrite_or_rephrase_quality(
                editor.model,
                editor.model_name,
                editor.hparams,
                editor.tok,
                [prompt for _idx, prompt, _target in rephrase_subset],
                [target for _idx, _prompt, target in rephrase_subset],
                device=editor.hparams.device,
                test_rephrase=True,
                eval_metric="token_em",
            )
            rephrase_scores = rephrase_result.get("rephrase_acc") or []
            for (local_idx, _prompt, _target), score in zip(rephrase_subset, rephrase_scores):
                batch_rows[local_idx]["post"]["rephrase_acc"] = _batched_metric_value(score)

        locality_keys = sorted({key for request in batch_requests for key in (request.get("locality") or {}).keys()})
        for locality_key in locality_keys:
            locality_subset = []
            for idx, request in enumerate(batch_requests):
                bucket = (request.get("locality") or {}).get(locality_key)
                if bucket is None:
                    continue
                prompt = _single_text(bucket.get("prompt"))
                ground_truth = _single_text(bucket.get("ground_truth"))
                if prompt is None or ground_truth is None:
                    continue
                locality_subset.append((idx, prompt, ground_truth))
            if not locality_subset:
                continue
            locality_result = compute_locality_quality(
                editor.model,
                editor.model_name,
                editor.hparams,
                editor.tok,
                locality_key,
                [prompt for _idx, prompt, _ground_truth in locality_subset],
                [ground_truth for _idx, _prompt, ground_truth in locality_subset],
                device=editor.hparams.device,
            )
            locality_outputs = locality_result.get(f"{locality_key}_output") or []
            for (local_idx, _prompt, _ground_truth), output in zip(locality_subset, locality_outputs):
                batch_rows[local_idx]["post"]["locality"][f"{locality_key}_output"] = _batched_locality_value(output)

        portability_keys = sorted({key for request in batch_requests for key in (request.get("portability") or {}).keys()})
        for portability_key in portability_keys:
            portability_subset = []
            for idx, request in enumerate(batch_requests):
                bucket = (request.get("portability") or {}).get(portability_key)
                if bucket is None:
                    continue
                prompt = _single_text(bucket.get("prompt"))
                ground_truth = _single_text(bucket.get("ground_truth"))
                if prompt is None or ground_truth is None:
                    continue
                portability_subset.append((idx, prompt, ground_truth))
            if not portability_subset:
                continue
            portability_result = compute_portability_quality(
                editor.model,
                editor.model_name,
                editor.hparams,
                editor.tok,
                portability_key,
                [prompt for _idx, prompt, _ground_truth in portability_subset],
                [ground_truth for _idx, _prompt, ground_truth in portability_subset],
                device=editor.hparams.device,
            )
            portability_scores = portability_result.get(f"{portability_key}_acc") or []
            for (local_idx, _prompt, _ground_truth), score in zip(portability_subset, portability_scores):
                batch_rows[local_idx]["post"]["portability"][f"{portability_key}_acc"] = _batched_metric_value(score)

    return rows


def attach_pre_metrics(metrics: list[dict], pre_metrics: list[dict] | None):
    if pre_metrics is None:
        return metrics
    for metric, pre in zip(metrics, pre_metrics):
        # finalize_locality mutates metric["pre"] after comparing pre/post
        # locality outputs. Reusing checkpoint-level pre_metrics across multiple
        # evaluation conditions therefore requires an isolated copy per call.
        metric["pre"] = copy.deepcopy(pre["pre"])
    return metrics


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def save_runtime_checkpoint_if_supported(editor: BaseEditor, output_dir: Path, runner_state: dict[str, Any] | None = None):
    controller = editor.model if hasattr(editor.model, "save_runtime_checkpoint") else None
    if controller is None:
        return None
    checkpoint_dir = output_dir / "runtime_checkpoint"
    start = time.time()
    controller.save_runtime_checkpoint(str(checkpoint_dir))
    save_seconds = time.time() - start
    if runner_state is not None:
        write_json(checkpoint_dir / "runner_state.json", runner_state)
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_save_seconds": save_seconds,
        "checkpoint_size_bytes": directory_size_bytes(checkpoint_dir),
    }


def write_deferred_eval_manifest(output_dir: Path, payload: dict[str, Any]):
    write_json(output_dir / "deferred_eval_manifest.json", payload)


def finalize_locality(metrics: list[dict], hparams):
    for metric in metrics:
        pre = metric.get("pre", {})
        post = metric.get("post", {})
        if "locality" not in post:
            continue
        for locality_key in list(metric.get("requested_rewrite", {}).get("locality", {}).keys()):
            output_key = f"{locality_key}_output"
            if output_key not in post["locality"]:
                continue
            if output_key not in pre.get("locality", {}):
                continue
            locality_result = []
            post_outputs = _extract_locality_outputs(post["locality"][output_key])
            pre_outputs = _extract_locality_outputs(pre["locality"][output_key])
            for ans, label in zip(post_outputs, pre_outputs):
                if isinstance(ans, str) or isinstance(label, str):
                    locality_result.append(string_match(ans, label))
                else:
                    locality_result.append(float(np.mean(np.equal(ans, label))))
            post["locality"][f"{locality_key}_acc"] = locality_result
            post["locality"].pop(output_key, None)
        if "locality" in pre:
            pre.pop("locality")
    return metrics


def _extract_locality_outputs(value):
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], list) and isinstance(value[1], list):
        # LLM-judge / generation path returns (scores, responses).
        if value[1] and all(isinstance(item, str) for item in value[1]):
            return value[1]
        return value[0]
    return value


def apply_single_edit(editor: BaseEditor, request: dict):
    method = editor.alg_name
    if method in {"IKE", "ICE"}:
        edited_model, _weights_copy, _icl_examples = editor.model, {}, editor.apply_algo(
            editor.model,
            editor.tok,
            [request],
            editor.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
            train_ds=None,
        )
        return edited_model

    edited_model, _weights_copy = editor.apply_algo(
        editor.model,
        editor.tok,
        [request],
        editor.hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
        train_ds=None,
    )

    if method in {"HOPEDIT", "LoRA", "QLoRA", "DPO", "MELO", "SERAC"}:
        editor.model = edited_model
    return edited_model


def clear_controller_logs(editor: BaseEditor):
    controller = editor.model if hasattr(editor.model, "route_logs") else None
    if controller is not None:
        controller.route_logs = []


def collect_controller_state(editor: BaseEditor):
    controller = editor.model if hasattr(editor.model, "route_logs") else None
    route_logs = []
    memory_entries = []
    memory_snapshot = []
    prototype_diagnostics = []
    stability_diagnostics = []
    hierarchy_diagnostics = []
    gate_diagnostics = []
    slot_diagnostics = []
    state_diagnostics = []
    factor_space_diagnostics = []
    shard_diagnostics = []
    support_diagnostics = []
    realization_diagnostics = []
    if controller is not None:
        route_logs = list(getattr(controller, "route_logs", []))
        memory_entries = list(getattr(controller, "memory_entries", []))
        if hasattr(controller, "export_memory_snapshot"):
            memory_snapshot = controller.export_memory_snapshot(include_keys=False)
        else:
            memory_snapshot = export_memory_snapshot(memory_entries, include_keys=False)
        if hasattr(controller, "export_prototype_diagnostics"):
            prototype_diagnostics = controller.export_prototype_diagnostics()
        if hasattr(controller, "export_stability_diagnostics"):
            stability_diagnostics = controller.export_stability_diagnostics()
        if hasattr(controller, "export_hierarchy_diagnostics"):
            hierarchy_diagnostics = controller.export_hierarchy_diagnostics()
        if hasattr(controller, "export_gate_diagnostics"):
            gate_diagnostics = controller.export_gate_diagnostics()
        if hasattr(controller, "export_slot_diagnostics"):
            slot_diagnostics = controller.export_slot_diagnostics()
        if hasattr(controller, "export_state_diagnostics"):
            state_diagnostics = controller.export_state_diagnostics()
        if hasattr(controller, "export_factor_space_diagnostics"):
            factor_space_diagnostics = controller.export_factor_space_diagnostics()
        if hasattr(controller, "export_shard_diagnostics"):
            shard_diagnostics = controller.export_shard_diagnostics()
        if hasattr(controller, "export_support_diagnostics"):
            support_diagnostics = controller.export_support_diagnostics()
        if hasattr(controller, "export_realization_diagnostics"):
            realization_diagnostics = controller.export_realization_diagnostics()
    return route_logs, memory_entries, memory_snapshot, prototype_diagnostics, stability_diagnostics, hierarchy_diagnostics, gate_diagnostics, slot_diagnostics, state_diagnostics, factor_space_diagnostics, shard_diagnostics, support_diagnostics, realization_diagnostics


def collect_memory_semantics(editor: BaseEditor, memory_snapshot: list[dict] | None = None):
    controller = editor.model if hasattr(editor, "model") else None
    return infer_memory_semantics(
        method_name(getattr(editor, "alg_name", "unknown")),
        controller=controller,
        memory_snapshot=memory_snapshot,
    )


def safe_mean(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def percentile(values: list[float | None], q: float) -> float | None:
    filtered = sorted(float(v) for v in values if v is not None)
    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]
    rank = (len(filtered) - 1) * q
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return filtered[low]
    weight = rank - low
    return float(filtered[low] * (1.0 - weight) + filtered[high] * weight)


def primary_route_entry_by_case(annotated_logs: list[dict]) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    priority = {"rewrite": 0, "rephrase": 1, "post_edit": 2}
    for entry in annotated_logs:
        case_id = entry.get("case_id")
        if case_id is None:
            continue
        event_type = entry.get("event_type")
        if event_type not in priority:
            continue
        case_id = int(case_id)
        existing = mapping.get(case_id)
        if existing is None or priority[event_type] < priority.get(existing.get("event_type"), 99):
            mapping[case_id] = entry
    return mapping


def build_failure_decomposition(
    run_config: dict,
    metrics: list[dict],
    summary: dict,
    annotated_logs: list[dict],
) -> dict:
    route_by_case = primary_route_entry_by_case(annotated_logs)
    exact_supported = bool(route_by_case)
    cases = []
    for row in summary.get("per_case", []):
        case_id = int(row.get("case_id", 0))
        route_entry = route_by_case.get(case_id)
        rewrite_acc = row.get("post_rewrite_acc")
        locality_acc = row.get("post_locality_acc")
        rewrite_ok = rewrite_acc is not None and rewrite_acc >= REWRITE_SUCCESS_THRESHOLD
        locality_ok = locality_acc is None or locality_acc >= LOCALITY_SUCCESS_THRESHOLD
        route_observed = route_entry is not None and route_entry.get("correct_route") is not None

        if route_observed:
            correct_route = bool(route_entry.get("correct_route"))
            chosen_memory_id = route_entry.get("chosen_memory_id") or route_entry.get("chosen_cell_id") or route_entry.get("chosen_edit_id")
            if (not correct_route) or chosen_memory_id is None:
                label = "retrieval_failure"
            elif correct_route and (not rewrite_ok) and locality_ok:
                label = "update_failure"
            elif correct_route and rewrite_ok and (not locality_ok):
                label = "locality_failure"
            elif correct_route and (not rewrite_ok) and (not locality_ok):
                label = "mixed_failure"
            elif correct_route and rewrite_ok:
                label = "correct"
            else:
                label = "unknown"
        else:
            if rewrite_ok and locality_ok:
                label = "correct"
            elif (not rewrite_ok) and locality_ok:
                label = "update_failure_proxy"
            elif rewrite_ok and (not locality_ok):
                label = "locality_failure_proxy"
            elif (not rewrite_ok) and (not locality_ok):
                label = "mixed_failure_proxy"
            else:
                label = "unknown"

        cases.append(
            {
                "case_id": case_id,
                "source_index": row.get("source_index"),
                "subject": row.get("subject"),
                "prompt": row.get("prompt"),
                "label": label,
                "post_rewrite_acc": rewrite_acc,
                "post_rephrase_acc": row.get("post_rephrase_acc"),
                "post_locality_acc": locality_acc,
                "route_observed": route_observed,
                "correct_route": None if route_entry is None else route_entry.get("correct_route"),
                "chosen_memory_id": None if route_entry is None else (route_entry.get("chosen_memory_id") or route_entry.get("chosen_cell_id") or route_entry.get("chosen_edit_id")),
                "expected_memory_id": None if route_entry is None else route_entry.get("expected_memory_id"),
                "chosen_edit_id": None if route_entry is None else route_entry.get("chosen_edit_id"),
                "expected_edit_id": None if route_entry is None else route_entry.get("expected_edit_id"),
                "top1_prob": None if route_entry is None else route_entry.get("top1_prob"),
                "route_margin": None if route_entry is None else route_entry.get("route_margin"),
            }
        )

    counts: dict[str, int] = {}
    for row in cases:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    total = len(cases)
    rates = {key: (value / total if total else None) for key, value in counts.items()}
    return {
        "applicable": True,
        "editing_method": run_config["editing_method"],
        "run_name": run_config["run_name"],
        "exact_supported": exact_supported,
        "mode": "exact" if exact_supported else "proxy",
        "rewrite_success_threshold": REWRITE_SUCCESS_THRESHOLD,
        "locality_success_threshold": LOCALITY_SUCCESS_THRESHOLD,
        "counts": counts,
        "rates": rates,
        "per_case": cases,
    }


def build_theory_metrics(
    run_config: dict,
    summary: dict,
    route_diagnostics: dict,
    conflict_diagnostics: dict,
    failure_decomposition: dict,
    prototype_diagnostics: dict | None = None,
    stability_diagnostics: dict | None = None,
    hierarchy_diagnostics: dict | None = None,
    gate_diagnostics: dict | None = None,
    slot_diagnostics: dict | None = None,
    state_diagnostics: dict | None = None,
    factor_space_diagnostics: dict | None = None,
) -> dict:
    rw_losses = []
    rp_losses = []
    loc_losses = []
    primary_distortions = []
    extended_distortions = []
    for row in summary.get("per_case", []):
        rewrite_acc = row.get("post_rewrite_acc")
        rephrase_acc = row.get("post_rephrase_acc")
        locality_acc = row.get("post_locality_acc")
        rw_loss = None if rewrite_acc is None else 1.0 - float(rewrite_acc)
        rp_loss = None if rephrase_acc is None else 1.0 - float(rephrase_acc)
        loc_loss = None if locality_acc is None else 1.0 - float(locality_acc)
        rw_losses.append(rw_loss)
        rp_losses.append(rp_loss)
        loc_losses.append(loc_loss)
        if rw_loss is not None or loc_loss is not None:
            primary_distortions.append(safe_mean([rw_loss, loc_loss]))
        if rw_loss is not None or rp_loss is not None or loc_loss is not None:
            extended_distortions.append(safe_mean([rw_loss, rp_loss, loc_loss]))

    routing_summary = route_diagnostics.get("routing", {}) if isinstance(route_diagnostics, dict) else {}
    rewrite_route = routing_summary.get("rewrite", {}) if isinstance(routing_summary, dict) else {}
    locality_route = routing_summary.get("locality", {}) if isinstance(routing_summary, dict) else {}

    return {
        "applicable": True,
        "editing_method": run_config["editing_method"],
        "run_name": run_config["run_name"],
        "distortion": {
            "primary_definition": "mean(1-rewrite_acc, 1-locality_acc) over available channels",
            "extended_definition": "mean(1-rewrite_acc, 1-rephrase_acc, 1-locality_acc) over available channels",
            "rewrite_loss_mean": safe_mean(rw_losses),
            "rephrase_loss_mean": safe_mean(rp_losses),
            "locality_loss_mean": safe_mean(loc_losses),
            "primary_mean": safe_mean(primary_distortions),
            "primary_p50": percentile(primary_distortions, 0.5),
            "primary_p90": percentile(primary_distortions, 0.9),
            "primary_max": max((float(v) for v in primary_distortions if v is not None), default=None),
            "extended_mean": safe_mean(extended_distortions),
        },
        "performance": {
            "post_rewrite_mean": summary.get("post_rewrite_mean"),
            "post_rephrase_mean": summary.get("post_rephrase_mean"),
            "post_locality_mean": summary.get("post_locality_mean"),
            "rewrite_delta_mean": summary.get("rewrite_delta_mean"),
            "rephrase_delta_mean": summary.get("rephrase_delta_mean"),
            "early_late_gap": summary.get("early_late_gap"),
        },
        "routing": {
            "applicable": bool(rewrite_route),
            "rewrite_route_accuracy": rewrite_route.get("route_accuracy"),
            "rephrase_route_accuracy": routing_summary.get("rephrase", {}).get("route_accuracy"),
            "cross_view_route_gap": route_diagnostics.get("cross_view", {}).get("cross_view_route_gap") if isinstance(route_diagnostics, dict) else None,
            "rewrite_route_margin_mean": rewrite_route.get("route_margin_mean"),
            "rewrite_top1_prob_mean": rewrite_route.get("top1_prob_mean"),
            "rewrite_coverage": rewrite_route.get("coverage"),
            "rephrase_coverage": routing_summary.get("rephrase", {}).get("coverage"),
            "locality_false_activation_rate": locality_route.get("false_activation_rate"),
            "locality_no_edit_rate": locality_route.get("no_edit_rate"),
        },
        "conflict": {
            "applicable": bool(conflict_diagnostics) and conflict_diagnostics.get("applicable", True),
            "num_edits": conflict_diagnostics.get("num_edits") if isinstance(conflict_diagnostics, dict) else None,
            "num_cells": conflict_diagnostics.get("num_cells") if isinstance(conflict_diagnostics, dict) else None,
            "mean_combined_offdiag": conflict_diagnostics.get("mean_combined_offdiag") if isinstance(conflict_diagnostics, dict) else None,
            "mean_max_offdiag_conflict": conflict_diagnostics.get("mean_max_offdiag_conflict") if isinstance(conflict_diagnostics, dict) else None,
            "max_pair_conflict": conflict_diagnostics.get("max_pair_conflict") if isinstance(conflict_diagnostics, dict) else None,
            "within_cell_conflict_mean": conflict_diagnostics.get("within_cell_conflict_mean") if isinstance(conflict_diagnostics, dict) else None,
            "within_cell_conflict_max": conflict_diagnostics.get("within_cell_conflict_max") if isinstance(conflict_diagnostics, dict) else None,
            "mean_cell_max_conflict": conflict_diagnostics.get("mean_cell_max_conflict") if isinstance(conflict_diagnostics, dict) else None,
        },
        "prototypes": prototype_diagnostics if isinstance(prototype_diagnostics, dict) else None,
        "stability": stability_diagnostics if isinstance(stability_diagnostics, dict) else None,
        "hierarchy": hierarchy_diagnostics if isinstance(hierarchy_diagnostics, dict) else None,
        "gate": gate_diagnostics if isinstance(gate_diagnostics, dict) else None,
        "slots": slot_diagnostics if isinstance(slot_diagnostics, dict) else None,
        "states": state_diagnostics if isinstance(state_diagnostics, dict) else None,
        "factor_space": factor_space_diagnostics if isinstance(factor_space_diagnostics, dict) else None,
        "failure": {
            "mode": failure_decomposition.get("mode"),
            "exact_supported": failure_decomposition.get("exact_supported"),
            "rates": failure_decomposition.get("rates"),
        },
    }


def build_efficiency_metrics(
    run_config: dict,
    summary: dict,
    route_diagnostics: dict,
    memory_snapshot: list[dict] | dict,
) -> dict:
    stream_length = run_config.get("stream_length") or 0
    cumulative_edits = run_config.get("cumulative_edits") or 0
    eval_wall = run_config.get("eval_wall_time_seconds")
    edit_wall = run_config.get("increment_edit_wall_time_seconds")
    total_wall = run_config.get("wall_time_seconds")
    if isinstance(memory_snapshot, list):
        memory_entries_final = len(memory_snapshot)
        memory_unit = "edit"
        retained_units_final = memory_entries_final
        if memory_snapshot:
            snapshot_unit = memory_snapshot[0].get("memory_unit")
            if snapshot_unit in {"cell", "state"}:
                memory_unit = snapshot_unit
                retained_units_final = len({row.get("cell_id") for row in memory_snapshot if row.get("cell_id") is not None})
    else:
        memory_entries_final = None
        memory_unit = None
        retained_units_final = None
    return {
        "applicable": True,
        "editing_method": run_config["editing_method"],
        "run_name": run_config["run_name"],
        "increment": run_config.get("increment"),
        "stream_type": run_config.get("stream_type"),
        "stream_length": stream_length,
        "cumulative_edits": cumulative_edits,
        "edit_wall_time_seconds": edit_wall,
        "eval_wall_time_seconds": eval_wall,
        "total_wall_time_seconds": total_wall,
        "edit_seconds_per_edit": None if not edit_wall or not stream_length else float(edit_wall / stream_length),
        "eval_seconds_per_case": None if not eval_wall or not stream_length else float(eval_wall / stream_length),
        "cumulative_seconds_per_edit": None if not total_wall or not cumulative_edits else float(total_wall / cumulative_edits),
        "current_eval_cases_per_second": None if not eval_wall or not stream_length else float(stream_length / eval_wall),
        "memory_entries_final": memory_entries_final,
        "memory_entries_per_edit": None if memory_entries_final is None or not cumulative_edits else float(memory_entries_final / cumulative_edits),
        "memory_unit": memory_unit,
        "retained_units_final": retained_units_final,
        "checkpoint_save_seconds": run_config.get("checkpoint_save_seconds"),
        "checkpoint_load_seconds": run_config.get("checkpoint_load_seconds"),
        "checkpoint_size_bytes": run_config.get("checkpoint_size_bytes"),
        "bytes_per_retained_edit": None
        if not run_config.get("checkpoint_size_bytes") or not memory_entries_final
        else float(run_config["checkpoint_size_bytes"] / max(1, memory_entries_final)),
        "bytes_per_retained_unit": None
        if not run_config.get("checkpoint_size_bytes") or not retained_units_final
        else float(run_config["checkpoint_size_bytes"] / max(1, retained_units_final)),
        "mean_case_time_reported": summary.get("mean_time"),
        "route_events_logged": route_diagnostics.get("summary", {}).get("num_logged_events") if isinstance(route_diagnostics, dict) else None,
    }


def write_checkpoint_artifacts(
    output_dir: Path,
    run_config: dict,
    metrics: list[dict],
    eval_records: list[dict],
    route_logs: list[dict],
    memory_entries: list[dict],
    memory_snapshot: list[dict],
    prototype_diagnostics: dict,
    stability_diagnostics: dict,
    hierarchy_diagnostics: dict,
    gate_diagnostics: dict,
    slot_diagnostics: dict,
    state_diagnostics: dict,
    factor_space_diagnostics: dict,
    shard_diagnostics: dict,
    support_diagnostics: dict,
    realization_diagnostics: dict,
    hparams,
):
    write_json(output_dir / "metrics.json", metrics)
    annotated_logs = annotate_route_logs(route_logs, metrics) if route_logs else []
    write_jsonl(output_dir / "annotated_route_logs.jsonl", annotated_logs)
    route_diagnostics = summarize_route_diagnostics(annotated_logs, metrics) if route_logs else placeholder_artifact("route_diagnostics", run_config)
    conflict_diagnostics = (
        summarize_conflicts(memory_entries, hparams.semantic_weight, hparams.activation_weight)
        if memory_entries and hasattr(hparams, "semantic_weight") and hasattr(hparams, "activation_weight")
        else placeholder_artifact("conflict_diagnostics", run_config)
    )
    confusion_audit = summarize_trace_confusion_audit(annotated_logs, memory_snapshot) if annotated_logs and memory_snapshot else placeholder_artifact("trace_confusion_audit", run_config)
    write_json(output_dir / "route_diagnostics.json", route_diagnostics)
    write_json(output_dir / "conflict_diagnostics.json", conflict_diagnostics)
    write_json(output_dir / "trace_confusion_audit.json", confusion_audit)
    write_json(output_dir / "prototype_diagnostics.json", prototype_diagnostics if prototype_diagnostics else placeholder_artifact("prototype_diagnostics", run_config))
    write_json(output_dir / "stability_diagnostics.json", stability_diagnostics if stability_diagnostics else placeholder_artifact("stability_diagnostics", run_config))
    write_json(output_dir / "hierarchy_diagnostics.json", hierarchy_diagnostics if hierarchy_diagnostics else placeholder_artifact("hierarchy_diagnostics", run_config))
    write_json(output_dir / "gate_diagnostics.json", gate_diagnostics if gate_diagnostics else placeholder_artifact("gate_diagnostics", run_config))
    write_json(output_dir / "slot_diagnostics.json", slot_diagnostics if slot_diagnostics else placeholder_artifact("slot_diagnostics", run_config))
    write_json(output_dir / "state_diagnostics.json", state_diagnostics if state_diagnostics else placeholder_artifact("state_diagnostics", run_config))
    write_json(output_dir / "factor_space_diagnostics.json", factor_space_diagnostics if factor_space_diagnostics else placeholder_artifact("factor_space_diagnostics", run_config))
    write_json(output_dir / "shard_diagnostics.json", shard_diagnostics if shard_diagnostics else placeholder_artifact("shard_diagnostics", run_config))
    write_json(output_dir / "support_diagnostics.json", support_diagnostics if support_diagnostics else placeholder_artifact("support_diagnostics", run_config))
    write_json(output_dir / "realization_diagnostics.json", realization_diagnostics if realization_diagnostics else placeholder_artifact("realization_diagnostics", run_config))
    write_json(output_dir / "cross_view_route_gap.json", route_diagnostics.get("cross_view", {}) if isinstance(route_diagnostics, dict) else placeholder_artifact("cross_view_route_gap", run_config))
    write_json(output_dir / "memory_snapshot.json", memory_snapshot if memory_snapshot else placeholder_artifact("memory_snapshot", run_config))
    summary = summarize_run(metrics, eval_records, run_config, memory_snapshot if isinstance(memory_snapshot, list) else [])
    write_json(output_dir / "summary.json", summary)
    failure_decomposition = build_failure_decomposition(run_config, metrics, summary, annotated_logs)
    failure_ladder = {
        "run_name": run_config.get("run_name"),
        "memory_unit": memory_snapshot[0].get("memory_unit") if isinstance(memory_snapshot, list) and memory_snapshot else None,
        "post_rewrite_mean": summary.get("post_rewrite_mean"),
        "post_rephrase_mean": summary.get("post_rephrase_mean"),
        "post_locality_mean": summary.get("post_locality_mean"),
        "cross_view_route_gap": route_diagnostics.get("cross_view", {}).get("cross_view_route_gap") if isinstance(route_diagnostics, dict) else None,
        "within_conflict_mean": conflict_diagnostics.get("within_cell_conflict_mean") if isinstance(conflict_diagnostics, dict) else None,
        "slot_transfer_attempts": slot_diagnostics.get("slot_transfer_attempts") if isinstance(slot_diagnostics, dict) else None,
    }
    theory_metrics = build_theory_metrics(
        run_config,
        summary,
        route_diagnostics,
        conflict_diagnostics,
        failure_decomposition,
        prototype_diagnostics,
        stability_diagnostics,
        hierarchy_diagnostics,
        gate_diagnostics,
        slot_diagnostics,
        state_diagnostics,
        factor_space_diagnostics,
    )
    efficiency_metrics = build_efficiency_metrics(run_config, summary, route_diagnostics, memory_snapshot if memory_snapshot else [])
    write_json(output_dir / "failure_decomposition.json", failure_decomposition)
    write_json(output_dir / "failure_ladder.json", failure_ladder)
    write_json(output_dir / "theory_metrics.json", theory_metrics)
    write_json(output_dir / "efficiency_metrics.json", efficiency_metrics)
    return summary


def evaluate_and_write(
    editor: BaseEditor,
    hparams,
    method: str,
    backbone: str,
    output_dir: Path,
    eval_records: list[dict],
    eval_requests: list[dict],
    eval_metric: str,
    run_config: dict,
    pre_metrics: list[dict] | None = None,
):
    clear_controller_logs(editor)
    eval_start = time.time()
    metrics = evaluate_requests(editor, eval_requests, eval_metric)
    eval_seconds = time.time() - eval_start
    metrics = attach_pre_metrics(metrics, pre_metrics)
    metrics = attach_free_generation_scores(metrics)
    metrics = finalize_locality(metrics, hparams)
    (
        route_logs,
        memory_entries,
        memory_snapshot,
        prototype_diagnostics,
        stability_diagnostics,
        hierarchy_diagnostics,
        gate_diagnostics,
        slot_diagnostics,
        state_diagnostics,
        factor_space_diagnostics,
        shard_diagnostics,
        support_diagnostics,
        realization_diagnostics,
    ) = collect_controller_state(editor)
    run_config["eval_wall_time_seconds"] = eval_seconds
    write_json(output_dir / "run_config.json", run_config)
    summary = write_checkpoint_artifacts(
        output_dir,
        run_config,
        metrics,
        eval_records,
        route_logs,
        memory_entries,
        memory_snapshot,
        prototype_diagnostics,
        stability_diagnostics,
        hierarchy_diagnostics,
        gate_diagnostics,
        slot_diagnostics,
        state_diagnostics,
        factor_space_diagnostics,
        shard_diagnostics,
        support_diagnostics,
        realization_diagnostics,
        hparams,
    )
    if method == "HOPEDIT":
        write_json(output_dir / "hopedit_route_diagnostics.json", json.loads((output_dir / "route_diagnostics.json").read_text()))
        write_json(output_dir / "hopedit_conflict_diagnostics.json", json.loads((output_dir / "conflict_diagnostics.json").read_text()))
        write_json(output_dir / "hopedit_trace_confusion_audit.json", json.loads((output_dir / "trace_confusion_audit.json").read_text()))
        write_json(output_dir / "hopedit_memory_snapshot.json", json.loads((output_dir / "memory_snapshot.json").read_text()))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", required=True, type=str)
    parser.add_argument("--hparams_dir", required=True, type=str)
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"), type=str)
    parser.add_argument("--increment_dir", default=None, type=str)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs" / "wikibigedit_lifelong"), type=str)
    parser.add_argument("--increments", default=" ".join(INCREMENTS), type=str)
    parser.add_argument("--ds_size_per_increment", default=0, type=int)
    parser.add_argument("--past_eval_max_samples", default=0, type=int)
    parser.add_argument("--checkpoint_interval", default=0, type=int)
    parser.add_argument("--evaluation_mode", default="teacher_forcing", choices=sorted(VALID_EVALUATION_MODES), type=str)
    parser.add_argument("--pre_eval_mode", default="match", choices=sorted(VALID_PRE_EVAL_MODES), type=str)
    parser.add_argument("--pre_metrics_root", default=None, type=str)
    parser.add_argument("--resume_runtime_checkpoint", default=None, type=str)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint_eval_policy", default="full", choices=sorted(VALID_EVAL_POLICIES), type=str)
    parser.add_argument("--current_eval_policy", default="full", choices=sorted(VALID_EVAL_POLICIES), type=str)
    parser.add_argument("--past_eval_policy", default="full", choices=sorted(VALID_EVAL_POLICIES), type=str)
    parser.add_argument("--gpu_keepalive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_keepalive_interval_seconds", default=10.0, type=float)
    parser.add_argument("--gpu_keepalive_burst_seconds", default=0.5, type=float)
    parser.add_argument("--gpu_keepalive_matrix_size", default=2048, type=int)
    parser.add_argument("--api_key", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    seed_everything(args.seed)

    method = method_name(args.editing_method)
    hparams_class = resolve_hparams_class(method)
    hparams = hparams_class.from_hparams(args.hparams_dir)
    if not hasattr(hparams, "sequential_edit"):
        setattr(hparams, "sequential_edit", True)
    else:
        hparams.sequential_edit = True
    hparams.eval_batch_size = args.eval_batch_size
    configure_evaluation_mode(hparams, args.evaluation_mode, args.api_key)
    pre_eval_mode = args.evaluation_mode if args.pre_eval_mode == "match" else args.pre_eval_mode

    increment_dir = Path(args.increment_dir) if args.increment_dir else Path(args.data_dir) / "wikibigedit"
    requested_increments = [token for token in args.increments.split() if token]
    backbone = backbone_slug(hparams.model_name)

    output_root = Path(args.output_root) / f"{method.lower()}_{backbone}"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "editing_method": method,
        "alg_name": hparams.alg_name,
        "model_name": hparams.model_name,
        "backbone": backbone,
        "increment_dir": str(increment_dir.resolve()),
        "increments": requested_increments,
        "ds_size_per_increment": args.ds_size_per_increment,
        "past_eval_max_samples": args.past_eval_max_samples,
        "checkpoint_interval": args.checkpoint_interval,
        "seed": args.seed,
        "evaluation_mode": args.evaluation_mode,
        "pre_eval_mode": pre_eval_mode,
        "pre_metrics_root": None if args.pre_metrics_root is None else str(Path(args.pre_metrics_root).resolve()),
        "eval_batch_size": args.eval_batch_size,
        "checkpoint_eval_policy": args.checkpoint_eval_policy,
        "current_eval_policy": args.current_eval_policy,
        "past_eval_policy": args.past_eval_policy,
        "gpu_keepalive": args.gpu_keepalive,
        "gpu_keepalive_interval_seconds": args.gpu_keepalive_interval_seconds,
        "gpu_keepalive_burst_seconds": args.gpu_keepalive_burst_seconds,
        "gpu_keepalive_matrix_size": args.gpu_keepalive_matrix_size,
        "hparams_path": str(Path(args.hparams_dir).resolve()),
    }
    write_json(output_root / "run_manifest.json", manifest)

    editor = BaseEditor.from_hparams(hparams)
    all_increment_records: dict[str, list[dict]] = {}
    cumulative_edits = 0
    resume_increment = None
    resume_checkpoint_step = 0
    if args.resume_runtime_checkpoint is not None:
        resume_checkpoint_dir = Path(args.resume_runtime_checkpoint)
        runner_state_path = resume_checkpoint_dir / "runner_state.json"
        if not runner_state_path.exists():
            raise FileNotFoundError(f"Missing runner_state.json for runtime checkpoint: {runner_state_path}")
        runner_state = json.loads(runner_state_path.read_text())
        resume_increment = runner_state.get("increment")
        resume_checkpoint_step = int(runner_state.get("checkpoint_step") or 0)
        cumulative_edits = int(runner_state.get("cumulative_edits") or 0)
        if resume_increment not in requested_increments:
            raise ValueError(
                f"Resume increment {resume_increment} is not present in requested increments: {requested_increments}"
            )
        if method != "HOPEDIT":
            raise ValueError("--resume_runtime_checkpoint is currently supported only for HOPEDIT")
        resume_load_start = time.time()
        with maybe_gpu_keepalive(args, hparams, f"resume_load_{resume_increment}_{resume_checkpoint_step:06d}"):
            editor.model = load_hopedit_runtime_checkpoint(
                editor.model,
                editor.tok,
                hparams,
                str(resume_checkpoint_dir),
                is_trainable=True,
            )
        resume_load_seconds = time.time() - resume_load_start
        manifest["resume_runtime_checkpoint"] = str(resume_checkpoint_dir.resolve())
        manifest["resume_runtime_checkpoint_load_seconds"] = resume_load_seconds
        manifest["resume_runtime_checkpoint_size_bytes"] = directory_size_bytes(resume_checkpoint_dir)
        write_json(output_root / "run_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "phase": "resume_loaded",
                    "resume_runtime_checkpoint": str(resume_checkpoint_dir.resolve()),
                    "increment": resume_increment,
                    "checkpoint_step": resume_checkpoint_step,
                    "cumulative_edits": cumulative_edits,
                    "checkpoint_load_seconds": resume_load_seconds,
                }
            ),
            flush=True,
        )
    start_time = time.time()

    resume_increment_idx = None if resume_increment is None else requested_increments.index(resume_increment)

    for increment_idx, increment in enumerate(requested_increments):
        raw_records, dataset_file = load_increment_records(increment_dir, increment)
        sampled_records = sample_records(raw_records, args.ds_size_per_increment, args.seed + increment_idx)
        current_records = normalize_wikibigedit_records(sampled_records)
        current_requests, eval_metric = build_editor_inputs(current_records)

        if resume_increment_idx is not None and increment_idx < resume_increment_idx:
            all_increment_records[increment] = current_records
            continue

        increment_dir_out = output_root / f"increment_{increment}"
        increment_dir_out.mkdir(parents=True, exist_ok=True)
        write_json(
            increment_dir_out / "increment_manifest.json",
            {
                "increment": increment,
                "dataset_file": str(dataset_file.resolve()),
                "records_in_source_file": len(raw_records),
                "records_used": len(current_records),
            },
        )

        print(
            json.dumps(
                {
                    "phase": "pre_eval",
                    "increment": increment,
                    "records": len(current_records),
                    "evaluation_mode": pre_eval_mode,
                }
            ),
            flush=True,
        )
        pre_metrics = None
        pre_metrics_source = "none"
        if args.pre_metrics_root is not None:
            pre_metrics_path = resolve_pre_metrics_path(Path(args.pre_metrics_root), method, backbone, increment)
            pre_metrics = json.loads(pre_metrics_path.read_text())
            if len(pre_metrics) != len(current_requests):
                raise ValueError(
                    f"Pre-metrics length mismatch for increment {increment}: "
                    f"{len(pre_metrics)} pre-metrics rows vs {len(current_requests)} current requests"
                )
            pre_metrics_source = str(pre_metrics_path.resolve())
            print(
                json.dumps(
                    {
                        "phase": "pre_eval_loaded",
                        "increment": increment,
                        "pre_metrics_path": pre_metrics_source,
                        "records": len(pre_metrics),
                    }
                ),
                flush=True,
            )
        elif pre_eval_mode != "skip":
            pre_eval_state = snapshot_evaluation_state(editor.hparams)
            configure_evaluation_mode(editor.hparams, pre_eval_mode, args.api_key)
            try:
                with maybe_gpu_keepalive(args, editor.hparams, f"pre_eval_{increment}"):
                    pre_metrics = compute_pre_metrics(editor, current_requests, eval_metric)
            finally:
                restore_evaluation_state(editor.hparams, pre_eval_state)
            pre_metrics_source = "computed_in_process"
        else:
            print(
                json.dumps(
                    {
                        "phase": "pre_eval_skipped",
                        "increment": increment,
                    }
                ),
                flush=True,
            )
        if pre_metrics is not None:
            write_json(increment_dir_out / "pre_metrics.json", pre_metrics)
        increment_manifest_path = increment_dir_out / "increment_manifest.json"
        increment_manifest = json.loads(increment_manifest_path.read_text())
        increment_manifest["pre_metrics_source"] = pre_metrics_source
        increment_manifest["pre_metrics_records"] = None if pre_metrics is None else len(pre_metrics)
        increment_manifest["eval_status"] = {
            "teacher_forcing": "pending",
            "non_teacher_forcing": "pending",
        }
        write_json(increment_manifest_path, increment_manifest)

        print(
            json.dumps(
                {
                    "phase": "edit",
                    "increment": increment,
                    "records": len(current_requests),
                    "evaluation_mode": args.evaluation_mode,
                }
            ),
            flush=True,
        )
        edit_start = time.time()
        checkpoint_steps = []
        if args.checkpoint_interval > 0:
            checkpoint_steps = list(range(args.checkpoint_interval, len(current_requests) + 1, args.checkpoint_interval))
            if checkpoint_steps and checkpoint_steps[-1] != len(current_requests):
                checkpoint_steps.append(len(current_requests))
            elif not checkpoint_steps and len(current_requests) > 0:
                checkpoint_steps = [len(current_requests)]
        next_checkpoint_idx = 0
        start_request_idx = 0
        if resume_increment == increment and resume_checkpoint_step > 0:
            start_request_idx = min(resume_checkpoint_step, len(current_requests))
            while next_checkpoint_idx < len(checkpoint_steps) and checkpoint_steps[next_checkpoint_idx] <= start_request_idx:
                next_checkpoint_idx += 1
            print(
                json.dumps(
                    {
                        "phase": "resume_edit",
                        "increment": increment,
                        "resume_checkpoint_step": start_request_idx,
                        "remaining_records": len(current_requests) - start_request_idx,
                    }
                ),
                flush=True,
            )

        for request_idx, request in enumerate(current_requests[start_request_idx:], start=start_request_idx + 1):
            apply_single_edit(editor, request)
            if next_checkpoint_idx < len(checkpoint_steps) and request_idx == checkpoint_steps[next_checkpoint_idx]:
                seen_records = current_records[:request_idx]
                seen_requests = current_requests[:request_idx]
                seen_pre_metrics = None if pre_metrics is None else pre_metrics[:request_idx]
                checkpoint_output_dir = increment_dir_out / f"checkpoint_{request_idx:06d}" / "current"
                checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
                with maybe_gpu_keepalive(args, editor.hparams, f"checkpoint_{increment}_{request_idx:06d}"):
                    checkpoint_runtime = save_runtime_checkpoint_if_supported(
                        editor,
                        checkpoint_output_dir,
                        {
                            "increment": increment,
                            "checkpoint_step": request_idx,
                            "cumulative_edits": cumulative_edits - start_request_idx + request_idx,
                            "increment_index": increment_idx,
                        },
                    )
                    checkpoint_manifest = build_checkpoint_manifest(
                        dataset="WikiBigEdit",
                        split_or_increment=increment,
                        backbone=backbone,
                        method=method,
                        hopedit_mode=getattr(editor.model, "hopedit_mode", None),
                        assignment_policy=getattr(editor.model, "cell_assignment_policy", None),
                        cell_budget=getattr(editor.hparams, "cell_budget", None),
                        edit_count=request_idx,
                        checkpoint_path=checkpoint_output_dir,
                        checkpoint_size_bytes=None if checkpoint_runtime is None else checkpoint_runtime.get("checkpoint_size_bytes"),
                        checkpoint_load_seconds=manifest.get("resume_runtime_checkpoint_load_seconds"),
                        runtime_checkpoint_path=None if checkpoint_runtime is None else checkpoint_runtime.get("checkpoint_dir"),
                        saved_memory_semantics=collect_memory_semantics(editor),
                        evaluation_mode=args.evaluation_mode,
                        extra={
                            "dataset_file": str(dataset_file.resolve()),
                            "hparams_path": str(Path(args.hparams_dir).resolve()),
                            "seed": args.seed,
                        },
                    )
                    write_checkpoint_manifest(checkpoint_output_dir, checkpoint_manifest)
                    if args.checkpoint_eval_policy == "full":
                        checkpoint_run_config = {
                            "run_name": f"{method.lower()}_{backbone}_wikibigedit_{increment}_current_{request_idx:06d}",
                            "editing_method": method,
                            "alg_name": hparams.alg_name,
                            "model_name": hparams.model_name,
                            "backbone": backbone,
                            "data_type": "WikiBigEdit",
                            "dataset_file": str(dataset_file.resolve()),
                            "stream_type": "increment_current_checkpoint",
                            "seed": args.seed,
                            "evaluation_mode": args.evaluation_mode,
                            "sequential_edit": True,
                            "stream_length": len(seen_records),
                            "requested_ds_size": len(seen_records),
                            "hparams_path": str(Path(args.hparams_dir).resolve()),
                            "output_dir": str(checkpoint_output_dir.resolve()),
                            "wall_time_seconds": time.time() - start_time,
                            "increment_edit_wall_time_seconds": time.time() - edit_start,
                            "increment": increment,
                            "increment_index": increment_idx,
                            "cumulative_edits": cumulative_edits - start_request_idx + request_idx,
                            "checkpoint_step": request_idx,
                            "checkpoint_interval": args.checkpoint_interval,
                            "checkpoint_save_seconds": None if checkpoint_runtime is None else checkpoint_runtime.get("checkpoint_save_seconds"),
                            "checkpoint_size_bytes": None if checkpoint_runtime is None else checkpoint_runtime.get("checkpoint_size_bytes"),
                            "checkpoint_load_seconds": manifest.get("resume_runtime_checkpoint_load_seconds"),
                        }
                        checkpoint_summary = evaluate_and_write(
                            editor,
                            hparams,
                            method,
                            backbone,
                            checkpoint_output_dir,
                            seen_records,
                            seen_requests,
                            eval_metric,
                            checkpoint_run_config,
                            pre_metrics=seen_pre_metrics,
                        )
                        update_checkpoint_manifest_status(
                            checkpoint_output_dir,
                            evaluation_mode=args.evaluation_mode,
                            status="done",
                            summary_path=checkpoint_output_dir / "summary.json",
                            eval_status_path=checkpoint_output_dir / "deferred_eval_status.json",
                        )
                    else:
                        write_deferred_eval_manifest(
                            checkpoint_output_dir,
                            {
                                "phase": "checkpoint_eval_deferred",
                                "increment": increment,
                                "checkpoint_step": request_idx,
                                "cumulative_edits": cumulative_edits - start_request_idx + request_idx,
                                "records_available": len(seen_records),
                                "evaluation_mode": args.evaluation_mode,
                                "editing_method": method,
                                "hparams_path": str(Path(args.hparams_dir).resolve()),
                                "data_type": "WikiBigEdit",
                                "data_dir": str(Path(args.data_dir).resolve()),
                                "increment_dir": str(increment_dir.resolve()),
                                "increment": increment,
                                "checkpoint_step": request_idx,
                                "runtime_checkpoint": None if checkpoint_runtime is None else str(checkpoint_runtime["checkpoint_dir"].resolve()),
                                "pre_metrics_root": None if args.pre_metrics_root is None else str(Path(args.pre_metrics_root).resolve()),
                                "eval_batch_size": args.eval_batch_size,
                                "seed": args.seed,
                            },
                        )
                        checkpoint_summary = None
                if checkpoint_summary is not None:
                    print(
                        json.dumps(
                            {
                                "phase": "checkpoint_eval_complete",
                                "increment": increment,
                                "checkpoint_step": request_idx,
                                "post_rewrite_mean": checkpoint_summary.get("post_rewrite_mean"),
                                "post_rephrase_mean": checkpoint_summary.get("post_rephrase_mean"),
                                "post_locality_mean": checkpoint_summary.get("post_locality_mean"),
                                "cumulative_edits": cumulative_edits - start_request_idx + request_idx,
                            }
                        ),
                        flush=True,
                    )
                else:
                    print(
                        json.dumps(
                            {
                                "phase": "checkpoint_eval_deferred",
                                "increment": increment,
                                "checkpoint_step": request_idx,
                                "cumulative_edits": cumulative_edits - start_request_idx + request_idx,
                            }
                        ),
                        flush=True,
                    )
                next_checkpoint_idx += 1
        edit_seconds = time.time() - edit_start
        cumulative_edits = cumulative_edits - start_request_idx + len(current_requests)
        all_increment_records[increment] = current_records
        if resume_increment == increment:
            resume_increment = None
            resume_increment_idx = None
            resume_checkpoint_step = 0

        current_output_dir = increment_dir_out / "current"
        current_output_dir.mkdir(parents=True, exist_ok=True)
        with maybe_gpu_keepalive(args, editor.hparams, f"increment_current_{increment}"):
            current_runtime = save_runtime_checkpoint_if_supported(
                editor,
                current_output_dir,
                {
                    "increment": increment,
                    "checkpoint_step": len(current_requests),
                    "cumulative_edits": cumulative_edits,
                    "increment_index": increment_idx,
                },
            )
            current_manifest = build_checkpoint_manifest(
                dataset="WikiBigEdit",
                split_or_increment=increment,
                backbone=backbone,
                method=method,
                hopedit_mode=getattr(editor.model, "hopedit_mode", None),
                assignment_policy=getattr(editor.model, "cell_assignment_policy", None),
                cell_budget=getattr(editor.hparams, "cell_budget", None),
                edit_count=len(current_requests),
                checkpoint_path=current_output_dir,
                checkpoint_size_bytes=None if current_runtime is None else current_runtime.get("checkpoint_size_bytes"),
                checkpoint_load_seconds=manifest.get("resume_runtime_checkpoint_load_seconds"),
                runtime_checkpoint_path=None if current_runtime is None else current_runtime.get("checkpoint_dir"),
                saved_memory_semantics=collect_memory_semantics(editor),
                evaluation_mode=args.evaluation_mode,
                extra={
                    "dataset_file": str(dataset_file.resolve()),
                    "hparams_path": str(Path(args.hparams_dir).resolve()),
                    "seed": args.seed,
                },
            )
            write_checkpoint_manifest(current_output_dir, current_manifest)
            if args.current_eval_policy == "full":
                run_config = {
                    "run_name": f"{method.lower()}_{backbone}_wikibigedit_{increment}_current",
                    "editing_method": method,
                    "alg_name": hparams.alg_name,
                    "model_name": hparams.model_name,
                    "backbone": backbone,
                    "data_type": "WikiBigEdit",
                    "dataset_file": str(dataset_file.resolve()),
                    "stream_type": "increment_current",
                    "seed": args.seed,
                    "evaluation_mode": args.evaluation_mode,
                    "sequential_edit": True,
                    "stream_length": len(current_records),
                    "requested_ds_size": len(current_records),
                    "hparams_path": str(Path(args.hparams_dir).resolve()),
                    "output_dir": str(current_output_dir.resolve()),
                    "wall_time_seconds": time.time() - start_time,
                    "increment_edit_wall_time_seconds": edit_seconds,
                    "increment": increment,
                    "increment_index": increment_idx,
                    "cumulative_edits": cumulative_edits,
                    "checkpoint_save_seconds": None if current_runtime is None else current_runtime.get("checkpoint_save_seconds"),
                    "checkpoint_size_bytes": None if current_runtime is None else current_runtime.get("checkpoint_size_bytes"),
                    "checkpoint_load_seconds": manifest.get("resume_runtime_checkpoint_load_seconds"),
                }
                current_summary = evaluate_and_write(
                    editor,
                    hparams,
                    method,
                    backbone,
                    current_output_dir,
                    current_records,
                    current_requests,
                    eval_metric,
                    run_config,
                    pre_metrics=pre_metrics,
                )
                update_checkpoint_manifest_status(
                    current_output_dir,
                    evaluation_mode=args.evaluation_mode,
                    status="done",
                    summary_path=current_output_dir / "summary.json",
                    eval_status_path=current_output_dir / "deferred_eval_status.json",
                )
            else:
                write_deferred_eval_manifest(
                    current_output_dir,
                    {
                        "phase": "current_eval_deferred",
                        "increment": increment,
                        "cumulative_edits": cumulative_edits,
                        "records_available": len(current_records),
                        "evaluation_mode": args.evaluation_mode,
                        "editing_method": method,
                        "hparams_path": str(Path(args.hparams_dir).resolve()),
                        "data_type": "WikiBigEdit",
                        "data_dir": str(Path(args.data_dir).resolve()),
                        "increment_dir": str(increment_dir.resolve()),
                        "increment": increment,
                        "checkpoint_step": len(current_requests),
                        "runtime_checkpoint": None if current_runtime is None else str(current_runtime["checkpoint_dir"].resolve()),
                        "pre_metrics_root": None if args.pre_metrics_root is None else str(Path(args.pre_metrics_root).resolve()),
                        "eval_batch_size": args.eval_batch_size,
                        "seed": args.seed,
                    },
                )
                current_summary = None

        if current_summary is not None:
            print(
                json.dumps(
                    {
                        "phase": "current_eval_complete",
                        "increment": increment,
                        "post_rewrite_mean": current_summary.get("post_rewrite_mean"),
                        "post_rephrase_mean": current_summary.get("post_rephrase_mean"),
                        "post_locality_mean": current_summary.get("post_locality_mean"),
                        "cumulative_edits": cumulative_edits,
                    }
                ),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {
                        "phase": "current_eval_deferred",
                        "increment": increment,
                        "cumulative_edits": cumulative_edits,
                    }
                ),
                flush=True,
            )

        past_eval_summaries = {}
        if args.past_eval_policy == "skip":
            write_json(increment_dir_out / "past_eval_summaries.json", past_eval_summaries)
            continue
        for previous_increment in requested_increments[:increment_idx]:
            previous_records = list(all_increment_records[previous_increment])
            if args.past_eval_max_samples > 0 and len(previous_records) > args.past_eval_max_samples:
                previous_records = sample_records(previous_records, args.past_eval_max_samples, args.seed + increment_idx + len(previous_increment))
            previous_requests, previous_metric = build_editor_inputs(previous_records)
            prev_output_dir = increment_dir_out / "past_eval" / previous_increment
            prev_output_dir.mkdir(parents=True, exist_ok=True)
            prev_config = {
                "run_name": f"{method.lower()}_{backbone}_wikibigedit_{increment}_past_{previous_increment}",
                "editing_method": method,
                "alg_name": hparams.alg_name,
                "model_name": hparams.model_name,
                "backbone": backbone,
                "data_type": "WikiBigEdit",
                "dataset_file": previous_increment,
                "stream_type": "increment_past",
                "seed": args.seed,
                "evaluation_mode": args.evaluation_mode,
                "sequential_edit": True,
                "stream_length": len(previous_records),
                "requested_ds_size": len(previous_records),
                "hparams_path": str(Path(args.hparams_dir).resolve()),
                "output_dir": str(prev_output_dir.resolve()),
                "wall_time_seconds": time.time() - start_time,
                "increment": increment,
                "evaluated_past_increment": previous_increment,
                "increment_index": increment_idx,
                "cumulative_edits": cumulative_edits,
            }
            with maybe_gpu_keepalive(args, editor.hparams, f"past_eval_{increment}_{previous_increment}"):
                summary = evaluate_and_write(
                    editor,
                    hparams,
                    method,
                    backbone,
                    prev_output_dir,
                    previous_records,
                    previous_requests,
                    previous_metric,
                    prev_config,
                    pre_metrics=None,
                )
            past_eval_summaries[previous_increment] = summary

        write_json(increment_dir_out / "past_eval_summaries.json", past_eval_summaries)


if __name__ == "__main__":
    main()
