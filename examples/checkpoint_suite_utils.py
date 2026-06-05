import json
from pathlib import Path
from typing import Any


LANE_TEACHER_FORCING = "teacher_forcing"
LANE_NON_TEACHER = "non_teacher_forcing"


def lane_name_for_mode(mode: str) -> str:
    if mode == "teacher_forcing":
        return LANE_TEACHER_FORCING
    return LANE_NON_TEACHER


def build_eval_status() -> dict[str, str]:
    return {
        LANE_TEACHER_FORCING: "pending",
        LANE_NON_TEACHER: "pending",
    }


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def infer_memory_semantics(editing_method: str, controller=None, memory_snapshot: list[dict] | None = None) -> dict[str, Any]:
    memory_unit = "edit"
    retained_units_final = None
    if editing_method.upper() == "HOPEDIT":
        hopedit_mode = getattr(controller, "hopedit_mode", None)
        if hopedit_mode == "v2_cell_bank":
            memory_unit = "cell"
    if memory_snapshot:
        first = memory_snapshot[0]
        snapshot_unit = first.get("memory_unit")
        if snapshot_unit in {"cell", "edit", "state"}:
            memory_unit = snapshot_unit
        if memory_unit in {"cell", "state"}:
            retained_units_final = len({row.get("cell_id") for row in memory_snapshot if row.get("cell_id") is not None})
        else:
            retained_units_final = len(memory_snapshot)
    return {
        "memory_unit": memory_unit,
        "retained_units_final": retained_units_final,
    }


def build_checkpoint_manifest(
    *,
    dataset: str,
    split_or_increment: str,
    backbone: str,
    method: str,
    hopedit_mode: str | None,
    assignment_policy: str | None,
    cell_budget: int | None,
    edit_count: int,
    checkpoint_path: Path,
    checkpoint_size_bytes: int | None,
    checkpoint_load_seconds: float | None,
    runtime_checkpoint_path: Path | None,
    saved_memory_semantics: dict[str, Any] | None,
    evaluation_mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "dataset": dataset,
        "split_or_increment": split_or_increment,
        "backbone": backbone,
        "method": method,
        "hopedit_mode": hopedit_mode,
        "assignment_policy": assignment_policy,
        "cell_budget": cell_budget,
        "edit_count": edit_count,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "runtime_checkpoint_path": None if runtime_checkpoint_path is None else str(runtime_checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "saved_memory_semantics": saved_memory_semantics or {},
        "default_evaluation_mode": evaluation_mode,
        "eval_status": build_eval_status(),
    }
    if extra:
        payload.update(extra)
    return payload


def write_checkpoint_manifest(checkpoint_dir: Path, payload: dict[str, Any]):
    write_json(checkpoint_dir / "checkpoint_manifest.json", payload)


def update_checkpoint_manifest_status(
    checkpoint_dir: Path,
    *,
    evaluation_mode: str,
    status: str,
    summary_path: Path | None = None,
    eval_status_path: Path | None = None,
):
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if not manifest_path.exists():
        return
    payload = read_json(manifest_path)
    lane = lane_name_for_mode(evaluation_mode)
    payload.setdefault("eval_status", build_eval_status())
    payload["eval_status"][lane] = status
    if summary_path is not None:
        payload.setdefault("eval_outputs", {})
        payload["eval_outputs"][lane] = {
            "evaluation_mode": evaluation_mode,
            "summary_path": str(summary_path.resolve()),
            "eval_status_path": None if eval_status_path is None else str(eval_status_path.resolve()),
        }
    write_checkpoint_manifest(checkpoint_dir, payload)
