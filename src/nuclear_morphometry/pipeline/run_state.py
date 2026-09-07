"""Run state and resume support (spec section 33).

A field only becomes "complete" after its outputs are committed, so
``resume`` can safely skip it. Writes are atomic (temp file + rename, spec
section 50) so a crash mid-write never leaves a corrupt run_state.json.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

FieldStatus = Literal["pending", "running", "complete", "failed"]


class FieldState(BaseModel):
    model_config = ConfigDict(frozen=False)

    image_id: str
    status: FieldStatus
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class RunState(BaseModel):
    model_config = ConfigDict(frozen=False)

    run_id: str
    config_hash: str
    fields: dict[str, FieldState]


def init_run_state(*, run_id: str, config_hash: str, image_ids: list[str]) -> RunState:
    return RunState(
        run_id=run_id,
        config_hash=config_hash,
        fields={
            image_id: FieldState(image_id=image_id, status="pending") for image_id in image_ids
        },
    )


def load_run_state(path: Path) -> RunState | None:
    if not path.exists():
        return None
    return RunState.model_validate_json(path.read_text())


def save_run_state(path: Path, state: RunState) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(state.model_dump_json(indent=2))
    tmp_path.replace(path)


def mark_running(state: RunState, image_id: str) -> None:
    state.fields[image_id].status = "running"
    state.fields[image_id].started_at = datetime.now(UTC).isoformat()
    state.fields[image_id].error = None


def mark_complete(state: RunState, image_id: str) -> None:
    state.fields[image_id].status = "complete"
    state.fields[image_id].completed_at = datetime.now(UTC).isoformat()


def mark_failed(state: RunState, image_id: str, error: str) -> None:
    state.fields[image_id].status = "failed"
    state.fields[image_id].error = error


def incomplete_image_ids(state: RunState) -> list[str]:
    """Fields resume must (re)process: pending, failed, or interrupted mid-run.

    "running" is included deliberately -- a field that was mid-processing
    when the process was killed never reached "complete" and must be
    retried (spec 33: resume "retries failed/incomplete fields"). This is
    safe because each field owns exactly one partial-table file
    (export.py), so reprocessing it just overwrites that file rather than
    duplicating rows.
    """
    return [
        image_id
        for image_id, field_state in state.fields.items()
        if field_state.status in ("pending", "failed", "running")
    ]
