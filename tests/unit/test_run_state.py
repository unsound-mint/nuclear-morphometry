from pathlib import Path

from nuclear_morphometry.pipeline.run_state import (
    incomplete_image_ids,
    init_run_state,
    load_run_state,
    mark_complete,
    mark_failed,
    mark_running,
    save_run_state,
)


def test_init_run_state_all_pending() -> None:
    state = init_run_state(run_id="r1", config_hash="abc", image_ids=["f1", "f2"])
    assert incomplete_image_ids(state) == ["f1", "f2"]


def test_state_transitions() -> None:
    state = init_run_state(run_id="r1", config_hash="abc", image_ids=["f1", "f2"])
    mark_running(state, "f1")
    mark_complete(state, "f1")
    mark_running(state, "f2")
    mark_failed(state, "f2", "boom")

    assert state.fields["f1"].status == "complete"
    assert state.fields["f2"].status == "failed"
    assert state.fields["f2"].error == "boom"
    assert incomplete_image_ids(state) == ["f2"]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    state = init_run_state(run_id="r1", config_hash="abc", image_ids=["f1"])
    mark_running(state, "f1")
    mark_complete(state, "f1")

    path = tmp_path / "run_state.json"
    save_run_state(path, state)
    assert not path.with_suffix(".json.tmp").exists()

    loaded = load_run_state(path)
    assert loaded is not None
    assert loaded.fields["f1"].status == "complete"


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_run_state(tmp_path / "missing.json") is None
