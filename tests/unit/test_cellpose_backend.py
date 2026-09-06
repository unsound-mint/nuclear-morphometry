"""Unit tests for the model-name/device resolution logic in cellpose_backend.py.

These do not load a real Cellpose model (no GPU/weights-download dependency) --
they test the fail-loud boundaries around backend construction, which is
where this module's correctness-critical logic lives. Actual inference is
covered by a GPU-marked integration test (tests/integration/test_cellpose_gpu.py)
that skips cleanly without CUDA/weights (spec 36.4).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from dayana_nuclei.models import PhysicalSpacing
from dayana_nuclei.segmentation.base import SegmenterUnavailableError
from dayana_nuclei.segmentation.cellpose_backend import (
    DEMO_DEFAULT_MODEL,
    _resolve_device,
    _resolve_model_name,
    _um_diameter_to_px,
)


def test_auto_model_without_opt_in_raises_actionable_error() -> None:
    with pytest.raises(SegmenterUnavailableError, match="validate-segmentation"):
        _resolve_model_name("auto", allow_unvalidated_model=False)


def test_auto_model_with_opt_in_resolves_to_demo_default() -> None:
    resolved = _resolve_model_name("auto", allow_unvalidated_model=True)
    assert resolved == DEMO_DEFAULT_MODEL


def test_existing_file_path_is_used_directly(tmp_path: Path) -> None:
    weights = tmp_path / "my_model.pth"
    weights.write_bytes(b"not a real model, just needs to exist")
    resolved = _resolve_model_name(str(weights), allow_unvalidated_model=False)
    assert resolved == str(weights)


def test_unknown_model_name_raises_rather_than_silently_falling_back() -> None:
    """Cellpose's own CellposeModel silently substitutes its default model on an
    unrecognized name (logs a warning, does not raise) -- this project's
    no-silent-fallback invariant requires raising instead, before ever
    calling into Cellpose."""
    with pytest.raises(SegmenterUnavailableError, match=r"not.*existing file path"):
        _resolve_model_name("definitely_not_a_real_model_xyz", allow_unvalidated_model=False)


def test_known_builtin_model_name_is_accepted() -> None:
    resolved = _resolve_model_name("cpsam_v2", allow_unvalidated_model=False)
    assert resolved == "cpsam_v2"


def test_cuda_device_without_availability_raises() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(SegmenterUnavailableError, match="no CUDA device"),
    ):
        _resolve_device("cuda")


def test_cpu_device_always_available() -> None:
    assert _resolve_device("cpu") == torch.device("cpu")


def test_cuda_device_resolves_when_available() -> None:
    with patch("torch.cuda.is_available", return_value=True):
        assert _resolve_device("cuda") == torch.device("cuda")


def test_diameter_zero_means_auto() -> None:
    spacing = PhysicalSpacing(x_um=0.2, y_um=0.2)
    assert _um_diameter_to_px(0.0, spacing) is None


def test_diameter_converted_using_x_calibration() -> None:
    spacing = PhysicalSpacing(x_um=0.5, y_um=0.5)
    assert _um_diameter_to_px(10.0, spacing) == pytest.approx(20.0)
