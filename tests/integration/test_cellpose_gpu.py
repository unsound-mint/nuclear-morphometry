"""Real Cellpose-SAM inference on a synthetic image (spec section 36.4).

Skips cleanly whenever CUDA or the Cellpose model weights are unavailable --
this must never run in ordinary CPU CI. To run explicitly on a GPU machine
with an internet connection (weights are a ~1.15 GB one-time download,
cached under ``~/.cellpose/models/``):

    uv run pytest tests/integration/test_cellpose_gpu.py -m "gpu and cellpose_model" -v

This suite does not judge segmentation *quality* -- that is spec section 14's
job (``validate-segmentation`` against a reference mask set). It only proves
the backend adapter actually drives the installed Cellpose API correctly:
runs on CUDA, returns an integer label image of the right shape, and the
model is not reloaded between calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk

from nuclear_morphometry.models import PhysicalSpacing

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


def _weights_available() -> bool:
    if not _CUDA_AVAILABLE:
        return False
    try:
        from cellpose.models import MODEL_DIR
    except ImportError:
        return False
    return (MODEL_DIR / "cpsam_v2").exists()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.cellpose_model,
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires a CUDA device"),
    pytest.mark.skipif(
        not _weights_available(),
        reason="requires the cpsam_v2 weights already cached under ~/.cellpose/models/",
    ),
]


def _synthetic_2d_field() -> np.ndarray:
    image = np.zeros((256, 256), dtype=np.float32)
    rr, cc = disk((80, 80), 30)
    image[rr, cc] = 1.0
    rr, cc = disk((180, 180), 25)
    image[rr, cc] = 1.0
    return image


def test_cellpose_2d_inference_returns_integer_labels() -> None:
    from nuclear_morphometry.segmentation.cellpose_backend import CellposeSegmenter

    segmenter = CellposeSegmenter(
        model="cpsam_v2",
        device="cuda",
        diameter_um=0.0,
        use_anisotropy=True,
        flow3d_smooth=0.0,
        batch_size=0,
    )
    spacing = PhysicalSpacing(x_um=0.2, y_um=0.2)

    result = segmenter.segment(_synthetic_2d_field(), spacing)

    assert result.labels.shape == (256, 256)
    assert np.issubdtype(result.labels.dtype, np.integer)
    assert result.labels.max() >= 1
    assert result.backend == "cellpose"
    assert result.model_id == "cpsam_v2"


def test_model_is_not_reloaded_between_segment_calls() -> None:
    from nuclear_morphometry.segmentation.cellpose_backend import CellposeSegmenter

    segmenter = CellposeSegmenter(
        model="cpsam_v2",
        device="cuda",
        diameter_um=0.0,
        use_anisotropy=True,
        flow3d_smooth=0.0,
        batch_size=0,
    )
    spacing = PhysicalSpacing(x_um=0.2, y_um=0.2)
    model_instance_id = id(segmenter._model)

    segmenter.segment(_synthetic_2d_field(), spacing)
    segmenter.segment(_synthetic_2d_field(), spacing)

    assert id(segmenter._model) == model_instance_id


def _synthetic_3d_sphere() -> np.ndarray:
    shape = (40, 100, 100)
    zz, yy, xx = np.indices(shape)
    center = np.array(shape) / 2
    sphere = ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= 15**2
    return np.where(sphere, 1.0, 0.0).astype(np.float32)


def test_cellpose_3d_inference_runs_and_returns_correct_shape() -> None:
    """Mechanics only, per this file's stated scope -- NOT a quality check.

    Normalizes the input first, matching what `pipeline/analyze.py` always does
    before calling `segment()` in production. A previous version of this test
    called `segment()` directly on un-normalized data and found severe
    over-segmentation; that was traced to the missing normalization step, not a
    genuine 3D limitation -- see
    docs/decisions/0008-cellpose-3d-oversegmentation.md. This test still does
    NOT assert a specific object count for this hard-edged binary synthetic
    sphere (which itself remains an unexplained edge case even normalized).
    Real 3D segmentation quality must be established via `validate-segmentation`
    against a real reference mask set (spec section 14) before trusting any 3D
    result.
    """
    from nuclear_morphometry.segmentation.cellpose_backend import CellposeSegmenter
    from nuclear_morphometry.segmentation.normalize import normalize_percentile

    segmenter = CellposeSegmenter(
        model="cpsam_v2",
        device="cuda",
        diameter_um=0.0,
        use_anisotropy=True,
        flow3d_smooth=0.0,
        batch_size=0,
    )
    spacing = PhysicalSpacing(x_um=0.2, y_um=0.2, z_um=0.2)
    normalized = normalize_percentile(
        _synthetic_3d_sphere(), percentile_low=1.0, percentile_high=99.8
    )

    result = segmenter.segment(normalized, spacing)

    assert result.labels.shape == (40, 100, 100)
    assert np.issubdtype(result.labels.dtype, np.integer)
    assert result.labels.max() >= 1
    assert result.backend_metadata["do_3D"] is True
