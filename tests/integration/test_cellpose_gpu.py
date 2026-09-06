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

from dayana_nuclei.models import PhysicalSpacing

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
    from dayana_nuclei.segmentation.cellpose_backend import CellposeSegmenter

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
    from dayana_nuclei.segmentation.cellpose_backend import CellposeSegmenter

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
