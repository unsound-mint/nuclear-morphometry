import pytest

from dayana_nuclei.config import Config
from dayana_nuclei.pipeline.analyze import build_segmenter
from dayana_nuclei.segmentation.base import SegmenterUnavailableError
from dayana_nuclei.segmentation.fixture import FixtureSegmenter


def _config(**segmentation_overrides: object) -> Config:
    return Config.model_validate(
        {
            "experiment": {"name": "t", "manifest": "manifest.csv"},
            "analysis": {"mode": "2d"},
            "input": {"hoechst_channel": "Hoechst"},
            "segmentation": segmentation_overrides,
        }
    )


def test_fixture_backend_returns_fixture_segmenter() -> None:
    config = _config(backend="fixture")
    assert isinstance(build_segmenter(config), FixtureSegmenter)


def test_cellpose_backend_with_auto_model_raises_without_flag() -> None:
    """This must fail before ever touching torch/CUDA/the model weights --
    resolve_model_id runs first inside CellposeSegmenter.__init__."""
    config = _config(backend="cellpose", model="auto")
    with pytest.raises(SegmenterUnavailableError, match="validate-segmentation"):
        build_segmenter(config, allow_unvalidated_model=False)
