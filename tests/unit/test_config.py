from pathlib import Path

import pytest
from pydantic import ValidationError

from dayana_nuclei.config import Config, load_config


def _base_dict() -> dict:
    return {
        "experiment": {"name": "t", "manifest": "manifest.csv"},
        "analysis": {"mode": "2d"},
        "input": {"hoechst_channel": "Hoechst"},
    }


def test_3d_mode_rejects_non_none_projection() -> None:
    data = _base_dict()
    data["analysis"] = {"mode": "3d", "projection": "max"}
    with pytest.raises(ValidationError, match="intact ZYX volume"):
        Config.model_validate(data)


def test_specific_plane_requires_index() -> None:
    data = _base_dict()
    data["analysis"] = {"mode": "2d", "projection": "specific_plane"}
    with pytest.raises(ValidationError, match="specific_plane"):
        Config.model_validate(data)


def test_percentile_low_must_be_below_high() -> None:
    data = _base_dict()
    data["segmentation"] = {"percentile_low": 99.9, "percentile_high": 1.0}
    with pytest.raises(ValidationError, match="percentile_low"):
        Config.model_validate(data)


def test_texture_scale_modes_are_mutually_exclusive() -> None:
    data = _base_dict()
    data["measurements"] = {
        "texture_distances_px": [3, 5],
        "texture_distances_um": [1.0, 2.0],
    }
    with pytest.raises(ValidationError, match="mutually exclusive"):
        Config.model_validate(data)


def test_cellpose_backend_rejects_disabled_normalization() -> None:
    data = _base_dict()
    data["segmentation"] = {"backend": "cellpose", "normalize_for_segmentation": False}
    with pytest.raises(ValidationError, match="normalize_for_segmentation"):
        Config.model_validate(data)


def test_fixture_backend_allows_disabled_normalization() -> None:
    data = _base_dict()
    data["segmentation"] = {"backend": "fixture", "normalize_for_segmentation": False}
    config = Config.model_validate(data)
    assert config.segmentation.normalize_for_segmentation is False


def test_3d_config_alias_and_defaults() -> None:
    data = _base_dict()
    data["analysis"] = {"mode": "3d"}
    data["segmentation"] = {"3d": {"use_anisotropy": False}}
    config = Config.model_validate(data)
    assert config.segmentation.three_d.use_anisotropy is False
    assert config.qc.random_seed == 0
    assert config.output.write_csv is True


def test_load_config_from_file_returns_hash(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.toml"
    config_path.write_text(
        """
        [experiment]
        name = "t"
        manifest = "manifest.csv"

        [analysis]
        mode = "2d"

        [input]
        hoechst_channel = "Hoechst"
        """
    )
    config, config_hash = load_config(config_path)
    assert isinstance(config, Config)
    assert len(config_hash) == 64
