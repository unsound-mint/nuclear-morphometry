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


def test_texture_2d_rejects_3d_mode() -> None:
    data = _base_dict()
    data["analysis"] = {"mode": "3d"}
    data["measurements"] = {"texture_2d": True}
    with pytest.raises(ValidationError, match="texture_2d"):
        Config.model_validate(data)


def test_radial_distribution_2d_rejects_3d_mode() -> None:
    data = _base_dict()
    data["analysis"] = {"mode": "3d"}
    data["measurements"] = {"radial_distribution_2d": True}
    with pytest.raises(ValidationError, match="radial_distribution_2d"):
        Config.model_validate(data)


def test_radial_distribution_2d_accepted_in_2d_mode() -> None:
    data = _base_dict()
    data["measurements"] = {"radial_distribution_2d": True, "radial_bins": 4}
    config = Config.model_validate(data)
    assert config.measurements.radial_distribution_2d is True
    assert config.measurements.radial_bins == 4


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


def test_additional_channel_must_be_listed_in_input_channels() -> None:
    data = _base_dict()
    data["measurements"] = {
        "additional_channels": [{"channel": "H3K9Ac", "prefix": "h3k9ac", "kind": "nuclear"}]
    }
    with pytest.raises(ValidationError, match="not listed in input"):
        Config.model_validate(data)


def test_additional_channel_nuclear_kind_is_accepted() -> None:
    data = _base_dict()
    data["input"]["additional_channels"] = ["H3K9Ac"]
    data["measurements"] = {
        "additional_channels": [{"channel": "H3K9Ac", "prefix": "h3k9ac", "kind": "nuclear"}]
    }
    config = Config.model_validate(data)
    assert config.measurements.additional_channels[0].prefix == "h3k9ac"


def test_additional_channel_prefix_cannot_be_hoechst() -> None:
    data = _base_dict()
    data["input"]["additional_channels"] = ["H3K9Ac"]
    data["measurements"] = {
        "additional_channels": [{"channel": "H3K9Ac", "prefix": "hoechst", "kind": "nuclear"}]
    }
    with pytest.raises(ValidationError, match='cannot be "hoechst"'):
        Config.model_validate(data)


def test_additional_channel_prefixes_must_be_unique() -> None:
    data = _base_dict()
    data["input"]["additional_channels"] = ["H3K9Ac", "H3K9me3"]
    data["measurements"] = {
        "additional_channels": [
            {"channel": "H3K9Ac", "prefix": "dup", "kind": "nuclear"},
            {"channel": "H3K9me3", "prefix": "dup", "kind": "nuclear"},
        ]
    }
    with pytest.raises(ValidationError, match="duplicate prefix"):
        Config.model_validate(data)


def test_lamin_shell_core_requires_positive_shell_width() -> None:
    data = _base_dict()
    data["input"]["additional_channels"] = ["LaminAC"]
    data["measurements"] = {
        "additional_channels": [
            {"channel": "LaminAC", "prefix": "laminac", "kind": "lamin_shell_core"}
        ]
    }
    with pytest.raises(ValidationError, match="shell_width_um"):
        Config.model_validate(data)


def test_lamin_shell_core_accepted_with_shell_width() -> None:
    data = _base_dict()
    data["input"]["additional_channels"] = ["LaminAC"]
    data["measurements"] = {
        "additional_channels": [
            {
                "channel": "LaminAC",
                "prefix": "laminac",
                "kind": "lamin_shell_core",
                "shell_width_um": 0.5,
            }
        ]
    }
    config = Config.model_validate(data)
    assert config.measurements.additional_channels[0].shell_width_um == 0.5


def test_mitotracker_rings_requires_both_bands() -> None:
    data = _base_dict()
    data["input"]["additional_channels"] = ["MitoTracker"]
    data["measurements"] = {
        "additional_channels": [
            {"channel": "MitoTracker", "prefix": "mito", "kind": "mitotracker_rings"}
        ]
    }
    with pytest.raises(ValidationError, match="near_ring_um and far_ring_um"):
        Config.model_validate(data)


def test_mitotracker_rings_reject_overlapping_bands() -> None:
    data = _base_dict()
    data["input"]["additional_channels"] = ["MitoTracker"]
    data["measurements"] = {
        "additional_channels": [
            {
                "channel": "MitoTracker",
                "prefix": "mito",
                "kind": "mitotracker_rings",
                "near_ring_um": [0.0, 10.0],
                "far_ring_um": [5.0, 15.0],
            }
        ]
    }
    with pytest.raises(ValidationError, match="ordered, non-overlapping"):
        Config.model_validate(data)


def test_mitotracker_rings_accepted_with_ordered_bands() -> None:
    data = _base_dict()
    data["input"]["additional_channels"] = ["MitoTracker"]
    data["measurements"] = {
        "additional_channels": [
            {
                "channel": "MitoTracker",
                "prefix": "mito",
                "kind": "mitotracker_rings",
                "near_ring_um": [0.0, 1.0],
                "far_ring_um": [3.0, 5.0],
            }
        ]
    }
    config = Config.model_validate(data)
    assert config.measurements.additional_channels[0].near_ring_um == (0.0, 1.0)


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
