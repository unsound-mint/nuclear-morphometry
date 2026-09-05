from pathlib import Path

import numpy as np
import pytest
import tifffile

from dayana_nuclei.io.metadata import inspect_image


def test_inspect_uncalibrated_tiff_reports_unknown_spacing(tmp_path: Path) -> None:
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((8, 10), dtype=np.uint16), metadata={"axes": "YX"})

    result = inspect_image(path)

    assert result.x_um is None
    assert result.y_um is None
    assert any("resolution tag" in w for w in result.warnings)


def test_inspect_calibrated_2d_tiff_reports_real_spacing(tmp_path: Path) -> None:
    path = tmp_path / "calibrated.tif"
    # 100000 px/cm -> 1e4 um/cm / 1e5 px/cm = 0.1 um/pixel
    tifffile.imwrite(
        path,
        np.zeros((8, 10), dtype=np.uint16),
        metadata={"axes": "YX"},
        resolution=(100000.0, 100000.0),
        resolutionunit="CENTIMETER",
    )

    result = inspect_image(path)

    assert result.x_um == 0.1
    assert result.y_um == 0.1
    assert not any("resolution tag" in w for w in result.warnings)


def test_inspect_3d_with_full_spacing(tmp_path: Path) -> None:
    path = tmp_path / "volume.tif"
    tifffile.imwrite(
        path,
        np.zeros((4, 8, 10), dtype=np.uint16),
        imagej=True,
        resolution=(10.0, 10.0),
        metadata={"axes": "ZYX", "spacing": 0.3, "unit": "um"},
    )

    result = inspect_image(path)

    assert result.z_um == 0.3
    assert result.x_um == 0.1
    assert result.y_um == 0.1
    assert result.n_z_planes == 4
    assert not any("Z step" in w for w in result.warnings)


def test_inspect_3d_missing_z_step_warns(tmp_path: Path) -> None:
    path = tmp_path / "volume_no_z.tif"
    tifffile.imwrite(path, np.zeros((4, 8, 10), dtype=np.uint16), metadata={"axes": "ZYX"})

    result = inspect_image(path)

    assert result.z_um is None
    assert any("Z step" in w for w in result.warnings)


def test_inspect_multiscene_without_scene_warns(tmp_path: Path) -> None:
    path = tmp_path / "multiscene.tif"
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.zeros((8, 10), dtype=np.uint16), metadata={"axes": "YX"})
        tw.write(np.ones((8, 10), dtype=np.uint16), metadata={"axes": "YX"})

    result = inspect_image(path)

    assert len(result.scenes) == 2
    assert any("scenes" in w for w in result.warnings)


def test_inspect_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_image(tmp_path / "does_not_exist.tif")
