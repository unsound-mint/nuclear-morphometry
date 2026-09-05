from pathlib import Path

import numpy as np
import pytest
import tifffile

from dayana_nuclei.io.images import load_channel_volume
from dayana_nuclei.models import ExperimentalMetadata, ImageSource

_META = ExperimentalMetadata(
    image_id="SW620_Sort01_low_48h_Field003",
    cell_line="SW620",
    sort_id="Sort01",
    condition="low",
    timepoint="48h",
    field="Field003",
    acquisition_batch=None,
)


def _source(
    path: Path, *, channel: str = "Channel:0:0", scene: str | int | None = None
) -> ImageSource:
    return ImageSource(path=path, scene=scene, channel=channel, metadata=_META)


def test_load_2d_calibrated(tmp_path: Path) -> None:
    path = tmp_path / "img2d.tif"
    tifffile.imwrite(
        path,
        np.arange(80, dtype=np.uint16).reshape(8, 10),
        metadata={"axes": "YX"},
        resolution=(100000.0, 100000.0),
        resolutionunit="CENTIMETER",
    )

    volume = load_channel_volume(_source(path), mode="2d", projection="none")

    assert volume.axes == "YX"
    assert volume.data.shape == (8, 10)
    assert volume.spacing.x_um == pytest.approx(0.1)
    assert volume.spacing.y_um == pytest.approx(0.1)
    assert volume.spacing.z_um is None


def test_load_3d_with_full_spacing(tmp_path: Path) -> None:
    path = tmp_path / "img3d.tif"
    data = np.arange(4 * 8 * 10, dtype=np.uint16).reshape(4, 8, 10)
    tifffile.imwrite(
        path,
        data,
        imagej=True,
        resolution=(10.0, 10.0),
        metadata={"axes": "ZYX", "spacing": 0.3, "unit": "um"},
    )

    volume = load_channel_volume(_source(path), mode="3d")

    assert volume.axes == "ZYX"
    assert volume.data.shape == (4, 8, 10)
    np.testing.assert_array_equal(volume.data, data)
    assert volume.spacing.z_um == pytest.approx(0.3)
    assert volume.spacing.x_um == pytest.approx(0.1)


def test_load_3d_missing_z_spacing_raises(tmp_path: Path) -> None:
    path = tmp_path / "img3d_no_z.tif"
    tifffile.imwrite(
        path,
        np.zeros((4, 8, 10), dtype=np.uint16),
        metadata={"axes": "ZYX"},
        resolution=(100000.0, 100000.0),
        resolutionunit="CENTIMETER",
    )

    with pytest.raises(ValueError, match="3D analysis requires physical X/Y/Z spacing"):
        load_channel_volume(_source(path), mode="3d")


def test_load_2d_multi_z_no_projection_raises(tmp_path: Path) -> None:
    path = tmp_path / "img_multiz.tif"
    tifffile.imwrite(
        path,
        np.zeros((4, 8, 10), dtype=np.uint16),
        metadata={"axes": "ZYX"},
        resolution=(100000.0, 100000.0),
        resolutionunit="CENTIMETER",
    )

    with pytest.raises(ValueError, match="will not silently max-project"):
        load_channel_volume(_source(path), mode="2d", projection="none")


def test_load_2d_max_projection(tmp_path: Path) -> None:
    path = tmp_path / "img_multiz_proj.tif"
    data = np.zeros((4, 8, 10), dtype=np.uint16)
    data[2, 3, 4] = 99
    tifffile.imwrite(
        path,
        data,
        metadata={"axes": "ZYX"},
        resolution=(100000.0, 100000.0),
        resolutionunit="CENTIMETER",
    )

    volume = load_channel_volume(_source(path), mode="2d", projection="max")

    assert volume.axes == "YX"
    assert volume.data.shape == (8, 10)
    assert volume.data[3, 4] == 99


def test_load_2d_specific_plane(tmp_path: Path) -> None:
    path = tmp_path / "img_multiz_plane.tif"
    data = np.zeros((4, 8, 10), dtype=np.uint16)
    data[2, 3, 4] = 77
    tifffile.imwrite(
        path,
        data,
        metadata={"axes": "ZYX"},
        resolution=(100000.0, 100000.0),
        resolutionunit="CENTIMETER",
    )

    volume = load_channel_volume(
        _source(path), mode="2d", projection="specific_plane", specific_plane=2
    )

    assert volume.axes == "YX"
    assert volume.data[3, 4] == 77


def test_multiscene_without_scene_raises(tmp_path: Path) -> None:
    path = tmp_path / "multiscene.tif"
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.zeros((8, 10), dtype=np.uint16), metadata={"axes": "YX"})
        tw.write(np.ones((8, 10), dtype=np.uint16), metadata={"axes": "YX"})

    with pytest.raises(ValueError, match="contains 2 scenes"):
        load_channel_volume(_source(path, scene=None), mode="2d", projection="none")


def test_unknown_channel_raises(tmp_path: Path) -> None:
    path = tmp_path / "img2d_badchannel.tif"
    tifffile.imwrite(
        path,
        np.zeros((8, 10), dtype=np.uint16),
        metadata={"axes": "YX"},
        resolution=(100000.0, 100000.0),
        resolutionunit="CENTIMETER",
    )

    with pytest.raises(ValueError, match="not found"):
        load_channel_volume(_source(path, channel="NoSuchChannel"), mode="2d", projection="none")
