from pathlib import Path

import numpy as np
import pytest

from nuclear_morphometry.io.masks import load_label_mask, mask_path_for, save_label_mask
from nuclear_morphometry.models import PhysicalSpacing


def test_mask_path_for_is_deterministic(tmp_path: Path) -> None:
    p1 = mask_path_for(tmp_path, "SW620_Sort01_low_48h_Field003")
    p2 = mask_path_for(tmp_path, "SW620_Sort01_low_48h_Field003")
    assert p1 == p2
    assert p1 == tmp_path / "masks" / "SW620_Sort01_low_48h_Field003_labels.tif"


def test_2d_mask_round_trip(tmp_path: Path) -> None:
    labels = np.zeros((8, 10), dtype=np.uint16)
    labels[2:4, 3:6] = 1
    labels[5:7, 7:9] = 2
    spacing = PhysicalSpacing(x_um=0.1, y_um=0.1, z_um=None)
    path = mask_path_for(tmp_path, "field001")

    save_label_mask(labels, path, spacing, "YX")
    loaded, axes = load_label_mask(path)

    assert axes == "YX"
    np.testing.assert_array_equal(loaded, labels)
    assert np.issubdtype(loaded.dtype, np.integer)


def test_3d_mask_round_trip_with_compression(tmp_path: Path) -> None:
    labels = np.zeros((4, 8, 10), dtype=np.uint32)
    labels[1, 2:4, 3:6] = 5
    spacing = PhysicalSpacing(x_um=0.1, y_um=0.1, z_um=0.3)
    path = mask_path_for(tmp_path, "field002")

    save_label_mask(labels, path, spacing, "ZYX", compress=True)
    loaded, axes = load_label_mask(path)

    assert axes == "ZYX"
    np.testing.assert_array_equal(loaded, labels)


def test_3d_mask_requires_z_spacing(tmp_path: Path) -> None:
    labels = np.zeros((4, 8, 10), dtype=np.uint16)
    spacing = PhysicalSpacing(x_um=0.1, y_um=0.1, z_um=None)
    path = mask_path_for(tmp_path, "field003")

    with pytest.raises(ValueError, match="z_um is None"):
        save_label_mask(labels, path, spacing, "ZYX")


def test_mask_ndim_mismatch_raises(tmp_path: Path) -> None:
    labels = np.zeros((8, 10), dtype=np.uint16)
    spacing = PhysicalSpacing(x_um=0.1, y_um=0.1, z_um=0.3)
    path = mask_path_for(tmp_path, "field004")

    with pytest.raises(ValueError, match="ndim"):
        save_label_mask(labels, path, spacing, "ZYX")


def test_load_label_mask_rejects_non_mask_tiff(tmp_path: Path) -> None:
    import tifffile

    path = tmp_path / "not_a_mask.tif"
    tifffile.imwrite(path, np.zeros((8, 10), dtype=np.uint16))

    with pytest.raises(ValueError, match="does not contain"):
        load_label_mask(path)


def test_save_label_mask_does_not_leave_tmp_file_on_success(tmp_path: Path) -> None:
    labels = np.zeros((8, 10), dtype=np.uint16)
    spacing = PhysicalSpacing(x_um=0.1, y_um=0.1, z_um=None)
    path = mask_path_for(tmp_path, "field005")

    save_label_mask(labels, path, spacing, "YX")

    leftover = list(path.parent.glob(".*tmp*"))
    assert leftover == []
