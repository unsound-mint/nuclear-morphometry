import numpy as np
import pytest
from pydantic import ValidationError

from dayana_nuclei.models import ImageVolume, PhysicalSpacing


def test_physical_spacing_require_z_raises_actionable_error() -> None:
    spacing = PhysicalSpacing(x_um=0.1, y_um=0.1)
    with pytest.raises(ValueError, match="3D analysis requires physical Z spacing"):
        spacing.require_z(context="test_image")


def test_physical_spacing_voxel_volume() -> None:
    spacing = PhysicalSpacing(x_um=0.2, y_um=0.2, z_um=0.5)
    assert spacing.voxel_volume_um3 == pytest.approx(0.02)


def test_image_volume_rejects_ndim_mismatch() -> None:
    spacing = PhysicalSpacing(x_um=0.1, y_um=0.1)
    with pytest.raises(ValidationError):
        ImageVolume(data=np.zeros((2, 3, 4)), spacing=spacing, axes="YX", dtype="uint16")


def test_image_volume_3d_requires_z_spacing() -> None:
    spacing = PhysicalSpacing(x_um=0.1, y_um=0.1)
    with pytest.raises(ValidationError):
        ImageVolume(data=np.zeros((2, 3, 4)), spacing=spacing, axes="ZYX", dtype="uint16")


def test_image_volume_valid_3d() -> None:
    spacing = PhysicalSpacing(x_um=0.1, y_um=0.1, z_um=0.3)
    volume = ImageVolume(data=np.zeros((2, 3, 4)), spacing=spacing, axes="ZYX", dtype="uint16")
    assert volume.axes == "ZYX"
