"""Load one channel of one field as an axis-normalized, calibrated ImageVolume.

Spec sections 10.2 (axes), 10.3 (scenes), 10.4 (calibration), 12 (2D/3D
semantics). This module must not import napari or Qt (spec section 6).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from bioio import BioImage

from nuclear_morphometry.io.metadata import read_physical_spacing
from nuclear_morphometry.models import ImageSource, ImageVolume, PhysicalSpacing

Mode = Literal["2d", "3d"]
Projection = Literal["none", "max", "mean", "specific_plane"]


def _resolve_spacing(
    source: ImageSource,
    embedded: tuple[float | None, float | None, float | None],
) -> tuple[float | None, float | None, float | None]:
    override = source.spacing_override
    if override is None:
        return embedded

    override_values = (override.x_um, override.y_um, override.z_um)
    names = ("x_um", "y_um", "z_um")
    conflicts = [
        f"{name}: embedded={found}, manifest={supplied}"
        for name, found, supplied in zip(names, embedded, override_values, strict=True)
        if found is not None
        and supplied is not None
        and not math.isclose(found, supplied, rel_tol=1e-6, abs_tol=1e-9)
    ]
    if conflicts:
        raise ValueError(
            f"Calibrated manifest spacing for {source.metadata.image_id!r}, channel "
            f"{source.channel!r} conflicts with embedded metadata "
            f"({'; '.join(conflicts)}). Correct the manifest or source metadata; the "
            f"pipeline will not choose one silently."
        )
    resolved = tuple(
        found if found is not None else supplied
        for found, supplied in zip(embedded, override_values, strict=True)
    )
    return resolved[0], resolved[1], resolved[2]


def _resolve_channel_index(img: BioImage, channel: str, path: Path) -> int:
    names = [str(c) for c in img.channel_names]
    if channel not in names:
        raise ValueError(
            f"Channel {channel!r} not found in {path}. Available channels: {names!r}.\n"
            f"Fix the manifest's channel column or the config's hoechst_channel/"
            f"additional_channels to match one of the available names."
        )
    return names.index(channel)


def _resolve_scene(img: BioImage, source: ImageSource) -> None:
    if len(img.scenes) > 1:
        if source.scene is None:
            raise ValueError(
                f"{source.path} contains {len(img.scenes)} scenes {img.scenes!r} but "
                f"the manifest row for image_id={source.metadata.image_id!r}, "
                f"channel={source.channel!r} does not specify one.\n\n"
                f"Inspect the file with:\n"
                f"  nuclear-morphometry inspect {source.path}\n\n"
                f"Then set the manifest's scene column to one of the scenes listed above."
            )
        img.set_scene(source.scene)
    elif source.scene is not None:
        img.set_scene(source.scene)


def load_channel_volume(
    source: ImageSource,
    *,
    mode: Mode,
    projection: Projection = "none",
    specific_plane: int | None = None,
) -> ImageVolume:
    """Load one channel of one field, normalized to YX (2D) or ZYX (3D).

    3D mode always loads the intact volume and requires real X/Y/Z spacing;
    it never projects. 2D mode requires an explicit projection strategy when
    the source has multiple Z planes -- ``projection="none"`` on a Z>1 source
    is a hard failure, not a silent max-projection.
    """
    if mode == "2d" and projection == "specific_plane" and specific_plane is None:
        raise ValueError(
            "projection='specific_plane' requires specific_plane to be set to a Z index."
        )

    img = BioImage(source.path)
    _resolve_scene(img, source)
    channel_index = _resolve_channel_index(img, source.channel, source.path)

    x_um, y_um, z_um, spacing_warnings = read_physical_spacing(img, source.path)
    del spacing_warnings  # load_channel_volume raises on missing spacing rather than warning
    x_um, y_um, z_um = _resolve_spacing(source, (x_um, y_um, z_um))

    n_z = img.dims.Z

    if mode == "3d":
        if x_um is None or y_um is None or z_um is None:
            raise ValueError(
                f"3D analysis requires physical X/Y/Z spacing, but "
                f"{source.metadata.image_id!r} has x_um={x_um}, y_um={y_um}, "
                f"z_um={z_um}.\n\n"
                f"Inspect the source with:\n"
                f"  nuclear-morphometry inspect {source.path}\n\n"
                f"Then either fix the source/manifest metadata or supply an "
                f"explicitly calibrated value from the acquisition record. "
                f"The pipeline will not assume isotropic (1/1/1) voxel spacing."
            )
        data = img.get_image_data("ZYX", C=channel_index)
        spacing = PhysicalSpacing(x_um=x_um, y_um=y_um, z_um=z_um)
        return ImageVolume(data=data, spacing=spacing, axes="ZYX", dtype=str(img.dtype))

    # mode == "2d"
    if x_um is None or y_um is None:
        raise ValueError(
            f"2D analysis requires physical X/Y spacing for calibrated output, but "
            f"{source.metadata.image_id!r} has x_um={x_um}, y_um={y_um}.\n\n"
            f"Inspect the source with:\n"
            f"  nuclear-morphometry inspect {source.path}\n\n"
            f"Then fix the source/manifest metadata before running 2D analysis."
        )
    spacing = PhysicalSpacing(x_um=x_um, y_um=y_um, z_um=None)

    if n_z <= 1:
        data = img.get_image_data("YX", C=channel_index)
        return ImageVolume(data=data, spacing=spacing, axes="YX", dtype=str(img.dtype))

    if projection == "none":
        raise ValueError(
            f"{source.metadata.image_id!r} has {n_z} Z planes but analysis.mode=2d "
            f"with analysis.projection='none' (the default).\n\n"
            f"The pipeline will not silently max-project a Z-stack. Either set "
            f"analysis.mode='3d' to analyze the intact volume, or explicitly set "
            f"analysis.projection to 'max', 'mean', or 'specific_plane' in the config."
        )

    if projection == "max":
        volume = img.get_image_data("ZYX", C=channel_index)
        data = volume.max(axis=0)
    elif projection == "mean":
        volume = img.get_image_data("ZYX", C=channel_index)
        data = volume.mean(axis=0).astype(volume.dtype)
    else:  # specific_plane
        assert specific_plane is not None
        if not (0 <= specific_plane < n_z):
            raise ValueError(
                f"projection='specific_plane' requested plane {specific_plane}, but "
                f"{source.metadata.image_id!r} only has {n_z} Z planes (valid range "
                f"0..{n_z - 1})."
            )
        data = img.get_image_data("YX", C=channel_index, Z=specific_plane)

    return ImageVolume(data=data, spacing=spacing, axes="YX", dtype=str(img.dtype))
