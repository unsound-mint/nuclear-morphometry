"""Structural and physical-calibration inspection of a microscopy file (spec section 10.5).

Backs ``dayana-nuclei inspect``. Reads structure and calibration without
requiring pixel data to be loaded, and never fabricates a physical pixel
size that the source file does not actually declare.
"""

from __future__ import annotations

from pathlib import Path

from bioio import BioImage
from pydantic import BaseModel, ConfigDict, Field

# tifffile's TIFF.RESUNIT.NONE. bioio-tifffile computes a physical pixel size
# from tifffile's default (resolution=(1, 1), resolutionunit=NONE) when a TIFF
# carries no real resolution tag at all, which is indistinguishable from a
# genuine 1.0 um/pixel calibration unless this specific unit code is checked.
# See docs/decisions/0006-tiff-uncalibrated-resolution-detection.md.
_TIFF_RESOLUTIONUNIT_NONE = 1


class ImageInspection(BaseModel):
    """Structural + physical-calibration summary of one microscopy file."""

    model_config = ConfigDict(frozen=True)

    path: Path
    format: str
    scenes: tuple[str, ...]
    current_scene: str
    dims_order: str
    shape: tuple[int, ...]
    channel_names: tuple[str, ...]
    dtype: str
    x_um: float | None
    y_um: float | None
    z_um: float | None
    n_z_planes: int
    n_channels: int
    warnings: tuple[str, ...] = Field(default_factory=tuple)


def _tiff_has_real_resolution_tag(path: Path, scene_index: int) -> bool:
    """True if the TIFF series declares an actual resolution unit, not tifffile's default.

    ImageJ-format TIFFs are exempt: bioio-tifffile derives their calibration
    from the ``imagej_metadata['unit']``/``['spacing']`` keys, not from the
    standard ResolutionUnit tag, and tifffile's ImageJ writer defaults that
    tag to NONE regardless of whether real calibration was provided via the
    ImageJ metadata keys. The ambiguous-default risk this function guards
    against is specific to the non-ImageJ "shaped" TIFF path.
    """
    import tifffile

    with tifffile.TiffFile(path) as tf:
        if tf.is_imagej:
            return True
        tags = tf.series[scene_index].keyframe.tags
        if "ResolutionUnit" not in tags:
            return False
        return tags["ResolutionUnit"].value != _TIFF_RESOLUTIONUNIT_NONE


def read_physical_spacing(
    img: BioImage, path: Path
) -> tuple[float | None, float | None, float | None, list[str]]:
    """Read (x_um, y_um, z_um) from an already-scened BioImage, plus any warnings.

    Shared by ``inspect_image`` and ``io.images.load_channel_volume`` so both
    apply the same TIFF-uncalibrated-default detection (see module docstring
    and docs/decisions/0006-tiff-uncalibrated-resolution-detection.md).
    """
    warnings_list: list[str] = []
    sizes = img.physical_pixel_sizes
    x_um, y_um, z_um = sizes.X, sizes.Y, sizes.Z

    reader_module = type(img.reader).__module__
    is_tiff_reader = reader_module.startswith("bioio_tifffile")

    if is_tiff_reader and (x_um is not None or y_um is not None):
        current_scene_index = img.scenes.index(img.current_scene)
        if not _tiff_has_real_resolution_tag(path, current_scene_index):
            warnings_list.append(
                "TIFF file has no resolution tag (or ResolutionUnit=NONE); the "
                "underlying reader's physical_pixel_sizes default cannot be "
                "distinguished from a genuine 1.0 um/pixel calibration, so X/Y "
                "spacing is being treated as unknown rather than trusted."
            )
            x_um = None
            y_um = None

    return x_um, y_um, z_um, warnings_list


def inspect_image(path: Path, scene: str | int | None = None) -> ImageInspection:
    """Read structural and physical-calibration metadata for one microscopy file.

    If the file has multiple scenes and ``scene`` is not given, the first
    scene is inspected and a warning is recorded -- this is informational
    only. Analysis code (``io.images.load_channel_volume``) enforces the
    hard requirement that a manifest specify the scene explicitly.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot inspect {path}: file does not exist.\n"
            f"Check the path and re-run: dayana-nuclei inspect {path}"
        )

    img = BioImage(path)
    warnings_list: list[str] = []

    if scene is not None:
        img.set_scene(scene)
    elif len(img.scenes) > 1:
        warnings_list.append(
            f"File contains {len(img.scenes)} scenes {img.scenes!r}; inspecting the "
            f"first ({img.current_scene!r}). Analysis requires the manifest to specify "
            f"a scene explicitly for multi-scene files."
        )

    x_um, y_um, z_um, spacing_warnings = read_physical_spacing(img, path)
    warnings_list.extend(spacing_warnings)
    reader_module = type(img.reader).__module__

    if z_um is None and img.dims.Z > 1:
        warnings_list.append(
            f"{img.dims.Z} Z planes present but no Z step (z_um) could be read "
            f"from file metadata. 3D analysis will fail until this is resolved."
        )

    return ImageInspection(
        path=path,
        format=reader_module,
        scenes=tuple(img.scenes),
        current_scene=img.current_scene,
        dims_order=img.dims.order,
        shape=tuple(img.dims.shape),
        channel_names=tuple(str(c) for c in img.channel_names),
        dtype=str(img.dtype),
        x_um=x_um,
        y_um=y_um,
        z_um=z_um,
        n_z_planes=img.dims.Z,
        n_channels=img.dims.C,
        warnings=tuple(warnings_list),
    )
