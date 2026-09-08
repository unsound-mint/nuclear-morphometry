"""Structural and physical-calibration inspection of a microscopy file (spec section 10.5).

Backs ``nuclear-morphometry inspect``. Reads structure and calibration without
requiring pixel data to be loaded, and never fabricates a physical pixel
size that the source file does not actually declare.
"""

from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

from bioio import BioImage
from pydantic import BaseModel, ConfigDict, Field

# tifffile's TIFF.RESUNIT.NONE. bioio-tifffile computes a physical pixel size
# from tifffile's default (resolution=(1, 1), resolutionunit=NONE) when a TIFF
# carries no real resolution tag at all, which is indistinguishable from a
# genuine 1.0 um/pixel calibration unless this specific unit code is checked.
# See docs/decisions/0006-tiff-uncalibrated-resolution-detection.md.
_TIFF_RESOLUTIONUNIT_NONE = 1
_MICROMETER_UNITS = frozenset({"um", "µm", "micron", "microns", "micrometer", "micrometers"})


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


def _parse_metamorph_descriptions(
    descriptions: list[str],
) -> tuple[float | None, float | None, float | None, list[str]]:
    """Extract calibrated X/Y and Z step from MetaMorph page XML.

    MetaMorph writes physical X/Y calibration and an absolute Z position into
    each page's ImageDescription instead of standard TIFF resolution tags.
    ``pixel-size-x/y`` are deliberately ignored: in these files they are the
    1200x1200 array dimensions, not physical pixel sizes.
    """
    warnings_list: list[str] = []
    page_properties: list[dict[str, str]] = []
    saw_metamorph = False

    for page_index, description in enumerate(descriptions):
        if not description or "<MetaData" not in description:
            page_properties.append({})
            continue
        saw_metamorph = True
        try:
            root = ET.fromstring(description)
        except ET.ParseError as exc:
            warnings_list.append(
                f"MetaMorph ImageDescription XML on TIFF page {page_index} is invalid: {exc}."
            )
            page_properties.append({})
            continue
        page_properties.append(
            {
                prop.attrib["id"]: prop.attrib.get("value", "")
                for prop in root.iter("prop")
                if "id" in prop.attrib
            }
        )

    if not saw_metamorph:
        return None, None, None, []

    xy_values: list[tuple[float, float]] = []
    for properties in page_properties:
        if properties.get("spatial-calibration-state", "").lower() != "on":
            continue
        units = properties.get("spatial-calibration-units", "").strip().lower()
        if units not in _MICROMETER_UNITS:
            continue
        try:
            x_value = float(properties["spatial-calibration-x"])
            y_value = float(properties["spatial-calibration-y"])
        except (KeyError, ValueError):
            continue
        if x_value > 0 and y_value > 0:
            xy_values.append((x_value, y_value))

    x_um: float | None = None
    y_um: float | None = None
    if xy_values:
        first_x, first_y = xy_values[0]
        if all(
            math.isclose(x, first_x, rel_tol=1e-6, abs_tol=1e-9)
            and math.isclose(y, first_y, rel_tol=1e-6, abs_tol=1e-9)
            for x, y in xy_values[1:]
        ):
            x_um, y_um = first_x, first_y
        else:
            warnings_list.append(
                "MetaMorph pages contain inconsistent X/Y physical calibration; "
                "X/Y spacing is being treated as unknown."
            )

    z_text = [properties.get("z-position") for properties in page_properties]
    z_um: float | None = None
    if len(descriptions) > 1 and all(value not in (None, "") for value in z_text):
        z_strings = [str(value) for value in z_text]
        try:
            z_positions = [float(value) for value in z_strings]
        except ValueError:
            warnings_list.append(
                "MetaMorph contains a non-numeric Z position; Z spacing is being treated "
                "as unknown."
            )
        else:
            signed_steps = [b - a for a, b in pairwise(z_positions)]
            if (
                signed_steps
                and all(step != 0 for step in signed_steps)
                and (
                    all(step > 0 for step in signed_steps) or all(step < 0 for step in signed_steps)
                )
            ):
                absolute_steps = [abs(step) for step in signed_steps]
                candidate = statistics.median(absolute_steps)
                decimal_places = [
                    len(value.partition(".")[2]) if "." in value else 0 for value in z_strings
                ]
                stored_resolution = 10.0 ** (-max(decimal_places, default=0))
                tolerance = max(2 * stored_resolution, candidate * 0.05, 1e-9)
                if candidate > 0 and all(
                    abs(step - candidate) <= tolerance for step in absolute_steps
                ):
                    z_um = round(candidate, max(decimal_places, default=0))
                else:
                    warnings_list.append(
                        "MetaMorph pages contain non-uniform Z positions; Z spacing is "
                        "being treated as unknown."
                    )
            else:
                warnings_list.append(
                    "MetaMorph Z positions are repeated or non-monotonic; Z spacing is "
                    "being treated as unknown."
                )
    elif len(descriptions) > 1 and any(value not in (None, "") for value in z_text):
        warnings_list.append(
            "MetaMorph Z positions are missing from one or more TIFF pages; Z spacing is "
            "being treated as unknown."
        )

    return x_um, y_um, z_um, warnings_list


def _read_metamorph_spacing(
    path: Path, scene_index: int
) -> tuple[float | None, float | None, float | None, list[str]]:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        page_indices = [page.index for page in tf.series[scene_index].pages if page is not None]
        descriptions = [
            str(getattr(tf.pages[index], "description", "") or "") for index in page_indices
        ]
    return _parse_metamorph_descriptions(descriptions)


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

    if is_tiff_reader:
        current_scene_index = img.scenes.index(img.current_scene)
        meta_x, meta_y, meta_z, metamorph_warnings = _read_metamorph_spacing(
            path, current_scene_index
        )
        warnings_list.extend(metamorph_warnings)
        if meta_x is not None and meta_y is not None:
            x_um, y_um = meta_x, meta_y
        elif (x_um is not None or y_um is not None) and not _tiff_has_real_resolution_tag(
            path, current_scene_index
        ):
            warnings_list.append(
                "TIFF file has no resolution tag (or ResolutionUnit=NONE); the "
                "underlying reader's physical_pixel_sizes default cannot be "
                "distinguished from a genuine 1.0 um/pixel calibration, so X/Y "
                "spacing is being treated as unknown rather than trusted."
            )
            x_um = None
            y_um = None
        if meta_z is not None:
            z_um = meta_z

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
            f"Check the path and re-run: nuclear-morphometry inspect {path}"
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
