"""Experimental manifest: the long-form, one-row-per-channel image index.

Spec section 9. The manifest is the only supported way to resolve which
files/channels/scenes belong to which experimental image at analysis time --
filename parsing (``manifest build``) is a convenience for *constructing* a
manifest, never a runtime fallback (spec 9, opening line).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from dayana_nuclei.models import ExperimentalMetadata, ImageSource, PhysicalSpacing

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "cell_line",
    "sort_id",
    "condition",
    "timepoint",
    "field",
    "channel",
    "path",
    "scene",
    "acquisition_batch",
)

SPACING_OVERRIDE_COLUMNS: tuple[str, ...] = (
    "spacing_x_um",
    "spacing_y_um",
    "spacing_z_um",
)

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".czi", ".tif", ".tiff"})

# Default thesis filename convention, e.g.:
#   SW620_Sort01_low_48h_Field003_Hoechst.tif
#   SW620_Sort01_low_48h_Field003_LaminA-C.tif
DEFAULT_FILENAME_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<cell_line>[^_]+)_"
    r"(?P<sort_id>[^_]+)_"
    r"(?P<condition>[^_]+)_"
    r"(?P<timepoint>[^_]+)_"
    r"Field(?P<field>[^_]+)_"
    r"(?P<channel>.+)$"
)


def manifest_row(
    *,
    cell_line: str,
    sort_id: str,
    condition: str,
    timepoint: str,
    field: str,
    channel: str,
    path: Path,
    scene: str | int | None = None,
    acquisition_batch: str | None = None,
    spacing_x_um: float | None = None,
    spacing_y_um: float | None = None,
    spacing_z_um: float | None = None,
) -> dict[str, object]:
    """Build one manifest row dict.

    Shared by the filename-parsing path (``build_manifest``) and any future
    path that derives rows from multichannel image metadata instead of
    filenames -- both must agree on how ``image_id`` is formed.
    """
    image_id = f"{cell_line}_{sort_id}_{condition}_{timepoint}_Field{field}"
    return {
        "image_id": image_id,
        "cell_line": cell_line,
        "sort_id": sort_id,
        "condition": condition,
        "timepoint": timepoint,
        "field": field,
        "channel": channel,
        "path": str(path),
        "scene": str(scene) if scene is not None else None,
        "acquisition_batch": acquisition_batch,
        "spacing_x_um": spacing_x_um,
        "spacing_y_um": spacing_y_um,
        "spacing_z_um": spacing_z_um,
    }


def _empty_manifest() -> pl.DataFrame:
    schema = {
        "image_id": pl.Utf8,
        "cell_line": pl.Utf8,
        "sort_id": pl.Utf8,
        "condition": pl.Utf8,
        "timepoint": pl.Utf8,
        "field": pl.Utf8,
        "channel": pl.Utf8,
        "path": pl.Utf8,
        "scene": pl.Utf8,
        "acquisition_batch": pl.Utf8,
        "spacing_x_um": pl.Float64,
        "spacing_y_um": pl.Float64,
        "spacing_z_um": pl.Float64,
    }
    return pl.DataFrame(schema=schema)


def build_manifest(
    input_root: Path,
    *,
    pattern: re.Pattern[str] | None = None,
) -> pl.DataFrame:
    """Scan ``input_root`` recursively and build a manifest from filenames.

    Files whose name does not match ``pattern`` are skipped with a logged
    warning rather than aborting the whole scan -- a single stray or
    differently-named file must not block manifest construction for the
    rest of the dataset. The returned frame is not guaranteed valid; it
    must be passed through :func:`validate_manifest` before use (spec 9.1).
    """
    active_pattern = pattern if pattern is not None else DEFAULT_FILENAME_PATTERN

    rows: list[dict[str, object]] = []
    for path in sorted(input_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        match = active_pattern.match(path.stem)
        if match is None:
            logger.warning(
                "Skipping %s: filename does not match the manifest pattern %r",
                path,
                active_pattern.pattern,
            )
            continue
        groups = match.groupdict()
        rows.append(
            manifest_row(
                cell_line=groups["cell_line"],
                sort_id=groups["sort_id"],
                condition=groups["condition"],
                timepoint=groups["timepoint"],
                field=groups["field"],
                channel=groups["channel"],
                path=path,
            )
        )

    if not rows:
        return _empty_manifest()
    return pl.DataFrame(rows, schema=dict.fromkeys(MANIFEST_COLUMNS, pl.Utf8))


@dataclass
class ManifestValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_REQUIRED_NONNULL_COLUMNS: tuple[str, ...] = (
    "image_id",
    "path",
    "channel",
    "cell_line",
    "sort_id",
    "condition",
    "timepoint",
    "field",
)


def validate_manifest(df: pl.DataFrame) -> ManifestValidationResult:
    """Validate a manifest frame before it is used for analysis.

    Path existence is reported as a warning, not a hard error: a manifest is
    routinely built/reviewed on one machine (e.g. a laptop) and consumed on
    another (e.g. a GPU workstation) where the data root differs, and the
    pipeline's own I/O layer will raise a hard, actionable error at load
    time if a path genuinely cannot be read. Everything else here is a
    manifest-authoring error and is a hard error.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing_columns:
        errors.append(f"Manifest is missing required columns: {missing_columns}")
        return ManifestValidationResult(is_valid=False, errors=errors, warnings=warnings)

    for col in _REQUIRED_NONNULL_COLUMNS:
        n_bad = df.filter(pl.col(col).is_null() | (pl.col(col).cast(pl.Utf8) == "")).height
        if n_bad:
            errors.append(f"Column {col!r} has {n_bad} null/empty value(s).")

    dup_key = (
        df.group_by(["image_id", "channel"]).agg(pl.len().alias("_n")).filter(pl.col("_n") > 1)
    )
    if dup_key.height:
        pairs = list(zip(dup_key["image_id"].to_list(), dup_key["channel"].to_list(), strict=True))
        errors.append(f"Duplicate (image_id, channel) entries found: {pairs}")

    if "path" in df.columns:
        missing_paths = [
            p for p in df["path"].unique().to_list() if p is not None and not Path(p).exists()
        ]
        if missing_paths:
            warnings.append(
                f"{len(missing_paths)} referenced path(s) do not exist on this machine: "
                f"{missing_paths[:5]}{'...' if len(missing_paths) > 5 else ''}"
            )

    present_spacing_columns = [column for column in SPACING_OVERRIDE_COLUMNS if column in df]
    if present_spacing_columns:
        for row_index, row in enumerate(df.iter_rows(named=True), start=1):
            values = {column: row.get(column) for column in SPACING_OVERRIDE_COLUMNS}
            if not any(value is not None for value in values.values()):
                continue
            if values["spacing_x_um"] is None or values["spacing_y_um"] is None:
                errors.append(
                    f"Manifest row {row_index}: spacing_x_um and spacing_y_um must both be "
                    f"provided when any calibrated spacing override is used."
                )
                continue
            for column, value in values.items():
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    errors.append(
                        f"Manifest row {row_index}: {column} must be a positive number in um "
                        f"(got {value!r})."
                    )
                    continue
                if numeric <= 0:
                    errors.append(
                        f"Manifest row {row_index}: {column} must be > 0 um (got {numeric})."
                    )

    scene_consistency = (
        df.group_by(["image_id", "path"])
        .agg(pl.col("scene").n_unique().alias("_n_scenes"))
        .filter(pl.col("_n_scenes") > 1)
    )
    if scene_consistency.height:
        bad_pairs = list(
            zip(
                scene_consistency["image_id"].to_list(),
                scene_consistency["path"].to_list(),
                strict=True,
            )
        )
        errors.append(f"Inconsistent scene values for the same (image_id, path): {bad_pairs}")

    return ManifestValidationResult(is_valid=not errors, errors=errors, warnings=warnings)


def read_manifest_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        schema_overrides={"scene": pl.Utf8, "acquisition_batch": pl.Utf8, "field": pl.Utf8},
    )
    casts = [
        pl.col(column).cast(pl.Float64, strict=False)
        for column in SPACING_OVERRIDE_COLUMNS
        if column in df.columns
    ]
    return df.with_columns(casts) if casts else df


def write_manifest_csv(df: pl.DataFrame, path: Path) -> None:
    df.write_csv(path)


def resolve_image_sources(
    df: pl.DataFrame,
    *,
    hoechst_channel: str,
    additional_channels: tuple[str, ...] = (),
) -> dict[str, dict[str, ImageSource]]:
    """Group a validated manifest into per-image_id {channel: ImageSource}.

    Must only be called on a manifest that has already passed
    ``validate_manifest``. Raises with an actionable message (spec 7.3) if
    the required Hoechst channel cannot be resolved for an image_id -- this
    is a hard requirement, not a warning, because segmentation has no other
    input.
    """
    required = {hoechst_channel, *additional_channels}
    sources: dict[str, dict[str, ImageSource]] = {}

    for row in df.iter_rows(named=True):
        image_id = row["image_id"]
        channel = row["channel"]
        if channel not in required:
            continue

        metadata = ExperimentalMetadata(
            image_id=image_id,
            cell_line=row["cell_line"],
            sort_id=row["sort_id"],
            condition=row["condition"],
            timepoint=row["timepoint"],
            field=row["field"],
            acquisition_batch=row["acquisition_batch"],
        )
        scene_raw = row["scene"]
        scene: str | int | None = scene_raw
        if scene_raw is not None and scene_raw.lstrip("-").isdigit():
            scene = int(scene_raw)

        spacing_override = None
        spacing_x_um = row.get("spacing_x_um")
        spacing_y_um = row.get("spacing_y_um")
        spacing_z_um = row.get("spacing_z_um")
        if spacing_x_um is not None and spacing_y_um is not None:
            spacing_override = PhysicalSpacing(
                x_um=float(spacing_x_um),
                y_um=float(spacing_y_um),
                z_um=float(spacing_z_um) if spacing_z_um is not None else None,
            )

        sources.setdefault(image_id, {})[channel] = ImageSource(
            path=Path(row["path"]),
            scene=scene,
            channel=channel,
            metadata=metadata,
            spacing_override=spacing_override,
        )

    missing_hoechst = [
        image_id for image_id, channels in sources.items() if hoechst_channel not in channels
    ]
    if missing_hoechst:
        raise ValueError(
            f"Required Hoechst channel {hoechst_channel!r} cannot be resolved for "
            f"{len(missing_hoechst)} image_id(s): {missing_hoechst[:5]}"
            f"{'...' if len(missing_hoechst) > 5 else ''}.\n\n"
            f"Check the manifest's 'channel' column against "
            f"input.hoechst_channel in the config."
        )
    return sources
