from pathlib import Path

import polars as pl
import pytest

from dayana_nuclei.io.manifest import (
    build_manifest,
    read_manifest_csv,
    resolve_image_sources,
    validate_manifest,
    write_manifest_csv,
)


def test_build_manifest_parses_default_filenames(tmp_path: Path) -> None:
    (tmp_path / "SW620_Sort01_low_48h_Field003_Hoechst.tif").touch()
    (tmp_path / "SW620_Sort01_low_48h_Field003_LaminA-C.tif").touch()

    df = build_manifest(tmp_path)

    assert df.height == 2
    rows = {row["channel"]: row for row in df.to_dicts()}
    assert set(rows) == {"Hoechst", "LaminA-C"}
    for row in rows.values():
        assert row["image_id"] == "SW620_Sort01_low_48h_Field003"
        assert row["cell_line"] == "SW620"
        assert row["sort_id"] == "Sort01"
        assert row["condition"] == "low"
        assert row["timepoint"] == "48h"
        assert row["field"] == "003"
        assert row["scene"] is None
        assert row["acquisition_batch"] is None


def test_build_manifest_skips_unparseable_filenames(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "SW620_Sort01_low_48h_Field003_Hoechst.tif").touch()
    (tmp_path / "random_unrelated_file.tif").touch()

    with caplog.at_level("WARNING"):
        df = build_manifest(tmp_path)

    assert df.height == 1
    assert any("random_unrelated_file.tif" in message for message in caplog.messages)


def test_build_manifest_empty_directory_returns_empty_frame(tmp_path: Path) -> None:
    df = build_manifest(tmp_path)
    assert df.height == 0
    assert list(df.columns) == [
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
        "spacing_x_um",
        "spacing_y_um",
        "spacing_z_um",
    ]


def _base_manifest(tmp_path: Path) -> pl.DataFrame:
    p1 = tmp_path / "a.tif"
    p1.touch()
    return pl.DataFrame(
        {
            "image_id": ["img1"],
            "cell_line": ["SW620"],
            "sort_id": ["Sort01"],
            "condition": ["low"],
            "timepoint": ["48h"],
            "field": ["003"],
            "channel": ["Hoechst"],
            "path": [str(p1)],
            "scene": [None],
            "acquisition_batch": [None],
        },
        schema_overrides={"scene": pl.Utf8, "acquisition_batch": pl.Utf8},
    )


def test_validate_manifest_accepts_clean_frame(tmp_path: Path) -> None:
    result = validate_manifest(_base_manifest(tmp_path))
    assert result.is_valid, result.errors


def test_validate_manifest_detects_missing_columns() -> None:
    df = pl.DataFrame({"image_id": ["img1"]})
    result = validate_manifest(df)
    assert not result.is_valid
    assert any("missing required columns" in e.lower() for e in result.errors)


def test_validate_manifest_detects_duplicate_image_channel(tmp_path: Path) -> None:
    df = pl.concat([_base_manifest(tmp_path), _base_manifest(tmp_path)])
    result = validate_manifest(df)
    assert not result.is_valid
    assert any("duplicate" in e.lower() for e in result.errors)


def test_validate_manifest_detects_inconsistent_scene(tmp_path: Path) -> None:
    df = _base_manifest(tmp_path)
    df2 = df.with_columns(
        pl.Series("channel", ["LaminAC"]),
        pl.Series("scene", ["1"]),
    )
    combined = pl.concat([df, df2])
    result = validate_manifest(combined)
    assert not result.is_valid
    assert any("inconsistent scene" in e.lower() for e in result.errors)


def test_validate_manifest_warns_on_missing_path(tmp_path: Path) -> None:
    df = _base_manifest(tmp_path).with_columns(pl.Series("path", [str(tmp_path / "missing.tif")]))
    result = validate_manifest(df)
    assert result.is_valid
    assert any("do not exist" in w for w in result.warnings)


def test_manifest_csv_round_trip(tmp_path: Path) -> None:
    (tmp_path / "SW620_Sort01_low_48h_Field003_Hoechst.tif").touch()
    df = build_manifest(tmp_path)

    csv_path = tmp_path / "manifest.csv"
    write_manifest_csv(df, csv_path)
    reloaded = read_manifest_csv(csv_path)

    assert reloaded.equals(df)


def test_validate_manifest_rejects_incomplete_spacing_override(tmp_path: Path) -> None:
    df = _base_manifest(tmp_path).with_columns(pl.lit(0.2).alias("spacing_x_um"))

    result = validate_manifest(df)

    assert not result.is_valid
    assert any("spacing_x_um and spacing_y_um" in error for error in result.errors)


def test_validate_manifest_accepts_positive_calibrated_spacing_override(tmp_path: Path) -> None:
    df = _base_manifest(tmp_path).with_columns(
        pl.lit(0.2).alias("spacing_x_um"),
        pl.lit(0.2).alias("spacing_y_um"),
        pl.lit(0.5).alias("spacing_z_um"),
    )

    result = validate_manifest(df)

    assert result.is_valid, result.errors

    sources = resolve_image_sources(df, hoechst_channel="Hoechst")
    source = sources["img1"]["Hoechst"]
    assert source.spacing_override is not None
    assert source.spacing_override.x_um == pytest.approx(0.2)
    assert source.spacing_override.y_um == pytest.approx(0.2)
    assert source.spacing_override.z_um == pytest.approx(0.5)
