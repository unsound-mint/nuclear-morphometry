import math
from pathlib import Path

import polars as pl
import pytest

from dayana_nuclei.compare_measurements import (
    MeasurementMapping,
    compare_measurements,
    compare_measurements_from_files,
    load_measurement_mapping,
)


def _mapping(**overrides: object) -> MeasurementMapping:
    defaults: dict[str, object] = {"columns": {"AreaShape_Area": "area_px"}}
    defaults.update(overrides)
    return MeasurementMapping.model_validate(defaults)


def test_mapping_rejects_mismatched_join_key_lengths() -> None:
    with pytest.raises(ValueError, match="same number of columns"):
        _mapping(reference_join_keys=("a", "b"), ours_join_keys=("a",))


def test_mapping_rejects_empty_columns() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        MeasurementMapping.model_validate({"columns": {}})


def test_load_measurement_mapping_reads_join_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "mapping.toml"
    path.write_text(
        """
        [join]
        reference = ["image_id", "ObjectNumber"]
        ours = ["image_id", "object_number"]

        [columns]
        "AreaShape_Area" = "area_px"
        "AreaShape_Eccentricity" = "eccentricity"
        """
    )
    mapping = load_measurement_mapping(path)
    assert mapping.reference_join_keys == ("image_id", "ObjectNumber")
    assert mapping.ours_join_keys == ("image_id", "object_number")
    assert mapping.columns == {
        "AreaShape_Area": "area_px",
        "AreaShape_Eccentricity": "eccentricity",
    }


def test_load_measurement_mapping_defaults_join_keys(tmp_path: Path) -> None:
    path = tmp_path / "mapping.toml"
    path.write_text('[columns]\n"AreaShape_Area" = "area_px"\n')
    mapping = load_measurement_mapping(path)
    assert mapping.reference_join_keys == ("image_id", "object_number")
    assert mapping.ours_join_keys == ("image_id", "object_number")


def test_load_measurement_mapping_raises_on_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "mapping.toml"
    path.write_text("not valid [[[ toml")
    with pytest.raises(ValueError, match="Failed to parse mapping"):
        load_measurement_mapping(path)


def test_identical_measurements_have_zero_difference_and_perfect_correlation() -> None:
    ours = pl.DataFrame(
        {"image_id": ["a", "a"], "object_number": [1, 2], "area_px": [100.0, 200.0]}
    )
    reference = pl.DataFrame(
        {"image_id": ["a", "a"], "ObjectNumber": [1, 2], "AreaShape_Area": [100.0, 200.0]}
    )
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    report = compare_measurements(ours, reference, mapping)

    assert report.n_matched_objects == 2
    assert report.n_only_in_ours == 0
    assert report.n_only_in_reference == 0
    col = report.columns[0]
    assert col.n_both_finite == 2
    assert col.mean_absolute_difference == pytest.approx(0.0)
    assert col.mean_relative_difference == pytest.approx(0.0)
    assert col.pearson_r == pytest.approx(1.0)


def test_unmatched_objects_are_reported_not_silently_dropped() -> None:
    ours = pl.DataFrame(
        {"image_id": ["a", "a", "a"], "object_number": [1, 2, 3], "area_px": [10.0, 20.0, 30.0]}
    )
    reference = pl.DataFrame(
        {"image_id": ["a", "a"], "ObjectNumber": [1, 2], "AreaShape_Area": [10.0, 25.0]}
    )
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    report = compare_measurements(ours, reference, mapping)

    assert report.n_ours_objects == 3
    assert report.n_reference_objects == 2
    assert report.n_matched_objects == 2
    assert report.n_only_in_ours == 1
    assert report.n_only_in_reference == 0


def test_relative_difference_is_none_when_all_references_are_zero() -> None:
    ours = pl.DataFrame({"image_id": ["a"], "object_number": [1], "area_px": [5.0]})
    reference = pl.DataFrame({"image_id": ["a"], "ObjectNumber": [1], "AreaShape_Area": [0.0]})
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    report = compare_measurements(ours, reference, mapping)

    col = report.columns[0]
    assert col.mean_absolute_difference == pytest.approx(5.0)
    assert col.mean_relative_difference is None


def test_no_comparable_rows_reports_none_not_nan_or_error() -> None:
    ours = pl.DataFrame({"image_id": ["a"], "object_number": [1], "area_px": [None]})
    reference = pl.DataFrame({"image_id": ["a"], "ObjectNumber": [1], "AreaShape_Area": [10.0]})
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    report = compare_measurements(ours, reference, mapping)

    col = report.columns[0]
    assert col.n_both_finite == 0
    assert col.mean_absolute_difference is None
    assert col.pearson_r is None


def test_colliding_column_name_on_both_sides_is_not_confused() -> None:
    """If ours and the reference both happen to use the same raw column name
    for something unrelated to the mapped measurement, the mapped reference
    column must never be silently shadowed by ours' column of that name."""
    ours = pl.DataFrame(
        {
            "image_id": ["a"],
            "object_number": [1],
            "area_px": [100.0],
            "AreaShape_Area": [999.0],  # decoy: same raw name as the reference's mapped column
        }
    )
    reference = pl.DataFrame({"image_id": ["a"], "ObjectNumber": [1], "AreaShape_Area": [100.0]})
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    report = compare_measurements(ours, reference, mapping)

    col = report.columns[0]
    assert col.mean_absolute_difference == pytest.approx(0.0)


def test_compare_measurements_from_files_supports_csv_and_parquet(tmp_path: Path) -> None:
    ours_path = tmp_path / "nuclei.parquet"
    reference_path = tmp_path / "legacy.csv"
    mapping_path = tmp_path / "mapping.toml"

    pl.DataFrame({"image_id": ["a"], "object_number": [1], "area_px": [42.0]}).write_parquet(
        ours_path
    )
    pl.DataFrame({"image_id": ["a"], "ObjectNumber": [1], "AreaShape_Area": [42.0]}).write_csv(
        reference_path
    )
    mapping_path.write_text(
        """
        [join]
        reference = ["image_id", "ObjectNumber"]
        ours = ["image_id", "object_number"]

        [columns]
        "AreaShape_Area" = "area_px"
        """
    )

    report = compare_measurements_from_files(ours_path, reference_path, mapping_path)

    assert report.n_matched_objects == 1
    assert report.columns[0].mean_absolute_difference == pytest.approx(0.0)


def test_duplicate_join_keys_in_ours_raise_instead_of_cartesian_join() -> None:
    ours = pl.DataFrame({"image_id": ["a", "a"], "object_number": [1, 1], "area_px": [10.0, 20.0]})
    reference = pl.DataFrame({"image_id": ["a"], "ObjectNumber": [1], "AreaShape_Area": [10.0]})
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    with pytest.raises(ValueError, match="duplicate values"):
        compare_measurements(ours, reference, mapping)


def test_duplicate_join_keys_in_reference_raise() -> None:
    ours = pl.DataFrame({"image_id": ["a"], "object_number": [1], "area_px": [10.0]})
    reference = pl.DataFrame(
        {"image_id": ["a", "a"], "ObjectNumber": [1, 1], "AreaShape_Area": [10.0, 20.0]}
    )
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    with pytest.raises(ValueError, match="duplicate values"):
        compare_measurements(ours, reference, mapping)


def test_missing_reference_column_raises_named_error() -> None:
    ours = pl.DataFrame({"image_id": ["a"], "object_number": [1], "area_px": [10.0]})
    reference = pl.DataFrame({"image_id": ["a"], "ObjectNumber": [1], "SomethingElse": [10.0]})
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    with pytest.raises(ValueError, match="AreaShape_Area"):
        compare_measurements(ours, reference, mapping)


def test_missing_ours_column_raises_named_error() -> None:
    ours = pl.DataFrame({"image_id": ["a"], "object_number": [1], "something_else": [10.0]})
    reference = pl.DataFrame({"image_id": ["a"], "ObjectNumber": [1], "AreaShape_Area": [10.0]})
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    with pytest.raises(ValueError, match="area_px"):
        compare_measurements(ours, reference, mapping)


def test_nan_and_inf_values_are_excluded_not_corrupting_statistics() -> None:
    ours = pl.DataFrame(
        {
            "image_id": ["a", "a", "a"],
            "object_number": [1, 2, 3],
            "area_px": [10.0, math.nan, math.inf],
        }
    )
    reference = pl.DataFrame(
        {"image_id": ["a", "a", "a"], "ObjectNumber": [1, 2, 3], "AreaShape_Area": [10.0, 5.0, 7.0]}
    )
    mapping = _mapping(
        reference_join_keys=("image_id", "ObjectNumber"),
        ours_join_keys=("image_id", "object_number"),
    )

    report = compare_measurements(ours, reference, mapping)

    col = report.columns[0]
    assert col.n_both_finite == 1
    assert col.mean_absolute_difference == pytest.approx(0.0)
    assert math.isfinite(col.mean_absolute_difference)


def test_unsupported_table_format_raises(tmp_path: Path) -> None:
    ours_path = tmp_path / "nuclei.txt"
    ours_path.write_text("not a table")
    reference_path = tmp_path / "legacy.csv"
    pl.DataFrame({"image_id": ["a"], "ObjectNumber": [1], "AreaShape_Area": [1.0]}).write_csv(
        reference_path
    )
    mapping_path = tmp_path / "mapping.toml"
    mapping_path.write_text('[columns]\n"AreaShape_Area" = "area_px"\n')

    with pytest.raises(ValueError, match="Unsupported table format"):
        compare_measurements_from_files(ours_path, reference_path, mapping_path)
