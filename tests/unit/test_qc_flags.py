from nuclear_morphometry.qc.flags import compute_object_qc
from nuclear_morphometry.schema import EXCLUSION_REASON_BORDER


def test_border_object_is_excluded_by_default_with_reason() -> None:
    qc = compute_object_qc(
        touches_border=True,
        flag_border_objects=True,
        exclude_border_from_default=True,
    )
    assert qc.qc_border is True
    assert qc.qc_excluded_default is True
    assert qc.qc_exclusion_reason == EXCLUSION_REASON_BORDER


def test_border_flagged_but_not_excluded_when_configured() -> None:
    qc = compute_object_qc(
        touches_border=True,
        flag_border_objects=True,
        exclude_border_from_default=False,
    )
    assert qc.qc_border is True
    assert qc.qc_excluded_default is False
    assert qc.qc_exclusion_reason is None


def test_non_border_object_is_never_excluded() -> None:
    """The negative case from spec 45/53.6: this function has no shape inputs at all,
    so it is structurally impossible for eccentricity/solidity/circularity to
    drive exclusion through this path."""
    qc = compute_object_qc(
        touches_border=False,
        flag_border_objects=True,
        exclude_border_from_default=True,
    )
    assert qc.qc_border is False
    assert qc.qc_excluded_default is False
    assert qc.qc_exclusion_reason is None


def test_compute_object_qc_has_no_shape_parameters() -> None:
    import inspect

    params = set(inspect.signature(compute_object_qc).parameters)
    assert params == {"touches_border", "flag_border_objects", "exclude_border_from_default"}
