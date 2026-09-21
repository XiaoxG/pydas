# tests/unit/test_state.py
import numpy as np

from pydas.core.state import (
    STATS_COLUMNS,
    empty_seg_statis,
    froude_scale_factors,
    normalize_sseg,
    require_channel,
)


def test_normalize_sseg_selectors(pydas_instance):
    """Integer / list / 'all' selectors share one normaliser."""
    assert normalize_sseg(pydas_instance, "all") == [0]
    assert normalize_sseg(pydas_instance, 0) == [0]
    assert normalize_sseg(pydas_instance, 99) == []
    assert normalize_sseg(pydas_instance, 99, on_invalid="all") == [0]
    assert normalize_sseg(pydas_instance, [0, 99]) == [0]


def test_require_channel(pydas_instance):
    assert require_channel(pydas_instance, "Wave1") is True
    assert require_channel(pydas_instance, "no_such_channel") is False


def test_empty_seg_statis_columns():
    df = empty_seg_statis()
    assert list(df.columns) == list(STATS_COLUMNS)


def test_froude_length_unit_scales_with_lam():
    new_unit, coeff, *_ = froude_scale_factors("m", lam=4.0)
    assert new_unit
    assert np.isclose(coeff, 4.0)
