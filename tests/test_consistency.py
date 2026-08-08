"""consistency._rescale(): confidence must land in the observed band, monotonically."""
import pandas as pd
import pytest

from router.consistency import TARGET_HI, TARGET_LO, _rescale


def test_output_always_in_band():
    conf = pd.Series([0.90, 0.93, 0.95, 0.97, 0.98])
    result = _rescale(conf)
    assert (result >= TARGET_LO).all()
    assert (result <= TARGET_HI).all()


def test_rank_preserving():
    conf = pd.Series([0.90, 0.93, 0.97])
    result = _rescale(conf)
    assert result.iloc[0] <= result.iloc[1] <= result.iloc[2]


def test_degenerate_returns_band_midpoint():
    # hi <= lo: no spread to preserve -> everything at the band midpoint.
    conf = pd.Series([0.92, 0.92, 0.92])
    result = _rescale(conf)
    midpoint = (TARGET_LO + TARGET_HI) / 2
    assert result.tolist() == pytest.approx([midpoint] * len(conf))


def test_single_element_stays_in_band():
    result = _rescale(pd.Series([0.95]))
    assert TARGET_LO <= float(result.iloc[0]) <= TARGET_HI
