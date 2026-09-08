"""Tests for TA-Lib compatibility fallback engine and No-GIL multithreading."""

import concurrent.futures

import numpy as np
import pandas as pd
import pytest

from yfinance_ta_patterns.pattern_analyzer import PatternAnalyzer
from yfinance_ta_patterns.talib_compat import ALL_CDL_PATTERNS, TALibWrapper


@pytest.fixture
def sample_data():
    dates = pd.date_range("2025-01-01", periods=10, freq="1d")
    return pd.DataFrame(
        {
            "Open": [10.0, 10.0, 10.0, 12.0, 15.0, 10.0, 11.0, 10.0, 14.0, 10.0],
            "High": [15.0, 16.0, 12.0, 14.0, 16.0, 12.0, 13.0, 11.0, 15.0, 12.0],
            "Low": [9.0, 9.0, 8.0, 11.0, 14.0, 9.0, 10.0, 9.0, 13.0, 9.0],
            "Close": [10.0, 15.0, 10.0, 13.0, 15.0, 11.0, 12.0, 10.0, 14.0, 11.0],
        },
        index=dates,
    )


def test_fallback_has_all_61_patterns():
    wrapper = TALibWrapper(force_fallback=True)
    cdl_names = [f for f in dir(wrapper) if f.startswith("CDL")]
    assert len(cdl_names) == 61
    assert set(cdl_names) == set(ALL_CDL_PATTERNS)


def test_fallback_cdl_doji(sample_data):
    wrapper = TALibWrapper(force_fallback=True)
    res = wrapper.CDLDOJI(
        sample_data["Open"].values,
        sample_data["High"].values,
        sample_data["Low"].values,
        sample_data["Close"].values,
    )
    assert isinstance(res, np.ndarray)
    assert len(res) == len(sample_data)
    # Index 0 has Open=10, Close=10 -> Doji!
    assert res[0] == 100


def test_fallback_cdl_hammer():
    wrapper = TALibWrapper(force_fallback=True)
    # Hammer candle: open=100, close=102 (small body 2), low=90 (lower shadow 10 >= 2*body), high=102.5 (small upper shadow)
    res = wrapper.CDLHAMMER(
        np.array([100.0]),
        np.array([102.5]),
        np.array([90.0]),
        np.array([102.0]),
    )
    assert res[0] == 100


def test_fallback_cdl_engulfing():
    wrapper = TALibWrapper(force_fallback=True)
    # Bullish engulfing: candle 1 is red (105 -> 95), candle 2 is green and engulfs (94 -> 106)
    res = wrapper.CDLENGULFING(
        np.array([105.0, 94.0]),
        np.array([106.0, 107.0]),
        np.array([94.0, 93.0]),
        np.array([95.0, 106.0]),
    )
    assert res[1] == 100


def test_pattern_analyzer_with_fallback(sample_data):
    import yfinance_ta_patterns.pattern_analyzer as pa
    orig_talib = pa.talib
    try:
        pa.talib = TALibWrapper(force_fallback=True)
        analyzer = PatternAnalyzer(sample_data)
        signals = analyzer.get_signals("DOJI")
        assert not signals.empty
        assert signals.iloc[0] == 100
    finally:
        pa.talib = orig_talib


def test_concurrent_multithreading_without_gil(sample_data):
    wrapper = TALibWrapper(force_fallback=True)

    def worker(i):
        return wrapper.CDLDOJI(
            sample_data["Open"].values,
            sample_data["High"].values,
            sample_data["Low"].values,
            sample_data["Close"].values,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(50)))

    assert len(results) == 50
    assert results[0][0] == 100
