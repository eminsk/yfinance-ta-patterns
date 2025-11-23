from yfinance_ta_patterns.cli import normalize_timeframe


def test_normalize_timeframe_aliases():
    assert normalize_timeframe("M5") == "5m"
    assert normalize_timeframe("h1") == "1h"


def test_normalize_timeframe_invalid():
    try:
        normalize_timeframe("invalid")
    except ValueError as exc:
        assert "Unsupported timeframe" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid timeframe")
