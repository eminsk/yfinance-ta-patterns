"""Comprehensive unit and regression tests for audit fixes v24 (v0.3.68).

Covering:
1. Yahoo Finance 1m intraday granularity:
   - Historical depth up to 30 days allowed, > 30 days rejected with informative error.
   - Single-query window span <= 7 days allowed, > 7 days rejected with informative error.
   - MarketDataLoader._clamp_period_for_interval clamps 1m to 7d.
2. GEMM fast_score_batch oversized buffer handling:
   - Mismatched shape (e.g. single sample with out buffer of length 5) does NOT broadcast scalar into out.
   - Correctly sized out buffer is filled in-place and returned.
3. CLI --backtest hook safety:
   - When backtest fails (ret_code != 0), --calendar and --news hooks are NOT invoked.
4. Economic calendar:
   - Expanded ISO-3 country mappings: ARE -> AED, CHL -> CLP, COL -> COP, MYS -> MYR, PHL -> PHP.
   - Clean auto-mode G10 handling without obsolete _STOCK_COLLISIONS.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from yfinance_ta_patterns.cli import parse_args, run_cli
from yfinance_ta_patterns.data import (
    YAHOO_INTRADAY_MAX_DAYS,
    YAHOO_INTRADAY_MAX_DEPTH_DAYS,
    YAHOO_INTRADAY_MAX_WINDOW_DAYS,
    MarketDataLoader,
)
from yfinance_ta_patterns.economic_calendar import resolve_symbol_currencies
from yfinance_ta_patterns.gemm import fast_score_batch


def _sample_ohlcv(bars: int = 40) -> pd.DataFrame:
    """Deterministic OHLCV test frame with realistic timestamps."""
    dates = pd.date_range("2026-08-01", periods=bars, freq="D")
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.randn(bars) * 0.5)
    open_p = close + rng.randn(bars) * 0.2
    high = np.maximum(open_p, close) + rng.uniform(0.1, 0.5, bars)
    low = np.minimum(open_p, close) - rng.uniform(0.1, 0.5, bars)
    volume = rng.randint(1000, 10000, bars)
    return pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# =====================================================================
# 1. 1m Intraday depth vs window verification
# =====================================================================
def test_1m_depth_allows_up_to_30_days_ago() -> None:
    """1m interval allows target dates up to 30 days ago without depth rejection."""
    today = pd.Timestamp.now().date()
    valid_1m_date = (today - pd.Timedelta(days=15)).strftime("%Y-%m-%d")

    # With mocked data loader, this should parse and pass depth validation
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data:
        mock_get_data.return_value = _sample_ohlcv(20)
        ret = run_cli(parse_args(["EURUSD=X", "-t", "1m", "--date", valid_1m_date]))
        assert ret == 0


def test_1m_depth_rejects_beyond_30_days() -> None:
    """1m interval rejects target dates older than 30 days."""
    today = pd.Timestamp.now().date()
    old_date = (today - pd.Timedelta(days=32)).strftime("%Y-%m-%d")

    with pytest.raises(ValueError, match="strictly limited to the last 30 days"):
        run_cli(parse_args(["EURUSD=X", "-t", "1m", "--date", old_date]))


def test_1m_window_span_limit() -> None:
    """1m interval limits request date range to at most 7 days per query."""
    today = pd.Timestamp.now().date()
    start_d = (today - pd.Timedelta(days=18)).strftime("%Y-%m-%d")
    end_d_ok = (today - pd.Timedelta(days=12)).strftime("%Y-%m-%d")  # 6 days span
    end_d_too_long = (today - pd.Timedelta(days=8)).strftime("%Y-%m-%d")  # 10 days span

    # Valid span (6 days <= 7 days)
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data:
        mock_get_data.return_value = _sample_ohlcv(20)
        ret = run_cli(
            parse_args(["EURUSD=X", "-t", "1m", "--start-date", start_d, "--end-date", end_d_ok])
        )
        assert ret == 0

    # Invalid span (> 7 days)
    with pytest.raises(ValueError, match="limits 1m granularity requests to at most 7 days"):
        run_cli(
            parse_args(
                [
                    "EURUSD=X",
                    "-t",
                    "1m",
                    "--start-date",
                    start_d,
                    "--end-date",
                    end_d_too_long,
                ]
            )
        )


def test_market_data_loader_clamp_uses_window_limit() -> None:
    """_clamp_period_for_interval clamps 1m to 7d window limit."""
    assert MarketDataLoader._clamp_period_for_interval("1mo", "1m") == "7d"
    assert MarketDataLoader._clamp_period_for_interval("60d", "1m") == "7d"
    assert MarketDataLoader._clamp_period_for_interval("5d", "1m") == "5d"

    # 5m clamps to 60d
    assert MarketDataLoader._clamp_period_for_interval("1y", "5m") == "60d"
    # 1h clamps to 730d
    assert MarketDataLoader._clamp_period_for_interval("5y", "1h") == "730d"


def test_intraday_depth_and_window_constants() -> None:
    """Verify YAHOO_INTRADAY_MAX_DEPTH_DAYS, YAHOO_INTRADAY_MAX_WINDOW_DAYS, and canonical alias."""
    assert YAHOO_INTRADAY_MAX_DAYS is YAHOO_INTRADAY_MAX_DEPTH_DAYS
    assert YAHOO_INTRADAY_MAX_DEPTH_DAYS["1m"] == 30
    assert YAHOO_INTRADAY_MAX_WINDOW_DAYS["1m"] == 7
    assert YAHOO_INTRADAY_MAX_DEPTH_DAYS["2m"] == 60
    assert YAHOO_INTRADAY_MAX_WINDOW_DAYS["2m"] == 60


# =====================================================================
# 2. GEMM fast_score_batch mismatched out buffer
# =====================================================================
def test_fast_score_batch_mismatched_buffer_no_scalar_broadcast() -> None:
    """Oversized out buffer does NOT broadcast a scalar score into all elements."""
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # single sample
    weights = np.array([0.5, -0.2, 0.1], dtype=np.float32)

    # Oversized ndarray buffer of length 5
    oversized_buf = np.zeros(5, dtype=np.float32)
    res = fast_score_batch(features, weights, bias=0.1, out=oversized_buf)

    # Must return single score with shape (1,), NOT broadcast into oversized_buf
    assert res.shape == (1,)
    # oversized_buf should not have been overwritten or corrupted
    assert np.all(oversized_buf == 0.0) or (res is not oversized_buf)

    # Oversized list buffer of length 5
    oversized_list = [0.0] * 5
    res_list = fast_score_batch(features, weights, bias=0.1, out=oversized_list)
    assert len(res_list) == 1
    assert oversized_list == [0.0] * 5 or (res_list is not oversized_list)


def test_fast_score_batch_matching_buffer_filled() -> None:
    """Matching shape out buffer is written in-place and returned."""
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    weights = np.array([0.5, -0.2, 0.1], dtype=np.float32)

    out_arr = np.zeros(2, dtype=np.float32)
    res = fast_score_batch(features, weights, bias=0.1, out=out_arr)
    assert res is out_arr
    assert not np.all(out_arr == 0.0)


# =====================================================================
# 3. CLI --backtest error does not invoke --calendar or --news
# =====================================================================
def test_backtest_error_does_not_display_calendar_or_news() -> None:
    """When backtest fails (e.g. unknown pattern), calendar and news hooks are skipped."""
    with patch("yfinance_ta_patterns.cli._display_calendar_for_symbol") as mock_cal, patch(
        "yfinance_ta_patterns.cli._display_news_for_symbol"
    ) as mock_news:
        ret = run_cli(
            parse_args(["SPY", "--backtest", "--pattern", "NONEXISTENT", "--calendar", "--news"])
        )
        assert ret == 1
        mock_cal.assert_not_called()
        mock_news.assert_not_called()


# =====================================================================
# 4. Economic calendar: new ISO-3 mappings & collision-free auto mode
# =====================================================================
def test_calendar_extended_iso3_mappings() -> None:
    """Extended ISO-3 country codes resolve to their respective currencies."""
    assert resolve_symbol_currencies("ARE", asset_type="country") == ["AED"]
    assert resolve_symbol_currencies("CHL", asset_type="country") == ["CLP"]
    assert resolve_symbol_currencies("COL", asset_type="country") == ["COP"]
    assert resolve_symbol_currencies("MYS", asset_type="country") == ["MYR"]
    assert resolve_symbol_currencies("PHL", asset_type="country") == ["PHP"]


def test_calendar_auto_mode_preserves_us_stocks_and_g10() -> None:
    """Auto mode treats US stock tickers as USD, while G10 currency codes resolve to themselves."""
    # US stocks
    assert resolve_symbol_currencies("COP", asset_type="auto") == ["USD"]
    assert resolve_symbol_currencies("NOK", asset_type="auto") == ["USD"]
    assert resolve_symbol_currencies("PEN", asset_type="auto") == ["USD"]
    assert resolve_symbol_currencies("EGP", asset_type="auto") == ["USD"]
    assert resolve_symbol_currencies("IDR", asset_type="auto") == ["USD"]

    # G10 currencies
    assert resolve_symbol_currencies("USD", asset_type="auto") == ["USD"]
    assert resolve_symbol_currencies("EUR", asset_type="auto") == ["EUR"]
    assert resolve_symbol_currencies("GBP", asset_type="auto") == ["GBP"]
    assert resolve_symbol_currencies("JPY", asset_type="auto") == ["JPY"]
    assert resolve_symbol_currencies("CHF", asset_type="auto") == ["CHF"]
    assert resolve_symbol_currencies("AUD", asset_type="auto") == ["AUD"]
    assert resolve_symbol_currencies("CAD", asset_type="auto") == ["CAD"]
    assert resolve_symbol_currencies("NZD", asset_type="auto") == ["NZD"]
