"""Comprehensive unit and regression tests for audit fixes v23 (v0.3.67).

Covering:
1. --backtest CLI invocation of --calendar and --news hooks (and exclusion in JSON mode).
2. --backtest pattern case-insensitivity, validation against get_all_patterns(), exit code 1 on unknown pattern, failure isolation, and dynamic date range header.
3. Intraday rolling limits: unified YAHOO_INTRADAY_MAX_DAYS, >= boundary check, 2m/90m support, and --start-date/--end-date validation.
4. Economic calendar: XAU-EUR delimiter stripping -> ['EUR'], asset_type="country" ValueError for invalid symbols and ISO-3 DEU -> ['EUR'], auto-mode G10 restriction with US equity collision protection.
5. GEMM: fast_matmul memory aliasing protection (np.shares_memory) and fast_score_batch out=list support on NumPy branch.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from yfinance_ta_patterns.cli import parse_args, run_cli
from yfinance_ta_patterns.data import (
    YAHOO_INTRADAY_MAX_DAYS,
    YAHOO_INTRADAY_MAX_DEPTH_DAYS,
    YAHOO_INTRADAY_MAX_WINDOW_DAYS,
)
from yfinance_ta_patterns.economic_calendar import resolve_symbol_currencies
from yfinance_ta_patterns.gemm import fast_matmul, fast_score_batch


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
# Point 1: --backtest CLI hooks for --calendar and --news
# =====================================================================
def test_backtest_cli_invokes_calendar_and_news_in_text_mode() -> None:
    """yftp EURUSD --backtest --calendar --news must trigger calendar and news display hooks."""
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data, patch(
        "yfinance_ta_patterns.pattern_tester.PatternRankingTester"
    ) as mock_tester_cls, patch(
        "yfinance_ta_patterns.cli._display_calendar_for_symbol"
    ) as mock_cal, patch("yfinance_ta_patterns.cli._display_news_for_symbol") as mock_news:
        mock_get_data.return_value = _sample_ohlcv(30)
        mock_inst = MagicMock()
        mock_inst.get_comparison_report.return_value = pd.DataFrame(
            [{"Pattern": "CDLENGULFING", "Win Rate %": 70.0}]
        )
        mock_tester_cls.return_value = mock_inst

        ret = run_cli(parse_args(["EURUSD=X", "--backtest", "--calendar", "--news"]))
        assert ret == 0
        mock_cal.assert_called_once_with("EURUSD=X", "text")
        mock_news.assert_called_once_with("EURUSD=X", "text")


def test_backtest_cli_suppresses_calendar_and_news_in_json_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In --format json mode, calendar and news hooks are skipped to preserve valid JSON."""
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data, patch(
        "yfinance_ta_patterns.pattern_tester.PatternRankingTester"
    ) as mock_tester_cls, patch(
        "yfinance_ta_patterns.cli._display_calendar_for_symbol"
    ) as mock_cal, patch("yfinance_ta_patterns.cli._display_news_for_symbol") as mock_news:
        mock_get_data.return_value = _sample_ohlcv(30)
        mock_inst = MagicMock()
        mock_res = MagicMock()
        mock_res.pattern_name = "CDLENGULFING"
        mock_res.total_signals = 5
        mock_res.total_trades = 4
        mock_res.winning_trades = 3
        mock_res.losing_trades = 1
        mock_res.win_rate = 0.75
        mock_res.total_pnl = 50.0
        mock_res.avg_pnl = 12.5
        mock_res.sharpe_ratio = 1.5
        mock_res.max_drawdown = 3.0
        mock_inst.get_top_patterns.return_value = [mock_res]
        mock_tester_cls.return_value = mock_inst

        ret = run_cli(
            parse_args(["EURUSD=X", "--backtest", "--calendar", "--news", "--format", "json"])
        )
        assert ret == 0
        mock_cal.assert_not_called()
        mock_news.assert_not_called()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert data[0]["pattern"] == "CDLENGULFING"


# =====================================================================
# Point 2: Backtest pattern validation, normalization, and isolation
# =====================================================================
def test_backtest_pattern_lowercase_normalization() -> None:
    """--pattern cdlengulfing is normalized to CDLENGULFING."""
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data, patch(
        "yfinance_ta_patterns.pattern_tester.PatternRankingTester"
    ) as mock_tester_cls:
        mock_get_data.return_value = _sample_ohlcv(30)
        mock_inst = MagicMock()
        mock_inst.get_comparison_report.return_value = pd.DataFrame()
        mock_tester_cls.return_value = mock_inst

        ret = run_cli(parse_args(["SPY", "--backtest", "--pattern", "cdlengulfing"]))
        assert ret == 0
        mock_inst.get_comparison_report.assert_called_once_with(patterns=["CDLENGULFING"])


def test_backtest_pattern_unknown_returns_error_code_1(capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid pattern name like FOO prints error and exits with code 1."""
    ret = run_cli(parse_args(["SPY", "--backtest", "--pattern", "FOO"]))
    assert ret == 1
    err = capsys.readouterr().err
    assert "Error: Unknown pattern 'FOO'" in err
    assert "Available:" in err


def test_backtest_multi_symbol_failure_isolation_and_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In multi-symbol backtest, one failing symbol does not crash the other, and exit code reflects failure."""
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data, patch(
        "yfinance_ta_patterns.pattern_tester.PatternRankingTester"
    ) as mock_tester_cls:
        # First symbol succeeds, second symbol returns empty
        mock_get_data.side_effect = [_sample_ohlcv(30), pd.DataFrame()]
        mock_inst = MagicMock()
        mock_res = MagicMock()
        mock_res.pattern_name = "CDLHAMMER"
        mock_res.total_signals = 2
        mock_res.total_trades = 2
        mock_res.winning_trades = 2
        mock_res.losing_trades = 0
        mock_res.win_rate = 1.0
        mock_res.total_pnl = 20.0
        mock_res.avg_pnl = 10.0
        mock_res.sharpe_ratio = 2.0
        mock_res.max_drawdown = 0.0
        mock_inst.get_top_patterns.return_value = [mock_res]
        mock_tester_cls.return_value = mock_inst

        ret = run_cli(parse_args(["AAPL", "BADSYM", "--backtest", "--format", "json"]))
        assert ret == 1  # Failure occurred on BADSYM
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "AAPL" in data
        assert isinstance(data["AAPL"], list)
        assert "BADSYM" in data
        assert "error" in data["BADSYM"]


def test_backtest_header_displays_actual_date_range(capsys: pytest.CaptureFixture[str]) -> None:
    """Backtest report header displays actual date window rather than hardcoded period."""
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data, patch(
        "yfinance_ta_patterns.pattern_tester.PatternRankingTester"
    ) as mock_tester_cls:
        mock_get_data.return_value = _sample_ohlcv(30)
        mock_inst = MagicMock()
        mock_inst.get_comparison_report.return_value = pd.DataFrame(
            [{"Pattern": "CDLHAMMER", "Trades": 5}]
        )
        mock_tester_cls.return_value = mock_inst

        ret = run_cli(
            parse_args(
                [
                    "SPY",
                    "-t",
                    "1d",
                    "--backtest",
                    "--start-date",
                    "2026-08-01",
                    "--end-date",
                    "2026-09-01",
                ]
            )
        )
        assert ret == 0
        captured = capsys.readouterr().out
        assert "2026-08-01 to 2026-09-01" in captured


# =====================================================================
# Point 3: Intraday rolling limits and date validation
# =====================================================================
def test_intraday_max_days_canonical_dict() -> None:
    """YAHOO_INTRADAY_MAX_DAYS and MAX_DEPTH/WINDOW_DAYS include 1m, 2m, 5m, 15m, 30m, 90m, 60m, 1h, 4h."""
    assert YAHOO_INTRADAY_MAX_DEPTH_DAYS["1m"] == 30
    assert YAHOO_INTRADAY_MAX_WINDOW_DAYS["1m"] == 7
    assert YAHOO_INTRADAY_MAX_DAYS["1m"] == 30
    assert YAHOO_INTRADAY_MAX_DAYS["2m"] == 60
    assert YAHOO_INTRADAY_MAX_DAYS["5m"] == 60
    assert YAHOO_INTRADAY_MAX_DAYS["15m"] == 60
    assert YAHOO_INTRADAY_MAX_DAYS["30m"] == 60
    assert YAHOO_INTRADAY_MAX_DAYS["90m"] == 60
    assert YAHOO_INTRADAY_MAX_DAYS["60m"] == 730
    assert YAHOO_INTRADAY_MAX_DAYS["1h"] == 730
    assert YAHOO_INTRADAY_MAX_DAYS["4h"] == 730


def test_intraday_boundary_rejection_at_exact_limit() -> None:
    """Target date exactly at or past rolling window is rejected with informative ValueError."""
    today = pd.Timestamp.now().date()
    boundary_60d = (today - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    with pytest.raises(ValueError, match="strictly limited to the last 60 days"):
        run_cli(parse_args(["EURUSD=X", "-t", "15m", "--date", boundary_60d]))


def test_intraday_start_date_exceeding_limit_raises_error() -> None:
    """--start-date exceeding intraday limit is rejected proactively."""
    today = pd.Timestamp.now().date()
    old_start = (today - pd.Timedelta(days=120)).strftime("%Y-%m-%d")

    with pytest.raises(ValueError, match="strictly limited to the last 60 days"):
        run_cli(parse_args(["EURUSD=X", "-t", "15m", "--start-date", old_start]))


# =====================================================================
# Point 4: Economic Calendar currency resolution & country mode
# =====================================================================
def test_calendar_precious_metal_tail_stripping() -> None:
    """Precious metal XAU-EUR with hyphen or slash delimiter resolves to quote fiat EUR."""
    assert resolve_symbol_currencies("XAU-EUR") == ["EUR"]
    assert resolve_symbol_currencies("XAU/EUR") == ["EUR"]
    assert resolve_symbol_currencies("XAU_EUR") == ["EUR"]
    assert resolve_symbol_currencies("XAUEUR") == ["EUR"]
    assert resolve_symbol_currencies("XAU-USD") == ["USD"]
    assert resolve_symbol_currencies("XAUUSD") == ["USD"]


def test_calendar_country_mode_strict_validation() -> None:
    """asset_type='country' raises ValueError for invalid country symbols instead of defaulting to USD."""
    with pytest.raises(ValueError, match=r"Invalid country identifier 'SAP\.DE'"):
        resolve_symbol_currencies("SAP.DE", asset_type="country")

    with pytest.raises(ValueError, match=r"Invalid country identifier 'EURUSD=X'"):
        resolve_symbol_currencies("EURUSD=X", asset_type="country")

    # Valid ISO-3 country codes resolve correctly
    assert resolve_symbol_currencies("DEU", asset_type="country") == ["EUR"]
    assert resolve_symbol_currencies("USA", asset_type="country") == ["USD"]
    assert resolve_symbol_currencies("GBR", asset_type="country") == ["GBP"]
    assert resolve_symbol_currencies("JPN", asset_type="country") == ["JPY"]


def test_calendar_auto_mode_us_ticker_collisions() -> None:
    """US equities NOK, PEN, EGP, IDR resolve to USD in auto mode; forex mode preserves currency codes."""
    assert resolve_symbol_currencies("NOK", "auto") == ["USD"]
    assert resolve_symbol_currencies("PEN", "auto") == ["USD"]
    assert resolve_symbol_currencies("EGP", "auto") == ["USD"]
    assert resolve_symbol_currencies("IDR", "auto") == ["USD"]

    assert resolve_symbol_currencies("NOK", "forex") == ["NOK"]
    assert resolve_symbol_currencies("PEN", "forex") == ["PEN"]
    assert resolve_symbol_currencies("EGP", "forex") == ["EGP"]
    assert resolve_symbol_currencies("IDR", "forex") == ["IDR"]


# =====================================================================
# Point 5: GEMM memory aliasing and buffer handling
# =====================================================================
def test_fast_matmul_memory_aliasing_protection() -> None:
    """In-place out=a or out=b in fast_matmul computes correct result without operand corruption."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    expected = a @ b

    out_buf = np.copy(a)
    result = fast_matmul(out_buf, b, out=out_buf)

    assert result is out_buf
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_fast_score_batch_out_list_support() -> None:
    """fast_score_batch supports out=list across NumPy and pure Python branches."""
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    weights = np.array([0.5, -0.2, 0.1], dtype=np.float32)
    out_list = [0.0, 0.0]

    res = fast_score_batch(features, weights, bias=0.1, activation="sigmoid", out=out_list)
    assert res is out_list
    assert len(out_list) == 2
    assert all(isinstance(v, float) for v in out_list)
    assert 0.0 < out_list[0] < 1.0
    assert 0.0 < out_list[1] < 1.0
