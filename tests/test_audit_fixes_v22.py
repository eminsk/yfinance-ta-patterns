"""Comprehensive regression and hardening tests for v22 / v0.3.66 audit fixes.

Covers all 6 audit points:
1. Multi-symbol and single-pattern backtesting in CLI (_run_backtest_cli) and elimination of 3x redundant runs.
2. Strict MCP parameter schema validation (clamp=False by default) preventing silent out-of-range masks.
3. Yahoo Finance intraday history limits and lookback buffer clamping for --date.
4. gemm.py memory layouts: out= buffer support, (K, 1) weights and empty batch [] handling in pure-Python.
5. Macroeconomic calendar currency resolution: precious metals (XAU/XAUUSD -> USD, XAUEUR -> EUR),
   asset_type='country' direct dispatch, and auto-mode ISO currency codes vs US equity ticker collisions.
6. Lookback warmup and short series documentation in talib_compat.py.
"""

from __future__ import annotations

import datetime as dt
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import yfinance_ta_patterns.talib_compat as talib_compat
from yfinance_ta_patterns.cli import parse_args, run_cli
from yfinance_ta_patterns.economic_calendar import resolve_symbol_currencies
from yfinance_ta_patterns.gemm import (
    fast_matmul,
    fast_score_batch,
)
from yfinance_ta_patterns.mcp_server import (
    YFinanceTAMCPServer,
    _parse_float_param,
    _parse_int_param,
)


def _sample_ohlcv(n: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="1d", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [10.0 + (i % 3) for i in range(n)],
            "High": [12.0 + (i % 3) for i in range(n)],
            "Low": [9.0 + (i % 3) for i in range(n)],
            "Close": [11.0 + (i % 3) for i in range(n)],
            "Volume": [1000 + i * 10 for i in range(n)],
        },
        index=dates,
    )


# =====================================================================
# Point 1: Multi-symbol backtesting and elimination of 3x redundant runs
# =====================================================================
def test_cli_multi_symbol_backtest_runs_all_symbols() -> None:
    """yftp EURUSD GBPUSD --backtest must run backtesting across both symbols."""
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data, patch(
        "yfinance_ta_patterns.pattern_tester.PatternRankingTester"
    ) as mock_tester_cls:
        mock_get_data.return_value = _sample_ohlcv(30)
        mock_inst = MagicMock()
        mock_inst.get_comparison_report.return_value = pd.DataFrame(
            [{"Pattern": "CDLHAMMER", "Win Rate % (No News)": 65.0}]
        )
        mock_tester_cls.return_value = mock_inst

        ret = run_cli(
            parse_args(["EURUSD=X", "GBPUSD=X", "--backtest", "--execution", "next_open"])
        )
        assert ret == 0
        assert mock_tester_cls.call_count == 2
        # get_comparison_report is called once per symbol, test_all_patterns is NOT called redundantly
        assert mock_inst.get_comparison_report.call_count == 2
        mock_inst.test_all_patterns.assert_not_called()


def test_cli_backtest_single_pattern_filter() -> None:
    """--pattern HAMMER --backtest passes targeted pattern to tester."""
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data, patch(
        "yfinance_ta_patterns.pattern_tester.PatternRankingTester"
    ) as mock_tester_cls:
        mock_get_data.return_value = _sample_ohlcv(30)
        mock_inst = MagicMock()
        mock_inst.get_comparison_report.return_value = pd.DataFrame()
        mock_tester_cls.return_value = mock_inst

        ret = run_cli(parse_args(["SPY", "--backtest", "--pattern", "HAMMER"]))
        assert ret == 0
        mock_inst.get_comparison_report.assert_called_once_with(patterns=["CDLHAMMER"])


def test_cli_multi_symbol_backtest_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    """Multi-symbol backtest in JSON mode executes test_all_patterns and outputs dict of lists."""
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data, patch(
        "yfinance_ta_patterns.pattern_tester.PatternRankingTester"
    ) as mock_tester_cls:
        mock_get_data.return_value = _sample_ohlcv(30)
        mock_inst = MagicMock()
        mock_res = MagicMock()
        mock_res.pattern_name = "CDLENGULFING"
        mock_res.total_signals = 10
        mock_res.total_trades = 8
        mock_res.winning_trades = 6
        mock_res.losing_trades = 2
        mock_res.win_rate = 0.75
        mock_res.total_pnl = 120.0
        mock_res.avg_pnl = 15.0
        mock_res.sharpe_ratio = 1.8
        mock_res.max_drawdown = 5.0
        mock_inst.get_top_patterns.return_value = [mock_res]
        mock_tester_cls.return_value = mock_inst

        ret = run_cli(
            parse_args(["AAPL", "MSFT", "--backtest", "--format", "json", "--min-signals", "5"])
        )
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict)
        assert "AAPL" in data
        assert "MSFT" in data
        assert len(data["AAPL"]) == 1
        assert data["AAPL"][0]["pattern"] == "CDLENGULFING"
        # In JSON mode, test_all_patterns is called once per symbol, comparison report is not called
        assert mock_inst.test_all_patterns.call_count == 2
        mock_inst.get_comparison_report.assert_not_called()


# =====================================================================
# Point 2: Strict MCP schema validation (clamp=False by default)
# =====================================================================
def test_mcp_strict_validation_rejects_out_of_bounds() -> None:
    """MCP helper functions reject out-of-range values with clean ValueError."""
    # float out-of-range
    with pytest.raises(ValueError, match=r"Parameter 'min_confidence' must be <= 1\.0, got 70\.0"):
        _parse_float_param(70.0, "min_confidence", 0.0, 1.0)

    with pytest.raises(ValueError, match=r"Parameter 'min_confidence' must be >= 0\.0, got -0\.1"):
        _parse_float_param(-0.1, "min_confidence", 0.0, 1.0)

    # int out-of-range
    with pytest.raises(ValueError, match=r"Parameter 'lookback_bars' must be <= 60, got 100000"):
        _parse_int_param(100000, "lookback_bars", 1, 60)

    with pytest.raises(ValueError, match=r"Parameter 'days' must be >= 1, got -5"):
        _parse_int_param(-5, "days", 1, 90)


def test_mcp_server_returns_error_on_schema_violation() -> None:
    """YFinanceTAMCPServer returns isError=True on out-of-bounds parameters."""
    server = YFinanceTAMCPServer(data_fetcher=lambda s, p, i: _sample_ohlcv(20))

    resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "ta_scan_symbol",
                "arguments": {"symbol": "AAPL", "min_confidence": 70.0},
            },
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is True
    assert "Parameter 'min_confidence' must be <= 1.0" in resp["result"]["content"][0]["text"]

    resp2 = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "ta_get_economic_calendar",
                "arguments": {"symbol": "USD", "days": -5},
            },
        }
    )
    assert resp2 is not None
    assert resp2["result"]["isError"] is True
    assert "Parameter 'days' must be >= 1" in resp2["result"]["content"][0]["text"]


# =====================================================================
# Point 3: Yahoo Finance intraday limits and --date clamping
# =====================================================================
def test_cli_date_intraday_limit_rejection() -> None:
    """Intraday target date exceeding Yahoo limit raises explicit ValueError."""
    old_date = (dt.date.today() - dt.timedelta(days=70)).strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="Yahoo Finance intraday history for interval '15m'"):
        run_cli(parse_args(["SPY", "-t", "15m", "--date", old_date]))

    old_date_1m = (dt.date.today() - dt.timedelta(days=35)).strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="Yahoo Finance intraday history for interval '1m'"):
        run_cli(parse_args(["SPY", "-t", "1m", "--date", old_date_1m]))


def test_cli_date_intraday_lookback_clamping() -> None:
    """Recent intraday date within Yahoo window clamps start_date to earliest allowed bar."""
    recent_date = (dt.date.today() - dt.timedelta(days=20)).strftime("%Y-%m-%d")
    with patch("yfinance_ta_patterns.cli.MarketDataLoader") as mock_loader:
        mock_loader.return_value.get_data.return_value = _sample_ohlcv(10)
        mock_loader.return_value.period = "60d"

        run_cli(parse_args(["SPY", "-t", "15m", "--date", recent_date]))
        mock_loader.assert_called_once()
        _, kwargs = mock_loader.call_args
        earliest_allowed = dt.date.today() - dt.timedelta(days=59)
        start_used = dt.datetime.strptime(kwargs["start"], "%Y-%m-%d").date()
        assert start_used >= earliest_allowed


# =====================================================================
# Point 4: gemm.py memory layouts, out= buffers, and fallback robust handling
# =====================================================================
def test_fast_matmul_with_out_buffer() -> None:
    """fast_matmul writes directly into caller-supplied out buffer."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    out_buf = np.zeros((2, 2), dtype=np.float64)

    res = fast_matmul(a, b, out=out_buf)
    assert res is out_buf
    np.testing.assert_allclose(out_buf, np.matmul(a, b))


def test_fast_score_batch_with_out_buffer() -> None:
    """fast_score_batch writes directly into caller-supplied out buffer."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    w = np.array([0.5, -0.5], dtype=np.float32)
    out_buf = np.zeros(2, dtype=np.float64)

    res = fast_score_batch(x, w, bias=0.1, activation="linear", out=out_buf)
    assert res is out_buf
    expected = np.matmul(x, w) + 0.1
    np.testing.assert_allclose(out_buf, expected, rtol=1e-5)


def test_fast_score_batch_pure_python_column_weights_and_empty() -> None:
    """Pure-Python fallback handles (K, 1) weights and empty batch without crashing."""
    with patch("yfinance_ta_patterns.gemm.HAS_NUMPY", False), patch(
        "yfinance_ta_patterns.gemm.HAS_NANOGEMM", False
    ):
        # (K, 1) column vector weights
        features = [[1.0, 2.0], [3.0, 4.0]]
        weights_col = [[0.5], [-0.5]]
        scores = fast_score_batch(features, weights_col, bias=0.0, activation="linear")
        assert len(scores) == 2
        assert abs(scores[0] - (-0.5)) < 1e-6
        assert abs(scores[1] - (-0.5)) < 1e-6

        # With out buffer in pure python
        out_buf = [0.0, 0.0]
        res = fast_score_batch(features, weights_col, bias=0.0, activation="linear", out=out_buf)
        assert res is out_buf
        assert abs(out_buf[0] - (-0.5)) < 1e-6
        assert abs(out_buf[1] - (-0.5)) < 1e-6

        # Empty features batch
        assert fast_score_batch([], weights_col) == []


# =====================================================================
# Point 5: Economic calendar currency resolution
# =====================================================================
def test_calendar_asset_type_country() -> None:
    """asset_type='country' resolves directly without throwing unsupported error."""
    currencies = resolve_symbol_currencies("germany", asset_type="country")
    assert currencies == ["EUR"]

    currencies_us = resolve_symbol_currencies("united states", asset_type="country")
    assert currencies_us == ["USD"]


def test_calendar_precious_metals_currency_resolution() -> None:
    """Precious metals resolve to quote fiat currency (USD or EUR), not metal code."""
    assert resolve_symbol_currencies("XAU", "forex") == ["USD"]
    assert resolve_symbol_currencies("XAUUSD", "forex") == ["USD"]
    assert resolve_symbol_currencies("XAUUSD=X", "forex") == ["USD"]
    assert resolve_symbol_currencies("XAUEUR", "forex") == ["EUR"]
    assert resolve_symbol_currencies("XAGUSD", "forex") == ["USD"]
    assert resolve_symbol_currencies("XPTUSD", "forex") == ["USD"]
    assert resolve_symbol_currencies("XPDUSD", "forex") == ["USD"]


def test_calendar_auto_mode_iso_currencies_and_stock_collision_defense() -> None:
    """Auto-mode recognizes valid G10 currencies but protects US equity tickers."""
    # Valid G10 currencies in auto mode
    assert resolve_symbol_currencies("EUR", "auto") == ["EUR"]
    assert resolve_symbol_currencies("GBP", "auto") == ["GBP"]
    assert resolve_symbol_currencies("JPY", "auto") == ["JPY"]
    assert resolve_symbol_currencies("CHF", "auto") == ["CHF"]
    assert resolve_symbol_currencies("AUD", "auto") == ["AUD"]
    assert resolve_symbol_currencies("CAD", "auto") == ["CAD"]
    assert resolve_symbol_currencies("NZD", "auto") == ["NZD"]

    # Equity collisions protected in auto mode (US equities -> USD)
    assert resolve_symbol_currencies("NOK", "auto") == ["USD"]  # Nokia
    assert resolve_symbol_currencies("PEN", "auto") == ["USD"]  # Penumbra
    assert resolve_symbol_currencies("EGP", "auto") == ["USD"]  # EastGroup
    assert resolve_symbol_currencies("IDR", "auto") == ["USD"]  # Idaho Strategic
    assert resolve_symbol_currencies("COP", "auto") == ["USD"]  # ConocoPhillips
    assert resolve_symbol_currencies("DE", "auto") == ["USD"]  # Deere & Co
    assert resolve_symbol_currencies("IT", "auto") == ["USD"]  # Gartner
    assert resolve_symbol_currencies("ES", "auto") == ["USD"]  # Eversource
    assert resolve_symbol_currencies("FR", "auto") == ["USD"]  # First Republic

    # Explicit forex mode or =X suffix resolves to currency codes
    assert resolve_symbol_currencies("NOK", "forex") == ["NOK"]
    assert resolve_symbol_currencies("PEN", "forex") == ["PEN"]
    assert resolve_symbol_currencies("EGP", "forex") == ["EGP"]
    assert resolve_symbol_currencies("IDR", "forex") == ["IDR"]
    assert resolve_symbol_currencies("MXN", "forex") == ["MXN"]
    assert resolve_symbol_currencies("NOK=X", "auto") == ["NOK", "USD"]


# =====================================================================
# Point 6: talib_compat lookback warmup documentation
# =====================================================================
def test_talib_compat_warmup_documentation_integrity() -> None:
    """talib_compat docstring explicitly documents lookback warmup behavior."""
    doc = talib_compat.__doc__
    assert doc is not None
    assert "Lookback" in doc or "lookback" in doc
    assert "warmup" in doc or "Warmup" in doc
