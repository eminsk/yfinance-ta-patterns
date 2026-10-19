"""Comprehensive regression and hardening tests for v21 / v0.3.60 audit fixes.

Covers all 12 points identified in audit:
1. `yftp --mcp` execution without 'unrecognized arguments: --mcp'.
2. Forex Friday weekly close (17:00 NY) candle retention over weekends.
3. `clean_forex_daily=True` download interval clamping to 730d.
4. CLI `--format json` output clean to stdout, banners routed to stderr, empty outputs as `[]`.
5. Backtester CLI flags `--execution`, `--commission`, `--slippage`, `--min-signals`.
6. Economic calendar equity tickers (COP, DE, IT, ES, FR, AU, BR, etc.) avoiding country confusion.
7. Precious metals (XAUUSD, XAGUSD, XAU/USD) normalized to `=X` and classified as forex.
8. Warmup index parity in `talib_compat.py` for `CDL3OUTSIDE` (starts at 3) and `CDLHIKKAKEMOD` (starts at 10).
9. CLI intermixed positional arguments, multi-arg `--symbol`, `--date` lookback scaling, and empty symbol guards.
10. MCP parameter bounds validation and JSON-RPC 2.0 batch array processing.
11. `MarketDataLoader.process()` callers' DataFrame columns immutability.
12. `TWO_WAY_PATTERNS` filtered to strictly bi-directional patterns.
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from yfinance_ta_patterns.cli import parse_args, resolve_symbols, run_cli
from yfinance_ta_patterns.data import (
    MarketDataLoader,
    classify_asset,
    normalize_ticker,
)
from yfinance_ta_patterns.economic_calendar import resolve_symbol_currencies
from yfinance_ta_patterns.mcp_server import YFinanceTAMCPServer
from yfinance_ta_patterns.pattern_tester import TWO_WAY_PATTERNS
from yfinance_ta_patterns.talib_compat import cdl_3outside, cdl_hikkakemod


# =====================================================================
# Point 1: yftp --mcp executes main_mcp([]) cleanly
# =====================================================================
def test_cli_mcp_flag_invokes_main_mcp_with_empty_args() -> None:
    with patch("yfinance_ta_patterns.mcp_server.main_mcp") as mock_mcp:
        mock_mcp.return_value = 0
        ret = run_cli(parse_args(["--mcp"]))
        assert ret == 0
        mock_mcp.assert_called_once_with([])


# =====================================================================
# Point 2: Forex Friday 17:00 NY close preserved over weekends
# =====================================================================
def test_forex_friday_close_candle_preserved_over_weekend() -> None:
    # Friday June 5, 2026: EDT (UTC-4), 17:00 NY = 21:00 UTC
    # Hourly candles for the full 4h bucket (17:00, 18:00, 19:00, 20:00 UTC)
    # The 20:00 UTC candle closes at 21:00 UTC (17:00 NY close)
    bar_times = pd.date_range("2026-06-05 17:00:00+00:00", "2026-06-05 20:00:00+00:00", freq="1h")
    df = pd.DataFrame(
        {
            "Open": [1.1000] * len(bar_times),
            "High": [1.1050] * len(bar_times),
            "Low": [1.0950] * len(bar_times),
            "Close": [1.1020] * len(bar_times),
            "Volume": [1000] * len(bar_times),
        },
        index=bar_times,
    )

    loader = MarketDataLoader("EURUSD=X", interval="4h", period="5d")
    # Simulate current time being Saturday June 6, 2026 12:00 UTC
    saturday_now = pd.Timestamp("2026-06-06 12:00:00", tz="UTC")
    cleaned = loader._resample_if_needed(df, now_utc=saturday_now)

    # Friday close candle must NOT be dropped over the weekend!
    assert not cleaned.empty
    assert len(cleaned) == 1


# =====================================================================
# Point 3: clean_forex_daily clamps period to 730d
# =====================================================================
def test_clean_forex_daily_clamps_period_to_730d() -> None:
    loader = MarketDataLoader("EURUSD=X", interval="1d", period="5y", clean_forex_daily=True)
    assert loader.period == "730d"

    # Also check regular 1h interval clamping
    loader_1h = MarketDataLoader("EURUSD=X", interval="1h", period="10y")
    assert loader_1h.period == "730d"


# =====================================================================
# Point 4: CLI --format json output purity & stderr banner routing
# =====================================================================
def test_cli_format_json_purity_and_stderr_banners(capsys: pytest.CaptureFixture) -> None:
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data:
        dates = pd.date_range("2026-01-01", periods=10, freq="1d", tz="UTC")
        mock_get_data.return_value = pd.DataFrame(
            {
                "Open": [10.0] * 10,
                "High": [12.0] * 10,
                "Low": [9.0] * 10,
                "Close": [11.0] * 10,
                "Volume": [500] * 10,
            },
            index=dates,
        )
        ret = run_cli(parse_args(["--pattern", "DOJI", "--symbol", "AAPL", "--format", "json"]))
        assert ret == 0
        captured = capsys.readouterr()
        # stdout must parse as valid JSON
        data = json.loads(captured.out)
        assert isinstance(data, list)


def test_cli_multi_symbol_format_json_empty_result(capsys: pytest.CaptureFixture) -> None:
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data:
        dates = pd.date_range("2026-01-01", periods=10, freq="1d", tz="UTC")
        mock_get_data.return_value = pd.DataFrame(
            {
                "Open": [10.0] * 10,
                "High": [12.0] * 10,
                "Low": [9.0] * 10,
                "Close": [11.0] * 10,
                "Volume": [500] * 10,
            },
            index=dates,
        )
        ret = run_cli(
            parse_args(
                [
                    "--symbol",
                    "AAPL,MSFT",
                    "--timeframe",
                    "1d",
                    "--all-patterns",
                    "--ai",
                    "--min-confidence",
                    "0.99",
                    "--format",
                    "json",
                ]
            )
        )
        assert ret == 0
        captured = capsys.readouterr()
        # stderr must have received progress/banners
        assert "SCANNING" in captured.err or "Scanned" in captured.err or len(captured.err) > 0
        # stdout must be clean JSON list
        data = json.loads(captured.out)
        assert isinstance(data, list)


# =====================================================================
# Point 5: Backtester CLI flags connected to PatternRankingTester
# =====================================================================
def test_cli_backtest_flags_connected() -> None:
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data") as mock_get_data, patch(
        "yfinance_ta_patterns.pattern_tester.PatternRankingTester"
    ) as mock_tester_cls:
        dates = pd.date_range("2026-01-01", periods=30, freq="1d", tz="UTC")
        mock_get_data.return_value = pd.DataFrame(
            {
                "Open": [10.0] * 30,
                "High": [12.0] * 30,
                "Low": [9.0] * 30,
                "Close": [11.0] * 30,
                "Volume": [500] * 30,
            },
            index=dates,
        )
        mock_inst = MagicMock()
        mock_inst.test_all_patterns.return_value = {}
        mock_inst.get_comparison_report.return_value = pd.DataFrame()
        mock_tester_cls.return_value = mock_inst

        ret = run_cli(
            parse_args(
                [
                    "--backtest",
                    "--symbol",
                    "EURUSD=X",
                    "--commission",
                    "2.5",
                    "--slippage",
                    "0.0001",
                    "--execution",
                    "next_open",
                    "--min-signals",
                    "8",
                ]
            )
        )
        assert ret == 0
        mock_tester_cls.assert_called_once()
        _, kwargs = mock_tester_cls.call_args
        assert kwargs["commission"] == 2.5
        assert kwargs["slippage"] == 0.0001
        assert kwargs["execution"] == "next_open"
        mock_inst.test_all_patterns.assert_called_once_with(min_signals=8)


# =====================================================================
# Point 6: Economic calendar avoids country confusion for equities
# =====================================================================
def test_economic_calendar_currency_resolution() -> None:
    # Stock tickers must resolve to USD and not trigger ISO country matches
    assert resolve_symbol_currencies("COP") == ["USD"]
    assert resolve_symbol_currencies("DE") == ["USD"]
    assert resolve_symbol_currencies("IT") == ["USD"]
    assert resolve_symbol_currencies("ES") == ["USD"]
    assert resolve_symbol_currencies("FR") == ["USD"]
    assert resolve_symbol_currencies("AU") == ["USD"]
    assert resolve_symbol_currencies("BR") == ["USD"]
    assert resolve_symbol_currencies("SE") == ["USD"]

    # Forex pairs resolve to base and quote
    assert resolve_symbol_currencies("EURUSD=X") == ["EUR", "USD"]
    assert resolve_symbol_currencies("GBPUSD=X") == ["GBP", "USD"]


# =====================================================================
# Point 7: Precious metals normalization and classification
# =====================================================================
def test_metals_normalization_and_classification() -> None:
    assert normalize_ticker("XAUUSD") == "XAUUSD=X"
    assert normalize_ticker("XAGUSD") == "XAGUSD=X"
    assert normalize_ticker("XAU/USD") == "XAUUSD=X"
    assert normalize_ticker("XAG/USD") == "XAGUSD=X"

    assert classify_asset("XAUUSD=X") == "forex"
    assert classify_asset("XAGUSD=X") == "forex"


# =====================================================================
# Point 8: talib_compat warmup index parity
# =====================================================================
def test_talib_compat_cdl3outside_warmup() -> None:
    # Bars 0, 1, 2 must strictly be 0 (requires 3 previous bars)
    n = 10
    open_arr = np.full(n, 100.0)
    high_arr = np.full(n, 105.0)
    low_arr = np.full(n, 95.0)
    close_arr = np.full(n, 102.0)
    res = cdl_3outside(open_arr, high_arr, low_arr, close_arr)
    assert res[0] == 0
    assert res[1] == 0
    assert res[2] == 0


def test_talib_compat_cdlhikkakemod_warmup() -> None:
    # Bars 0 through 9 must strictly be 0
    n = 15
    open_arr = np.full(n, 100.0)
    high_arr = np.full(n, 105.0)
    low_arr = np.full(n, 95.0)
    close_arr = np.full(n, 102.0)
    res = cdl_hikkakemod(open_arr, high_arr, low_arr, close_arr)
    for i in range(10):
        assert res[i] == 0


# =====================================================================
# Point 9: CLI positional tickers, date lookbacks, and comma guards
# =====================================================================
def test_cli_intermixed_positional_arguments() -> None:
    # Intermixed: SPY --timeframe 1d NVDA
    args = parse_args(["SPY", "--timeframe", "1d", "NVDA", "--all-patterns"])
    assert "SPY" in args.symbol
    assert "NVDA" in args.symbol

    # Multi-arg --symbol
    args2 = parse_args(["--symbol", "AAPL", "MSFT", "--all-patterns"])
    assert "AAPL" in args2.symbol
    assert "MSFT" in args2.symbol


def test_cli_date_lookback_scaling() -> None:
    with patch("yfinance_ta_patterns.cli.MarketDataLoader") as mock_loader:
        mock_instance = MagicMock()
        mock_instance.get_data.return_value = pd.DataFrame()
        mock_loader.return_value = mock_instance

        # 3mo timeframe should use ~10y lookback buffer
        with contextlib.suppress(Exception):
            run_cli(
                parse_args(
                    [
                        "--symbol",
                        "AAPL",
                        "--date",
                        "2026-01-15",
                        "--timeframe",
                        "3mo",
                        "--all-patterns",
                    ]
                )
            )
        assert mock_loader.call_args is not None
        _, kwargs_3mo = mock_loader.call_args
        start_dt_3mo = pd.to_datetime(kwargs_3mo["start"])
        assert start_dt_3mo.year <= 2016

        # 5d timeframe should use ~3y lookback buffer
        with contextlib.suppress(Exception):
            run_cli(
                parse_args(
                    [
                        "--symbol",
                        "AAPL",
                        "--date",
                        "2026-01-15",
                        "--timeframe",
                        "5d",
                        "--all-patterns",
                    ]
                )
            )
        assert mock_loader.call_args is not None
        _, kwargs_5d = mock_loader.call_args
        start_dt_5d = pd.to_datetime(kwargs_5d["start"])
        assert start_dt_5d.year <= 2023


def test_resolve_symbols_guards() -> None:
    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        resolve_symbols("")

    with pytest.raises(ValueError, match="No valid symbols specified"):
        resolve_symbols(",")


# =====================================================================
# Point 10: MCP parameter validation & JSON-RPC 2.0 batch support
# =====================================================================
def test_mcp_parameter_validation() -> None:
    server = YFinanceTAMCPServer()

    # Invalid timeframe must fail
    resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "ta_scan_symbol",
                "arguments": {"symbol": "AAPL", "timeframe": "invalid_tf"},
            },
        }
    )
    assert resp is not None
    assert resp.get("result", {}).get("isError") is True or "error" in resp

    # Invalid non-numeric min_confidence must fail
    resp2 = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "ta_scan_symbol",
                "arguments": {"symbol": "AAPL", "min_confidence": "invalid_number"},
            },
        }
    )
    assert resp2 is not None
    assert resp2.get("result", {}).get("isError") is True or "error" in resp2


def test_mcp_json_rpc_batch_requests() -> None:
    server = YFinanceTAMCPServer()
    batch = [
        {"jsonrpc": "2.0", "id": 10, "method": "ping"},
        {"jsonrpc": "2.0", "id": 20, "method": "tools/list"},
    ]
    resp = server.handle_request(batch)
    assert isinstance(resp, list)
    assert len(resp) == 2
    assert resp[0]["id"] == 10
    assert resp[0]["result"] == {}
    assert resp[1]["id"] == 20
    assert "tools" in resp[1]["result"]


# =====================================================================
# Point 11: MarketDataLoader.process callers' DataFrame immutability
# =====================================================================
def test_market_data_loader_process_immutability() -> None:
    dates = pd.date_range("2026-01-01", periods=10, freq="1d", tz="UTC")
    # MultiIndex columns as returned by yf.download(group_by="column")
    cols = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"), ("Open", "AAPL"), ("Volume", "AAPL")]
    )
    raw_df = pd.DataFrame(np.ones((10, 5)), index=dates, columns=cols)
    original_cols = list(raw_df.columns)

    loader = MarketDataLoader("AAPL", interval="1d", period="10d")
    processed = loader.process(raw_df)

    # Processed has single-level columns
    assert not isinstance(processed.columns, pd.MultiIndex)
    # Original raw_df columns must remain completely unchanged!
    assert list(raw_df.columns) == original_cols


# =====================================================================
# Point 12: TWO_WAY_PATTERNS filtered to strictly bi-directional
# =====================================================================
def test_two_way_patterns_excludes_unidirectional() -> None:
    strictly_one_way = [
        "CDL2CROWS",
        "CDL3BLACKCROWS",
        "CDL3WHITESOLDIERS",
        "CDLADVANCEBLOCK",
        "CDLDARKCLOUDCOVER",
        "CDLEVENINGDOJISTAR",
        "CDLEVENINGSTAR",
        "CDLHANGINGMAN",
        "CDLHAMMER",
        "CDLINVERTEDHAMMER",
        "CDLMORNINGDOJISTAR",
        "CDLMORNINGSTAR",
        "CDLPIERCING",
        "CDLSHOOTINGSTAR",
        "CDLUPSIDEGAP2CROWS",
        "CDLONNECK",
        "CDLINNECK",
    ]
    for pat in strictly_one_way:
        assert pat not in TWO_WAY_PATTERNS, f"{pat} should not be in TWO_WAY_PATTERNS"
