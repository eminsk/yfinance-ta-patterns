"""Comprehensive tests for yfinance-ta-patterns Model Context Protocol (MCP) JSON-RPC 2.0 Server."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from yfinance_ta_patterns.mcp_server import MAX_WATCHLIST_SYMBOLS, YFinanceTAMCPServer


def _make_sample_ohlcv(seed: int = 42, bullish_spike: float = 2.5) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=60, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0.1, 1.5, size=60))
    opens = closes - rng.normal(0.0, 0.8, size=60)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0.5, 0.5, size=60))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0.5, 0.5, size=60))
    # Force a clear bullish hammer / engulfing on the final bar
    opens[-1] = closes[-2] - 1.5
    closes[-1] = closes[-2] + bullish_spike
    highs[-1] = closes[-1] + 0.3
    lows[-1] = opens[-1] - 3.0
    vols = rng.integers(100_000, 500_000, size=60).astype(float)
    vols[-1] = 950_000.0
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": vols,
        },
        index=dates,
    )


def test_yfinance_ta_mcp_server_lifecycle(capsys: pytest.CaptureFixture[str]) -> None:
    df = _make_sample_ohlcv()
    server = YFinanceTAMCPServer(data_fetcher=lambda sym, per, tf: df.copy())

    # 1. initialize (includes instructions and negotiated version)
    init_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        }
    )
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "yfinance-ta-patterns-mcp"
    assert init_resp["result"]["protocolVersion"] == "2025-03-26"
    assert "instructions" in init_resp["result"]
    assert "yfinance-ta-patterns" in init_resp["result"]["instructions"]

    # 2. notifications return None (including non-prefixed notifications)
    assert server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle_request({"jsonrpc": "2.0", "method": "$/cancelRequest"}) is None

    # 3. tools/list
    list_resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None
    tools = [t["name"] for t in list_resp["result"]["tools"]]
    assert "ta_scan_symbol" in tools
    assert "ta_scan_watchlist" in tools
    assert "ta_backtest_patterns" in tools
    assert "ta_list_patterns" in tools

    # 4. ta_scan_symbol (compact by default: no markdown_brief unless requested)
    scan_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ta_scan_symbol",
                "arguments": {
                    "symbol": "BTC-USD",
                    "timeframe": "15m",
                    "period": "6mo",
                    "min_confidence": 0.0,
                    "lookback_bars": 5,
                },
            },
        }
    )
    assert scan_resp is not None
    assert scan_resp["result"]["isError"] is False
    report = json.loads(scan_resp["result"]["content"][0]["text"])
    assert report["symbol"] == "BTC-USD"
    assert report["period"] == "60d"  # 15m period is auto-clamped from 6mo to 60d
    assert report["timezone"] == "UTC"
    assert "market_summary" in report
    assert "markdown_brief" not in report

    # With include_brief=True
    scan_brief_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/call",
            "params": {
                "name": "ta_scan_symbol",
                "arguments": {
                    "symbol": "BTC-USD",
                    "timeframe": "15m",
                    "min_confidence": 0.0,
                    "lookback_bars": 5,
                    "include_brief": True,
                },
            },
        }
    )
    assert scan_brief_resp is not None
    report_with_brief = json.loads(scan_brief_resp["result"]["content"][0]["text"])
    assert "markdown_brief" in report_with_brief
    assert "UTC" in report_with_brief["markdown_brief"]

    # 5. ta_backtest_patterns (must NOT print "No signals generated" to stdout!)
    bt_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "ta_backtest_patterns",
                "arguments": {"symbol": "BTC-USD", "top_n": 3},
            },
        }
    )
    assert bt_resp is not None
    assert bt_resp["result"]["isError"] is False
    bt_data = json.loads(bt_resp["result"]["content"][0]["text"])
    assert bt_data["symbol"] == "BTC-USD"
    assert "top_patterns" in bt_data

    # 6. ta_list_patterns
    pat_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "ta_list_patterns", "arguments": {}},
        }
    )
    assert pat_resp is not None
    pat_data = json.loads(pat_resp["result"]["content"][0]["text"])
    assert len(pat_data["fallback_patterns"]) > 0

    # Verify zero stray stdout output across the entire lifecycle
    captured = capsys.readouterr()
    assert captured.out == ""


def test_mcp_watchlist_normalization_sorting_and_error_handling() -> None:
    df_low = _make_sample_ohlcv(seed=10, bullish_spike=1.2)
    df_high = _make_sample_ohlcv(seed=42, bullish_spike=4.0)

    def fetcher(sym: str, per: str, tf: str) -> pd.DataFrame:
        if sym == "FAIL":
            raise ConnectionError("Yahoo Finance timeout for FAIL → connection reset")
        if sym == "NVDA":
            return df_high.copy()
        return df_low.copy()

    server = YFinanceTAMCPServer(data_fetcher=fetcher)

    # 1. Comma-separated string is normalized into discrete symbols (not iterated char-by-char)
    resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "ta_scan_watchlist",
                "arguments": {
                    "symbols": "AAPL, NVDA, FAIL",
                    "timeframe": "1d",
                    "min_confidence": 0.0,
                    "max_results": 5,
                },
            },
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is False
    data = json.loads(resp["result"]["content"][0]["text"])
    assert data["symbols_requested"] == 3
    assert data["symbols_scanned"] == 2
    assert "FAIL" in data["errors"]
    assert len(data["opportunities"]) <= 5

    # Verify descending sort by confluence_score
    scores = [op["confluence_score"] for op in data["opportunities"]]
    assert scores == sorted(scores, reverse=True)

    # 2. Total outage (all symbols fail) returns isError=True instead of pretending 0 opportunities
    all_fail_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "ta_scan_watchlist",
                "arguments": {"symbols": ["FAIL"]},
            },
        }
    )
    assert all_fail_resp is not None
    assert all_fail_resp["result"]["isError"] is True
    assert (
        "Failed to scan all 1 requested symbol(s)" in all_fail_resp["result"]["content"][0]["text"]
    )

    # 3. Exceeding MAX_WATCHLIST_SYMBOLS returns isError=True
    too_many = [f"SYM{i}" for i in range(MAX_WATCHLIST_SYMBOLS + 1)]
    limit_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "ta_scan_watchlist",
                "arguments": {"symbols": too_many},
            },
        }
    )
    assert limit_resp is not None
    assert limit_resp["result"]["isError"] is True
    assert "exceeds maximum allowed limit" in limit_resp["result"]["content"][0]["text"]


def test_mcp_run_stdio_purity_garbage_resilience_and_cp1251() -> None:
    df = _make_sample_ohlcv()

    def noisy_fetcher(sym: str, per: str, tf: str) -> pd.DataFrame:
        # Simulate a rogue dependency printing non-ASCII text to sys.stdout
        print(f"1 Failed download: ['{sym}'] → noisy stdout spam")
        return df.copy()

    server = YFinanceTAMCPServer(data_fetcher=noisy_fetcher)

    requests = [
        "not valid json at all {{{",
        "12345",
        '["an", "array", "not", "object"]',
        '{"jsonrpc": "2.0", "id": 1, "method": 999}',
        '{"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": "bad_params"}',
        '{"jsonrpc": "2.0", "method": "notifications/initialized"}',
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "ta_backtest_patterns",
                    "arguments": {"symbol": "BTC-USD", "top_n": 2},
                },
            }
        ),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/read",
                "params": {"uri": "ta://forex/56-pairs"},
            }
        ),
    ]

    stdin_buf = io.StringIO("\n".join(requests) + "\n")
    # Simulate a strict Windows cp1251 stdout pipe that crashes if raw '→' is written unescaped
    raw_bytes_out = io.BytesIO()
    cp1251_stdout = io.TextIOWrapper(raw_bytes_out, encoding="cp1251", errors="strict")

    server.run_stdio(stdin=stdin_buf, stdout=cp1251_stdout)
    cp1251_stdout.flush()

    output_lines = raw_bytes_out.getvalue().decode("cp1251").splitlines()
    # Every single output line must be a valid JSON-RPC 2.0 object (zero stray print lines)
    parsed_msgs: list[dict[str, Any]] = [json.loads(line) for line in output_lines]

    assert len(parsed_msgs) == 7
    assert parsed_msgs[0]["error"]["code"] == -32700  # Parse error
    assert parsed_msgs[1]["error"]["code"] == -32600  # Invalid Request (int)
    assert parsed_msgs[2]["error"]["code"] == -32600  # Invalid Request (list)
    assert parsed_msgs[3]["error"]["code"] == -32600  # Invalid Request (bad method)
    assert parsed_msgs[4]["error"]["code"] == -32602  # Invalid params
    assert parsed_msgs[5]["id"] == 3
    assert parsed_msgs[5]["result"]["isError"] is False
    assert parsed_msgs[6]["id"] == 4
    fx_res = json.loads(parsed_msgs[6]["result"]["contents"][0]["text"])
    assert fx_res["count"] == 56


def test_mcp_economic_calendar_for_selected_asset() -> None:
    """Verify ta_get_economic_calendar resolves asset currencies, parses live HTML feed, and falls back."""
    from yfinance_ta_patterns.economic_calendar import (
        InvestingCalendar,
        resolve_symbol_currencies,
    )

    assert resolve_symbol_currencies("EURUSD=X") == ["EUR", "USD"]
    assert resolve_symbol_currencies("EUR/USD") == ["EUR", "USD"]
    assert resolve_symbol_currencies("GBPJPY") == ["GBP", "JPY"]
    assert resolve_symbol_currencies("AAPL") == ["USD"]
    assert resolve_symbol_currencies("SAP.DE") == ["EUR"]
    assert resolve_symbol_currencies("VOD.L") == ["GBP"]
    assert resolve_symbol_currencies("7203.T") == ["JPY"]
    assert resolve_symbol_currencies("BTC-USD") == ["USD"]
    assert resolve_symbol_currencies("BTC-EUR") == ["EUR"]
    assert resolve_symbol_currencies("^GDAXI") == ["EUR"]

    sample_te_html = """
    <table>
      <tr data-id="101" data-country="united states" data-event="non farm payrolls">
        <td class=" 2026-10-06"><span class="event-0 calendar-date-3">08:30 AM</span></td>
        <td class="calendar-item">
          <table><tr><td><div class="flag flag-us"></div></td><td class="calendar-iso">US</td></tr></table>
        </td>
        <td><a class="calendar-event">Non Farm Payrolls</a></td>
        <td><span id="actual">254K</span></td>
        <td><span id="previous">159K</span><span id="revised">\u00ae</span></td>
        <td><a id="consensus"></a></td>
        <td><a id="forecast">140K</a></td>
      </tr>
      <tr data-id="102" data-country="germany" data-event="ifo business climate" class="calendar-date-2">
        <td class="2026-10-06">09:00 AM</td>
        <td>DE</td>
        <td>IFO Business Climate</td>
        <td>85.4</td>
        <td>85.0</td>
        <td>85.2</td>
      </tr>
      <tr data-id="103" data-country="japan" data-event="cpi" class="calendar-date-3">
        <td class="2026-10-06">11:30 PM</td>
        <td>JP</td>
        <td>Tokyo Core CPI</td>
        <td>2.8%</td>
        <td>2.8%</td>
        <td>2.7%</td>
      </tr>
    </table>
    """

    server = YFinanceTAMCPServer(
        data_fetcher=lambda s, p, i: _make_sample_ohlcv(),
        calendar_fetcher=lambda _url: sample_te_html,
    )

    list_resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert list_resp is not None
    tools = [t["name"] for t in list_resp["result"]["tools"]]
    assert "ta_get_economic_calendar" in tools

    # 1. EUR/USD should include EUR and USD events (101, 102) and exclude JPY (103)
    fx_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "ta_get_economic_calendar",
                "arguments": {
                    "symbol": "EUR/USD",
                    "date_from": "2026-10-05",
                    "date_to": "2026-10-07",
                },
            },
        }
    )
    assert fx_resp is not None
    assert fx_resp["result"]["isError"] is False
    fx_cal = json.loads(fx_resp["result"]["content"][0]["text"])
    assert fx_cal["normalized_symbol"] == "EURUSD=X"
    assert fx_cal["currencies"] == ["EUR", "USD"]
    assert fx_cal["source"] == "live"
    assert fx_cal["total_events"] == 2
    assert {e["currency"] for e in fx_cal["events"]} == {"EUR", "USD"}

    # 2. AAPL with importance=['high'] should return only the high-importance USD event (101)
    stock_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ta_get_economic_calendar",
                "arguments": {
                    "symbol": "AAPL",
                    "date_from": "2026-10-05",
                    "date_to": "2026-10-07",
                    "importances": ["high"],
                },
            },
        }
    )
    assert stock_resp is not None
    assert stock_resp["result"]["isError"] is False
    stock_cal = json.loads(stock_resp["result"]["content"][0]["text"])
    assert stock_cal["currencies"] == ["USD"]
    assert stock_cal["total_events"] == 1
    assert stock_cal["events"][0]["id"] == "101"
    assert stock_cal["events"][0]["time"] == "2026-10-06 08:30"

    # 3. Fallback when live feed is unavailable
    offline_server = YFinanceTAMCPServer(
        data_fetcher=lambda s, p, i: _make_sample_ohlcv(),
        calendar_fetcher=lambda _url: None,
    )
    fb_resp = offline_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "ta_get_economic_calendar",
                "arguments": {
                    "symbol": "USDJPY",
                    "date_from": "2026-10-05",
                    "date_to": "2026-10-09",
                },
            },
        }
    )
    assert fb_resp is not None
    assert fb_resp["result"]["isError"] is False
    fb_cal = json.loads(fb_resp["result"]["content"][0]["text"])
    assert fb_cal["source"] == "scheduled_fallback"
    assert fb_cal["currencies"] == ["USD", "JPY"]
    assert fb_cal["total_events"] > 0
    assert all(e["currency"] in ("USD", "JPY") for e in fb_cal["events"])

    # 4. Verify Investing.com HTML parser compatibility
    investing_html = """
    <table>
      <tr><td class="theDay">6 октября 2026</td></tr>
      <tr id="eventRowId_555" data-event-id="555">
        <td class="time">15:30</td>
        <td class="flagCur"><span class="ceFlags" title="США"></span> USD</td>
        <td class="sentiment">
          <i class="grayFullBullishIcon"></i>
          <i class="grayFullBullishIcon"></i>
          <i class="grayFullBullishIcon"></i>
        </td>
        <td class="event"><a href="#">Core CPI (MoM)</a></td>
        <td class="act">0.3%</td>
        <td class="fore">0.2%</td>
        <td class="prev">0.2%</td>
      </tr>
    </table>
    """
    parsed_inv = list(InvestingCalendar.parse_events_html(investing_html))
    assert len(parsed_inv) == 1
    assert parsed_inv[0]["id"] == "555"
    assert parsed_inv[0]["time"] == "2026-10-06 15:30:00"
    assert parsed_inv[0]["currency"] == "USD"
    assert parsed_inv[0]["importance"] == "3"
    assert parsed_inv[0]["event"] == "Core CPI (MoM)"
