"""Tests for yfinance-ta-patterns Model Context Protocol (MCP) JSON-RPC 2.0 Server."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yfinance_ta_patterns.mcp_server import YFinanceTAMCPServer


def _make_sample_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=60, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    closes = 100.0 + np.cumsum(rng.normal(0.1, 1.5, size=60))
    opens = closes - rng.normal(0.0, 0.8, size=60)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0.5, 0.5, size=60))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0.5, 0.5, size=60))
    # Force a clear bullish hammer / engulfing on the final bar
    opens[-1] = closes[-2] - 1.5
    closes[-1] = closes[-2] + 2.5
    highs[-1] = closes[-1] + 0.3
    lows[-1] = opens[-1] - 3.0
    vols = rng.integers(100_000, 500_000, size=60).astype(float)
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


def test_yfinance_ta_mcp_server_lifecycle() -> None:
    df = _make_sample_ohlcv()
    server = YFinanceTAMCPServer(data_fetcher=lambda sym, per, tf: df.copy())

    # 1. initialize
    init_resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "yfinance-ta-patterns-mcp"

    # 2. tools/list
    list_resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None
    tools = [t["name"] for t in list_resp["result"]["tools"]]
    assert "ta_scan_symbol" in tools
    assert "ta_scan_watchlist" in tools
    assert "ta_backtest_patterns" in tools
    assert "ta_list_patterns" in tools

    # 3. ta_scan_symbol
    scan_resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ta_scan_symbol",
                "arguments": {
                    "symbol": "BTC-USD",
                    "timeframe": "1d",
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
    assert "market_summary" in report
    assert "markdown_brief" in report

    # 4. ta_backtest_patterns
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

    # 5. ta_list_patterns
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
