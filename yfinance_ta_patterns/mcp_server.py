"""Native Model Context Protocol (MCP) JSON-RPC 2.0 Server for yfinance-ta-patterns.

Exposes institutional candlestick pattern scanning, AI Confluence Scoring (Trend + RSI + Volume + ATR),
actionable Trade Setups (Entry / Stop-Loss / Take-Profit), and multi-asset backtesting to Claude Desktop,
Cursor, Windsurf, Antigravity, and any MCP client over stdio.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from . import __version__
from .ai.analyst import AIMarketAnalyst, _json_safe
from .data import MarketDataLoader
from .pattern_analyzer import PatternAnalyzer
from .pattern_tester import PatternRankingTester
from .talib_compat import get_talib_status

MCP_PROTOCOL_VERSION = "2024-11-05"

MCP_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "ta_scan_symbol",
        "description": (
            "Scan a market symbol (Stocks, Crypto, Forex, Indices, Commodities, e.g. 'BTC-USD', "
            "'NVDA', 'AAPL', 'EURUSD=X') for candlestick patterns, compute AI Confluence Scores "
            "(Trend + RSI + Volume + ATR), and generate actionable Trade Setups (Entry, Stop-Loss, TP1, TP2)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol (e.g. 'BTC-USD', 'NVDA', 'EURUSD=X')"},
                "timeframe": {"type": "string", "default": "1d", "description": "Timeframe interval ('15m', '1h', '4h', '1d', '1wk')"},
                "period": {"type": "string", "default": "6mo", "description": "Historical lookback period ('1mo', '3mo', '6mo', '1y', '2y')"},
                "min_confidence": {"type": "number", "default": 0.4, "description": "Minimum AI Confluence score in [0.0, 1.0]"},
                "lookback_bars": {"type": "integer", "default": 3, "description": "Number of recent bars to evaluate for active patterns"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "ta_scan_watchlist",
        "description": (
            "Scan multiple ticker symbols in batch and rank all detected candlestick pattern "
            "setups across the watchlist by AI Confluence Score descending."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols (e.g. ['AAPL', 'NVDA', 'MSFT', 'BTC-USD'])",
                },
                "timeframe": {"type": "string", "default": "1d", "description": "Timeframe interval (default: '1d')"},
                "period": {"type": "string", "default": "6mo", "description": "Historical lookback period (default: '6mo')"},
                "min_confidence": {"type": "number", "default": 0.55, "description": "Minimum AI Confluence score in [0.0, 1.0]"},
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "ta_backtest_patterns",
        "description": (
            "Run quantitative historical backtesting of all candlestick patterns on a symbol "
            "and return the top-performing patterns ranked by win rate, PnL, Profit Factor, and Sharpe ratio."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol (e.g. 'BTC-USD', 'AAPL')"},
                "timeframe": {"type": "string", "default": "1d", "description": "Timeframe interval (default: '1d')"},
                "period": {"type": "string", "default": "1y", "description": "Backtest period (default: '1y')"},
                "holding_period": {"type": "integer", "default": 5, "description": "Bars to hold each trade (default: 5)"},
                "top_n": {"type": "integer", "default": 10, "description": "Number of top-ranked patterns to return"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "ta_list_patterns",
        "description": "List all supported candlestick patterns, TA-Lib binary status, and Python No-GIL runtime status.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


class YFinanceTAMCPServer:
    """Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 server for yfinance-ta-patterns."""

    def __init__(
        self,
        data_fetcher: Callable[[str, str, str], pd.DataFrame] | None = None,
    ) -> None:
        self._data_fetcher = data_fetcher

    def _fetch_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        if self._data_fetcher is not None:
            return self._data_fetcher(symbol, period, interval)
        loader = MarketDataLoader(symbol, period=period, interval=interval)
        return loader.get_data()

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC 2.0 MCP message."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params") or {}

        if req_id is None and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "yfinance-ta-patterns-mcp",
                            "version": __version__,
                        },
                    },
                }

            if method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}

            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS_SCHEMA}}

            if method == "tools/call":
                tool_name = params.get("name", "")
                args = params.get("arguments") or {}
                output = _json_safe(self._call_tool(tool_name, args))
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(output, ensure_ascii=False, indent=2, default=str),
                            }
                        ],
                        "isError": False,
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    def _call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "ta_scan_symbol":
            symbol = str(args["symbol"])
            timeframe = str(args.get("timeframe", "1d"))
            period = str(args.get("period", "6mo"))
            min_conf = float(args.get("min_confidence", 0.4))
            lookback = int(args.get("lookback_bars", 3))

            df = self._fetch_ohlcv(symbol, period, timeframe)
            analyst = AIMarketAnalyst(df, symbol=symbol, timeframe=timeframe)
            analyst.analyze(min_confidence=min_conf, lookback_bars=lookback)
            report = analyst.to_dict()
            report["markdown_brief"] = analyst.generate_brief()
            return report

        if name == "ta_scan_watchlist":
            symbols = [str(s) for s in (args.get("symbols") or [])]
            timeframe = str(args.get("timeframe", "1d"))
            period = str(args.get("period", "6mo"))
            min_conf = float(args.get("min_confidence", 0.55))

            opportunities: list[dict[str, Any]] = []
            for sym in symbols:
                try:
                    df = self._fetch_ohlcv(sym, period, timeframe)
                    if df.empty or len(df) < 20:
                        continue
                    analyst = AIMarketAnalyst(df, symbol=sym, timeframe=timeframe)
                    scored = analyst.analyze(min_confidence=min_conf, lookback_bars=2)
                    for r in scored:
                        d = r.to_dict()
                        d["symbol"] = sym
                        opportunities.append(d)
                except Exception:
                    continue

            opportunities.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
            return {
                "symbols_scanned": len(symbols),
                "total_opportunities": len(opportunities),
                "opportunities": opportunities,
            }

        if name == "ta_backtest_patterns":
            symbol = str(args["symbol"])
            timeframe = str(args.get("timeframe", "1d"))
            period = str(args.get("period", "1y"))
            holding_period = int(args.get("holding_period", 5))
            top_n = int(args.get("top_n", 10))

            df = self._fetch_ohlcv(symbol, period, timeframe)
            tester = PatternRankingTester(
                df,
                timeframe=timeframe,
                holding_period=holding_period,
                symbol=symbol,
            )
            tester.test_all_patterns(filter_news=False)
            top = tester.get_top_patterns(n=top_n)
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "period": period,
                "holding_period": holding_period,
                "top_patterns": [
                    {
                        "pattern_name": r.pattern_name,
                        "total_signals": r.total_signals,
                        "total_trades": r.total_trades,
                        "win_rate": round(r.win_rate, 2),
                        "total_pnl": round(r.total_pnl, 2),
                        "profit_factor": round(r.profit_factor, 2),
                        "sharpe_ratio": round(r.sharpe_ratio, 2),
                        "max_drawdown": round(r.max_drawdown, 2),
                    }
                    for r in top
                ],
            }

        if name == "ta_list_patterns":
            status = get_talib_status()
            return {
                "version": __version__,
                "talib_status": status,
                "fallback_patterns": PatternAnalyzer.get_supported_fallback_patterns(),
            }

        raise ValueError(f"Unknown MCP tool: {name}")

    def run_stdio(self) -> None:
        """Run the yfinance-ta-patterns MCP JSON-RPC 2.0 server over standard input/output."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False, default=str) + "\n")
                sys.stdout.flush()


def main_mcp(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for yfinance-ta-mcp."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="yfinance-ta-mcp",
        description="Start yfinance-ta-patterns Model Context Protocol (MCP) Server over stdio",
    )
    parser.parse_args(argv)
    server = YFinanceTAMCPServer()
    server.run_stdio()
    return 0


if __name__ == "__main__":
    sys.exit(main_mcp())
