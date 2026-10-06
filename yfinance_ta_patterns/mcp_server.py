"""Native Model Context Protocol (MCP) JSON-RPC 2.0 Server for yfinance-ta-patterns.

Exposes institutional candlestick pattern scanning, AI Confluence Scoring (Trend + RSI + Volume + ATR),
actionable Trade Setups (Entry / Stop-Loss / Take-Profit), and multi-asset backtesting to Claude Code,
Claude Desktop, Cursor, Windsurf, Antigravity, and any MCP client over stdio.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from . import __version__
from .ai.analyst import AIMarketAnalyst, _json_safe
from .ai.scorer import PATTERN_CANDLE_COUNTS
from .data import MarketDataLoader, normalize_interval
from .economic_calendar import InvestingCalendar
from .forex_data_loader import FOREX_56_PAIRS
from .pattern_analyzer import PatternAnalyzer
from .pattern_tester import PatternRankingTester
from .talib_compat import get_talib_status

MCP_PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
)

MAX_WATCHLIST_SYMBOLS = 25
DEFAULT_MAX_RESULTS = 25
MAX_RESULTS_LIMIT = 50

MCP_SERVER_INSTRUCTIONS = (
    "yfinance-ta-patterns MCP server for multi-asset candlestick pattern scanning, "
    "deterministic AI Confluence Scoring (EMA 20/50/200 + RVOL + Wilder RSI-14 + Wilder ATR-14), "
    "ATR-based Trade Setups (Entry, Stop-Loss, TP1, TP2), unbiased next-open backtesting, "
    "and macroeconomic calendar events filtered for a selected asset. "
    "Supported tickers: Stocks ('AAPL', 'NVDA'), Crypto ('BTC-USD'), Forex ('EURUSD=X'), "
    "Indices ('^GSPC'), Commodities ('GC=F'). Yahoo Finance lookback limits apply to intraday "
    "intervals ('1m' <= '7d', '5m'/'15m'/'30m' <= '60d', '1h'/'4h' <= '730d') and are "
    "automatically clamped when omitted or exceeded."
)

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
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol (e.g. 'BTC-USD', 'NVDA', 'EURUSD=X')",
                },
                "timeframe": {
                    "type": "string",
                    "default": "1d",
                    "description": "Timeframe interval ('15m', '1h', '4h', '1d', '1wk')",
                },
                "period": {
                    "type": "string",
                    "default": "6mo",
                    "description": (
                        "Historical lookback period ('7d' for 1m, '60d' for 5m/15m/30m, "
                        "'6mo'/'1y'/'2y' for 1h/4h/1d; auto-clamped to Yahoo Finance limits)"
                    ),
                },
                "min_confidence": {
                    "type": "number",
                    "default": 0.4,
                    "description": "Minimum AI Confluence score in [0.0, 1.0]",
                },
                "lookback_bars": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of recent bars to evaluate for active patterns",
                },
                "include_brief": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, include a human-readable markdown_brief alongside structured JSON patterns",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "ta_scan_watchlist",
        "description": (
            "Scan multiple ticker symbols in batch (up to 25 symbols) and rank all detected "
            "candlestick pattern setups across the watchlist by AI Confluence Score descending."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": MAX_WATCHLIST_SYMBOLS,
                    "description": (
                        f"List of ticker symbols (max {MAX_WATCHLIST_SYMBOLS}, "
                        "e.g. ['AAPL', 'NVDA', 'MSFT', 'BTC-USD'])"
                    ),
                },
                "timeframe": {
                    "type": "string",
                    "default": "1d",
                    "description": "Timeframe interval (default: '1d')",
                },
                "period": {
                    "type": "string",
                    "default": "6mo",
                    "description": "Historical lookback period (auto-clamped for intraday intervals)",
                },
                "min_confidence": {
                    "type": "number",
                    "default": 0.55,
                    "description": "Minimum AI Confluence score in [0.0, 1.0]",
                },
                "lookback_bars": {
                    "type": "integer",
                    "default": 2,
                    "description": "Number of recent bars per symbol to evaluate (default: 2)",
                },
                "max_results": {
                    "type": "integer",
                    "default": DEFAULT_MAX_RESULTS,
                    "description": f"Maximum number of ranked setups to return (default: {DEFAULT_MAX_RESULTS}, max: {MAX_RESULTS_LIMIT})",
                },
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "ta_backtest_patterns",
        "description": (
            "Run quantitative historical backtesting of all candlestick patterns on a symbol "
            "(next-open execution) and return the top-performing patterns ranked by composite score or win rate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol (e.g. 'BTC-USD', 'AAPL')",
                },
                "timeframe": {
                    "type": "string",
                    "default": "1d",
                    "description": "Timeframe interval (default: '1d')",
                },
                "period": {
                    "type": "string",
                    "default": "1y",
                    "description": "Backtest period (default: '1y'; auto-clamped for intraday intervals)",
                },
                "holding_period": {
                    "type": "integer",
                    "default": 5,
                    "description": "Bars to hold each trade (default: 5)",
                },
                "min_signals": {
                    "type": "integer",
                    "default": 1,
                    "description": "Minimum total signals required to include a pattern (default: 1)",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["composite", "win_rate"],
                    "default": "composite",
                    "description": "Ranking metric: 'composite' (balances win rate, PF, Sharpe, log trades) or 'win_rate'",
                },
                "top_n": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of top-ranked patterns to return (default: 10)",
                },
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
    {
        "name": "ta_get_economic_calendar",
        "description": (
            "Get macroeconomic calendar events (live feed with resilient scheduled fallback) "
            "automatically filtered by the currencies impacting a selected asset symbol "
            "(e.g. 'EURUSD=X' -> EUR, USD; 'AAPL' -> USD; 'GBPJPY' -> GBP, JPY; 'BTC-USD' -> USD)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": (
                        "Selected asset ticker symbol or currency pair "
                        "(e.g. 'EURUSD=X', 'EUR/USD', 'GBPUSD', 'AAPL', 'BTC-USD', 'SAP.DE')"
                    ),
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format (defaults to today UTC)",
                },
                "date_to": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (defaults to date_from + days)",
                },
                "days": {
                    "type": "integer",
                    "default": 7,
                    "description": "Number of days ahead from date_from when date_to is omitted (default: 7, max: 60)",
                },
                "importances": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by event importance level: '1' (low), '2' (medium), '3' (high). "
                        "Also accepts 'low', 'medium', 'high'. Omit for all levels."
                    ),
                },
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional override list of currency codes or countries (e.g. ['USD', 'EUR']). "
                        "When omitted, automatically resolved from 'symbol'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "default": 30,
                    "description": "Maximum number of calendar events to return (default: 30, max: 100)",
                },
            },
            "required": ["symbol"],
        },
    },
]

MCP_RESOURCES_SCHEMA: list[dict[str, Any]] = [
    {
        "uri": "ta://patterns/catalog",
        "name": "TA-Lib 61 Candlestick Patterns Catalog",
        "description": "Complete list of all 61 canonical TA-Lib candlestick patterns with candle lookback lengths and runtime status.",
        "mimeType": "application/json",
    },
    {
        "uri": "ta://forex/56-pairs",
        "name": "56 Major & Cross Forex Currency Pairs",
        "description": "Canonical list of 56 major and cross Forex ticker symbols supported by yfinance-ta-patterns.",
        "mimeType": "application/json",
    },
]

MCP_PROMPTS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "ta_market_analyst_prompt",
        "description": "Generate an institutional Quantitative Portfolio Manager prompt with live pattern setups for a symbol.",
        "arguments": [
            {
                "name": "symbol",
                "description": "Ticker symbol (e.g. 'BTC-USD', 'AAPL', 'EURUSD=X')",
                "required": True,
            },
            {
                "name": "timeframe",
                "description": "Timeframe interval (default: '1d')",
                "required": False,
            },
        ],
    }
]


def _resolve_mcp_period(timeframe: str, period: Any, default_period: str = "6mo") -> str:
    """Resolve and clamp lookback period to match Yahoo Finance's intraday interval limits."""
    norm_tf = normalize_interval(timeframe or "1d")
    raw_period = str(period).strip() if period is not None and str(period).strip() else ""
    if not raw_period:
        if norm_tf == "1m":
            raw_period = "7d"
        elif norm_tf in ("2m", "5m", "15m", "30m", "90m"):
            raw_period = "60d"
        else:
            raw_period = default_period
    return MarketDataLoader._clamp_period_for_interval(raw_period, norm_tf)


def _normalize_watchlist_symbols(raw_symbols: Any) -> list[str]:
    """Validate and normalize watchlist symbols input (accepts list or comma-separated string)."""
    if raw_symbols is None:
        raise ValueError("Parameter 'symbols' is required and must be a non-empty list of tickers.")

    tokens: list[str] = []
    if isinstance(raw_symbols, str):
        tokens = [s.strip() for s in re.split(r"[,;\s]+", raw_symbols) if s.strip()]
    elif isinstance(raw_symbols, Sequence) and not isinstance(raw_symbols, (bytes, bytearray)):
        for item in raw_symbols:
            if not isinstance(item, str):
                raise ValueError(
                    f"Invalid symbol entry {item!r}: each symbol must be a string ticker."
                )
            for part in re.split(r"[,;\s]+", item.strip()):
                if part:
                    tokens.append(part)
    else:
        raise ValueError(
            "Parameter 'symbols' must be an array of ticker strings (e.g. ['AAPL', 'NVDA'])."
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for sym in tokens:
        upper = sym.upper()
        if upper not in seen:
            seen.add(upper)
            deduped.append(sym)

    if not deduped:
        raise ValueError("Parameter 'symbols' must contain at least one valid ticker symbol.")

    if len(deduped) > MAX_WATCHLIST_SYMBOLS:
        raise ValueError(
            f"Watchlist size ({len(deduped)}) exceeds maximum allowed limit of "
            f"{MAX_WATCHLIST_SYMBOLS} symbols per request."
        )

    return deduped


class YFinanceTAMCPServer:
    """Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 server for yfinance-ta-patterns."""

    def __init__(
        self,
        data_fetcher: Callable[[str, str, str], pd.DataFrame] | None = None,
        calendar_fetcher: Callable[[str], str | None] | None = None,
    ) -> None:
        self._data_fetcher = data_fetcher
        self._calendar_fetcher = calendar_fetcher

    def _fetch_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        if self._data_fetcher is not None:
            return self._data_fetcher(symbol, period, interval)
        loader = MarketDataLoader(symbol, period=period, interval=interval, timezone="UTC")
        return loader.get_data()

    def handle_request(self, request: Any) -> dict[str, Any] | None:
        """Process a single JSON-RPC 2.0 MCP message."""
        if not isinstance(request, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request: expected JSON object"},
            }

        req_id = request.get("id")
        has_id = "id" in request and req_id is not None
        method = request.get("method")

        if not isinstance(method, str) or not method.strip():
            return {
                "jsonrpc": "2.0",
                "id": req_id if has_id else None,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: 'method' must be a non-empty string",
                },
            }

        # JSON-RPC 2.0 specification: any request without an id is a notification and must not be replied to
        if not has_id:
            return None

        raw_params = request.get("params")
        if raw_params is not None and not isinstance(raw_params, dict):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: expected JSON object"},
            }
        params: dict[str, Any] = raw_params or {}

        try:
            if method == "initialize":
                requested_version = params.get("protocolVersion")
                negotiated_version = (
                    requested_version
                    if isinstance(requested_version, str)
                    and requested_version in SUPPORTED_PROTOCOL_VERSIONS
                    else MCP_PROTOCOL_VERSION
                )
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": negotiated_version,
                        "capabilities": {
                            "tools": {},
                            "resources": {},
                            "prompts": {},
                        },
                        "serverInfo": {
                            "name": "yfinance-ta-patterns-mcp",
                            "version": __version__,
                        },
                        "instructions": MCP_SERVER_INSTRUCTIONS,
                    },
                }

            if method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}

            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS_SCHEMA}}

            if method == "resources/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"resources": MCP_RESOURCES_SCHEMA},
                }

            if method == "resources/templates/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"resourceTemplates": []},
                }

            if method == "resources/read":
                uri = str(params.get("uri", ""))
                res_payload = _json_safe(self._read_resource(uri))
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(
                                    res_payload,
                                    ensure_ascii=True,
                                    separators=(",", ":"),
                                    default=str,
                                ),
                            }
                        ]
                    },
                }

            if method == "prompts/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"prompts": MCP_PROMPTS_SCHEMA},
                }

            if method == "prompts/get":
                prompt_name = str(params.get("name", ""))
                prompt_args = params.get("arguments") or {}
                if prompt_name != "ta_market_analyst_prompt":
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": f"Unknown prompt: {prompt_name}"},
                    }
                symbol = str(prompt_args.get("symbol", "")).strip()
                if not symbol:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Missing required argument: 'symbol'"},
                    }
                timeframe = normalize_interval(str(prompt_args.get("timeframe", "1d")))
                period = _resolve_mcp_period(timeframe, prompt_args.get("period"), "6mo")
                with contextlib.redirect_stdout(sys.stderr):
                    df = self._fetch_ohlcv(symbol, period, timeframe)
                    analyst = AIMarketAnalyst(df, symbol=symbol, timeframe=timeframe)
                    analyst.analyze(min_confidence=0.4, lookback_bars=3)
                    prompt_text = analyst.to_llm_prompt()
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "description": f"Quantitative technical analysis prompt for {symbol} ({timeframe})",
                        "messages": [
                            {
                                "role": "user",
                                "content": {"type": "text", "text": prompt_text},
                            }
                        ],
                    },
                }

            if method == "tools/call":
                tool_name = params.get("name", "")
                args = params.get("arguments") or {}
                if not isinstance(args, dict):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid tool arguments: expected JSON object",
                        },
                    }
                with contextlib.redirect_stdout(sys.stderr):
                    output = _json_safe(self._call_tool(str(tool_name), args))
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    output,
                                    ensure_ascii=True,
                                    separators=(",", ":"),
                                    default=str,
                                ),
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
            if method == "tools/call":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": f"Internal error: {exc}"},
            }

    def _read_resource(self, uri: str) -> dict[str, Any]:
        if uri == "ta://patterns/catalog":
            patterns = PatternAnalyzer.get_supported_fallback_patterns()
            return {
                "version": __version__,
                "total_patterns": len(patterns),
                "patterns": [
                    {
                        "name": pat.replace("CDL", ""),
                        "talib_function": pat,
                        "candle_lookback": PATTERN_CANDLE_COUNTS.get(pat.replace("CDL", ""), 1),
                    }
                    for pat in patterns
                ],
                "talib_status": get_talib_status(),
            }
        if uri == "ta://forex/56-pairs":
            return {
                "count": len(FOREX_56_PAIRS),
                "pairs": list(FOREX_56_PAIRS),
            }
        raise ValueError(f"Unknown resource URI: {uri}")

    def _call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "ta_scan_symbol":
            symbol = str(args.get("symbol", "")).strip()
            if not symbol:
                raise ValueError("Parameter 'symbol' is required and cannot be empty.")
            timeframe = normalize_interval(str(args.get("timeframe", "1d")))
            period = _resolve_mcp_period(timeframe, args.get("period"), default_period="6mo")
            min_conf = float(args.get("min_confidence", 0.4))
            lookback = int(args.get("lookback_bars", 3))
            include_brief = bool(args.get("include_brief", False))

            df = self._fetch_ohlcv(symbol, period, timeframe)
            analyst = AIMarketAnalyst(df, symbol=symbol, timeframe=timeframe)
            analyst.analyze(min_confidence=min_conf, lookback_bars=lookback)
            report = analyst.to_dict()
            report["period"] = period
            tz_obj = getattr(df.index, "tz", None)
            report["timezone"] = str(tz_obj) if tz_obj is not None else "UTC"
            if include_brief:
                report["markdown_brief"] = analyst.generate_brief()
            return report

        if name == "ta_scan_watchlist":
            symbols = _normalize_watchlist_symbols(args.get("symbols"))
            timeframe = normalize_interval(str(args.get("timeframe", "1d")))
            period = _resolve_mcp_period(timeframe, args.get("period"), default_period="6mo")
            min_conf = float(args.get("min_confidence", 0.55))
            lookback = int(args.get("lookback_bars", 2))
            raw_max_results = int(args.get("max_results", DEFAULT_MAX_RESULTS))
            max_results = max(1, min(raw_max_results, MAX_RESULTS_LIMIT))

            opportunities: list[dict[str, Any]] = []
            errors: dict[str, str] = {}
            symbols_succeeded = 0

            for sym in symbols:
                try:
                    df = self._fetch_ohlcv(sym, period, timeframe)
                    if df.empty or len(df) < 20:
                        errors[sym] = (
                            f"Insufficient market data ({len(df)} bars returned, minimum 20 required)."
                        )
                        continue
                    symbols_succeeded += 1
                    analyst = AIMarketAnalyst(df, symbol=sym, timeframe=timeframe)
                    scored = analyst.analyze(min_confidence=min_conf, lookback_bars=lookback)
                    for r in scored:
                        d = r.to_dict()
                        d["symbol"] = sym
                        opportunities.append(d)
                except Exception as exc:
                    errors[sym] = str(exc)
                    continue

            if symbols_succeeded == 0 and errors:
                first_sym, first_err = next(iter(errors.items()))
                raise RuntimeError(
                    f"Failed to scan all {len(symbols)} requested symbol(s). "
                    f"First error ({first_sym}): {first_err}"
                )

            opportunities.sort(
                key=lambda x: (
                    float(
                        x.get("confluence_score")
                        or x.get("confidence_score")
                        or x.get("confidence")
                        or 0.0
                    ),
                    str(x.get("timestamp", "")),
                ),
                reverse=True,
            )

            total_found = len(opportunities)
            truncated = opportunities[:max_results]
            result: dict[str, Any] = {
                "symbols_requested": len(symbols),
                "symbols_scanned": symbols_succeeded,
                "timeframe": timeframe,
                "period": period,
                "total_opportunities": total_found,
                "opportunities": truncated,
            }
            if total_found > max_results:
                result["truncated_to"] = max_results
            if errors:
                result["errors"] = errors
            return result

        if name == "ta_backtest_patterns":
            symbol = str(args.get("symbol", "")).strip()
            if not symbol:
                raise ValueError("Parameter 'symbol' is required and cannot be empty.")
            timeframe = normalize_interval(str(args.get("timeframe", "1d")))
            period = _resolve_mcp_period(timeframe, args.get("period"), default_period="1y")
            holding_period = int(args.get("holding_period", 5))
            min_signals = max(1, int(args.get("min_signals", 1)))
            sort_by = str(args.get("sort_by", "composite")).strip().lower()
            if sort_by not in ("composite", "win_rate"):
                raise ValueError("Parameter 'sort_by' must be 'composite' or 'win_rate'.")
            top_n = max(1, min(int(args.get("top_n", 10)), 61))

            df = self._fetch_ohlcv(symbol, period, timeframe)
            tester = PatternRankingTester(
                df,
                timeframe=timeframe,
                holding_period=holding_period,
                min_signals=min_signals,
                symbol=symbol,
                verbose=False,
            )
            tester.test_all_patterns(filter_news=False, min_signals=min_signals, sort_by=sort_by)
            top = tester.get_top_patterns(n=top_n)
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "period": period,
                "holding_period": holding_period,
                "min_signals": min_signals,
                "sort_by": sort_by,
                "top_patterns": [
                    {
                        "pattern_name": r.pattern_name,
                        "total_signals": r.total_signals,
                        "total_trades": r.total_trades,
                        "win_rate": round(r.win_rate, 2),
                        "total_pnl": round(r.total_pnl, 2),
                        "profit_factor": round(r.profit_factor, 2)
                        if pd.notna(r.profit_factor) and r.profit_factor != float("inf")
                        else None,
                        "sharpe_ratio": round(r.sharpe_ratio, 2),
                        "max_drawdown": round(r.max_drawdown, 2),
                        "score": round(r.score, 2),
                    }
                    for r in top
                ],
            }

        if name == "ta_list_patterns":
            status = get_talib_status()
            fallback_patterns = PatternAnalyzer.get_supported_fallback_patterns()
            return {
                "version": __version__,
                "talib_status": status,
                "total_patterns": len(fallback_patterns),
                "fallback_patterns": fallback_patterns,
            }

        if name == "ta_get_economic_calendar":
            symbol = str(args.get("symbol", "")).strip()
            if not symbol:
                raise ValueError("Parameter 'symbol' is required and cannot be empty.")
            date_from = args.get("date_from")
            date_to = args.get("date_to")
            days = int(args.get("days", 7))
            importances = args.get("importances")
            countries = args.get("countries")
            max_results = int(args.get("max_results", 30))

            with InvestingCalendar(http_fetcher=self._calendar_fetcher) as calendar:
                return calendar.get_events_for_symbol(
                    symbol=symbol,
                    date_from=date_from,
                    date_to=date_to,
                    days=days,
                    importances=importances,
                    countries=countries,
                    limit=max_results,
                )

        raise ValueError(f"Unknown MCP tool: {name}")

    def run_stdio(
        self,
        stdin: Any | None = None,
        stdout: Any | None = None,
    ) -> None:
        """Run the yfinance-ta-patterns MCP JSON-RPC 2.0 server over standard input/output."""
        in_stream = stdin if stdin is not None else sys.stdin
        proto_out = stdout if stdout is not None else sys.stdout
        saved_stdout = sys.stdout
        # Redirect global sys.stdout to sys.stderr so third-party libraries (yfinance, etc.)
        # can never write non-JSON text onto the MCP protocol wire.
        sys.stdout = sys.stderr
        try:
            for raw_line in in_stream:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as exc:
                    err_resp: dict[str, Any] = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {exc.msg}"},
                    }
                    proto_out.write(
                        json.dumps(err_resp, ensure_ascii=True, separators=(",", ":")) + "\n"
                    )
                    proto_out.flush()
                    continue

                try:
                    resp = self.handle_request(req)
                except Exception as exc:
                    req_id = req.get("id") if isinstance(req, dict) else None
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32603, "message": f"Internal error: {exc}"},
                    }

                if resp is not None:
                    proto_out.write(
                        json.dumps(resp, ensure_ascii=True, separators=(",", ":"), default=str)
                        + "\n"
                    )
                    proto_out.flush()
        finally:
            sys.stdout = saved_stdout


def main_mcp(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for yfinance-ta-mcp."""
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
