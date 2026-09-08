"""Command-line interface for scanning TA-Lib candlestick patterns with AI intelligence."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from . import __version__
from .ai.analyst import AIMarketAnalyst
from .ai.scorer import AIPatternScorer, PatternConfidenceResult
from .data import MarketDataLoader, normalize_interval
from .pattern_analyzer import PatternAnalyzer

TIMEFRAME_MAP: dict[str, str] = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}
VALID_INTERVALS: set[str] = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "4h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}


def normalize_timeframe(timeframe: str) -> str:
    """Convert human-friendly timeframe names into yfinance intervals."""
    interval = normalize_interval(timeframe)
    if interval not in VALID_INTERVALS:
        allowed = ", ".join(sorted(VALID_INTERVALS))
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Allowed: {allowed}")
    return interval


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="yfinance-ta-patterns",
        description="Scan candlestick patterns with optional AI probabilistic confidence scoring and trade setups.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # AI-scored patterns with trade setups\n"
            "  yftp --all-patterns --symbol EURUSD --timeframe 1h --ai --min-confidence 0.65\n\n"
            "  # AI Market Analyst executive brief\n"
            "  yftp --all-patterns --symbol AAPL --timeframe 1d --ai-analyst\n\n"
            "  # Export structured JSON for trading bots or LLMs\n"
            "  yftp --all-patterns --symbol BTC-USD --timeframe 4h --ai --format json\n\n"
            "  # Single pattern classic scan\n"
            "  yftp --pattern HAMMER --symbol EURUSD --timeframe 15m --period 30d\n"
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Ticker symbol (e.g. EURUSD, AAPL, BTC-USD, NVDA, GC=F).",
    )
    parser.add_argument("--period", default="60d", help="History period, e.g. 60d, 1y, max")
    parser.add_argument(
        "--timeframe",
        default="15m",
        help=(
            "Timeframe (M1, M5, M15, M30, H1, H4, D1) "
            "or raw yfinance interval (1m, 5m, 15m, 1h, 4h, 1d...)."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pattern",
        help="Single candlestick pattern, e.g. HAMMER (CDL prefix optional).",
    )
    group.add_argument(
        "--all-patterns",
        action="store_true",
        help="Scan and show signals for all TA-Lib candlestick patterns.",
    )
    parser.add_argument(
        "--date",
        help="Optional filter by date (YYYY-MM-DD). If omitted, show all signals.",
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date (YYYY-MM-DD) for range filter (inclusive).",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date (YYYY-MM-DD) for range filter (inclusive).",
    )
    # AI Options
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Enable AI multi-factor confidence scoring and automated trade setup calculation.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum AI confidence score threshold (0.0 to 1.0, e.g. 0.65).",
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        default=False,
        help="Enable split and dividend price adjustments (useful for long historical equity backtests).",
    )
    parser.add_argument(
        "--execution",
        choices=["next_open", "close"],
        default="next_open",
        help="Trade execution timing: 'next_open' (unbiased, enters on bar i+1) or 'close' (legacy).",
    )
    parser.add_argument(
        "--min-signals",
        type=int,
        default=1,
        help="Minimum total signals required to include a pattern in ranking (reduces overfitting).",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Transaction fee deducted per completed trade in backtester.",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Slippage in price units applied adversely to entries and exits in backtester.",
    )
    parser.add_argument(
        "--ai-analyst",
        action="store_true",
        help="Generate an executive AI market intelligence brief.",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Generate a structured prompt optimized for LLMs (ChatGPT, Claude, Gemini).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output presentation format.",
    )
    return parser.parse_args(args)


def run_cli(args: argparse.Namespace) -> int:
    """Run CLI logic with parsed arguments."""
    interval = normalize_timeframe(args.timeframe)

    if args.date and (args.start_date or args.end_date):
        raise ValueError("Use either --date or --start-date/--end-date, not both.")

    start_arg = args.start_date
    end_arg = args.end_date

    if args.date:
        # Load with 60-day warm-up lookback buffer so indicators (EMA, RSI, ATR) compute accurately
        target_dt = pd.to_datetime(args.date)
        start_arg = (target_dt - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        end_arg = (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    loader = MarketDataLoader(
        args.symbol,
        period=args.period,
        interval=interval,
        start=start_arg,
        end=end_arg,
        auto_adjust=args.auto_adjust,
    )
    data = loader.get_data()
    period = loader.period

    if data.empty:
        print(
            f"Error: No data retrieved for symbol '{args.symbol}' ({period}, {interval}).",
            file=sys.stderr,
        )
        return 1

    analyzer = PatternAnalyzer(data)
    range_info = ""
    if args.date:
        range_info = f" on {args.date}"
    elif args.start_date or args.end_date:
        range_info = f" from {args.start_date or 'beginning'} to {args.end_date or 'end'}"

    patterns_to_scan = [args.pattern] if args.pattern else sorted(analyzer.pattern_functions)
    use_ai = args.ai or args.ai_analyst or args.prompt

    all_scored_results: list[PatternConfidenceResult] = []
    classic_signals: dict[str, Any] = {}

    scorer = AIPatternScorer(data) if use_ai else None

    for pat in patterns_to_scan:
        signals = analyzer.get_signals(
            pat,
            date=args.date,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        if signals.empty:
            continue

        clean_name = pat.replace("CDL", "")
        if scorer:
            scored = scorer.score_all_signals(
                signals, clean_name, min_confidence=args.min_confidence
            )
            all_scored_results.extend(scored)
        else:
            classic_signals[clean_name] = signals

    # Align historical analysis when a historical --date is specified (Issue 20)
    analyst_data = data
    if args.date and not data.empty:
        target_cutoff = pd.to_datetime(args.date) + pd.Timedelta(days=1)
        if (
            isinstance(data.index, pd.DatetimeIndex)
            and data.index.tz is not None
            and target_cutoff.tz is None
        ):
            target_cutoff = target_cutoff.tz_localize(data.index.tz)
        sliced = data[data.index < target_cutoff]
        if not sliced.empty:
            analyst_data = sliced

    # 1. LLM Prompt Output
    if args.prompt:
        analyst = AIMarketAnalyst(analyst_data, all_scored_results)
        print(analyst.to_llm_prompt(args.symbol, interval))
        return 0

    # 2. AI Analyst Brief
    if args.ai_analyst:
        analyst = AIMarketAnalyst(analyst_data, all_scored_results)
        if args.format == "json":
            print(analyst.to_json(args.symbol, interval))
        else:
            print(analyst.generate_brief(args.symbol, interval))
        return 0

    # 3. AI Mode Output
    if args.ai:
        if not all_scored_results:
            print(f"No AI-scored signals found for {args.symbol} matching criteria{range_info}.")
            return 0

        if args.format == "json":
            analyst = AIMarketAnalyst(analyst_data, all_scored_results)
            print(analyst.to_json(args.symbol, interval))
            return 0


        print(
            f"=== AI Pattern Intelligence: {args.symbol} ({interval}, {period}){range_info} ==="
        )
        for res in all_scored_results:
            conf_pct = f"{res.confidence_score * 100:.1f}%"
            print(
                f"\n[{res.grade.value}] {res.pattern_name} at {res.timestamp} | AI Confidence: {conf_pct}"
            )
            print(
                f"  Regime: {res.trend_regime} | RVOL: {res.rvol:.2f}x | RSI: {res.rsi:.1f} | ATR: {res.atr:.5f}"
            )
            if res.trade_setup:
                ts = res.trade_setup
                print(
                    f"  Setup: {ts.direction} @ {ts.entry_price:.5f} | "
                    f"Stop: {ts.stop_loss:.5f} | TP1: {ts.take_profit_1:.5f} | TP2: {ts.take_profit_2:.5f} "
                    f"(R/R: {ts.risk_reward_ratio:.1f}:1)"
                )
            if res.confluence_factors:
                print(f"  Confluence: {', '.join(res.confluence_factors)}")
            if res.risk_factors:
                print(f"  Risks: {', '.join(res.risk_factors)}")
            print("-" * 60)
        return 0

    # 4. Classic TA-Lib Mode Output
    if not classic_signals:
        if args.pattern:
            print(
                f"No signals for pattern {args.pattern} "
                f"on period {period} timeframe {interval}{range_info}"
            )
        else:
            print(
                f"No signals for any pattern on period {period} "
                f"timeframe {interval}{range_info}"
            )
        return 0

    print(f"Scanning patterns for {args.symbol} ({interval}, {period}){range_info}...")
    for pat_name, sig in classic_signals.items():
        print(f"{pat_name}:")
        print(sig.to_string())
        print("-" * 40)

    return 0


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI entrypoint."""
    parsed = parse_args(argv)
    try:
        sys.exit(run_cli(parsed))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
