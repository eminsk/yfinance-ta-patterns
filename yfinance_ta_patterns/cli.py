"""Command-line interface for scanning TA-Lib candlestick patterns with AI intelligence."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from . import __version__
from .ai.analyst import AIMarketAnalyst
from .ai.scorer import AIPatternScorer, PatternConfidenceResult
from .data import MarketDataLoader, normalize_interval
from .forex_data_loader import FOREX_56_PAIRS
from .pattern_analyzer import PatternAnalyzer
from .talib_compat import HAS_NATIVE_TALIB

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


def resolve_symbols(symbol_arg: str | None, all_pairs_flag: bool = False) -> list[str]:
    """Resolve target symbols from CLI arguments.

    If symbol_arg is None, empty, 'ALL', 'FOREX', 'PAIRS', or all_pairs_flag is True,
    returns all 56 major Forex currency pairs.
    If symbol_arg contains comma-separated values (e.g. 'EURUSD,GBPUSD'), returns the list.
    Otherwise returns [symbol_arg].
    """
    if all_pairs_flag:
        return list(FOREX_56_PAIRS)
    if not symbol_arg or symbol_arg.strip().upper() in (
        "ALL",
        "FOREX",
        "PAIRS",
        "ALL_PAIRS",
        "ALL-PAIRS",
        "*",
    ):
        return list(FOREX_56_PAIRS)
    if "," in symbol_arg:
        return [s.strip() for s in symbol_arg.split(",") if s.strip()]
    return [symbol_arg.strip()]


def get_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="yfinance-ta-patterns",
        description="Scan candlestick patterns with optional AI multi-factor confluence scoring and trade setups.",
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
        default=None,
        help=(
            "Ticker symbol (e.g. EURUSD, AAPL, BTC-USD, NVDA, GC=F) or comma-separated list. "
            "If omitted or set to 'ALL'/'FOREX', scans all 56 major Forex currency pairs automatically."
        ),
    )
    parser.add_argument(
        "--symbols",
        dest="symbol_alias",
        help="Alias for --symbol (supports comma-separated list of symbols or 'ALL'/'FOREX').",
    )
    parser.add_argument(
        "--all-pairs",
        "--forex",
        dest="all_pairs",
        action="store_true",
        help="Scan all 56 major Forex currency pairs (equivalent to omitting --symbol).",
    )
    parser.add_argument(
        "--lookback-bars",
        type=int,
        default=None,
        help="Number of recent bars to scan in multi-symbol mode (default: 2 for live scanning; 0 for all bars).",
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
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--pattern",
        help="Single candlestick pattern, e.g. HAMMER (CDL prefix optional).",
    )
    group.add_argument(
        "--all-patterns",
        action="store_true",
        default=False,
        help="Scan and show signals for all TA-Lib candlestick patterns (default).",
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
        help="Enable AI multi-factor confluence scoring and automated trade setup calculation.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum AI multi-factor confluence score threshold (0.0 to 1.0, e.g. 0.65; default: 0.55 for multi-pair, 0.0 for single symbol).",
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        default=False,
        help="Adjust OHLC prices for splits and dividends (total-return analysis). Default is raw unadjusted prices.",
    )
    repair_group = parser.add_mutually_exclusive_group()
    repair_group.add_argument(
        "--repair",
        dest="repair",
        action="store_true",
        default=None,
        help="Explicitly enable yfinance price anomaly repair (requires scikit-learn).",
    )
    repair_group.add_argument(
        "--no-repair",
        dest="repair",
        action="store_false",
        help="Explicitly disable yfinance price anomaly repair.",
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
    return parser


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return get_parser().parse_args(args)


build_parser = get_parser


def run_cli(args: argparse.Namespace) -> int:
    """Run CLI logic with parsed arguments."""
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    interval = normalize_timeframe(args.timeframe)

    if not args.pattern and not args.all_patterns:
        args.all_patterns = True

    symbol_input = args.symbol or getattr(args, "symbol_alias", None)
    symbols = resolve_symbols(symbol_input, getattr(args, "all_pairs", False))

    min_confidence: float = (
        float(args.min_confidence)
        if args.min_confidence is not None
        else (0.55 if len(symbols) > 1 else 0.0)
    )

    if args.date and (args.start_date or args.end_date):
        raise ValueError("Use either --date or --start-date/--end-date, not both.")

    start_arg = args.start_date
    end_arg = args.end_date

    if args.date:
        # Scale lookback buffer based on timeframe so indicators (EMA200, RSI, ATR) have enough warm-up history
        target_dt = pd.to_datetime(args.date)
        if interval in ("1d", "1wk", "1mo"):
            lookback_days = 365  # 1 full year (~252 trading days) to ensure full EMA200 warm-up
        elif interval in ("4h", "1h", "60m"):
            lookback_days = 90
        elif interval in ("15m", "30m"):
            lookback_days = 60
        elif interval == "5m":
            lookback_days = 30
        else:  # 1m, 2m
            lookback_days = 7

        start_arg = (target_dt - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_arg = (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    use_ai = args.ai or args.ai_analyst or args.prompt or (len(symbols) > 1 and not args.pattern)

    # ---------------------------------------------------------
    # MULTI-SYMBOL / FOREX PORTFOLIO SCANNING MODE
    # ---------------------------------------------------------
    if len(symbols) > 1:
        print("=" * 95)
        print(
            f"🌍 СКАНИРОВАНИЕ {len(symbols)} ВАЛЮТНЫХ ПАР FOREX (Таймфрейм: {interval}, Период: {args.period})"
        )
        print(
            "Поиск самых выгодных точек входа на основе AI Confluence Scoring (Trend + RSI + Volume + ATR)"
        )
        print("=" * 95)

        all_signals_records: list[dict[str, Any]] = []
        active_pairs_count = 0
        lookback = args.lookback_bars if args.lookback_bars is not None else 2

        for idx, sym in enumerate(symbols, 1):
            clean_sym = sym.replace("=X", "")
            if sys.stdout and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
                print(f"[{idx:2d}/{len(symbols)}] Опрос {clean_sym:<7} ...", end="\r", flush=True)

            try:
                loader = MarketDataLoader(
                    sym,
                    period=args.period,
                    interval=interval,
                    start=start_arg,
                    end=end_arg,
                    auto_adjust=args.auto_adjust,
                    repair=args.repair,
                )
                df = loader.get_data()
                if df.empty or len(df) < 20:
                    continue

                active_pairs_count += 1
                analyzer = PatternAnalyzer(df)
                scorer = AIPatternScorer(df) if use_ai else None

                if args.date or args.start_date or args.end_date or lookback <= 0:
                    recent_timestamps = df.index
                else:
                    recent_timestamps = df.index[-lookback:] if len(df) >= lookback else df.index

                patterns_to_scan = (
                    [args.pattern] if args.pattern else sorted(analyzer.pattern_functions)
                )

                for pat in patterns_to_scan:
                    try:
                        sig_series = analyzer.get_signals(
                            pat,
                            date=args.date,
                            start_date=args.start_date,
                            end_date=args.end_date,
                        )
                    except NotImplementedError:
                        continue

                    if sig_series.empty:
                        continue

                    for ts in recent_timestamps:
                        if ts not in sig_series.index:
                            continue
                        signal_val = sig_series.loc[ts]
                        if signal_val == 0:
                            continue

                        clean_pat = pat.replace("CDL", "")
                        if scorer:
                            res = scorer.score_signal(pat, ts, signal_val)
                            score = res.confidence
                            if score >= min_confidence:
                                setup = res.trade_setup
                                grade_label = (
                                    res.grade.value
                                    if hasattr(res.grade, "value")
                                    else str(res.grade)
                                )
                                direction = (
                                    setup.direction
                                    if setup
                                    else ("BUY" if signal_val > 0 else "SELL")
                                )
                                entry = setup.entry_price if setup else float(df.loc[ts, "Close"])
                                sl = setup.stop_loss if setup else 0.0
                                tp1 = setup.take_profit_1 if setup else 0.0
                                tp2 = setup.take_profit_2 if setup else 0.0
                                rr = setup.risk_reward_ratio if setup else 1.5

                                all_signals_records.append(
                                    {
                                        "Symbol": clean_sym,
                                        "RawSymbol": sym,
                                        "Time": (
                                            ts.strftime("%d.%m %H:%M")
                                            if hasattr(ts, "strftime")
                                            else str(ts)
                                        ),
                                        "Direction": "🟢 BUY" if direction == "BUY" else "🔴 SELL",
                                        "Pattern": clean_pat,
                                        "Confluence": f"{score * 100:.1f}%",
                                        "Grade": grade_label,
                                        "Entry": entry,
                                        "StopLoss": sl,
                                        "TakeProfit_1": tp1,
                                        "TakeProfit_2": tp2,
                                        "RR": f"1:{rr:.1f}",
                                        "Trend": res.trend_regime,
                                        "RSI": round(res.rsi, 1),
                                        "Score_Raw": score,
                                        "RR_Raw": rr,
                                        "df": df,
                                        "result": res,
                                    }
                                )
                        else:
                            all_signals_records.append(
                                {
                                    "Symbol": clean_sym,
                                    "RawSymbol": sym,
                                    "Time": (
                                        ts.strftime("%d.%m %H:%M")
                                        if hasattr(ts, "strftime")
                                        else str(ts)
                                    ),
                                    "Direction": "🟢 BUY" if signal_val > 0 else "🔴 SELL",
                                    "Pattern": clean_pat,
                                    "Price": float(df.loc[ts, "Close"]),
                                    "df": df,
                                }
                            )
            except Exception:
                continue

        if sys.stdout and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            print(" " * 60, end="\r")
        print(f"✔ Просканировано: {active_pairs_count} активных пар из {len(symbols)}")

        if not all_signals_records:
            bars_info = (
                f"за последние {lookback} бара(ов)" if lookback > 0 else "за выбранный период"
            )
            print(
                f"\n⚠️ {bars_info.capitalize()} надежных сетапов с уверенностью >= {min_confidence*100:.0f}% не найдено."
            )
            print(
                "Рынок находится в фазе консолидации / флэта. Попробуйте сменить таймфрейм на 15m или 1h."
            )
            return 0

        if not use_ai:
            classic_df = pd.DataFrame(all_signals_records)[
                ["Time", "Symbol", "Direction", "Pattern", "Price"]
            ]
            print("\n" + "=" * 80)
            print(f"📊 НАЙДЕННЫЕ СИГНАЛЫ ({len(classic_df)} паттернов):")
            print("=" * 80)
            print(classic_df.to_string(index=False))
            return 0

        signals_df = pd.DataFrame(all_signals_records).sort_values(
            by=["Score_Raw", "RR_Raw"], ascending=[False, False]
        )

        if args.format == "json":
            import json

            export_data = [
                {k: v for k, v in row.items() if k not in ("df", "result")}
                for row in signals_df.to_dict(orient="records")
            ]
            print(json.dumps(export_data, indent=2, ensure_ascii=False))
            return 0

        print("\n" + "=" * 105)
        print(
            f"📊 РЕЙТИНГ НАЙДЕННЫХ ВОЗМОЖНОСТЕЙ ({len(signals_df)} сигналов, отсортированы по качеству входа):"
        )
        print("=" * 105)

        display_cols = [
            "Time",
            "Symbol",
            "Direction",
            "Pattern",
            "Confluence",
            "Grade",
            "Entry",
            "StopLoss",
            "TakeProfit_1",
            "RR",
            "Trend",
            "RSI",
        ]
        print(signals_df[display_cols].to_string(index=False))

        best = signals_df.iloc[0]

        print("\n" + "🔥" * 38)
        print(f"  🏆 НАИБОЛЕЕ ВЫГОДНАЯ ПАРА ДЛЯ ВХОДА ПРЯМО СЕЙЧАС: {best['Symbol']}")
        print("🔥" * 38)
        print(f"  • Направление сделки:   {best['Direction']} (Паттерн: {best['Pattern']})")
        print(f"  • Качество слияния:     {best['Confluence']} [Грейд: {best['Grade']}]")
        print(f"  • Подтверждение тренда: {best['Trend']}")
        print(f"  • Индекс силы RSI(14):  {best['RSI']}")
        print(f"  • Точка входа (Entry):  {best['Entry']}")
        print(f"  • Стоп-лосс (StopLoss): {best['StopLoss']}")
        print(f"  • Тейк-профит 1 (TP1):  {best['TakeProfit_1']} (Риск/Прибыль {best['RR']})")
        print(f"  • Тейк-профит 2 (TP2):  {best['TakeProfit_2']}")
        print(f"  • Свеча сигнала:        {best['Time']}")
        print("=" * 76)

        if args.ai_analyst:
            print("\n" + "=" * 76)
            print(f"📋 AI TECHNICAL INTELLIGENCE BRIEF ДЛЯ ТОП-ПАРЫ: {best['Symbol']}")
            print("=" * 76 + "\n")
            analyst = AIMarketAnalyst(best["df"], [best["result"]])
            print(analyst.generate_brief(best["Symbol"], interval))

        if args.prompt:
            print("\n" + "=" * 76)
            print(f"🤖 LLM PROMPT ДЛЯ ТОП-ПАРЫ: {best['Symbol']}")
            print("=" * 76 + "\n")
            analyst = AIMarketAnalyst(best["df"], [best["result"]])
            print(analyst.to_llm_prompt(best["Symbol"], interval))

        return 0

    # ---------------------------------------------------------
    # SINGLE-SYMBOL MODE (100% Backwards Compatible)
    # ---------------------------------------------------------
    symbol = symbols[0]
    loader = MarketDataLoader(
        symbol,
        period=args.period,
        interval=interval,
        start=start_arg,
        end=end_arg,
        auto_adjust=args.auto_adjust,
        repair=args.repair,
    )
    data = loader.get_data()
    period = loader.period

    if data.empty:
        print(
            f"Error: No data retrieved for symbol '{symbol}' ({period}, {interval}).",
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
    if not HAS_NATIVE_TALIB and args.all_patterns:
        print(
            f"Notice: Native TA-Lib binary not found. Scanning {len(analyzer.pattern_functions)} pure-Python fallback patterns.",
            file=sys.stderr,
        )

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
                signals, clean_name, min_confidence=min_confidence
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
        print(analyst.to_llm_prompt(symbol, interval))
        return 0

    # 2. AI Analyst Brief
    if args.ai_analyst:
        analyst = AIMarketAnalyst(analyst_data, all_scored_results)
        if args.format == "json":
            print(analyst.to_json(symbol, interval))
        else:
            print(analyst.generate_brief(symbol, interval))
        return 0

    # 3. AI Mode Output
    if args.ai:
        if not all_scored_results:
            print(f"No AI-scored signals found for {symbol} matching criteria{range_info}.")
            return 0

        if args.format == "json":
            analyst = AIMarketAnalyst(analyst_data, all_scored_results)
            print(analyst.to_json(symbol, interval))
            return 0

        print(f"=== AI Pattern Intelligence: {symbol} ({interval}, {period}){range_info} ===")
        for res in all_scored_results:
            conf_score = f"{res.confidence * 100:.1f}/100"
            print(
                f"\n[{res.grade.value}] {res.pattern_name} at {res.timestamp} | AI Confluence: {conf_score}"
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
            print(f"No signals for any pattern on period {period} timeframe {interval}{range_info}")
        return 0

    print(f"Scanning patterns for {symbol} ({interval}, {period}){range_info}...")
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
