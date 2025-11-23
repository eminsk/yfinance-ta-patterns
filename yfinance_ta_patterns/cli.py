import argparse

from .forex_data_loader import ForexDataLoader
from .pattern_analyzer import PatternAnalyzer


TIMEFRAME_MAP = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "D1": "1d",
}
VALID_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}


def normalize_timeframe(timeframe: str) -> str:
    """Convert human-friendly timeframe names into yfinance intervals."""
    tf = timeframe.upper()
    interval = TIMEFRAME_MAP.get(tf, timeframe.lower())
    if interval not in VALID_INTERVALS:
        allowed = ", ".join(sorted(VALID_INTERVALS))
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Allowed: {allowed}")
    return interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show TA-Lib candlestick pattern signals for a symbol.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # All patterns on a single day\n"
            "  python main.py --all-patterns --symbol EURUSD --timeframe 5m --period 60d --date 2025-04-01\n\n"
            "  # All patterns over a date range\n"
            "  python main.py --all-patterns --symbol EURUSD --timeframe 5m --period 60d "
            "--start-date 2025-04-01 --end-date 2025-04-10\n\n"
            "  # Single pattern without date filter\n"
            "  python main.py --pattern KICKING --symbol EURUSD --timeframe 5m --period 60d\n"
        ),
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Ticker without suffix (e.g. EURUSD, GBPUSD); '=X' will be appended automatically.",
    )
    parser.add_argument("--period", default="60d", help="History period, e.g. 60d")
    parser.add_argument(
        "--timeframe",
        default="15m",
        help="Timeframe (M1, M5, M15, M30, H1, D1) or raw yfinance interval (1m, 5m, 15m, 1h, 1d...).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pattern",
        help="Single candlestick pattern, e.g. KICKING (CDL prefix optional).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interval = normalize_timeframe(args.timeframe)

    if args.date and (args.start_date or args.end_date):
        raise ValueError("Use either --date or --start-date/--end-date, not both.")

    data = ForexDataLoader(
        args.symbol,
        period=args.period,
        interval=interval,
    ).get_data()

    analyzer = PatternAnalyzer(data)
    range_info = ""
    if args.date:
        range_info = f" on {args.date}"
    elif args.start_date or args.end_date:
        range_info = f" from {args.start_date or 'beginning'} to {args.end_date or 'end'}"

    if args.pattern:
        signals = analyzer.get_signals(
            args.pattern,
            date=args.date,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        if signals.empty:
            print(
                f"No signals for pattern {args.pattern} "
                f"on period {args.period} timeframe {interval}{range_info}"
            )
        else:
            print(f"Found signals for {args.pattern} ({interval}, {args.period}){range_info}:")
            print(signals.to_string())
    else:
        print(
            f"Scanning all patterns for {args.symbol} "
            f"({interval}, {args.period}){range_info}..."
        )
        found_any = False
        for pattern in sorted(analyzer.pattern_functions):
            signals = analyzer.get_signals(
                pattern,
                date=args.date,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            pattern_name = pattern.replace("CDL", "")
            if signals.empty:
                print(f"{pattern_name}: no signals")
                continue

            found_any = True
            print(f"{pattern_name}:")
            print(signals.to_string())
            print("-" * 40)

        if not found_any:
            print(
                f"No signals for any pattern on period {args.period} "
                f"timeframe {interval}{range_info}"
            )


if __name__ == "__main__":
    main()
