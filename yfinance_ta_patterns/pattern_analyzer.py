"""Pattern Analyzer for detecting TA-Lib candlestick patterns in market data."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from yfinance_ta_patterns.talib_compat import talib


class PatternAnalyzer:
    """Analyze OHLC market data for TA-Lib candlestick patterns."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.pattern_functions: list[str] = [f for f in dir(talib) if f.startswith("CDL")]

    def _normalize_pattern(self, pattern: str) -> str:
        pattern_upper = pattern.upper()
        return pattern_upper if pattern_upper.startswith("CDL") else f"CDL{pattern_upper}"

    def _normalize_dates(
        self,
        date: str | None,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
        idx = self.data.index
        tz = idx.tz if isinstance(idx, pd.DatetimeIndex) else None

        def convert(dt: str | None) -> pd.Timestamp | None:
            if dt is None:
                return None
            parsed = pd.to_datetime(dt)
            if parsed.tzinfo is None and tz is not None:
                parsed = parsed.tz_localize(tz)
            elif parsed.tzinfo is not None and tz is not None:
                parsed = parsed.tz_convert(tz)
            return parsed.normalize()

        return convert(date), convert(start_date), convert(end_date)

    def get_signals(
        self,
        pattern: str,
        date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Return non-zero signals for a single candlestick pattern with optional date filters."""
        normalized = self._normalize_pattern(pattern)
        if normalized not in self.pattern_functions:
            available = ", ".join(p.replace("CDL", "") for p in sorted(self.pattern_functions))
            raise ValueError(f"Unknown pattern '{pattern}'. Available: {available}")

        pattern_func = getattr(talib, normalized)
        result = pattern_func(
            self.data["Open"],
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
        )
        series = pd.Series(result, index=self.data.index, name=normalized)
        signals = series[series != 0]

        target_date, start_dt, end_dt = self._normalize_dates(date, start_date, end_date)

        if target_date is not None or start_dt is not None or end_dt is not None:
            idx = signals.index
            day_index = (
                idx.normalize()
                if isinstance(idx, pd.DatetimeIndex)
                else pd.to_datetime(idx).normalize()
            )

            if target_date is not None:
                signals = signals[day_index == target_date]
            else:
                mask = pd.Series(True, index=signals.index)
                if start_dt is not None:
                    mask &= day_index >= start_dt
                if end_dt is not None:
                    mask &= day_index <= end_dt
                signals = signals[mask]

        return signals

    def analyze_all_for_date(self, date: str) -> Iterable[str]:
        """Analyze and format pattern signals for a specific date."""
        messages: list[str] = []
        for pattern in sorted(self.pattern_functions):
            try:
                signals = self.get_signals(pattern, date=date)
                pattern_name = pattern.replace("CDL", "")
                if not signals.empty:
                    messages.append(
                        f"{pattern_name} on {date} (non-zero values):\n{signals.to_string()}"
                    )
                else:
                    messages.append(f"{pattern_name} on {date}: all values are 0")
            except Exception as exc:
                messages.append(f"Error in {pattern}: {exc}")
        return messages
