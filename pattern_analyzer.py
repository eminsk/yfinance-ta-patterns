from typing import Iterable, Optional, Tuple

import pandas as pd
import talib


class PatternAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.pattern_functions = [f for f in dir(talib) if f.startswith("CDL")]

    def _normalize_pattern(self, pattern: str) -> str:
        pattern_upper = pattern.upper()
        return pattern_upper if pattern_upper.startswith("CDL") else f"CDL{pattern_upper}"

    def _normalize_dates(
        self, date: Optional[str], start_date: Optional[str], end_date: Optional[str]
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        tz = self.data.index.tz

        def convert(dt: Optional[str]) -> Optional[pd.Timestamp]:
            if dt is None:
                return None
            parsed = pd.to_datetime(dt)
            if parsed.tzinfo is None:
                parsed = parsed.tz_localize(tz)
            else:
                parsed = parsed.tz_convert(tz)
            return parsed.normalize()

        return convert(date), convert(start_date), convert(end_date)

    def get_signals(
        self,
        pattern: str,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Return non-zero signals for a single candlestick pattern with optional date filters."""
        normalized = self._normalize_pattern(pattern)
        if normalized not in self.pattern_functions:
            available = ", ".join(p.replace("CDL", "") for p in self.pattern_functions)
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

        if target_date:
            day_index = signals.index.normalize()
            signals = signals[day_index == target_date]

        if start_dt or end_dt:
            day_index = signals.index.normalize()
            mask = pd.Series(True, index=signals.index)
            if start_dt:
                mask &= day_index >= start_dt
            if end_dt:
                mask &= day_index <= end_dt
            signals = signals[mask]

        return signals

    def analyze_all_for_date(self, date: str) -> Iterable[str]:
        """Keep legacy all-patterns behavior for a specific date."""
        messages = []
        for pattern in self.pattern_functions:
            pattern_func = getattr(talib, pattern)
            try:
                result = pattern_func(
                    self.data["Open"],
                    self.data["High"],
                    self.data["Low"],
                    self.data["Close"],
                )
                values = result.loc[date]
                values_series = (
                    values
                    if isinstance(values, pd.Series)
                    else pd.Series([values], index=[pd.Timestamp(date)])
                )
                non_zero = values_series[values_series != 0]
                messages.append(
                    f"{pattern} on {date} (non-zero values):\n{non_zero.to_string()}"
                    if not non_zero.empty
                    else f"{pattern} on {date}: all values are 0"
                )
            except KeyError:
                messages.append(f"{pattern}: No data available for {date}")
            except Exception as e:
                messages.append(f"Error in {pattern}: {e}")
        return messages
