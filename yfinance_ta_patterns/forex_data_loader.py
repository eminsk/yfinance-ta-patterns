"""Data loader module for fetching and preparing Yahoo Finance market data."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytz
import yfinance as yf


class ForexDataLoader:
    """Generic Data Loader for Yahoo Finance tickers (forex, stocks, crypto, etc.)."""

    def __init__(
        self,
        symbol: str,
        period: str = "60d",
        interval: str = "15m",
        timezone: str = "Europe/Moscow",
    ) -> None:
        # Allow full tickers or append suffix if missing
        suffix = "=X"
        self.ticker: str = symbol if symbol.endswith(suffix) else f"{symbol}{suffix}"
        self.period: str = period
        self.interval: str = interval
        self.timezone: str = timezone
        self._download_interval: str = self._resolve_download_interval(interval)
        self._resample_rule: str | None = "4h" if interval == "4h" else None

    def _resolve_download_interval(self, interval: str) -> str:
        """Map custom intervals to supported yfinance intervals."""
        if interval == "4h":
            return "1h"  # fetch 1h data and resample to 4h
        return interval

    def fetch(self) -> pd.DataFrame:
        """Fetch raw data via yfinance."""
        data = yf.download(
            self.ticker,
            period=self.period,
            interval=self._download_interval,
            auto_adjust=True,
        )
        return cast(pd.DataFrame, data)

    def _resample_if_needed(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample OHLCV data when the requested interval is not natively supported."""
        if not self._resample_rule or data.empty:
            return data

        agg: dict[Any, Any] = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
        if "Adj Close" in data.columns:
            agg["Adj Close"] = "last"
        if "Volume" in data.columns:
            agg["Volume"] = "sum"

        # Keep any other columns by taking the last value in the window
        for col in data.columns:
            col_str = str(col)
            if col_str not in agg:
                agg[col_str] = "last"

        resampled = cast(pd.DataFrame, data.resample(self._resample_rule).agg(agg).dropna())
        return resampled

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize columns and timezone."""
        if data.empty:
            return data

        # Drop extra MultiIndex level if present (common in newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        idx = data.index
        if isinstance(idx, pd.DatetimeIndex):
            if idx.tz is None:
                data.index = idx.tz_localize(pytz.UTC).tz_convert(self.timezone)
            else:
                data.index = idx.tz_convert(pytz.UTC).tz_convert(self.timezone)

        return data

    def get_data(self) -> pd.DataFrame:
        """Full pipeline: fetch then process."""
        processed = self.process(self.fetch())
        return self._resample_if_needed(processed)


if __name__ == "__main__":
    # Example usage
    loader = ForexDataLoader("EURUSD=X")
    data = loader.get_data()
    print(data)
