"""Universal market data loader module supporting Stocks, Forex, Crypto, Indices, and Commodities."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytz
import yfinance as yf

# Common standard 6-character currency pairs
_COMMON_FOREX_PAIRS = {
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "EURCHF",
    "AUDJPY",
    "EURAUD",
    "CADJPY",
    "GBPAUD",
    "NZDJPY",
    "AUDNZD",
    "AUDCAD",
    "USDRUB",
    "EURRUB",
}


def normalize_ticker(symbol: str, asset_type: str = "auto") -> str:
    """Intelligently normalize symbol for Yahoo Finance API.

    Rules:
    - If asset_type is 'forex' or auto-detected as a 6-letter currency pair, append '=X'.
    - If already formatted with suffix (=X, =F, -USD, ^) or standard stock ticker, preserve.
    """
    clean = symbol.strip().upper()

    if asset_type.lower() == "forex":
        return clean if clean.endswith("=X") else f"{clean}=X"

    if asset_type.lower() == "auto":
        # Check if already has a suffix or special prefix
        if clean.endswith("=X") or clean.endswith("=F") or clean.startswith("^") or "-" in clean:
            return clean

        # If it matches known currency pair or 6-letter alphabetic forex code
        if clean in _COMMON_FOREX_PAIRS or (
            len(clean) == 6 and clean.isalpha() and not clean.startswith(("AAPL", "GOOG"))
        ):
            return f"{clean}=X"

    return clean



TIMEFRAME_MAP: dict[str, str] = {
    "M1": "1m",
    "M2": "2m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "W1": "1wk",
    "MN1": "1mo",
}


def normalize_interval(interval: str) -> str:
    """Normalize interval notation (e.g. 'm1', 'M1' -> '1m', 'h4', 'H4' -> '4h')."""
    clean = interval.strip()
    upper = clean.upper()
    if upper in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[upper]
    lower = clean.lower()
    if lower in ("m1", "m2", "m5", "m15", "m30"):
        return f"{lower[1:]}m"
    if lower in ("h1", "h4"):
        return f"{lower[1:]}h"
    if lower == "d1":
        return "1d"
    return lower


class MarketDataLoader:
    """Universal market data loader for Yahoo Finance tickers."""

    def __init__(
        self,
        symbol: str,
        period: str = "60d",
        interval: str = "15m",
        timezone: str = "Europe/Moscow",
        asset_type: str = "auto",
    ) -> None:
        """Initialize data loader with symbol and timeframe parameters."""
        norm_interval = normalize_interval(interval)
        self.symbol: str = symbol
        self.asset_type: str = asset_type
        self.ticker: str = normalize_ticker(symbol, asset_type=asset_type)
        # Yahoo Finance restricts 1m/2m data to the last 7-8 days.
        # Auto-adjust period to '7d' if default '60d' is passed with 1m/2m.
        if norm_interval in ("1m", "2m") and period == "60d":
            self.period: str = "7d"
        else:
            self.period: str = period
        self.interval: str = norm_interval
        self.timezone: str = timezone
        self._download_interval: str = self._resolve_download_interval(norm_interval)
        self._resample_rule: str | None = "4h" if norm_interval == "4h" else None

    @staticmethod
    def _resolve_download_interval(interval: str) -> str:
        """Map custom intervals to supported yfinance download intervals."""
        if interval == "4h":
            return "1h"  # fetch 1h data and resample to 4h
        return interval

    def fetch(self) -> pd.DataFrame:
        """Fetch raw market data via yfinance."""
        data = yf.download(
            self.ticker,
            period=self.period,
            interval=self._download_interval,
            auto_adjust=True,
            progress=False,
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

        for col in data.columns:
            col_str = str(col)
            if col_str not in agg:
                agg[col_str] = "last"

        return cast(pd.DataFrame, data.resample(self._resample_rule).agg(agg).dropna())

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize column indexing and apply timezone conversion."""
        if data.empty:
            return data

        # Drop extra MultiIndex level if present (yfinance >= 0.2.40)
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
        """Full pipeline: fetch, process, and resample if needed."""
        processed = self.process(self.fetch())
        return self._resample_if_needed(processed)
