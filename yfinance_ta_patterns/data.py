"""Universal market data loader module supporting Stocks, Forex, Crypto, Indices, and Commodities."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except (ImportError, ModuleNotFoundError):
    HAS_SKLEARN = False


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


def validate_ohlc(data: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate OHLC price integrity, drop corrupted rows, and check invariants.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV market dataframe
    strict : bool
        If True, raises ValueError on bad data or missing OHLC; if False, validates available columns

    Returns:
    --------
    pd.DataFrame: Cleaned and validated dataframe
    """
    if data.empty:
        return data

    if strict and hasattr(data.index, "has_duplicates") and data.index.has_duplicates:
        raise ValueError("Corrupted OHLC data: duplicate candle timestamps detected in index.")

    req_cols = [c for c in ["Open", "High", "Low", "Close"] if c in data.columns]

    if strict and len(req_cols) < 4:
        missing = [c for c in ["Open", "High", "Low", "Close"] if c not in data.columns]
        raise ValueError(f"Missing required OHLC columns: {missing}")

    if not req_cols:
        return data

    # Check for non-finite values (np.inf, -np.inf)
    has_inf = np.isinf(data[req_cols]).any().any()
    if strict and has_inf:
        raise ValueError("Corrupted OHLC data: non-finite (inf/-inf) values found.")

    finite_mask = np.isfinite(data[req_cols]).all(axis=1)
    cleaned = data.loc[cast(Any, finite_mask)].copy()

    # Check for non-positive prices
    bad_prices = (cleaned[req_cols] <= 0).any(axis=1)
    if bad_prices.any():
        if strict:
            raise ValueError("Corrupted OHLC data: non-positive prices found.")
        cleaned = cleaned[~bad_prices]

    # Check bar geometry invariants if High and Low are present
    if "High" in cleaned.columns and "Low" in cleaned.columns:
        compare_cols = [c for c in ["Open", "Close"] if c in cleaned.columns]
        if compare_cols:
            top = cleaned[compare_cols].max(axis=1)
            bot = cleaned[compare_cols].min(axis=1)
            broken_bars = (
                (cleaned["High"] < top)
                | (cleaned["Low"] > bot)
                | (cleaned["High"] < cleaned["Low"])
            )
            if broken_bars.any():
                if strict:
                    raise ValueError("Inconsistent OHLC bar geometry detected.")
                cleaned = cleaned[~broken_bars]

    return cast(pd.DataFrame, cleaned)


class MarketDataLoader:
    """Universal market data loader for Yahoo Finance tickers."""

    def __init__(
        self,
        symbol: str,
        period: str = "60d",
        interval: str = "15m",
        timezone: str = "Europe/Moscow",
        asset_type: str = "auto",
        start: str | None = None,
        end: str | None = None,
        auto_adjust: bool = False,
        repair: bool = True,
        timeframe: str | None = None,
        closed_only: bool = True,
    ) -> None:
        """Initialize data loader with symbol and timeframe parameters."""
        effective_interval = timeframe if timeframe is not None else interval
        norm_interval = normalize_interval(effective_interval)
        self.symbol: str = symbol
        self.asset_type: str = asset_type
        self.closed_only: bool = closed_only
        self.ticker: str = normalize_ticker(symbol, asset_type=asset_type)
        # Yahoo Finance restricts 1m data to the last 7-8 days (2m is supported up to 60d).
        # Auto-adjust period to '7d' if default '60d' is passed with 1m.
        self.period: str = "7d" if (norm_interval == "1m" and period == "60d") else period
        self.interval: str = norm_interval
        self.timezone: str = timezone
        self.start: str | None = start
        self.end: str | None = end
        self.start_date: str | None = start
        self.end_date: str | None = end
        self.auto_adjust: bool = auto_adjust
        self.repair: bool = repair
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
        start_val = self.start or self.start_date
        end_val = self.end or self.end_date
        repair_opt = self.repair and HAS_SKLEARN

        if start_val or end_val:
            data = yf.download(
                self.ticker,
                start=start_val,
                end=end_val,
                interval=self._download_interval,
                auto_adjust=self.auto_adjust,
                repair=repair_opt,
                progress=False,
            )
        else:
            data = yf.download(
                self.ticker,
                period=self.period,
                interval=self._download_interval,
                auto_adjust=self.auto_adjust,
                repair=repair_opt,
                progress=False,
            )
        return cast(pd.DataFrame, data)

    def _resample_if_needed(
        self, data: pd.DataFrame, now_utc: pd.Timestamp | None = None
    ) -> pd.DataFrame:
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

        # Anchor resampling in UTC (origin='epoch') to preserve institutional 4h candle bounds
        resampled = data.resample(self._resample_rule, origin="epoch").agg(agg).dropna()
        if "Volume" in resampled.columns and (resampled["Volume"] > 0).any():
            resampled = resampled[resampled["Volume"] > 0]

        if self.closed_only and not resampled.empty and isinstance(resampled.index, pd.DatetimeIndex):
            ends = resampled.index + pd.Timedelta(self._resample_rule)
            now = now_utc if now_utc is not None else pd.Timestamp.now(tz=pytz.UTC)
            ends_tz = getattr(ends, "tz", None)
            if ends_tz is not None and now.tz is None:
                now = now.tz_localize(pytz.UTC)
            elif ends_tz is None and now.tz is not None:
                now = now.tz_localize(None)

            input_td = pd.Timedelta(self._download_interval)
            max_input_time = cast(pd.Timestamp, data.index.max())
            max_input_end = max_input_time + input_td
            if ends_tz is not None and max_input_end.tz is None:
                max_input_end = max_input_end.tz_localize(pytz.UTC)
            elif ends_tz is None and max_input_end.tz is not None:
                max_input_end = max_input_end.tz_localize(None)

            cutoff = min(now, max_input_end)
            resampled = resampled.loc[ends <= cutoff]

        return cast(pd.DataFrame, resampled)

    def process(
        self, data: pd.DataFrame, now_utc: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Normalize column indexing, validate OHLC, resample in UTC, and convert timezone."""
        if data.empty:
            return data

        # Drop extra MultiIndex level if present (yfinance >= 0.2.40)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Check for duplicate timestamps in index
        if hasattr(data.index, "has_duplicates") and data.index.has_duplicates:
            raise ValueError("Corrupted market data: duplicate candle timestamps detected in index.")

        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        data = validate_ohlc(data)
        if data.empty:
            return data

        idx = data.index
        if isinstance(idx, pd.DatetimeIndex):
            if idx.tz is None:
                data.index = idx.tz_localize(pytz.UTC)
            else:
                data.index = idx.tz_convert(pytz.UTC)

        # Resample in UTC before local timezone conversion (Issue 8)
        if self._resample_rule:
            data = self._resample_if_needed(data, now_utc=now_utc)

        # Convert to target user timezone
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.tz_convert(self.timezone)

        return data

    def get_data(self) -> pd.DataFrame:
        """Full pipeline: fetch raw OHLCV and process with validation and resampling."""
        return self.process(self.fetch())

    def load_data(self) -> pd.DataFrame:
        """Alias for get_data to ensure CLI and script compatibility."""
        return self.get_data()
