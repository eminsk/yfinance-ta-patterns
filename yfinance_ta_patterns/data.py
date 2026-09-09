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


# Standard ISO 4217 fiat and crypto currency 3-letter codes for Forex validation
_CURRENCY_CODES = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "CAD",
    "AUD",
    "NZD",
    "RUB",
    "CNH",
    "CNY",
    "SEK",
    "NOK",
    "SGD",
    "HKD",
    "TRY",
    "ZAR",
    "MXN",
    "PLN",
    "INR",
    "BRL",
    "KRW",
    "DKK",
    "THB",
    "IDR",
    "HUF",
    "CZK",
    "ILS",
    "CLP",
    "PHP",
    "AED",
    "COP",
    "SAR",
    "MYR",
    "RON",
}

_KNOWN_CRYPTO_SYMBOLS: set[str] = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "ADA",
    "BNB",
    "AVAX",
    "DOT",
    "MATIC",
    "LINK",
    "LTC",
    "BCH",
    "UNI",
    "NEAR",
    "SHIB",
    "TRX",
    "TON",
    "XLM",
    "ATOM",
    "XMR",
    "ETC",
    "ALGO",
    "FIL",
    "ICP",
    "APT",
    "SUI",
    "RENDER",
    "PEPE",
    "HBAR",
}


def normalize_ticker(symbol: str, asset_type: str = "auto") -> str:
    """Intelligently normalize symbol for Yahoo Finance API.

    Rules:
    - Normalizes slashes:
      - Crypto: "BTC/USD" -> "BTC-USD", "ETH/USD" -> "ETH-USD"
      - Forex: "EUR/USD" -> "EURUSD=X"
    - If asset_type is 'crypto' or auto-detected as crypto pair, format with hyphen (e.g. "BTC-USD").
    - If asset_type is 'forex' or auto-detected as valid currency pair, append '=X'.
    - If already formatted with suffix (=X, =F, -USD, ^) or standard stock ticker, preserve.
    """
    clean = symbol.strip().upper()

    # Explicit crypto handling: converts slashes to hyphen, preserves existing hyphens
    if asset_type.lower() == "crypto":
        if "/" in clean:
            parts = clean.split("/")
            return f"{parts[0]}-{parts[1]}"
        if "-" in clean:
            return clean
        for quote in ("USD", "EUR", "USDT", "USDC", "BTC"):
            if clean.endswith(quote) and len(clean) > len(quote):
                return f"{clean[: -len(quote)]}-{quote}"
        return clean

    # Slashes in auto or forex mode
    if "/" in clean:
        parts = clean.split("/")
        if len(parts) == 2:
            base, quote = parts[0], parts[1]
            if asset_type.lower() == "auto" and base in _KNOWN_CRYPTO_SYMBOLS:
                return f"{base}-{quote}"
            if base in _CURRENCY_CODES and quote in _CURRENCY_CODES:
                return f"{base}{quote}=X"
            if len(base) == 3 and len(quote) == 3:
                clean = f"{base}{quote}"

    if asset_type.lower() == "forex":
        return clean if clean.endswith("=X") else f"{clean}=X"

    if asset_type.lower() == "auto":
        # Check if already has a suffix or special prefix
        if clean.endswith("=X") or clean.endswith("=F") or clean.startswith("^") or "-" in clean:
            return clean

        # If it matches known currency pair or valid 6-letter currency code pair
        if clean in _COMMON_FOREX_PAIRS:
            return f"{clean}=X"

        if len(clean) == 6 and clean.isalpha():
            base, quote = clean[:3], clean[3:]
            if base in _KNOWN_CRYPTO_SYMBOLS and (
                quote in _CURRENCY_CODES or quote in ("USDT", "USDC")
            ):
                return f"{base}-{quote}"
            if base in _CURRENCY_CODES and quote in _CURRENCY_CODES:
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

INTERVAL_DELTAS: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "2m": pd.Timedelta(minutes=2),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "60m": pd.Timedelta(hours=1),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta("1D"),
    "1wk": pd.Timedelta("7D"),
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

    # Check for non-finite values (np.inf, -np.inf, nan)
    finite_mask = np.isfinite(data[req_cols]).all(axis=1)
    if strict and not finite_mask.all():
        raise ValueError("Corrupted OHLC data: non-finite values (NaN or inf/-inf) found.")

    cleaned = data.loc[cast(Any, finite_mask)].copy()

    # Check for non-positive prices
    bad_prices = (cleaned[req_cols] <= 0).any(axis=1)
    if bad_prices.any():
        if strict:
            raise ValueError("Corrupted OHLC data: non-positive prices found.")
        cleaned = cleaned[~bad_prices]

    # Check bar geometry invariants if High and Low are present
    if "High" in cleaned.columns and "Low" in cleaned.columns:
        broken_bars = cleaned["High"] < cleaned["Low"]
        compare_cols = [c for c in ["Open", "Close"] if c in cleaned.columns]
        if compare_cols:
            top = cleaned[compare_cols].max(axis=1)
            bot = cleaned[compare_cols].min(axis=1)
            broken_bars = broken_bars | (cleaned["High"] < top) | (cleaned["Low"] > bot)

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

        # Check input bar completeness per bucket (Issue 3: YF-003, Screenshot 2)
        bar_counts = data.resample(self._resample_rule, origin="epoch")["Close"].count()
        is_crypto = (
            "-USD" in self.ticker or "-EUR" in self.ticker or self.asset_type.lower() == "crypto"
        )
        is_forex = self.asset_type.lower() == "forex" or self.ticker.endswith("=X")

        def _is_bucket_complete(ts: pd.Timestamp, count: int) -> bool:
            if is_crypto:
                return count >= 4
            if is_forex:
                day = ts.dayofweek
                hour = ts.hour
                # Boundary sessions (Sunday open or Friday close) allow partial
                is_boundary = (day == 6 and hour >= 20) or (day == 4 and hour >= 20)
                if is_boundary:
                    return count >= 1
                return count >= 4
            # Stocks and other assets: partial session blocks allowed
            return count >= 1

        valid_buckets = [
            ts
            for ts, count in bar_counts.items()
            if _is_bucket_complete(cast(pd.Timestamp, ts), int(count))
        ]
        resampled = resampled.loc[resampled.index.isin(valid_buckets)]

        if (
            self.closed_only
            and not resampled.empty
            and isinstance(resampled.index, pd.DatetimeIndex)
        ):
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

    def process(self, data: pd.DataFrame, now_utc: pd.Timestamp | None = None) -> pd.DataFrame:
        """Normalize column indexing, validate OHLC, resample in UTC, and convert timezone."""
        if data.empty:
            return data

        # Drop extra MultiIndex level if present (yfinance >= 0.2.40)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Check for duplicate timestamps in index
        if hasattr(data.index, "has_duplicates") and data.index.has_duplicates:
            raise ValueError(
                "Corrupted market data: duplicate candle timestamps detected in index."
            )

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

        # Universal closed-only filtering for all intervals (Issue 1: YF-001, Screenshot 1)
        if self.closed_only and not data.empty and isinstance(data.index, pd.DatetimeIndex):
            if self.interval == "1mo":
                candle_ends = data.index + pd.DateOffset(months=1)
            else:
                delta = INTERVAL_DELTAS.get(self.interval, pd.Timedelta("1D"))
                candle_ends = data.index + delta
            now = now_utc if now_utc is not None else pd.Timestamp.now(tz=pytz.UTC)
            if candle_ends.tz is not None and now.tz is not None:
                now = now.tz_convert(candle_ends.tz)
            elif candle_ends.tz is not None and now.tz is None:
                now = now.tz_localize(pytz.UTC).tz_convert(candle_ends.tz)
            elif candle_ends.tz is None and now.tz is not None:
                now = now.tz_localize(None)
            data = data.loc[candle_ends <= now]

        return data

    def get_data(self) -> pd.DataFrame:
        """Full pipeline: fetch raw OHLCV and process with validation and resampling."""
        return self.process(self.fetch())

    def load_data(self) -> pd.DataFrame:
        """Alias for get_data to ensure CLI and script compatibility."""
        return self.get_data()
