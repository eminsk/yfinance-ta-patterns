"""Universal market data loader module supporting Stocks, Forex, Crypto, Indices, and Commodities."""

from __future__ import annotations

import datetime
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
    "FET",
    "TAO",
    "INJ",
    "TIA",
    "SEI",
    "KAS",
}


_KNOWN_CRYPTO_QUOTES: tuple[str, ...] = (
    "USDT",
    "USDC",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CAD",
    "AUD",
    "BTC",
    "ETH",
)

_EXCHANGE_SUFFIX_MAP: dict[str, str] = {
    # Germany / Eurozone
    ".DE": "EUR",
    ".F": "EUR",
    ".PA": "EUR",
    ".AS": "EUR",
    ".BR": "EUR",
    ".LS": "EUR",
    ".MI": "EUR",
    ".MC": "EUR",
    ".VI": "EUR",
    ".IR": "EUR",
    ".HE": "EUR",
    ".AT": "EUR",
    # UK
    ".L": "GBp",
    ".IL": "USD",
    # Canada
    ".TO": "CAD",
    ".V": "CAD",
    ".CN": "CAD",
    # Australia & New Zealand
    ".AX": "AUD",
    ".NZ": "NZD",
    # Japan
    ".T": "JPY",
    # Hong Kong & China
    ".HK": "HKD",
    ".SS": "CNY",
    ".SZ": "CNY",
    # Switzerland
    ".SW": "CHF",
    # Nordic
    ".ST": "SEK",
    ".OL": "NOK",
    ".CO": "DKK",
    # Global
    ".KS": "KRW",
    ".KQ": "KRW",
    ".TW": "TWD",
    ".TWO": "TWD",
    ".SA": "BRL",
    ".MX": "MXN",
    ".TA": "ILS",
    ".SI": "SGD",
    ".SG": "SGD",
    ".JK": "IDR",
    ".BK": "THB",
    ".KL": "MYR",
    ".NS": "INR",
    ".BO": "INR",
    ".IS": "TRY",
    ".JO": "ZAR",
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
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    f"Invalid crypto ticker format: '{symbol}'. Expected format like 'BTC/USD' or 'BTC-USD'."
                )
            return f"{parts[0]}-{parts[1]}"
        if "-" in clean:
            parts = clean.split("-")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    f"Invalid crypto ticker format: '{symbol}'. Expected format like 'BTC-USD'."
                )
            return clean
        for quote in _KNOWN_CRYPTO_QUOTES:
            if clean.endswith(quote) and len(clean) > len(quote):
                return f"{clean[: -len(quote)]}-{quote}"
        return clean

    # Slashes in auto or forex mode
    if "/" in clean:
        parts = clean.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Invalid ticker format: '{symbol}'. Slashes must separate exactly two symbols (e.g. 'EUR/USD' or 'BTC/USD')."
            )
        base, quote = parts[0], parts[1]
        if base in _CURRENCY_CODES and quote in _CURRENCY_CODES:
            return f"{base}{quote}=X"
        if (
            base in _KNOWN_CRYPTO_SYMBOLS
            or quote in _KNOWN_CRYPTO_SYMBOLS
            or quote in _KNOWN_CRYPTO_QUOTES
            or quote in _CURRENCY_CODES
            or quote in ("USDT", "USDC")
        ):
            return f"{base}-{quote}"
        if len(base) == 3 and len(quote) == 3:
            clean = f"{base}{quote}"

    if asset_type.lower() == "forex":
        if "-" in clean:
            parts = clean.split("-")
            if len(parts) == 2 and parts[0] in _CURRENCY_CODES and parts[1] in _CURRENCY_CODES:
                return f"{parts[0]}{parts[1]}=X"
        return clean if clean.endswith("=X") else f"{clean}=X"

    if asset_type.lower() == "auto":
        # Check if hyphen separates two ISO currency codes (e.g. EUR-USD -> EURUSD=X)
        if "-" in clean:
            parts = clean.split("-")
            if len(parts) == 2 and parts[0] in _CURRENCY_CODES and parts[1] in _CURRENCY_CODES:
                return f"{parts[0]}{parts[1]}=X"

        # Check if already has a suffix or special prefix
        if clean.endswith("=X") or clean.endswith("=F") or clean.startswith("^") or "-" in clean:
            return clean

        # Check unseparated crypto ticker first (e.g. BTCUSDT, DOGEUSD, BTCUSDC, SHIBUSDT, BTCGBP)
        for quote in _KNOWN_CRYPTO_QUOTES:
            if clean.endswith(quote) and len(clean) > len(quote):
                base = clean[: -len(quote)]
                if base in _KNOWN_CRYPTO_SYMBOLS:
                    return f"{base}-{quote}"

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


def classify_asset(symbol: str, asset_type: str = "auto") -> str:
    """Classify an asset into 'crypto', 'forex', 'commodity', 'index', or 'stock'.

    Works seamlessly whether symbol is raw ('BTC/EUR', 'ETH-BTC', 'EURUSD=X', 'BTCUSDT') or normalized.
    """
    if asset_type and asset_type.lower() != "auto":
        return asset_type.lower()

    # Normalize first to detect unseparated cryptos like BTCUSDT, DOGEUSD, SHIBUSDT
    norm = normalize_ticker(symbol, asset_type="auto")
    clean = norm.strip().upper()
    if clean.endswith("=F"):
        return "commodity"
    if clean.startswith("^"):
        return "index"
    if clean.endswith("=X"):
        return "forex"

    # Check separators: '-' or '/'
    sep = "-" if "-" in clean else ("/" if "/" in clean else None)
    if sep:
        parts = clean.split(sep)
        if len(parts) == 2:
            base, quote = parts[0], parts[1]
            if base in _CURRENCY_CODES and quote in _CURRENCY_CODES:
                return "forex"
            if (
                base in _KNOWN_CRYPTO_SYMBOLS
                or quote in _KNOWN_CRYPTO_SYMBOLS
                or quote in {"USDT", "USDC"}
            ):
                return "crypto"
            if (
                (quote in _CURRENCY_CODES or quote in _KNOWN_CRYPTO_QUOTES)
                and base not in _CURRENCY_CODES
                and len(quote) >= 3
            ):
                return "crypto"

    # Check 6-letter FX e.g. EURUSD
    if clean in _COMMON_FOREX_PAIRS:
        return "forex"
    if len(clean) == 6 and clean.isalpha():
        base, quote = clean[:3], clean[3:]
        if base in _KNOWN_CRYPTO_SYMBOLS and (
            quote in _CURRENCY_CODES or quote in {"USDT", "USDC", "BTC", "ETH"}
        ):
            return "crypto"
        if base in _CURRENCY_CODES and quote in _CURRENCY_CODES:
            return "forex"

    return "stock"


def resolve_asset_currencies(symbol: str, asset_type: str = "auto") -> tuple[str, str]:
    """Resolve base and quote currency for any asset symbol.

    Returns (base_currency, quote_currency).
    Examples:
      'BTC-EUR' -> ('BTC', 'EUR')
      'ETH-BTC' -> ('ETH', 'BTC')
      'EURUSD=X' -> ('EUR', 'USD')
      'EUR/GBP' -> ('EUR', 'GBP')
      'USDJPY' -> ('USD', 'JPY')
      'SAP.DE' -> ('SAP.DE', 'EUR')
      'VOD.L' -> ('VOD.L', 'GBp')
      'AAPL' -> ('AAPL', 'USD')
    """
    clean = symbol.strip().upper()
    if clean.endswith("=X"):
        clean_fx = clean[:-2]
        if len(clean_fx) == 6:
            return clean_fx[:3], clean_fx[3:]

    # Check exchange suffix for stocks (e.g. .DE -> EUR, .L -> GBp, .TO -> CAD)
    for suffix, quote_curr in _EXCHANGE_SUFFIX_MAP.items():
        if clean.endswith(suffix):
            return clean, quote_curr

    # If explicitly or auto-classified as stock, treat entire ticker as base and USD as quote
    a_type = asset_type.lower() if asset_type else "auto"
    if a_type == "stock" or (a_type == "auto" and classify_asset(clean, "auto") == "stock"):
        return clean, "USD"

    sep = "-" if "-" in clean else ("/" if "/" in clean else None)
    if sep:
        parts = clean.split(sep)
        if len(parts) == 2:
            base, quote = parts[0], parts[1]
            if (
                base in _KNOWN_CRYPTO_SYMBOLS
                or quote in _KNOWN_CRYPTO_SYMBOLS
                or quote in _KNOWN_CRYPTO_QUOTES
                or (base in _CURRENCY_CODES and quote in _CURRENCY_CODES)
            ):
                return base, quote
            if a_type in ("crypto", "forex"):
                return base, quote

    # Check unseparated crypto (e.g. BTCUSDT, DOGEUSD)
    for quote in _KNOWN_CRYPTO_QUOTES:
        if clean.endswith(quote) and len(clean) > len(quote):
            base = clean[: -len(quote)]
            if base in _KNOWN_CRYPTO_SYMBOLS:
                return base, quote

    if len(clean) == 6 and clean.isalpha():
        base, quote = clean[:3], clean[3:]
        if (base in _CURRENCY_CODES and quote in _CURRENCY_CODES) or (
            base in _KNOWN_CRYPTO_SYMBOLS and quote in _KNOWN_CRYPTO_SYMBOLS
        ):
            return base, quote

    # Default to USD quote for standard equities, commodities, etc.
    return clean, "USD"


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
    "90m": pd.Timedelta(minutes=90),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta("1D"),
    "5d": pd.Timedelta("5D"),
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


def _get_market_session_hours(
    symbol: str, d: datetime.date
) -> tuple[str, datetime.time, datetime.time]:
    """Get exchange timezone, open time, and close time for an equity symbol."""
    clean_sym = symbol.strip().upper()
    if clean_sym.endswith(".L"):
        return "Europe/London", datetime.time(8, 0), datetime.time(16, 30)
    if any(clean_sym.endswith(sfx) for sfx in (".DE", ".PA", ".AS", ".BR", ".MI", ".MC", ".VI", ".HE", ".F", ".AT")):
        return "Europe/Berlin", datetime.time(9, 0), datetime.time(17, 30)
    if clean_sym.endswith(".T"):
        close_min = 30 if d >= datetime.date(2024, 11, 5) else 0
        return "Asia/Tokyo", datetime.time(9, 0), datetime.time(15, close_min)
    if clean_sym.endswith(".HK"):
        return "Asia/Hong_Kong", datetime.time(9, 30), datetime.time(16, 0)
    # Default US Equities
    is_early_close = False
    if d.month == 11 and d.weekday() == 4 and 23 <= d.day <= 29:
        is_early_close = True  # Black Friday
    elif d.month == 12 and d.day == 24 and d.weekday() < 5:
        is_early_close = True  # Christmas Eve
    elif d.month == 7 and d.day == 3 and d.weekday() < 4:
        is_early_close = True  # Day before July 4th
    close_hour = 13 if is_early_close else 16
    return "America/New_York", datetime.time(9, 30), datetime.time(close_hour, 0)


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

        # Check input bar completeness per bucket (Issue 3: YF-003, Screenshot 2 & Screenshot 5)
        bar_counts = data.resample(self._resample_rule, origin="epoch")["Close"].count()
        resolved_type = classify_asset(self.ticker, self.asset_type)
        is_crypto = resolved_type == "crypto"
        is_forex = resolved_type == "forex"

        def _is_bucket_complete(ts: pd.Timestamp, count: int) -> bool:
            rule_str = str(self._resample_rule or "4h")
            bucket_start = ts
            bucket_end = ts + pd.Timedelta(rule_str)
            dt_index = cast(pd.DatetimeIndex, data.index)
            bucket_stamps = dt_index[(dt_index >= bucket_start) & (dt_index < bucket_end)]
            if len(bucket_stamps) == 0:
                return False

            if is_crypto:
                # Crypto trades 24/7. Exactly 4 hourly bars expected on the regular 1h grid:
                # ts, ts+1h, ts+2h, ts+3h.
                # Must have exactly 4 bars, no irregular displaced timestamps, and match expected slots 1-to-1.
                if len(bucket_stamps) != 4:
                    return False
                expected_crypto = [bucket_start + pd.Timedelta(hours=k) for k in range(4)]
                matched_indices = set()
                for exp in expected_crypto:
                    found = None
                    for i, act in enumerate(bucket_stamps):
                        if i in matched_indices:
                            continue
                        if abs((act - exp).total_seconds()) <= 300:  # within 5 minutes of the hour
                            found = i
                            break
                    if found is None:
                        return False
                    matched_indices.add(found)
                return True

            if is_forex:
                open_slots: list[pd.Timestamp] = []
                for k in range(4):
                    slot = ts + pd.Timedelta(hours=k)
                    slot_ny = (
                        slot.tz_convert("America/New_York")
                        if slot.tz is not None
                        else slot.tz_localize("UTC").tz_convert("America/New_York")
                    )
                    wd = slot_ny.weekday()
                    hr = slot_ny.hour
                    is_open = (
                        (wd == 6 and hr >= 17)
                        or (wd in (0, 1, 2, 3))
                        or (wd == 4 and hr < 17)
                    )
                    if is_open:
                        open_slots.append(slot)
                if len(open_slots) == 0:
                    return False
                if len(bucket_stamps) < len(open_slots):
                    return False
                matched_fx_indices = set()
                for exp in open_slots:
                    found = None
                    for i, act in enumerate(bucket_stamps):
                        if i in matched_fx_indices:
                            continue
                        if abs((act - exp).total_seconds()) <= 300:
                            found = i
                            break
                    if found is None:
                        return False
                    matched_fx_indices.add(found)
                if len(bucket_stamps) > 1:
                    diffs = np.diff(bucket_stamps.values).astype("timedelta64[m]").astype(int)
                    if np.any(diffs > 75):
                        return False
                return True

            # Stocks and other assets:
            # Build expected hourly session grid from exchange open_time to close_time.
            # Compare actual timestamps 1-to-1 without bar reuse.
            dates_to_check = {bucket_start.date(), (bucket_end - pd.Timedelta(seconds=1)).date()}
            expected_stock_slots: list[pd.Timestamp] = []
            for d in sorted(dates_to_check):
                if d.weekday() >= 5:
                    continue  # Weekend
                tz_name, open_time, close_time = _get_market_session_hours(self.ticker, d)
                s_open = pd.Timestamp(
                    year=d.year,
                    month=d.month,
                    day=d.day,
                    hour=open_time.hour,
                    minute=open_time.minute,
                    tz=tz_name,
                )
                s_close = pd.Timestamp(
                    year=d.year,
                    month=d.month,
                    day=d.day,
                    hour=close_time.hour,
                    minute=close_time.minute,
                    tz=tz_name,
                )
                curr = s_open
                while curr < s_close:
                    curr_utc = (
                        curr.tz_convert("UTC")
                        if bucket_start.tz is not None
                        else curr.tz_convert("UTC").tz_localize(None)
                    )
                    if bucket_start <= curr_utc < bucket_end:
                        expected_stock_slots.append(curr_utc)
                    curr += pd.Timedelta(hours=1)

            if len(expected_stock_slots) > 0:
                if len(bucket_stamps) < len(expected_stock_slots):
                    return False
                matched_stock_indices = set()
                for exp in expected_stock_slots:
                    found = None
                    for i, act in enumerate(bucket_stamps):
                        if i in matched_stock_indices:
                            continue
                        if abs((act - exp).total_seconds()) <= 300:
                            found = i
                            break
                    if found is None:
                        return False
                    matched_stock_indices.add(found)

            if len(bucket_stamps) > 1:
                diffs = np.diff(bucket_stamps.values).astype("timedelta64[m]").astype(int)
                if np.any(diffs > 75):
                    return False

            return True

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
        orig_dates = list(idx.date) if isinstance(idx, pd.DatetimeIndex) else None
        if isinstance(idx, pd.DatetimeIndex):
            if idx.tz is None:
                data.index = idx.tz_localize(pytz.UTC)
            else:
                data.index = idx.tz_convert(pytz.UTC)

        # Resample in UTC before local timezone conversion (Issue 8)
        if self._resample_rule:
            data = self._resample_if_needed(data, now_utc=now_utc)

        # Universal closed-only filtering in UTC / market session time before user timezone conversion
        if self.closed_only and not data.empty and isinstance(data.index, pd.DatetimeIndex):
            now = now_utc if now_utc is not None else pd.Timestamp.now(tz=pytz.UTC)
            if now.tz is None:
                now = now.tz_localize(pytz.UTC)

            if self.interval == "1mo":
                candle_ends = data.index + pd.DateOffset(months=1)
            elif self.interval == "3mo":
                candle_ends = data.index + pd.DateOffset(months=3)
            elif self.interval in ("1d", "1D"):
                # Session-aware daily candle close
                asset_class = classify_asset(self.ticker, self.asset_type)
                clean_sym = self.ticker.strip().upper()

                candle_end_list = []
                for i, ts in enumerate(data.index):
                    if asset_class == "crypto":
                        # 24/7 calendar: closes at next day 00:00 UTC
                        d = ts.date()
                        close_ts = pd.Timestamp(d, tz=pytz.UTC) + pd.Timedelta(days=1)
                    elif asset_class == "forex":
                        # Forex daily rollover: 17:00 America/New_York
                        d = orig_dates[i] if orig_dates is not None and i < len(orig_dates) else ts.date()
                        close_ts = pd.Timestamp(
                            year=d.year, month=d.month, day=d.day, hour=17, minute=0, tz="America/New_York"
                        ).tz_convert(pytz.UTC)
                    else:
                        # Stock / Commodity / Index
                        # Asian exchanges and international markets: determine session date in local exchange timezone
                        tz_name, _open_t, _ = _get_market_session_hours(clean_sym, ts.date())
                        ts_local = ts.tz_convert(tz_name) if ts.tz is not None else ts
                        d = ts_local.date()
                        if orig_dates is not None and i < len(orig_dates):
                            d = orig_dates[i]
                        tz_name, _open_t, close_t = _get_market_session_hours(clean_sym, d)
                        close_ts = pd.Timestamp(
                            year=d.year,
                            month=d.month,
                            day=d.day,
                            hour=close_t.hour,
                            minute=close_t.minute,
                            tz=tz_name,
                        ).tz_convert(pytz.UTC)

                    candle_end_list.append(close_ts)

                candle_ends = pd.DatetimeIndex(candle_end_list)
            elif self.interval in INTERVAL_DELTAS:
                delta = INTERVAL_DELTAS[self.interval]
                candle_ends = data.index + delta
            else:
                raise ValueError(f"Unsupported interval for closed_only filtering: {self.interval}")

            if candle_ends.tz is not None and now.tz is not None:
                now_cmp = now.tz_convert(candle_ends.tz)
            elif candle_ends.tz is not None and now.tz is None:
                now_cmp = now.tz_localize(pytz.UTC).tz_convert(candle_ends.tz)
            elif candle_ends.tz is None and now.tz is not None:
                now_cmp = now.tz_localize(None)
            else:
                now_cmp = now
            data = data.loc[candle_ends <= now_cmp]

        # Convert to target user timezone after closed-only filtering
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.tz_convert(self.timezone)

        return data

    def get_data(self) -> pd.DataFrame:
        """Full pipeline: fetch raw OHLCV and process with validation and resampling."""
        return self.process(self.fetch())

    def load_data(self) -> pd.DataFrame:
        """Alias for get_data to ensure CLI and script compatibility."""
        return self.get_data()
