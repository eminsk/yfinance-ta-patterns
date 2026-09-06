"""Unit tests for universal MarketDataLoader and ticker normalization."""

from __future__ import annotations

import pandas as pd

from yfinance_ta_patterns.data import MarketDataLoader, normalize_ticker
from yfinance_ta_patterns.forex_data_loader import ForexDataLoader


def test_normalize_ticker_forex() -> None:
    """Forex symbols should append =X properly."""
    assert normalize_ticker("EURUSD") == "EURUSD=X"
    assert normalize_ticker("GBPUSD=X") == "GBPUSD=X"
    assert normalize_ticker("usdchf") == "USDCHF=X"
    assert normalize_ticker("EURUSD", asset_type="forex") == "EURUSD=X"


def test_normalize_ticker_stocks() -> None:
    """Stock symbols must not have =X appended."""
    assert normalize_ticker("AAPL") == "AAPL"
    assert normalize_ticker("NVDA") == "NVDA"
    assert normalize_ticker("TSLA") == "TSLA"
    assert normalize_ticker("MSFT") == "MSFT"


def test_normalize_ticker_crypto_and_commodities() -> None:
    """Crypto pairs and commodity futures should preserve symbols."""
    assert normalize_ticker("BTC-USD") == "BTC-USD"
    assert normalize_ticker("ETH-USD") == "ETH-USD"
    assert normalize_ticker("GC=F") == "GC=F"
    assert normalize_ticker("^GSPC") == "^GSPC"


def test_forex_data_loader_subclass() -> None:
    """ForexDataLoader maintains subclass compatibility with MarketDataLoader."""
    loader = ForexDataLoader("EURUSD")
    assert isinstance(loader, MarketDataLoader)
    assert loader.ticker == "EURUSD=X"
    assert loader.asset_type == "forex"


def test_market_data_loader_process_timezone() -> None:
    """MarketDataLoader normalizes timezone and column format."""
    loader = MarketDataLoader("AAPL", timezone="UTC")
    dates = pd.date_range("2025-01-01", periods=5, freq="1D")
    df = pd.DataFrame(
        {
            "Open": [150.0] * 5,
            "High": [155.0] * 5,
            "Low": [149.0] * 5,
            "Close": [154.0] * 5,
            "Volume": [1000] * 5,
        },
        index=dates,
    )

    processed = loader.process(df)
    assert isinstance(processed.index, pd.DatetimeIndex)
    assert processed.index.tz is not None
