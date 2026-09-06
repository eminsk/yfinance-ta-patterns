"""Forex data loader (maintained for backwards compatibility, inherits MarketDataLoader)."""

from __future__ import annotations

from .data import MarketDataLoader


class ForexDataLoader(MarketDataLoader):
    """Data loader specifically targeted for Currency / Forex pairs.

    Maintains 100% backwards compatibility with earlier versions of yfinance-ta-patterns.
    """

    def __init__(
        self,
        symbol: str,
        period: str = "60d",
        interval: str = "15m",
        timezone: str = "Europe/Moscow",
    ) -> None:
        """Initialize ForexDataLoader forcing forex asset type."""
        super().__init__(
            symbol=symbol,
            period=period,
            interval=interval,
            timezone=timezone,
            asset_type="forex",
        )
