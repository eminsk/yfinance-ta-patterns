import pandas as pd
import yfinance as yf
import pytz

class ForexDataLoader:
    """
    Generic Data Loader for Yahoo Finance tickers (forex, stocks, crypto, etc.).
    """

    def __init__(
        self,
        symbol: str,
        period: str = '60d',
        interval: str = '15m',
        timezone: str = 'Europe/Moscow'
    ):
        # Allow full tickers or append suffix if missing
        self.ticker = symbol if symbol.endswith(suffix := '=X') else f"{symbol}{suffix}"
        self.period = period
        self.interval = interval
        self.timezone = timezone
        self._download_interval = self._resolve_download_interval(interval)
        self._resample_rule = "4h" if interval == "4h" else None

    def _resolve_download_interval(self, interval: str) -> str:
        """Map custom intervals to supported yfinance intervals."""
        if interval == "4h":
            return "1h"  # fetch 1h data and resample to 4h
        return interval

    def fetch(self) -> pd.DataFrame:
        """Fetch raw data via yfinance."""
        return yf.download(
            self.ticker,
            period=self.period,
            interval=self._download_interval,
            auto_adjust=True
        )

    def _resample_if_needed(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample OHLCV data when the requested interval is not natively supported."""
        if not self._resample_rule or data.empty:
            return data

        agg = {
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
            if col not in agg:
                agg[col] = "last"

        return data.resample(self._resample_rule).agg(agg).dropna()

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize columns and timezone."""
        # Drop extra MultiIndex level if present
        data.columns = data.columns.droplevel(1) if data.columns.nlevels > 1 else data.columns

        if data.index.tz is None:
            data.index = data.index.tz_localize(pytz.UTC)
        else:
            data.index = data.index.tz_convert(pytz.UTC)
        # Потом конвертируем в нужную зону
        data.index = data.index.tz_convert(self.timezone)
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
