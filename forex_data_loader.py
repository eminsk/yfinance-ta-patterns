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

    def fetch(self) -> pd.DataFrame:
        """Fetch raw data via yfinance."""
        return yf.download(
            self.ticker,
            period=self.period,
            interval=self.interval,
            auto_adjust=True
        )

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
        return self.process(self.fetch())

if __name__ == "__main__":
    # Example usage
    loader = ForexDataLoader("EURUSD=X")
    data = loader.get_data()
    print(data)