import pandas as pd

from yfinance_ta_patterns.forex_data_loader import ForexDataLoader


def test_resample_to_4h():
    loader = ForexDataLoader("EURUSD", interval="4h")
    assert loader._download_interval == "1h"  # noqa: SLF001

    idx = pd.date_range("2024-01-01 00:00", periods=4, freq="1H", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [2, 3, 5, 10],
            "Low": [1, 0, 2, 3],
            "Close": [1.1, 2.2, 3.3, 4.4],
            "Volume": [10, 20, 30, 40],
        },
        index=idx,
    )

    resampled = loader._resample_if_needed(data)  # noqa: SLF001
    assert len(resampled) == 1

    row = resampled.iloc[0]
    assert row["Open"] == 1
    assert row["High"] == 10
    assert row["Low"] == 0
    assert row["Close"] == 4.4
    assert row["Volume"] == 100
