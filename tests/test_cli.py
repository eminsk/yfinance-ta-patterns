from unittest.mock import patch

import pandas as pd
import pytest

from yfinance_ta_patterns.cli import normalize_timeframe, parse_args, run_cli


def test_normalize_timeframe_aliases():
    assert normalize_timeframe("M1") == "1m"
    assert normalize_timeframe("M5") == "5m"
    assert normalize_timeframe("M15") == "15m"
    assert normalize_timeframe("M30") == "30m"
    assert normalize_timeframe("h1") == "1h"
    assert normalize_timeframe("h4") == "4h"
    assert normalize_timeframe("D1") == "1d"
    assert normalize_timeframe("1wk") == "1wk"


def test_normalize_timeframe_invalid():
    with pytest.raises(ValueError, match="Unsupported timeframe 'invalid'"):
        normalize_timeframe("invalid")


def test_parse_args_pattern():
    args = parse_args(["--pattern", "HAMMER", "--symbol", "GBPUSD", "--timeframe", "1h"])
    assert args.pattern == "HAMMER"
    assert args.symbol == "GBPUSD"
    assert args.timeframe == "1h"
    assert not args.all_patterns
    assert not args.ai


def test_parse_args_all_patterns_with_ai():
    args = parse_args(["--all-patterns", "--ai", "--min-confidence", "0.65"])
    assert args.all_patterns is True
    assert args.ai is True
    assert args.min_confidence == 0.65


def test_parse_args_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--pattern", "HAMMER", "--all-patterns"])


def test_run_cli_date_conflict():
    args = parse_args(["--pattern", "HAMMER", "--date", "2025-01-01", "--start-date", "2025-01-01"])
    with pytest.raises(ValueError, match="Use either --date or --start-date/--end-date"):
        run_cli(args)


@pytest.fixture
def mock_ohlcv_df():
    idx = pd.date_range("2025-01-01 00:00", periods=20, freq="1d", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(20)],
            "High": [102.0 + i for i in range(20)],
            "Low": [99.0 + i for i in range(20)],
            "Close": [101.0 + i for i in range(20)],
            "Volume": [50000.0] * 20,
        },
        index=idx,
    )


def test_run_cli_pattern_flow(capsys, mock_ohlcv_df):
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data", return_value=mock_ohlcv_df):
        args = parse_args(["--pattern", "DOJI", "--symbol", "EURUSD"])
        exit_code = run_cli(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "DOJI" in captured.out


def test_run_cli_ai_flow(capsys, mock_ohlcv_df):
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data", return_value=mock_ohlcv_df):
        args = parse_args(["--pattern", "DOJI", "--symbol", "AAPL", "--ai"])
        exit_code = run_cli(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        # Either found AI score or cleanly reported no signals
        assert "AI" in captured.out or "No" in captured.out


def test_run_cli_ai_analyst_flow(capsys, mock_ohlcv_df):
    with patch("yfinance_ta_patterns.cli.MarketDataLoader.get_data", return_value=mock_ohlcv_df):
        args = parse_args(["--pattern", "HAMMER", "--symbol", "BTC-USD", "--ai-analyst"])
        exit_code = run_cli(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "AI Technical Intelligence Brief" in captured.out
