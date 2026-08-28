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


def test_parse_args_all_patterns():
    args = parse_args(["--all-patterns", "--date", "2025-01-10"])
    assert args.all_patterns is True
    assert args.pattern is None
    assert args.date == "2025-01-10"


def test_parse_args_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--pattern", "HAMMER", "--all-patterns"])


def test_run_cli_date_conflict():
    args = parse_args(["--pattern", "HAMMER", "--date", "2025-01-01", "--start-date", "2025-01-01"])
    with pytest.raises(ValueError, match="Use either --date or --start-date/--end-date"):
        run_cli(args)


def test_run_cli_pattern_flow(capsys):
    idx = pd.date_range("2025-01-01 00:00", periods=5, freq="1d", tz="UTC")
    mock_df = pd.DataFrame(
        {
            "Open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "High": [1.2, 1.3, 1.4, 1.5, 1.6],
            "Low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "Close": [1.1, 1.2, 1.3, 1.4, 1.5],
        },
        index=idx,
    )

    with patch("yfinance_ta_patterns.cli.ForexDataLoader.get_data", return_value=mock_df):
        args = parse_args(["--pattern", "DOJI", "--symbol", "EURUSD"])
        exit_code = run_cli(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "DOJI" in captured.out
