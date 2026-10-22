"""Regression tests for the review follow-ups on v0.3.68.

Covering:
1. ``--end-date`` is documented as inclusive but yfinance's ``end`` is exclusive: the CLI must hand
   the loader the next midnight (a single-day range used to become an empty ``[D, D)`` window).
2. ``--start-date`` later than ``--end-date`` is rejected up front.
3. The 1m per-request window (7 days) is enforced for single-sided ranges as well, and the
   inclusive end day is counted.
4. ``MarketDataLoader._clamp_period_for_interval`` clamps every period longer than the window,
   not just a fixed list of strings.
5. ``fast_score_batch(out=...)`` semantics are identical in the pure-Python branch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from yfinance_ta_patterns import cli, gemm
from yfinance_ta_patterns.cli import parse_args, run_cli
from yfinance_ta_patterns.data import MarketDataLoader


def _ohlcv(bars: int = 60) -> pd.DataFrame:
    """Deterministic daily OHLCV frame."""
    rng = np.random.RandomState(7)
    close = 100.0 + np.cumsum(rng.randn(bars) * 0.5)
    open_p = close + rng.randn(bars) * 0.2
    return pd.DataFrame(
        {
            "Open": open_p,
            "High": np.maximum(open_p, close) + 0.3,
            "Low": np.minimum(open_p, close) - 0.3,
            "Close": close,
            "Volume": rng.randint(1000, 10000, bars),
        },
        index=pd.date_range("2026-06-01", periods=bars, freq="D"),
    )


def _days_ago(n: int) -> str:
    return (pd.Timestamp.now().normalize() - pd.Timedelta(days=n)).strftime("%Y-%m-%d")


@pytest.fixture
def loader_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the CLI's MarketDataLoader with a recorder of the start/end it is given."""
    calls: list[dict[str, Any]] = []

    class _Loader:
        def __init__(self, symbol: str, period: str = "60d", **kwargs: Any) -> None:
            calls.append({"symbol": symbol, "start": kwargs.get("start"), "end": kwargs.get("end")})
            self.period = period

        def get_data(self) -> pd.DataFrame:
            return _ohlcv()

    monkeypatch.setattr(cli, "MarketDataLoader", _Loader)
    return calls


# =====================================================================
# 1-2. Inclusive --end-date / inverted ranges
# =====================================================================
def test_end_date_is_made_exclusive_for_yfinance(loader_calls: list[dict[str, Any]]) -> None:
    argv = ["SPY", "-t", "1d", "--start-date", "2026-08-03", "--end-date", "2026-08-07"]
    assert run_cli(parse_args([*argv, "--pattern", "DOJI"])) == 0
    assert (loader_calls[0]["start"], loader_calls[0]["end"]) == ("2026-08-03", "2026-08-08")


def test_single_day_range_is_not_an_empty_window(loader_calls: list[dict[str, Any]]) -> None:
    argv = ["SPY", "-t", "1d", "--start-date", "2026-08-05", "--end-date", "2026-08-05"]
    assert run_cli(parse_args([*argv, "--pattern", "DOJI"])) == 0
    assert (loader_calls[0]["start"], loader_calls[0]["end"]) == ("2026-08-05", "2026-08-06")


def test_end_date_with_time_of_day_is_passed_through(loader_calls: list[dict[str, Any]]) -> None:
    argv = ["SPY", "-t", "1d", "--start-date", "2026-08-03", "--end-date", "2026-08-07 17:00"]
    assert run_cli(parse_args([*argv, "--pattern", "DOJI"])) == 0
    assert loader_calls[0]["end"] == "2026-08-07 17:00"


def test_open_ended_ranges_keep_their_open_side(loader_calls: list[dict[str, Any]]) -> None:
    assert run_cli(parse_args(["SPY", "-t", "1d", "--start-date", "2026-08-03"])) == 0
    assert (loader_calls[-1]["start"], loader_calls[-1]["end"]) == ("2026-08-03", None)
    assert run_cli(parse_args(["SPY", "-t", "1d", "--end-date", "2026-08-07"])) == 0
    assert (loader_calls[-1]["start"], loader_calls[-1]["end"]) == (None, "2026-08-08")


def test_backtest_gets_exclusive_end_but_header_stays_inclusive(
    loader_calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("yfinance_ta_patterns.pattern_tester.PatternRankingTester") as tester_cls:
        tester_cls.return_value = MagicMock(
            get_comparison_report=MagicMock(return_value=pd.DataFrame([{"Pattern": "CDLDOJI"}]))
        )
        argv = ["SPY", "-t", "1d", "--backtest", "--start-date", "2026-08-03"]
        assert run_cli(parse_args([*argv, "--end-date", "2026-08-07"])) == 0
    assert loader_calls[0]["end"] == "2026-08-08"
    assert "2026-08-03 to 2026-08-07" in capsys.readouterr().out


def test_start_after_end_is_rejected() -> None:
    argv = ["SPY", "-t", "1d", "--start-date", "2026-08-10", "--end-date", "2026-08-03"]
    with pytest.raises(ValueError, match="must not be later than"):
        run_cli(parse_args(argv))


# =====================================================================
# 3. 1m request window: inclusive end day and single-sided ranges
# =====================================================================
def test_1m_window_counts_the_inclusive_end_day(loader_calls: list[dict[str, Any]]) -> None:
    ok = ["EURUSD=X", "-t", "1m", "--start-date", _days_ago(12), "--end-date", _days_ago(6)]
    assert run_cli(parse_args(ok)) == 0  # 7 calendar days, both ends included
    assert loader_calls[0]["end"] == _days_ago(5)

    too_long = ["EURUSD=X", "-t", "1m", "--start-date", _days_ago(13), "--end-date", _days_ago(6)]
    with pytest.raises(ValueError, match="at most 7 days"):
        run_cli(parse_args(too_long))  # 8 calendar days


def test_1m_start_only_must_fit_the_window(loader_calls: list[dict[str, Any]]) -> None:
    assert run_cli(parse_args(["EURUSD=X", "-t", "1m", "--start-date", _days_ago(6)])) == 0
    with pytest.raises(ValueError, match="at most 7 days"):
        run_cli(parse_args(["EURUSD=X", "-t", "1m", "--start-date", _days_ago(7)]))
    with pytest.raises(ValueError, match="at most 7 days"):
        run_cli(parse_args(["EURUSD=X", "-t", "1m", "--start-date", _days_ago(20)]))


def test_1m_end_only_is_rejected() -> None:
    # yfinance fills a missing start with `end - 1 month`, which can never fit a 7-day window.
    with pytest.raises(ValueError, match="--start-date together with --end-date"):
        run_cli(parse_args(["EURUSD=X", "-t", "1m", "--end-date", _days_ago(5)]))


def test_other_intraday_intervals_keep_open_ended_ranges(
    loader_calls: list[dict[str, Any]],
) -> None:
    # 5m: window == depth (60 days), so open-ended ranges inside the depth stay valid.
    assert run_cli(parse_args(["EURUSD=X", "-t", "5m", "--start-date", _days_ago(40)])) == 0
    assert run_cli(parse_args(["EURUSD=X", "-t", "5m", "--end-date", _days_ago(5)])) == 0


# =====================================================================
# 4. Table-driven period clamp
# =====================================================================
@pytest.mark.parametrize(
    ("interval", "period", "expected"),
    [
        ("1m", "8d", "7d"),
        ("1m", "20d", "7d"),
        ("1m", "29d", "7d"),
        ("1m", "2mo", "7d"),
        ("1m", "5d", "5d"),
        ("1m", "7d", "7d"),
        ("1m", "1wk", "1wk"),
        ("1m", "garbage", "garbage"),
        ("5m", "61d", "60d"),
        ("5m", "75d", "60d"),
        ("5m", "59d", "59d"),
        ("5m", "2mo", "2mo"),
        ("5m", "ytd", "60d"),
        ("15m", "1y", "60d"),
        ("1h", "731d", "730d"),
        ("1h", "3y", "730d"),
        ("1h", "2y", "2y"),
        ("1h", "ytd", "ytd"),
        ("1h", "max", "730d"),
        ("1d", "10y", "10y"),
        ("1mo", "max", "max"),
    ],
)
def test_clamp_period_is_table_driven(interval: str, period: str, expected: str) -> None:
    assert MarketDataLoader._clamp_period_for_interval(period, interval) == expected


def test_clamp_period_keeps_every_legacy_allow_list_entry() -> None:
    clamp = MarketDataLoader._clamp_period_for_interval
    legacy_1m = ("14d", "15d", "30d", "60d", "90d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y")
    assert all(clamp(p, "1m") == "7d" for p in (*legacy_1m, "ytd", "max"))
    legacy_60d = ("90d", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
    assert all(clamp(p, i) == "60d" for p in legacy_60d for i in ("2m", "5m", "15m", "30m", "90m"))
    assert all(clamp(p, i) == "730d" for p in ("5y", "10y", "max") for i in ("60m", "1h", "4h"))


# =====================================================================
# 5. fast_score_batch(out=...) parity between the NumPy and pure-Python branches
# =====================================================================
def test_pure_python_branch_ignores_a_list_of_the_wrong_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gemm, "HAS_NUMPY", False)
    monkeypatch.setattr(gemm, "HAS_NANOGEMM", False)
    features, weights = [[1.0, 2.0, 3.0]], [0.5, -0.2, 0.1]
    expected = 0.5 - 0.4 + 0.3 + 0.1

    oversized = [0.0] * 5
    res = gemm.fast_score_batch(features, weights, bias=0.1, activation="linear", out=oversized)
    assert oversized == [0.0] * 5  # neither resized nor overwritten
    assert res is not oversized
    assert res == pytest.approx([expected])

    exact = [0.0]
    res = gemm.fast_score_batch(features, weights, bias=0.1, activation="linear", out=exact)
    assert res is exact
    assert exact == pytest.approx([expected])


def test_numpy_branch_leaves_a_mismatched_list_untouched() -> None:
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    weights = np.array([0.5, -0.2, 0.1], dtype=np.float32)
    short = [9.0]
    res = gemm.fast_score_batch(features, weights, bias=0.1, activation="linear", out=short)
    assert short == [9.0]
    assert len(res) == 2
