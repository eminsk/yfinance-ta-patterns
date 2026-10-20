"""Regression tests for defects found while reviewing v0.3.64.

Every test here fails on the v0.3.64 code base and passes with the accompanying fixes.
None of them needs network access, native TA-Lib or a native NanoGEMM build.
"""

from __future__ import annotations

import types
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from yfinance_ta_patterns import cli
from yfinance_ta_patterns import gemm as gemm_mod
from yfinance_ta_patterns import talib_compat as tc
from yfinance_ta_patterns.data import MarketDataLoader

# --------------------------------------------------------------------------- helpers


def _ohlcv(n: int = 120, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + 0.3,
            "Low": np.minimum(open_, close) - 0.3,
            "Close": close,
            "Volume": 1000.0,
        },
        index=idx,
    )


class _RecordingLoader:
    """Stand-in for MarketDataLoader that records the symbols the CLI asks for."""

    symbols: ClassVar[list[str]] = []

    def __init__(self, symbol: str, *args: Any, **kwargs: Any) -> None:
        type(self).symbols.append(symbol)
        self.symbol = symbol
        self.period = kwargs.get("period", "60d")
        self.interval = kwargs.get("interval", "1h")

    def get_data(self) -> pd.DataFrame:
        return _ohlcv()


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int | str | None:
    _RecordingLoader.symbols = []
    monkeypatch.setattr(cli, "MarketDataLoader", _RecordingLoader)
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--timeframe", "1h", "--lookback-bars", "2", "--min-confidence", "0"])
    return exc.value.code


def _fx_hourly(first_utc: str, last_utc: str) -> pd.DataFrame:
    """Hourly EURUSD bars; the FX week opens Sun 17:00 and closes Fri 17:00 New York time."""
    idx = pd.date_range(first_utc, last_utc, freq="h", tz="UTC")
    keep = []
    for ts in idx:
        ny = ts.tz_convert("America/New_York")
        wd, hr = ny.weekday(), ny.hour
        keep.append((wd == 6 and hr >= 17) or wd in (0, 1, 2, 3) or (wd == 4 and hr < 17))
    idx = idx[np.array(keep)]
    rng = np.random.default_rng(3)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0004, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + 1e-4,
            "Low": np.minimum(open_, close) - 1e-4,
            "Close": close,
            "Volume": 0.0,
        },
        index=idx,
    )


# --------------------------------------------------------------------------- CLI


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["SPY"], ["SPY"]),
        (["NVDA", "AAPL"], ["NVDA", "AAPL"]),
        (["SPY", "--lookback-bars", "2", "NVDA"], ["SPY", "NVDA"]),
        (["--symbol", "AAPL", "MSFT"], ["AAPL", "MSFT"]),
        (["--symbol", "AAPL,MSFT"], ["AAPL", "MSFT"]),
    ],
)
def test_cli_requests_every_symbol_exactly_once(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: list[str]
) -> None:
    """parse_args() already merges positionals into --symbol; run_cli() must not merge again."""
    assert _run_cli(monkeypatch, argv) in (0, None)
    assert _RecordingLoader.symbols == expected


def test_cli_single_positional_symbol_keeps_single_symbol_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_cli(monkeypatch, ["SPY"])
    assert "SCANNING" not in capsys.readouterr().out  # multi-symbol banner


def test_cli_rejects_empty_symbol_instead_of_scanning_all_pairs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run_cli(monkeypatch, ["--symbol", ""])
    assert code not in (0, None)
    assert _RecordingLoader.symbols == []
    assert "empty" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- talib_compat

_O = [
    100.0,
    100.5,
    98.0,
    99.5,
    96.5,
    99.0,
    99.5,
    98.5,
    97.5,
    97.0,
    98.0,
    99.0,
    99.5,
    101.0,
    98.5,
    100.5,
    101.5,
    101.0,
    101.5,
    101.5,
]
_H = [
    101.0,
    100.5,
    98.5,
    100.0,
    99.0,
    99.5,
    100.5,
    99.5,
    99.0,
    98.5,
    100.0,
    100.0,
    101.5,
    102.0,
    100.5,
    101.5,
    101.5,
    102.0,
    101.5,
    102.0,
]
_L = [
    98.5,
    97.5,
    97.0,
    96.5,
    96.0,
    98.5,
    96.5,
    97.0,
    97.5,
    96.0,
    97.5,
    98.0,
    99.0,
    98.0,
    98.0,
    100.5,
    100.5,
    100.5,
    100.0,
    101.0,
]
_C = [
    99.5,
    98.5,
    98.5,
    97.0,
    98.0,
    98.5,
    97.5,
    97.5,
    98.0,
    98.5,
    100.0,
    100.0,
    100.5,
    99.0,
    100.0,
    100.5,
    100.5,
    101.5,
    100.5,
    101.0,
]


def test_hikkakemod_setup_inside_warmup_window_is_confirmed_like_native_talib() -> None:
    """Native TA-Lib (lookback 10) pre-scans bars 7..9, so a setup there is confirmed at bar 10.

    Expected values were produced by native TA-Lib 0.6 (`talib.CDLHIKKAKEMOD`).
    """
    out = [
        int(v) for v in tc.cdl_hikkakemod(np.array(_O), np.array(_H), np.array(_L), np.array(_C))
    ]
    assert out == [0] * 10 + [200] + [0] * 9


# --------------------------------------------------------------------------- data (FX weekend)


def test_fx_4h_friday_last_bar_is_released_at_the_weekly_close() -> None:
    df = _fx_hourly("2026-10-11 21:00", "2026-10-16 20:00")  # Friday close = 21:00 UTC (EDT)
    loader = MarketDataLoader("EURUSD=X", interval="4h", timezone="UTC")

    before_close = loader.process(df.copy(), now_utc=pd.Timestamp("2026-10-16 20:30", tz="UTC"))
    assert before_close.index[-1] == pd.Timestamp("2026-10-16 16:00", tz="UTC")

    after_close = loader.process(df.copy(), now_utc=pd.Timestamp("2026-10-16 21:30", tz="UTC"))
    assert after_close.index[-1] == pd.Timestamp("2026-10-16 20:00", tz="UTC")


# --------------------------------------------------------------------------- gemm


@pytest.mark.parametrize("order", ["C", "F"])
def test_fast_score_batch_hands_native_kernel_float32_contiguous_arrays(
    monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    """The native kernel reinterprets raw memory as float32 (it does not validate dtype/layout).

    pandas / NumPy default to float64, so the wrapper must sanitize before dispatching.
    """
    seen: dict[str, Any] = {}

    def score_batch(features: Any, weights: Any, **kwargs: Any) -> Any:
        seen["features"], seen["weights"] = features, weights
        return np.zeros(len(features), dtype=np.float32)

    monkeypatch.setattr(gemm_mod, "_ng", types.SimpleNamespace(score_batch=score_batch))
    monkeypatch.setattr(gemm_mod, "HAS_NANOGEMM", True)

    rng = np.random.default_rng(0)
    features = np.asarray(rng.normal(size=(5, 4)), order=order)  # float64, maybe Fortran-ordered
    weights = rng.normal(size=4)  # float64
    gemm_mod.fast_score_batch(features, weights, activation="linear")

    assert seen["features"].dtype == np.float32 and seen["features"].flags.c_contiguous
    assert seen["weights"].dtype == np.float32 and seen["weights"].flags.c_contiguous


def test_fast_score_batch_zero_vectors_and_shapes() -> None:
    """fast_score_batch handles zero vectors cleanly across fallback and fast paths."""
    features = np.zeros((10, 5), dtype=np.float64)
    weights = np.zeros(5, dtype=np.float64)
    scores = gemm_mod.fast_score_batch(features, weights, bias=0.0, activation="linear")
    assert np.allclose(scores, 0.0)
    assert len(scores) == 10


def test_resolve_symbol_currencies_precious_metals_excludes_commodity_codes() -> None:
    """Precious metal symbols like XAUUSD should resolve to the fiat currency, not XAU."""
    from yfinance_ta_patterns.economic_calendar import resolve_symbol_currencies

    assert resolve_symbol_currencies("XAUUSD") == ["USD"]
    assert resolve_symbol_currencies("XAGUSD") == ["USD"]
    assert resolve_symbol_currencies("XPTUSD") == ["USD"]
    assert resolve_symbol_currencies("XPDUSD") == ["USD"]
