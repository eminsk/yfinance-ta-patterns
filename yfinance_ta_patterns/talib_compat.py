"""
TA-Lib Compatibility and Pure-Python / NumPy Fallback Engine.
Allows yfinance-ta-patterns to run on any Python version (including
Python 3.13t, 3.14t, 3.15t Free-Threaded No-GIL, PyPy 3.8-3.12, and ARM)
even if native C ta-lib binaries or C-extensions are not installed.
Provides pure-Python vectorized implementations of all 61 canonical TA-Lib
candlestick patterns.
"""

from __future__ import annotations

import importlib
import os
import sys
import sysconfig
from typing import Any

# Try importing numpy; if missing or unsupported (e.g. PyPy 3.12 cpyext), provide pure-Python shim
try:
    import numpy as np

    HAS_NUMPY = True
except (ImportError, ModuleNotFoundError, RuntimeError, TypeError):  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False


class _Vec:
    """Lightweight pure-Python vectorized 1D array with slice-views, mask assignment, and broadcasting."""

    __slots__ = ("_data", "_start", "_stop")

    def __init__(self, data: Any, start: int = 0, stop: int | None = None) -> None:
        if isinstance(data, _Vec):
            self._data = data._data
            self._start = data._start + start
            self._stop = data._start + (data._stop - data._start if stop is None else stop)
        elif isinstance(data, list):
            self._data = data
            self._start = start
            self._stop = len(data) if stop is None else stop
        elif hasattr(data, "tolist"):
            self._data = data.tolist()
            self._start = 0
            self._stop = len(self._data)
        else:
            self._data = list(data)
            self._start = 0
            self._stop = len(self._data)

    def __len__(self) -> int:
        return max(0, self._stop - self._start)

    def __iter__(self):
        for i in range(self._start, self._stop):
            yield self._data[i]

    def tolist(self) -> list[Any]:
        return self._data[self._start : self._stop]

    @property
    def dtype(self) -> Any:
        return int

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self),)

    def unique(self) -> _Vec:
        seen: list[Any] = []
        for x in self:
            if x not in seen:
                seen.append(x)
        return _Vec(seen)

    def __getitem__(self, item: Any) -> Any:
        length = len(self)
        if isinstance(item, slice):
            start, stop, step = item.indices(length)
            if step != 1:
                return _Vec([self._data[self._start + i] for i in range(start, stop, step)])
            return _Vec(self._data, start=self._start + start, stop=self._start + stop)

        if isinstance(item, _Vec):
            res = []
            for i, m in enumerate(item):
                if m:
                    res.append(self._data[self._start + i])
            return _Vec(res)

        if isinstance(item, int):
            if item < 0:
                item += length
            if item < 0 or item >= length:
                raise IndexError("Index out of bounds")
            return self._data[self._start + item]

        raise TypeError(f"Invalid index type: {type(item)}")

    def __setitem__(self, item: Any, val: Any) -> None:
        length = len(self)
        if isinstance(item, slice):
            start, stop, step = item.indices(length)
            if isinstance(val, (_Vec, list, tuple)):
                val_list = val.tolist() if hasattr(val, "tolist") else list(val)
                for idx, v in zip(range(start, stop, step), val_list):
                    self._data[self._start + idx] = v
            else:
                for idx in range(start, stop, step):
                    self._data[self._start + idx] = val
            return

        if isinstance(item, _Vec):
            val_iter = (
                iter(val)
                if hasattr(val, "__iter__") and not isinstance(val, (int, float, bool))
                else None
            )
            for i, m in enumerate(item):
                if m:
                    v = next(val_iter) if val_iter is not None else val
                    self._data[self._start + i] = v
            return

        if isinstance(item, int):
            if item < 0:
                item += length
            if item < 0 or item >= length:
                raise IndexError("Index out of bounds")
            self._data[self._start + item] = val
            return

        raise TypeError(f"Invalid index type: {type(item)}")

    def _binop(self, other: Any, op: Any) -> _Vec:
        if isinstance(other, _Vec):
            return _Vec([op(a, b) for a, b in zip(self, other)])
        return _Vec([op(a, other) for a in self])

    def _rbinop(self, other: Any, op: Any) -> _Vec:
        if isinstance(other, _Vec):
            return _Vec([op(b, a) for a, b in zip(self, other)])
        return _Vec([op(other, a) for a in self])

    def __add__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> _Vec:
        return self._rbinop(other, lambda a, b: a + b)

    def __sub__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> _Vec:
        return self._rbinop(other, lambda a, b: a - b)

    def __mul__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> _Vec:
        return self._rbinop(other, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a / b if b != 0 else 0.0)

    def __rtruediv__(self, other: Any) -> _Vec:
        return self._rbinop(other, lambda a, b: a / b if b != 0 else 0.0)

    def __abs__(self) -> _Vec:
        return _Vec([abs(x) for x in self])

    def __gt__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a > b)

    def __lt__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a < b)

    def __ge__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a >= b)

    def __le__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a <= b)

    def __eq__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a == b)  # type: ignore[override]

    def __ne__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: a != b)  # type: ignore[override]

    def __and__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: bool(a) and bool(b))

    def __or__(self, other: Any) -> _Vec:
        return self._binop(other, lambda a, b: bool(a) or bool(b))

    def __invert__(self) -> _Vec:
        return _Vec([not bool(x) for x in self])

    def __repr__(self) -> str:
        return f"_Vec({self.tolist()})"


class _NumpyShim:
    """Minimal pure-Python shim providing required numpy functions when numpy is unavailable."""

    @staticmethod
    def asarray(a: Any, dtype: Any = None) -> _Vec:
        return a if isinstance(a, _Vec) else _Vec(a)

    @staticmethod
    def array(a: Any, dtype: Any = None) -> _Vec:
        return _Vec(a.tolist() if isinstance(a, _Vec) else a)

    @staticmethod
    def zeros(n: int, dtype: Any = None) -> _Vec:
        return _Vec([0] * n)

    @staticmethod
    def abs(a: Any) -> Any:
        return abs(a)

    @staticmethod
    def maximum(a: Any, b: Any) -> Any:
        if isinstance(a, _Vec):
            return a._binop(b, max)
        elif isinstance(b, _Vec):
            return b._rbinop(a, max)
        return max(a, b)

    @staticmethod
    def minimum(a: Any, b: Any) -> Any:
        if isinstance(a, _Vec):
            return a._binop(b, min)
        elif isinstance(b, _Vec):
            return b._rbinop(a, min)
        return min(a, b)

    @staticmethod
    def mean(a: Any) -> float:
        if isinstance(a, _Vec):
            return sum(a) / len(a) if len(a) > 0 else 1.0
        seq = list(a)
        return sum(seq) / len(seq) if len(seq) > 0 else 1.0

    @staticmethod
    def unique(a: Any) -> Any:
        if hasattr(a, "unique"):
            return a.unique()
        return list(dict.fromkeys(a))

    int32 = int
    float64 = float
    bool = bool
    ndarray = _Vec


if not HAS_NUMPY:
    np = _NumpyShim()  # type: ignore[assignment]


# TA-Lib does not publish the required Windows CPython 3.15 or free-threaded
# wheels to PyPI. This release contains matching TA-Lib 0.7.1 x64 wheels.
WINDOWS_TALIB_WHEELHOUSE_TAG = "v0.3.26"

_talib: Any = None
HAS_NATIVE_TALIB: bool = False
TALIB_IMPORT_ERROR: Exception | None = None

try:
    _talib = importlib.import_module("talib")
    HAS_NATIVE_TALIB = True
except (ImportError, ModuleNotFoundError) as exc:
    _talib = None
    HAS_NATIVE_TALIB = False
    TALIB_IMPORT_ERROR = exc


# Full canonical list of 61 TA-Lib candlestick patterns
ALL_CDL_PATTERNS = [
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3INSIDE",
    "CDL3LINESTRIKE",
    "CDL3OUTSIDE",
    "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK",
    "CDLBELTHOLD",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER",
    "CDLDOJI",
    "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI",
    "CDLHAMMER",
    "CDLHANGINGMAN",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLHIGHWAVE",
    "CDLHIKKAKE",
    "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS",
    "CDLINNECK",
    "CDLINVERTEDHAMMER",
    "CDLKICKING",
    "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLMATCHINGLOW",
    "CDLMATHOLD",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
    "CDLONNECK",
    "CDLPIERCING",
    "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
]


def _to_arrays(open_, high, low, close):
    if HAS_NUMPY:
        o = np.asarray(open_, dtype=np.float64)
        h = np.asarray(high, dtype=np.float64)
        lo = np.asarray(low, dtype=np.float64)
        c = np.asarray(close, dtype=np.float64)
        return o, h, lo, c
    return _Vec(open_), _Vec(high), _Vec(low), _Vec(close)


def _prior_trend_down(close: Any, lookback: int = 5) -> Any:
    """Boolean mask: True where price declined over the `lookback` bars strictly
    preceding each index. Safe to use as a same-bar reversal-context filter with zero lookahead.
    """
    n = len(close)
    trend = np.zeros(n, dtype=bool)
    if n <= lookback:
        return trend
    trend[lookback:] = close[: n - lookback] > close[lookback - 1 : n - 1]
    return trend


def _prior_trend_up(close: Any, lookback: int = 5) -> Any:
    """Boolean mask: True where price advanced over the `lookback` bars strictly
    preceding each index.
    """
    n = len(close)
    trend = np.zeros(n, dtype=bool)
    if n <= lookback:
        return trend
    trend[lookback:] = close[: n - lookback] < close[lookback - 1 : n - 1]
    return trend


# ==============================================================================
# Vectorized Pattern Implementations for all 61 Canonical TA-Lib Patterns
# ==============================================================================


def cdl_doji(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (body <= 0.1 * hl)
    res[mask] = 100
    return res


def cdl_hammer(open_, high, low, close, trend_lookback: int = 5):
    """Bullish reversal: long lower shadow + small body near the top, after a decline."""
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    lower_shadow = np.minimum(o, c) - lo
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    shape = (hl > 0) & (lower_shadow >= 2.0 * body) & (upper_shadow <= 0.25 * hl) & (body > 0)
    mask = shape & _prior_trend_down(c, trend_lookback)
    res[mask] = 100
    return res


def cdl_invertedhammer(open_, high, low, close, trend_lookback: int = 5):
    """Bullish reversal: long upper shadow + small body near the bottom, after a decline."""
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    lower_shadow = np.minimum(o, c) - lo
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    shape = (hl > 0) & (upper_shadow >= 2.0 * body) & (lower_shadow <= 0.25 * hl) & (body > 0)
    mask = shape & _prior_trend_down(c, trend_lookback)
    res[mask] = 100
    return res


def cdl_shootingstar(open_, high, low, close, trend_lookback: int = 5):
    """Bearish reversal: long upper shadow + small body near the bottom, after an advance."""
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    lower_shadow = np.minimum(o, c) - lo
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    shape = (hl > 0) & (upper_shadow >= 2.0 * body) & (lower_shadow <= 0.25 * hl) & (body > 0)
    mask = shape & _prior_trend_up(c, trend_lookback)
    res[mask] = -100
    return res


def cdl_hangingman(open_, high, low, close, trend_lookback: int = 5):
    """Bearish reversal: long lower shadow + small body near the top, after an advance."""
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    lower_shadow = np.minimum(o, c) - lo
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    shape = (hl > 0) & (lower_shadow >= 2.0 * body) & (upper_shadow <= 0.25 * hl) & (body > 0)
    mask = shape & _prior_trend_up(c, trend_lookback)
    res[mask] = -100
    return res


def cdl_engulfing(open_, high, low, close):
    o, _, _, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    prev_o, prev_c = o[:-1], c[:-1]
    curr_o, curr_c = o[1:], c[1:]
    bullish = (prev_c < prev_o) & (curr_c > curr_o) & (curr_o <= prev_c) & (curr_c >= prev_o)
    bearish = (prev_c > prev_o) & (curr_c < curr_o) & (curr_o >= prev_c) & (curr_c <= prev_o)
    res[1:][bullish] = 100
    res[1:][bearish] = -100
    return res


def cdl_harami(open_, high, low, close):
    o, _, _, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    prev_o, prev_c = o[:-1], c[:-1]
    curr_o, curr_c = o[1:], c[1:]
    prev_top = np.maximum(prev_o, prev_c)
    prev_bot = np.minimum(prev_o, prev_c)
    curr_top = np.maximum(curr_o, curr_c)
    curr_bot = np.minimum(curr_o, curr_c)
    inside = (curr_top <= prev_top) & (curr_bot >= prev_bot)
    bullish = inside & (prev_c < prev_o) & (curr_c > curr_o)
    bearish = inside & (prev_c > prev_o) & (curr_c < curr_o)
    res[1:][bullish] = 100
    res[1:][bearish] = -100
    return res


def cdl_marubozu(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (body >= 0.9 * hl)
    res[mask & (c > o)] = 100
    res[mask & (c < o)] = -100
    return res


def cdl_spinningtop(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    lower_shadow = np.minimum(o, c) - lo
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (body <= 0.3 * hl) & (upper_shadow >= body) & (lower_shadow >= body)
    res[mask & (c > o)] = 100
    res[mask & (c < o)] = -100
    return res


def cdl_3whitesoldiers(open_, high, low, close):
    o, _, _, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    g1 = c[:-2] > o[:-2]
    g2 = c[1:-1] > o[1:-1]
    g3 = c[2:] > o[2:]
    higher = (c[2:] > c[1:-1]) & (c[1:-1] > c[:-2])
    res[2:][g1 & g2 & g3 & higher] = 100
    return res


def cdl_3blackcrows(open_, high, low, close):
    o, _, _, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    r1 = c[:-2] < o[:-2]
    r2 = c[1:-1] < o[1:-1]
    r3 = c[2:] < o[2:]
    lower = (c[2:] < c[1:-1]) & (c[1:-1] < c[:-2])
    res[2:][r1 & r2 & r3 & lower] = -100
    return res


def cdl_2crows(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1 = o[1:-1], c[1:-1]
    o2, c2 = o[2:], c[2:]
    c1_white = (c0 > o0) & ((c0 - o0) >= 0.5 * (h0 - lo0))
    c2_gap_black = (c1 < o1) & (c1 > c0) & (o1 > c0)
    c3_black = (c2 < o2) & (o2 > c1) & (o2 < o1) & (c2 > o0) & (c2 < c0) & (c2 < c1)
    res[2:][c1_white & c2_gap_black & c3_black] = -100
    return res


def cdl_3inside(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0 = o[:-2], c[:-2]
    o1, c1 = o[1:-1], c[1:-1]
    _o2, c2 = o[2:], c[2:]
    prev_top = np.maximum(o0, c0)
    prev_bot = np.minimum(o0, c0)
    curr_top = np.maximum(o1, c1)
    curr_bot = np.minimum(o1, c1)
    inside = (curr_top <= prev_top) & (curr_bot >= prev_bot)
    bullish = inside & (c0 < o0) & (c1 > o1) & (c2 > o0) & (c2 > c1)
    bearish = inside & (c0 > o0) & (c1 < o1) & (c2 < o0) & (c2 < c1)
    res[2:][bullish] = 100
    res[2:][bearish] = -100
    return res


def cdl_3outside(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0 = o[:-2], c[:-2]
    o1, c1 = o[1:-1], c[1:-1]
    _o2, c2 = o[2:], c[2:]
    bull_engulf = (c0 < o0) & (c1 > o1) & (o1 <= c0) & (c1 >= o0)
    bear_engulf = (c0 > o0) & (c1 < o1) & (o1 >= c0) & (c1 <= o0)
    res[2:][bull_engulf & (c2 > c1)] = 100
    res[2:][bear_engulf & (c2 < c1)] = -100
    return res


def cdl_3linestrike(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 4:
        return res
    o0, c0 = o[:-3], c[:-3]
    o1, c1 = o[1:-2], c[1:-2]
    o2, c2 = o[2:-1], c[2:-1]
    o3, c3 = o[3:], c[3:]
    three_black = (c0 < o0) & (c1 < o1) & (c2 < o2) & (c1 < c0) & (c2 < c1)
    bull_strike = (c3 > o3) & (o3 <= c2) & (c3 >= o0)
    three_white = (c0 > o0) & (c1 > o1) & (c2 > o2) & (c1 > c0) & (c2 > c1)
    bear_strike = (c3 < o3) & (o3 >= c2) & (c3 <= o0)
    res[3:][three_black & bull_strike] = 100
    res[3:][three_white & bear_strike] = -100
    return res


def cdl_3starsinsouth(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, _h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, h1, lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2, h2, lo2 = o[2:], c[2:], h[2:], lo[2:]
    trend = _prior_trend_down(c, 5)[2:]
    three_black = (c0 < o0) & (c1 < o1) & (c2 < o2)
    c1_shadow = (c0 - lo0) >= (o0 - c0)
    c2_inside = (o1 < o0) & (o1 > c0) & (lo1 > lo0)
    c3_small = (h2 - lo2 <= h1 - lo1) & (lo2 >= lo1) & (h2 <= h1)
    res[2:][trend & three_black & c1_shadow & c2_inside & c3_small] = 100
    return res


def cdl_abandonedbaby(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, h1, lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2, h2, lo2 = o[2:], c[2:], h[2:], lo[2:]
    doji = np.abs(c1 - o1) <= 0.1 * (h1 - lo1)
    bullish = (c0 < o0) & doji & (h1 < lo0) & (c2 > o2) & (lo2 > h1)
    bearish = (c0 > o0) & doji & (lo1 > h0) & (c2 < o2) & (h2 < lo1)
    res[2:][bullish] = 100
    res[2:][bearish] = -100
    return res


def cdl_advanceblock(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, _h0, _lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, h1, _lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2, h2, _lo2 = o[2:], c[2:], h[2:], lo[2:]
    trend = _prior_trend_up(c, 5)[2:]
    three_white = (c0 > o0) & (c1 > o1) & (c2 > o2) & (c2 > c1) & (c1 > c0)
    opens_inside = (o1 > o0) & (o1 < c0) & (o2 > o1) & (o2 < c1)
    weakening = ((c2 - o2) < (c1 - o1)) | ((h2 - c2) > (h1 - c1))
    res[2:][trend & three_white & opens_inside & weakening] = -100
    return res


def cdl_belthold(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    long_body = (hl > 0) & (body >= 0.6 * hl)
    bullish = long_body & (c > o) & ((o - lo) <= 0.05 * hl)
    bearish = long_body & (c < o) & ((h - o) <= 0.05 * hl)
    res[bullish] = 100
    res[bearish] = -100
    return res


def cdl_breakaway(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 5:
        return res
    o0, c0, h0, lo0 = o[:-4], c[:-4], h[:-4], lo[:-4]
    o1, c1, h1, lo1 = o[1:-3], c[1:-3], h[1:-3], lo[1:-3]
    _o2, c2, _h2, _lo2 = o[2:-2], c[2:-2], h[2:-2], lo[2:-2]
    _o3, c3, _h3, _lo3 = o[3:-1], c[3:-1], h[3:-1], lo[3:-1]
    o4, c4, _h4, _lo4 = o[4:], c[4:], h[4:], lo[4:]
    bullish = (
        (c0 < o0)
        & (c1 < o1)
        & (h1 < lo0)
        & (c2 < c1)
        & (c3 < c2)
        & (c4 > o4)
        & (c4 > h1)
        & (c4 < lo0)
    )
    bearish = (
        (c0 > o0)
        & (c1 > o1)
        & (lo1 > h0)
        & (c2 > c1)
        & (c3 > c2)
        & (c4 < o4)
        & (c4 < lo1)
        & (c4 > h0)
    )
    res[4:][bullish] = 100
    res[4:][bearish] = -100
    return res


def cdl_closingmarubozu(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    long_body = (hl > 0) & (body >= 0.7 * hl)
    bullish = long_body & (c > o) & ((h - c) <= 0.02 * hl)
    bearish = long_body & (c < o) & ((c - lo) <= 0.02 * hl)
    res[bullish] = 100
    res[bearish] = -100
    return res


def cdl_concealbabyswall(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 4:
        return res
    o0, c0, h0, lo0 = o[:-3], c[:-3], h[:-3], lo[:-3]
    o1, c1, h1, lo1 = o[1:-2], c[1:-2], h[1:-2], lo[1:-2]
    o2, c2, h2, lo2 = o[2:-1], c[2:-1], h[2:-1], lo[2:-1]
    o3, c3, _h3, _lo3 = o[3:], c[3:], h[3:], lo[3:]
    four_black = (c0 < o0) & (c1 < o1) & (c2 < o2) & (c3 < o3)
    c1_maru = (o0 - c0) >= 0.7 * (h0 - lo0)
    c2_maru = ((o1 - c1) >= 0.7 * (h1 - lo1)) & (o1 < o0) & (c1 < c0)
    c3_gap = (o2 < c1) & (h2 > c1)
    c4_engulf = (o3 > h2) & (c3 < lo2)
    res[3:][four_black & c1_maru & c2_maru & c3_gap & c4_engulf] = 100
    return res


def cdl_counterattack(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1, _h1, _lo1 = o[1:], c[1:], h[1:], lo[1:]
    equal_close = np.abs(c1 - c0) <= 0.003 * c0
    bullish = (c0 < o0) & (c1 > o1) & (o1 < lo0) & equal_close
    bearish = (c0 > o0) & (c1 < o1) & (o1 > h0) & equal_close
    res[1:][bullish] = 100
    res[1:][bearish] = -100
    return res


def cdl_darkcloudcover(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1, _h1, _lo1 = o[1:], c[1:], h[1:], lo[1:]
    c1_white = (c0 > o0) & ((c0 - o0) >= 0.5 * (h0 - lo0))
    c2_black = c1 < o1
    gap_open = o1 > h0
    pierce_half = (c1 < (o0 + c0) / 2.0) & (c1 > o0)
    res[1:][c1_white & c2_black & gap_open & pierce_half] = -100
    return res


def cdl_dojistar(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1, h1, lo1 = o[1:], c[1:], h[1:], lo[1:]
    long_body = np.abs(c0 - o0) >= 0.5 * (h0 - lo0)
    doji = (h1 - lo1 > 0) & (np.abs(c1 - o1) <= 0.1 * (h1 - lo1))
    bullish = (c0 < o0) & long_body & doji & (np.maximum(o1, c1) < c0)
    bearish = (c0 > o0) & long_body & doji & (np.minimum(o1, c1) > c0)
    res[1:][bullish] = 100
    res[1:][bearish] = -100
    return res


def cdl_dragonflydoji(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - lo
    mask = (hl > 0) & (body <= 0.1 * hl) & (upper_shadow <= 0.05 * hl) & (lower_shadow >= 0.6 * hl)
    res[mask] = 100
    return res


def cdl_eveningdojistar(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, h1, lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2, _h2, _lo2 = o[2:], c[2:], h[2:], lo[2:]
    c1_white = (c0 > o0) & ((c0 - o0) >= 0.5 * (h0 - lo0))
    doji_gap = (h1 - lo1 > 0) & (np.abs(c1 - o1) <= 0.1 * (h1 - lo1)) & (np.minimum(o1, c1) > c0)
    c3_black = (c2 < o2) & (c2 < (o0 + c0) / 2.0) & (c2 > o0)
    res[2:][c1_white & doji_gap & c3_black] = -100
    return res


def cdl_eveningstar(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, _h1, _lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2, _h2, _lo2 = o[2:], c[2:], h[2:], lo[2:]
    c1_white = (c0 > o0) & ((c0 - o0) >= 0.5 * (h0 - lo0))
    star_gap = (np.minimum(o1, c1) > c0) & (np.abs(c1 - o1) < 0.5 * (c0 - o0))
    c3_black = (c2 < o2) & (c2 < (o0 + c0) / 2.0) & (c2 > o0)
    res[2:][c1_white & star_gap & c3_black] = -100
    return res


def cdl_gapsidesidewhite(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    _o0, _c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, h1, _lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2, _h2, _lo2 = o[2:], c[2:], h[2:], lo[2:]
    two_white = (c1 > o1) & (c2 > o2)
    side_by_side = (np.abs(o2 - o1) <= 0.01 * o1) & (np.abs(c2 - c1) <= 0.01 * c1)
    res[2:][two_white & side_by_side & (o1 > h0)] = 100
    res[2:][two_white & side_by_side & (h1 < lo0)] = -100
    return res


def cdl_gravestonedoji(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - lo
    mask = (hl > 0) & (body <= 0.1 * hl) & (lower_shadow <= 0.05 * hl) & (upper_shadow >= 0.6 * hl)
    res[mask] = 100
    return res


def cdl_haramicross(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1, h1, lo1 = o[1:], c[1:], h[1:], lo[1:]
    long_body = np.abs(c0 - o0) >= 0.5 * (h0 - lo0)
    doji = (h1 - lo1 > 0) & (np.abs(c1 - o1) <= 0.1 * (h1 - lo1))
    inside = (np.maximum(o1, c1) <= np.maximum(o0, c0)) & (np.minimum(o1, c1) >= np.minimum(o0, c0))
    res[1:][(c0 < o0) & long_body & doji & inside] = 100
    res[1:][(c0 > o0) & long_body & doji & inside] = -100
    return res


def cdl_highwave(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - lo
    shape = (hl > 0) & (body <= 0.25 * hl) & (upper_shadow >= 0.3 * hl) & (lower_shadow >= 0.3 * hl)
    res[shape & (c > o)] = 100
    res[shape & (c < o)] = -100
    return res


def cdl_hikkake(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    n = len(o)
    if n < 4:
        return res
    for i in range(3, n):
        inside = (h[i - 2] < h[i - 3]) & (lo[i - 2] > lo[i - 3])
        if not inside:
            continue
        if (lo[i - 1] < lo[i - 2]) and (c[i] > h[i - 2]):
            res[i] = 100
        elif (h[i - 1] > h[i - 2]) and (c[i] < lo[i - 2]):
            res[i] = -100
    return res


def cdl_hikkakemod(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    n = len(o)
    if n < 5:
        return res
    for i in range(4, n):
        inside = (h[i - 2] < h[i - 3]) & (lo[i - 2] > lo[i - 3])
        if not inside:
            continue
        if (lo[i - 1] < lo[i - 2]) and (c[i] > h[i - 2]):
            res[i] = 100
        elif (h[i - 1] > h[i - 2]) and (c[i] < lo[i - 2]):
            res[i] = -100
    return res


def cdl_homingpigeon(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1, _h1, _lo1 = o[1:], c[1:], h[1:], lo[1:]
    two_black = (c0 < o0) & (c1 < o1)
    long_black = (o0 - c0) >= 0.5 * (h0 - lo0)
    inside_body = (o1 <= o0) & (c1 >= c0)
    res[1:][two_black & long_black & inside_body] = 100
    return res


def cdl_identical3crows(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0 = o[:-2], c[:-2]
    o1, c1 = o[1:-1], c[1:-1]
    o2, c2 = o[2:], c[2:]
    three_black = (c0 < o0) & (c1 < o1) & (c2 < o2) & (c2 < c1) & (c1 < c0)
    identical_opens = (np.abs(o1 - c0) <= 0.003 * c0) & (np.abs(o2 - c1) <= 0.003 * c1)
    res[2:][three_black & identical_opens] = -100
    return res


def cdl_inneck(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1, _h1, _lo1 = o[1:], c[1:], h[1:], lo[1:]
    c1_black = (c0 < o0) & ((o0 - c0) >= 0.5 * (h0 - lo0))
    c2_white = (c1 > o1) & (o1 < lo0)
    in_neck = (c1 >= c0) & (c1 <= c0 + 0.05 * (o0 - c0))
    res[1:][c1_black & c2_white & in_neck] = -100
    return res


def cdl_kicking(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1, h1, lo1 = o[1:], c[1:], h[1:], lo[1:]
    maru0 = np.abs(c0 - o0) >= 0.8 * (h0 - lo0)
    maru1 = np.abs(c1 - o1) >= 0.8 * (h1 - lo1)
    bullish = maru0 & maru1 & (c0 < o0) & (c1 > o1) & (lo1 > h0)
    bearish = maru0 & maru1 & (c0 > o0) & (c1 < o1) & (h1 < lo0)
    res[1:][bullish] = 100
    res[1:][bearish] = -100
    return res


def cdl_kickingbylength(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1, h1, lo1 = o[1:], c[1:], h[1:], lo[1:]
    body0 = np.abs(c0 - o0)
    body1 = np.abs(c1 - o1)
    maru0 = body0 >= 0.7 * (h0 - lo0)
    maru1 = body1 >= 0.7 * (h1 - lo1)
    opposite = (c0 > o0) != (c1 > o1)
    bullish = maru0 & maru1 & opposite & (c1 > o1) & (body1 > body0)
    bearish = maru0 & maru1 & opposite & (c1 < o1) & (body1 > body0)
    res[1:][bullish] = 100
    res[1:][bearish] = -100
    return res


def cdl_ladderbottom(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 5:
        return res
    o0, c0 = o[:-4], c[:-4]
    o1, c1 = o[1:-3], c[1:-3]
    o2, c2 = o[2:-2], c[2:-2]
    o3, c3, h3 = o[3:-1], c[3:-1], h[3:-1]
    o4, c4 = o[4:], c[4:]
    three_black = (c0 < o0) & (c1 < o1) & (c2 < o2) & (c1 < c0) & (c2 < c1)
    c4_inverted = (c3 < o3) & ((h3 - np.maximum(o3, c3)) >= (o3 - c3))
    c5_white = (c4 > o4) & (o4 > o3)
    res[4:][three_black & c4_inverted & c5_white] = 100
    return res


def cdl_longleggeddoji(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - lo
    mask = (hl > 0) & (body <= 0.1 * hl) & (upper_shadow >= 0.3 * hl) & (lower_shadow >= 0.3 * hl)
    res[mask] = 100
    return res


def cdl_longline(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    long_body = (hl > 0) & (body >= 0.7 * hl)
    res[long_body & (c > o)] = 100
    res[long_body & (c < o)] = -100
    return res


def cdl_matchinglow(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0 = o[:-1], c[:-1]
    o1, c1 = o[1:], c[1:]
    two_black = (c0 < o0) & (c1 < o1)
    matching_close = np.abs(c1 - c0) <= 0.002 * c0
    res[1:][two_black & matching_close] = 100
    return res


def cdl_mathold(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 5:
        return res
    o0, c0, h0, lo0 = o[:-4], c[:-4], h[:-4], lo[:-4]
    o1, c1 = o[1:-3], c[1:-3]
    _o2, c2 = o[2:-2], c[2:-2]
    _o3, c3 = o[3:-1], c[3:-1]
    o4, c4 = o[4:], c[4:]
    c1_white = (c0 > o0) & ((c0 - o0) >= 0.5 * (h0 - lo0))
    gap_up = o1 > c0
    hold_above = (c1 > o0) & (c2 > o0) & (c3 > o0)
    c5_breakout = (c4 > o4) & (c4 > h0)
    res[4:][c1_white & gap_up & hold_above & c5_breakout] = 100
    return res


def cdl_morningdojistar(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, h1, lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2, _h2, _lo2 = o[2:], c[2:], h[2:], lo[2:]
    c1_black = (c0 < o0) & ((o0 - c0) >= 0.5 * (h0 - lo0))
    doji_gap = (h1 - lo1 > 0) & (np.abs(c1 - o1) <= 0.1 * (h1 - lo1)) & (np.maximum(o1, c1) < c0)
    c3_white = (c2 > o2) & (c2 > (o0 + c0) / 2.0) & (c2 < o0)
    res[2:][c1_black & doji_gap & c3_white] = 100
    return res


def cdl_morningstar(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1 = o[1:-1], c[1:-1]
    o2, c2 = o[2:], c[2:]
    c1_black = (c0 < o0) & ((o0 - c0) >= 0.5 * (h0 - lo0))
    star_gap = (np.maximum(o1, c1) < c0) & (np.abs(c1 - o1) < 0.5 * (o0 - c0))
    c3_white = (c2 > o2) & (c2 > (o0 + c0) / 2.0) & (c2 < o0)
    res[2:][c1_black & star_gap & c3_white] = 100
    return res


def cdl_onneck(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1 = o[1:], c[1:]
    c1_black = (c0 < o0) & ((o0 - c0) >= 0.5 * (h0 - lo0))
    c2_white = (c1 > o1) & (o1 < lo0)
    on_neck = np.abs(c1 - lo0) <= 0.003 * lo0
    res[1:][c1_black & c2_white & on_neck] = -100
    return res


def cdl_piercing(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1 = o[1:], c[1:]
    c1_black = (c0 < o0) & ((o0 - c0) >= 0.5 * (h0 - lo0))
    c2_white = (c1 > o1) & (o1 < lo0)
    pierce_half = (c1 > (o0 + c0) / 2.0) & (c1 < o0)
    res[1:][c1_black & c2_white & pierce_half] = 100
    return res


def cdl_rickshawman(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - lo
    body_mid = (o + c) / 2.0
    candle_mid = (h + lo) / 2.0
    mask = (
        (hl > 0)
        & (body <= 0.1 * hl)
        & (upper_shadow >= 0.3 * hl)
        & (lower_shadow >= 0.3 * hl)
        & (np.abs(body_mid - candle_mid) <= 0.1 * hl)
    )
    res[mask] = 100
    return res


def cdl_risefall3methods(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 5:
        return res
    o0, c0, h0, lo0 = o[:-4], c[:-4], h[:-4], lo[:-4]
    _o1, _c1, h1, lo1 = o[1:-3], c[1:-3], h[1:-3], lo[1:-3]
    _o2, _c2, h2, lo2 = o[2:-2], c[2:-2], h[2:-2], lo[2:-2]
    _o3, _c3, h3, lo3 = o[3:-1], c[3:-1], h[3:-1], lo[3:-1]
    o4, c4 = o[4:], c[4:]
    c1_white = (c0 > o0) & ((c0 - o0) >= 0.5 * (h0 - lo0))
    inside_3 = (h1 < h0) & (lo1 > lo0) & (h2 < h0) & (lo2 > lo0) & (h3 < h0) & (lo3 > lo0)
    bullish = c1_white & inside_3 & (c4 > o4) & (c4 > c0)
    c1_black = (c0 < o0) & ((o0 - c0) >= 0.5 * (h0 - lo0))
    bearish = c1_black & inside_3 & (c4 < o4) & (c4 < c0)
    res[4:][bullish] = 100
    res[4:][bearish] = -100
    return res


def cdl_separatinglines(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0 = o[:-1], c[:-1]
    o1, c1 = o[1:], c[1:]
    equal_open = np.abs(o1 - o0) <= 0.003 * o0
    res[1:][(c0 < o0) & (c1 > o1) & equal_open] = 100
    res[1:][(c0 > o0) & (c1 < o1) & equal_open] = -100
    return res


def cdl_shortline(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    avg_hl = np.mean(hl) if len(hl) > 0 else 1.0
    short = (hl > 0) & (body <= 0.4 * hl) & (hl <= 0.5 * avg_hl)
    res[short & (c > o)] = 100
    res[short & (c < o)] = -100
    return res


def cdl_stalledpattern(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, h1, lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2 = o[2:], c[2:]
    trend = _prior_trend_up(c, 5)[2:]
    three_white = (c0 > o0) & (c1 > o1) & (c2 > o2) & (c2 > c1) & (c1 > c0)
    c1_c2_long = (c0 - o0 >= 0.4 * (h0 - lo0)) & (c1 - o1 >= 0.4 * (h1 - lo1))
    c3_stalled = (c2 - o2 <= 0.3 * (c1 - o1)) & (o2 >= c1 - 0.1 * (c1 - o1))
    res[2:][trend & three_white & c1_c2_long & c3_stalled] = -100
    return res


def cdl_sticksandwich(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0 = o[:-2], c[:-2]
    o1, c1 = o[1:-1], c[1:-1]
    o2, c2 = o[2:], c[2:]
    black_white_black = (c0 < o0) & (c1 > o1) & (c2 < o2)
    higher_white = c1 > c0
    matching_close = np.abs(c2 - c0) <= 0.003 * c0
    res[2:][black_white_black & higher_white & matching_close] = 100
    return res


def cdl_takuri(open_, high, low, close, trend_lookback: int = 5):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - lo
    res = np.zeros(len(o), dtype=np.int32)
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - lo
    shape = (
        (hl > 0) & (body <= 0.1 * hl) & (upper_shadow <= 0.05 * hl) & (lower_shadow >= 0.65 * hl)
    )
    mask = shape & _prior_trend_down(c, trend_lookback)
    res[mask] = 100
    return res


def cdl_tasukigap(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0 = o[:-2], c[:-2]
    o1, c1 = o[1:-1], c[1:-1]
    o2, c2 = o[2:], c[2:]
    bull_gap = (c0 > o0) & (c1 > o1) & (o1 > c0)
    bull_retrace = (c2 < o2) & (o2 < c1) & (o2 > o1) & (c2 < o1) & (c2 > c0)
    bear_gap = (c0 < o0) & (c1 < o1) & (o1 < c0)
    bear_retrace = (c2 > o2) & (o2 > c1) & (o2 < o1) & (c2 > o1) & (c2 < c0)
    res[2:][bull_gap & bull_retrace] = 100
    res[2:][bear_gap & bear_retrace] = -100
    return res


def cdl_thrusting(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 2:
        return res
    o0, c0, h0, lo0 = o[:-1], c[:-1], h[:-1], lo[:-1]
    o1, c1 = o[1:], c[1:]
    c1_black = (c0 < o0) & ((o0 - c0) >= 0.5 * (h0 - lo0))
    c2_white = (c1 > o1) & (o1 < lo0)
    thrust = (c1 > c0) & (c1 < (o0 + c0) / 2.0)
    res[1:][c1_black & c2_white & thrust] = -100
    return res


def cdl_tristar(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, h1, lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2, h2, lo2 = o[2:], c[2:], h[2:], lo[2:]
    d0 = (h0 - lo0 > 0) & (np.abs(c0 - o0) <= 0.1 * (h0 - lo0))
    d1 = (h1 - lo1 > 0) & (np.abs(c1 - o1) <= 0.1 * (h1 - lo1))
    d2 = (h2 - lo2 > 0) & (np.abs(c2 - o2) <= 0.1 * (h2 - lo2))
    three_doji = d0 & d1 & d2
    bullish = (
        three_doji
        & (np.maximum(o1, c1) < np.minimum(o0, c0))
        & (np.minimum(o2, c2) > np.maximum(o1, c1))
    )
    bearish = (
        three_doji
        & (np.minimum(o1, c1) > np.maximum(o0, c0))
        & (np.maximum(o2, c2) < np.minimum(o1, c1))
    )
    res[2:][bullish] = 100
    res[2:][bearish] = -100
    return res


def cdl_unique3river(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, h1, lo1 = o[1:-1], c[1:-1], h[1:-1], lo[1:-1]
    o2, c2, h2, lo2 = o[2:], c[2:], h[2:], lo[2:]
    trend = _prior_trend_down(c, 5)[2:]
    c1_black = (c0 < o0) & ((o0 - c0) >= 0.5 * (h0 - lo0))
    c2_harami_low = (c1 < o1) & (lo1 < lo0) & (o1 <= o0) & (c1 >= c0)
    c3_white = (c2 > o2) & (h2 < h1) & (lo2 > lo1)
    res[2:][trend & c1_black & c2_harami_low & c3_white] = 100
    return res


def cdl_upsidegap2crows(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0, h0, lo0 = o[:-2], c[:-2], h[:-2], lo[:-2]
    o1, c1, lo1 = o[1:-1], c[1:-1], lo[1:-1]
    o2, c2 = o[2:], c[2:]
    c1_white = (c0 > o0) & ((c0 - o0) >= 0.5 * (h0 - lo0))
    c2_gap_black = (c1 < o1) & (lo1 > c0)
    c3_black = (c2 < o2) & (o2 > o1) & (c2 < c1) & (c2 > c0)
    res[2:][c1_white & c2_gap_black & c3_black] = -100
    return res


def cdl_xsidegap3methods(open_, high, low, close):
    o, h, lo, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    o0, c0 = o[:-2], c[:-2]
    o1, c1 = o[1:-1], c[1:-1]
    o2, c2 = o[2:], c[2:]
    bull_two = (c0 > o0) & (c1 > o1) & (o1 > c0)
    bull_fill = (c2 < o2) & (o2 < c1) & (o2 > o1) & (c2 < o1) & (c2 > o0)
    bear_two = (c0 < o0) & (c1 < o1) & (o1 < c0)
    bear_fill = (c2 > o2) & (o2 > c1) & (o2 < o1) & (c2 > o1) & (c2 < c0)
    res[2:][bull_two & bull_fill] = 100
    res[2:][bear_two & bear_fill] = -100
    return res


# All 61 canonical TA-Lib patterns registered
CUSTOM_PATTERNS = {
    "CDL2CROWS": cdl_2crows,
    "CDL3BLACKCROWS": cdl_3blackcrows,
    "CDL3INSIDE": cdl_3inside,
    "CDL3LINESTRIKE": cdl_3linestrike,
    "CDL3OUTSIDE": cdl_3outside,
    "CDL3STARSINSOUTH": cdl_3starsinsouth,
    "CDL3WHITESOLDIERS": cdl_3whitesoldiers,
    "CDLABANDONEDBABY": cdl_abandonedbaby,
    "CDLADVANCEBLOCK": cdl_advanceblock,
    "CDLBELTHOLD": cdl_belthold,
    "CDLBREAKAWAY": cdl_breakaway,
    "CDLCLOSINGMARUBOZU": cdl_closingmarubozu,
    "CDLCONCEALBABYSWALL": cdl_concealbabyswall,
    "CDLCOUNTERATTACK": cdl_counterattack,
    "CDLDARKCLOUDCOVER": cdl_darkcloudcover,
    "CDLDOJI": cdl_doji,
    "CDLDOJISTAR": cdl_dojistar,
    "CDLDRAGONFLYDOJI": cdl_dragonflydoji,
    "CDLENGULFING": cdl_engulfing,
    "CDLEVENINGDOJISTAR": cdl_eveningdojistar,
    "CDLEVENINGSTAR": cdl_eveningstar,
    "CDLGAPSIDESIDEWHITE": cdl_gapsidesidewhite,
    "CDLGRAVESTONEDOJI": cdl_gravestonedoji,
    "CDLHAMMER": cdl_hammer,
    "CDLHANGINGMAN": cdl_hangingman,
    "CDLHARAMI": cdl_harami,
    "CDLHARAMICROSS": cdl_haramicross,
    "CDLHIGHWAVE": cdl_highwave,
    "CDLHIKKAKE": cdl_hikkake,
    "CDLHIKKAKEMOD": cdl_hikkakemod,
    "CDLHOMINGPIGEON": cdl_homingpigeon,
    "CDLIDENTICAL3CROWS": cdl_identical3crows,
    "CDLINNECK": cdl_inneck,
    "CDLINVERTEDHAMMER": cdl_invertedhammer,
    "CDLKICKING": cdl_kicking,
    "CDLKICKINGBYLENGTH": cdl_kickingbylength,
    "CDLLADDERBOTTOM": cdl_ladderbottom,
    "CDLLONGLEGGEDDOJI": cdl_longleggeddoji,
    "CDLLONGLINE": cdl_longline,
    "CDLMARUBOZU": cdl_marubozu,
    "CDLMATCHINGLOW": cdl_matchinglow,
    "CDLMATHOLD": cdl_mathold,
    "CDLMORNINGDOJISTAR": cdl_morningdojistar,
    "CDLMORNINGSTAR": cdl_morningstar,
    "CDLONNECK": cdl_onneck,
    "CDLPIERCING": cdl_piercing,
    "CDLRICKSHAWMAN": cdl_rickshawman,
    "CDLRISEFALL3METHODS": cdl_risefall3methods,
    "CDLSEPARATINGLINES": cdl_separatinglines,
    "CDLSHOOTINGSTAR": cdl_shootingstar,
    "CDLSHORTLINE": cdl_shortline,
    "CDLSPINNINGTOP": cdl_spinningtop,
    "CDLSTALLEDPATTERN": cdl_stalledpattern,
    "CDLSTICKSANDWICH": cdl_sticksandwich,
    "CDLTAKURI": cdl_takuri,
    "CDLTASUKIGAP": cdl_tasukigap,
    "CDLTHRUSTING": cdl_thrusting,
    "CDLTRISTAR": cdl_tristar,
    "CDLUNIQUE3RIVER": cdl_unique3river,
    "CDLUPSIDEGAP2CROWS": cdl_upsidegap2crows,
    "CDLXSIDEGAP3METHODS": cdl_xsidegap3methods,
}

SUPPORTED_FALLBACK_PATTERNS: frozenset[str] = frozenset(CUSTOM_PATTERNS.keys())
UNSUPPORTED_FALLBACK_PATTERNS: frozenset[str] = frozenset()


def _make_unsupported_pattern(name: str):
    def _unsupported(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Candlestick pattern '{name}' is not recognized. "
            f"Pure-Python fallback engine implements all {len(CUSTOM_PATTERNS)} TA-Lib patterns: "
            f"{', '.join(sorted(CUSTOM_PATTERNS.keys()))}."
        )

    return _unsupported


class TALibWrapper:
    """Wrapper that routes to native TA-Lib if available, or pure-Python / NumPy fallback engine."""

    def __init__(self, force_fallback: bool = False) -> None:
        self._has_native = HAS_NATIVE_TALIB and not force_fallback

    def __getattr__(self, name: str) -> Any:
        if self._has_native and hasattr(_talib, name):
            return getattr(_talib, name)

        upper = name.upper().replace("_", "")
        if not upper.startswith("CDL") and f"CDL{upper}" in CUSTOM_PATTERNS:
            upper = f"CDL{upper}"

        if upper in CUSTOM_PATTERNS:
            return CUSTOM_PATTERNS[upper]
        if upper.startswith("CDL") and upper in ALL_CDL_PATTERNS:
            return _make_unsupported_pattern(upper)

        if name == "HAS_NATIVE_TALIB":
            return False

        raise AttributeError(f"Module 'talib' has no attribute '{name}'")

    def __dir__(self) -> list[str]:
        if self._has_native:
            return dir(_talib)
        return [*ALL_CDL_PATTERNS, "HAS_NATIVE_TALIB"]


talib: Any = _talib if HAS_NATIVE_TALIB else TALibWrapper()


def is_freethreaded() -> bool:
    """Return True if running under a free-threaded (No-GIL, PEP 703) CPython build."""
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED") == 1)


def is_gil_enabled() -> bool:
    """Return True if the Global Interpreter Lock (GIL) is currently active.

    On free-threaded CPython (3.13t+), queries sys._is_gil_enabled().
    On standard CPython and PyPy, returns True.
    """
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    if callable(gil_probe):
        try:
            return bool(gil_probe())
        except Exception:  # pragma: no cover
            return True
    return True


def get_talib_status() -> dict[str, Any]:
    """Return diagnostic details regarding native TA-Lib availability and system environment."""
    is_pypy = sys.implementation.name == "pypy"
    is_windows = sys.platform.startswith("win")
    is_macos = sys.platform == "darwin"
    is_linux = sys.platform.startswith("linux")
    is_free_threaded = is_freethreaded()
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    try:
        gil_enabled = bool(gil_probe()) if callable(gil_probe) else None
    except Exception:  # pragma: no cover - interpreter-specific diagnostic API
        gil_enabled = None

    gil_cause: str | None = None
    gil_recommendation: str | None = None

    if is_free_threaded:
        if gil_enabled is True:
            env_gil = os.environ.get("PYTHON_GIL")
            if env_gil == "1":
                gil_cause = "Forced enabled by PYTHON_GIL=1 environment variable."
            elif HAS_NATIVE_TALIB:
                gil_cause = (
                    "Automatically re-enabled by CPython PEP 703 runtime protection upon importing "
                    "native 'talib._ta_lib' (which lacks Py_MOD_GIL_NOT_USED declaration)."
                )
            else:
                gil_cause = (
                    "Re-enabled by non-free-threaded C extension or default interpreter policy."
                )
            gil_recommendation = "Run with 'python -X gil=0 <script>' or set PYTHON_GIL=0 in environment to maintain No-GIL execution."
        elif gil_enabled is False:
            gil_cause = "GIL is disabled (running in true multi-core No-GIL mode)."
            gil_recommendation = None

    err_str = str(TALIB_IMPORT_ERROR) if TALIB_IMPORT_ERROR is not None else None
    if HAS_NATIVE_TALIB:
        reason = "Native C TA-Lib successfully loaded."
    elif is_windows and is_pypy:
        reason = (
            "Native TA-Lib binary not found for Windows PyPy. PyPI does not host "
            "precompiled PyPy Windows wheels for TA-Lib. Use --find-links with GitHub Releases "
            "to install prebuilt wheels."
        )
    elif TALIB_IMPORT_ERROR is not None:
        reason = f"Native TA-Lib import failed: {err_str}"
    else:
        reason = "Native TA-Lib package is not installed."

    return {
        "has_native_talib": HAS_NATIVE_TALIB,
        "import_error": err_str,
        "is_pypy": is_pypy,
        "is_windows": is_windows,
        "is_macos": is_macos,
        "is_linux": is_linux,
        "is_free_threaded": is_free_threaded,
        "gil_enabled": gil_enabled,
        "gil_cause": gil_cause,
        "gil_recommendation": gil_recommendation,
        "python_version": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "reason": reason,
    }


def get_talib_install_hint(release_tag: str = "v0.3.30") -> str:
    """Return actionable platform-specific instructions to install or repair native TA-Lib."""
    if HAS_NATIVE_TALIB:
        return "Native TA-Lib is already installed and available."

    is_pypy = sys.implementation.name == "pypy"
    is_windows = sys.platform.startswith("win")
    is_macos = sys.platform == "darwin"
    is_linux = sys.platform.startswith("linux")

    find_links_url = (
        f"https://github.com/eminsk/yfinance-ta-patterns/releases/expanded_assets/{release_tag}"
    )
    is_free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    windows_wheelhouse_url = (
        "https://github.com/eminsk/yfinance-ta-patterns/releases/expanded_assets/"
        f"{WINDOWS_TALIB_WHEELHOUSE_TAG}"
    )
    needs_windows_wheelhouse = (
        is_windows and not is_pypy and (is_free_threaded or sys.version_info >= (3, 15))
    )

    lines: list[str] = []
    if is_windows and is_free_threaded and sys.version_info[:2] == (3, 13):
        lines.append("Note on Windows Free-Threaded Python 3.13t:")
        lines.append(
            "  PyPI does not host precompiled cp313t-win_amd64 wheels for core dependencies "
            "(lxml, numpy, scipy). Building from source requires MSVC and libxml2 development headers."
        )
        lines.append(
            "  Recommended action: Upgrade to Python 3.14t or 3.15t where official prebuilt wheels "
            "are available on PyPI, or install yfinance-ta-patterns without the [all] extra:"
        )
        lines.append("    uv add yfinance-ta-patterns")
        lines.append("  If native TA-Lib is desired on Windows, use Python 3.14t or 3.15t:")

    if needs_windows_wheelhouse:
        runtime = "Windows free-threaded CPython" if is_free_threaded else "Windows CPython 3.15+"
        lines.append(f"To enable native TA-Lib on {runtime}:")
        lines.append("  [PowerShell + uv]:")
        if is_free_threaded:
            lines.append('    $env:PYTHON_GIL = "0"')
        lines.append(
            f'    uv add "yfinance-ta-patterns[all]" --find-links {windows_wheelhouse_url}'
        )
        lines.append("  [PowerShell + pip]:")
        if is_free_threaded:
            lines.append('    $env:PYTHON_GIL = "0"')
        lines.append(
            f'    pip install "yfinance-ta-patterns[all]" --find-links {windows_wheelhouse_url}'
        )
        if is_free_threaded:
            lines.append(
                "  The wheelhouse supplies matching pandas, curl_cffi, and TA-Lib wheels "
                "for CPython 3.14t and 3.15t."
            )
        else:
            lines.append(
                "  The wheelhouse supplies a TA-Lib cp315 wheel, but PyPI does not yet provide "
                "a Windows pandas cp315 wheel; pandas must be built from source."
            )
    elif is_windows and is_pypy:
        lines.append("To enable all 61 native TA-Lib candlestick patterns on Windows PyPy:")
        lines.append("  [uv]:")
        lines.append(f'    uv add "yfinance-ta-patterns[all]" --find-links {find_links_url}')
        lines.append("  [pip]:")
        lines.append(f'    pip install "ta-lib>=0.4.19,<=0.7.1" --find-links {find_links_url}')
        lines.append("  [pyproject.toml]:")
        lines.append("    [tool.uv]")
        lines.append(f'    find-links = ["{find_links_url}"]')
    elif is_macos:
        lines.append("To install native TA-Lib on macOS:")
        lines.append("  brew install ta-lib")
        lines.append("  pip install ta-lib")
    elif is_linux:
        lines.append("To install native TA-Lib on Linux (Ubuntu/Debian):")
        lines.append("  sudo apt-get install -y libta-lib0 libta-lib-dev")
        lines.append("  pip install ta-lib")
    else:
        lines.append("To install native TA-Lib on Windows (CPython):")
        lines.append("  pip install ta-lib")
        lines.append(
            f"  or install prebuilt wheels via: pip install ta-lib --find-links {find_links_url}"
        )

    return "\n".join(lines)
