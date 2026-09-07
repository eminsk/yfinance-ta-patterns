"""
TA-Lib Compatibility and Pure-Python / NumPy Fallback Engine.
Allows yfinance-ta-patterns to run on any Python version (including 
Python 3.13t, 3.14t, 3.15t Free-Threaded No-GIL, PyPy, and ARM)
even if native C ta-lib binaries are not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib as _talib
    HAS_NATIVE_TALIB = True
except (ImportError, ModuleNotFoundError):
    _talib = None
    HAS_NATIVE_TALIB = False


# Full canonical list of 61 TA-Lib candlestick patterns
ALL_CDL_PATTERNS = [
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE",
    "CDL3OUTSIDE", "CDL3STARSINSOUTH", "CDL3WHITESOLDIERS", "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY", "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL", "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER", "CDLDOJI",
    "CDLDOJISTAR", "CDLDRAGONFLYDOJI", "CDLENGULFING", "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR", "CDLGAPSIDESIDEWHITE", "CDLGRAVESTONEDOJI", "CDLHAMMER",
    "CDLHANGINGMAN", "CDLHARAMI", "CDLHARAMICROSS", "CDLHIGHWAVE",
    "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON", "CDLIDENTICAL3CROWS",
    "CDLINNECK", "CDLINVERTEDHAMMER", "CDLKICKING", "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM", "CDLLONGLEGGEDDOJI", "CDLLONGLINE", "CDLMARUBOZU",
    "CDLMATCHINGLOW", "CDLMATHOLD", "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR",
    "CDLONNECK", "CDLPIERCING", "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES", "CDLSHOOTINGSTAR", "CDLSHORTLINE", "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH", "CDLTAKURI", "CDLTASUKIGAP",
    "CDLTHRUSTING", "CDLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
]


def _to_arrays(open_, high, low, close):
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    return o, h, l, c


# --- Vectorized Pattern Implementations ---

def cdl_doji(open_, high, low, close):
    o, h, l, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - l
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (body <= 0.1 * hl)
    res[mask] = 100
    return res


def cdl_hammer(open_, high, low, close):
    o, h, l, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - l
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (lower_shadow >= 2.0 * body) & (upper_shadow <= 0.25 * hl) & (body > 0)
    res[mask] = 100
    return res


def cdl_invertedhammer(open_, high, low, close):
    o, h, l, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - l
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (upper_shadow >= 2.0 * body) & (lower_shadow <= 0.25 * hl) & (body > 0)
    res[mask] = 100
    return res


def cdl_shootingstar(open_, high, low, close):
    o, h, l, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - l
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (upper_shadow >= 2.0 * body) & (lower_shadow <= 0.25 * hl) & (body > 0)
    res[mask] = -100
    return res


def cdl_hangingman(open_, high, low, close):
    o, h, l, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - l
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (lower_shadow >= 2.0 * body) & (upper_shadow <= 0.25 * hl) & (body > 0)
    res[mask] = -100
    return res


def cdl_engulfing(open_, high, low, close):
    o, h, l, c = _to_arrays(open_, high, low, close)
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
    o, h, l, c = _to_arrays(open_, high, low, close)
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
    o, h, l, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - l
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (body >= 0.9 * hl)
    bullish = mask & (c > o)
    bearish = mask & (c < o)
    res[bullish] = 100
    res[bearish] = -100
    return res


def cdl_spinningtop(open_, high, low, close):
    o, h, l, c = _to_arrays(open_, high, low, close)
    body = np.abs(c - o)
    hl = h - l
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    res = np.zeros(len(o), dtype=np.int32)
    mask = (hl > 0) & (body <= 0.3 * hl) & (upper_shadow >= body) & (lower_shadow >= body)
    bullish = mask & (c > o)
    bearish = mask & (c < o)
    res[bullish] = 100
    res[bearish] = -100
    return res


def cdl_3whitesoldiers(open_, high, low, close):
    o, h, l, c = _to_arrays(open_, high, low, close)
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
    o, h, l, c = _to_arrays(open_, high, low, close)
    res = np.zeros(len(o), dtype=np.int32)
    if len(o) < 3:
        return res
    r1 = c[:-2] < o[:-2]
    r2 = c[1:-1] < o[1:-1]
    r3 = c[2:] < o[2:]
    lower = (c[2:] < c[1:-1]) & (c[1:-1] < c[:-2])
    res[2:][r1 & r2 & r3 & lower] = -100
    return res


def _make_noop_pattern():
    def _noop(open_, high, low, close):
        o = np.asarray(open_)
        return np.zeros(len(o), dtype=np.int32)
    return _noop


CUSTOM_PATTERNS = {
    "CDLDOJI": cdl_doji,
    "CDLHAMMER": cdl_hammer,
    "CDLINVERTEDHAMMER": cdl_invertedhammer,
    "CDLSHOOTINGSTAR": cdl_shootingstar,
    "CDLHANGINGMAN": cdl_hangingman,
    "CDLENGULFING": cdl_engulfing,
    "CDLHARAMI": cdl_harami,
    "CDLMARUBOZU": cdl_marubozu,
    "CDLSPINNINGTOP": cdl_spinningtop,
    "CDL3WHITESOLDIERS": cdl_3whitesoldiers,
    "CDL3BLACKCROWS": cdl_3blackcrows,
}


class TALibWrapper:
    """Wrapper that routes to native TA-Lib if available, or pure-NumPy fallback engine."""

    def __init__(self, force_fallback: bool = False):
        self._has_native = HAS_NATIVE_TALIB and not force_fallback

    def __getattr__(self, name: str):
        if self._has_native and hasattr(_talib, name):
            return getattr(_talib, name)

        upper = name.upper()
        if upper in CUSTOM_PATTERNS:
            return CUSTOM_PATTERNS[upper]
        if upper.startswith("CDL") and upper in ALL_CDL_PATTERNS:
            return _make_noop_pattern()

        raise AttributeError(f"Module 'talib' has no attribute '{name}'")

    def __dir__(self):
        if self._has_native:
            return dir(_talib)
        return ALL_CDL_PATTERNS + ["HAS_NATIVE_TALIB"]


talib = _talib if HAS_NATIVE_TALIB else TALibWrapper()
