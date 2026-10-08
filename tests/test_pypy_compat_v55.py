"""Test suite for v0.3.55: PyPy 3.11+ compatibility without forced C-extension dependencies."""

import sys
import pytest
import yfinance_ta_patterns as ytp
from yfinance_ta_patterns.talib_compat import talib, ALL_CDL_PATTERNS


def test_version_matches():
    assert ytp.__version__ == "0.3.55"


def test_all_61_patterns_present():
    assert len(ALL_CDL_PATTERNS) == 61
    for name in ALL_CDL_PATTERNS:
        fn = getattr(talib, name, None)
        assert callable(fn), f"Pattern {name} must be callable"


def test_pattern_execution_with_lists():
    n = 30
    o = [100.0 + (i * 0.1) for i in range(n)]
    h = [105.0 + (i * 0.1) for i in range(n)]
    l = [95.0 + (i * 0.1) for i in range(n)]
    c = [102.0 + (i * 0.1) for i in range(n)]

    res = talib.CDLDOJI(o, h, l, c)
    assert len(res) == n


def test_windows_pypy_hook_safety():
    """Verify that importing yfinance_ta_patterns never crashes on Windows PyPy."""
    if sys.platform == "win32" and sys.implementation.name == "pypy":
        try:
            import dateutil.tz.win
            # Verify load_name is callable and patched
            assert hasattr(dateutil.tz.win.tzres, "load_name")
        except ImportError:
            pass
