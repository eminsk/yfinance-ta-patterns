#!/usr/bin/env python3
"""Verify native extras keep the GIL disabled on free-threaded CPython."""

from __future__ import annotations

import sys
import sysconfig


def _gil_enabled() -> bool:
    probe = getattr(sys, "_is_gil_enabled", None)
    if not callable(probe):
        raise RuntimeError("This Python build does not expose sys._is_gil_enabled().")
    return bool(probe())


def main() -> int:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("This verifier requires a free-threaded CPython build.")
    if _gil_enabled():
        raise RuntimeError("GIL is enabled. Set PYTHON_GIL=0 or pass -X gil=0 when starting Python.")

    import numpy as np
    import pandas as pd
    import sklearn
    import talib
    import yfinance

    from yfinance_ta_patterns.pattern_analyzer import PatternAnalyzer
    from yfinance_ta_patterns.talib_compat import ALL_CDL_PATTERNS, HAS_NATIVE_TALIB

    count = 64
    base = np.linspace(1.0, 1.2, count)
    opens = base + np.sin(np.arange(count)) * 0.001
    closes = base + np.cos(np.arange(count)) * 0.001
    data = pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(opens, closes) + 0.002,
            "Low": np.minimum(opens, closes) - 0.002,
            "Close": closes,
        }
    )

    analyzer = PatternAnalyzer(data)
    if not HAS_NATIVE_TALIB:
        raise AssertionError("Native TA-Lib is required for the all-extras verifier.")
    if set(analyzer.pattern_functions) != set(ALL_CDL_PATTERNS):
        raise AssertionError("PatternAnalyzer does not expose all 61 native TA-Lib patterns.")

    for pattern in analyzer.pattern_functions:
        analyzer.get_signals(pattern)

    if _gil_enabled():
        raise AssertionError("A native dependency enabled the GIL during verification.")

    print(
        "Free-threaded verification passed: "
        f"Python {sys.version.split()[0]}, TA-Lib {talib.__version__}, "
        f"scikit-learn {sklearn.__version__}, yfinance {yfinance.__version__}, "
        f"patterns {len(analyzer.pattern_functions)}, GIL disabled."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
