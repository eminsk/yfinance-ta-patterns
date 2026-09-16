"""YFinance TA Patterns: Advanced Technical Candlestick Analysis with AI Intelligence."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("yfinance-ta-patterns")
except PackageNotFoundError:  # pragma: no cover - during editable installs
    __version__ = "0.3.34"

import sys

if sys.version_info < (3, 9) and sys.implementation.name != "pypy":
    from ._compat_hook import install_compat_hook
    install_compat_hook()

from .ai import (
    AIMarketAnalyst,
    AIPatternScorer,
    PatternConfidenceResult,
    SignalGrade,
    TradeSetup,
)
from .data import (
    ALLOWED_ASSET_TYPES,
    MarketDataLoader,
    classify_asset,
    normalize_ticker,
    resolve_asset_currencies,
    validate_asset_type,
)
from .forex_data_loader import (
    FOREX_56_PAIRS,
    FOREX_MAJOR_CURRENCIES,
    ForexDataLoader,
)
from .pattern_analyzer import PatternAnalyzer
from .pattern_tester import PatternRankingTester, PatternResult
from .talib_compat import (
    HAS_NATIVE_TALIB,
    TALIB_IMPORT_ERROR,
    get_talib_install_hint,
    get_talib_status,
)

__all__ = [
    "ALLOWED_ASSET_TYPES",
    "FOREX_56_PAIRS",
    "FOREX_MAJOR_CURRENCIES",
    "HAS_NATIVE_TALIB",
    "TALIB_IMPORT_ERROR",
    "AIMarketAnalyst",
    "AIPatternScorer",
    "ForexDataLoader",
    "MarketDataLoader",
    "PatternAnalyzer",
    "PatternConfidenceResult",
    "PatternRankingTester",
    "PatternResult",
    "SignalGrade",
    "TradeSetup",
    "__version__",
    "classify_asset",
    "get_talib_install_hint",
    "get_talib_status",
    "normalize_ticker",
    "resolve_asset_currencies",
    "validate_asset_type",
]
