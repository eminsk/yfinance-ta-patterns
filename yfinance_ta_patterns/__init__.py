"""YFinance TA Patterns: Advanced Technical Candlestick Analysis with AI Intelligence."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("yfinance-ta-patterns")
except PackageNotFoundError:  # pragma: no cover - during editable installs
    __version__ = "0.3.12"


from .ai import (
    AIMarketAnalyst,
    AIPatternScorer,
    PatternConfidenceResult,
    SignalGrade,
    TradeSetup,
)
from .data import MarketDataLoader, classify_asset, normalize_ticker, resolve_asset_currencies
from .forex_data_loader import ForexDataLoader
from .pattern_analyzer import PatternAnalyzer
from .pattern_tester import PatternRankingTester, PatternResult

__all__ = [
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
    "normalize_ticker",
    "resolve_asset_currencies",
]
