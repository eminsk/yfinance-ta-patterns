"""YFinance TA Patterns package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("yfinance-ta-patterns")
except PackageNotFoundError:  # pragma: no cover - during editable installs
    __version__ = "0.0.0"

from .forex_data_loader import ForexDataLoader
from .pattern_analyzer import PatternAnalyzer
from .pattern_tester import PatternRankingTester, PatternResult

__all__ = [
    "ForexDataLoader",
    "PatternAnalyzer",
    "PatternRankingTester",
    "PatternResult",
    "__version__",
]
