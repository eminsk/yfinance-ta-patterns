"""YFinance TA Patterns package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("yfinance-ta-patterns")
except PackageNotFoundError:  # pragma: no cover - during editable installs
    __version__ = "0.0.0"

from .forex_data_loader import ForexDataLoader
from .pattern_analyzer import PatternAnalyzer

__all__ = ["ForexDataLoader", "PatternAnalyzer", "__version__"]
