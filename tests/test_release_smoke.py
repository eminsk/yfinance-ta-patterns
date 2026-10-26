"""Release and code-change smoke test suite for yfinance-ta-patterns.

Validates:
- Package metadata and semantic version integrity
- CLI entrypoint and argument parser behavior
- Runtime compatibility (datetime timedelta in INTERVAL_DELTAS, lazy imports)
- Candlestick pattern recognition on synthetic data (both native TA-Lib and fallback)
- End-to-end backtesting pipeline and metrics calculation
- AI confluence scoring and trade setup generation
- Concurrency and free-threaded (PEP 703) safety across threads
"""

from __future__ import annotations

import concurrent.futures
import datetime
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yfinance_ta_patterns
from yfinance_ta_patterns import (
    AIMarketAnalyst,
    AIPatternScorer,
    PatternAnalyzer,
    PatternConfidenceResult,
    PatternRankingTester,
    SignalGrade,
    TradeSetup,
    __version__,
)
from yfinance_ta_patterns.data import INTERVAL_DELTAS
from yfinance_ta_patterns.talib_compat import (
    HAS_NATIVE_TALIB,
    SUPPORTED_FALLBACK_PATTERNS,
)


@pytest.fixture
def synthetic_market_candles() -> pd.DataFrame:
    """Generate 60 bars of geometrically valid synthetic OHLCV data."""
    t0 = datetime.datetime(2025, 1, 1)
    dates = [t0 + datetime.timedelta(days=i) for i in range(60)]
    np.random.seed(12345)

    base = 100.0 + np.cumsum(np.random.normal(0, 0.5, 60))
    opens = base + np.random.uniform(-0.5, 0.5, 60)
    closes = base + np.random.uniform(-0.5, 0.5, 60)

    # Strictly guarantee OHLC invariants: High >= max(Open, Close), Low <= min(Open, Close)
    highs = np.maximum(opens, closes) + np.random.uniform(0.2, 1.5, 60)
    lows = np.minimum(opens, closes) - np.random.uniform(0.2, 1.5, 60)
    volumes = np.random.uniform(50000, 150000, 60)

    # Insert a Doji at bar 20
    opens[20] = 105.0
    closes[20] = 105.0
    highs[20] = 106.0
    lows[20] = 104.0

    # Insert a Hammer at bar 40
    opens[40] = 102.0
    closes[40] = 103.0
    highs[40] = 103.2
    lows[40] = 98.0  # long lower shadow

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# 1. Package Metadata & Public API Exports
# ---------------------------------------------------------------------------


def _load_verify_release():
    """Return scripts/verify_release.py module (None when scripts/ is not shipped, e.g. sdist)."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "verify_release.py"
    if not script.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_verify_release_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_version_format():
    """Version string must follow standard semantic versioning (X.Y.Z) and match across packaging files."""
    assert isinstance(__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+(\.post\d+)?$", __version__), (
        f"Invalid version format: {__version__}"
    )

    verify_mod = _load_verify_release()
    if verify_mod is not None:
        verify_mod.check_version_sync(Path(__file__).resolve().parent.parent, __version__)


def test_version_sync_mismatch_and_missing_sources_detected(tmp_path: Path):
    """check_version_sync must raise AssertionError on missing sources, mismatches, or stale installs."""
    verify_mod = _load_verify_release()
    if verify_mod is None:
        pytest.skip("scripts/verify_release.py not present in sdist")

    repo_root = Path(__file__).resolve().parent.parent
    (tmp_path / ".git").mkdir()

    # 1. Missing sources in strict mode
    with pytest.raises(AssertionError) as exc_info:
        verify_mod.check_version_sync(tmp_path, __version__)
    assert "Version source missing: pyproject.toml" in str(exc_info.value)

    # 2. Populate all files from repo_root
    for _label, rel_path, _pattern in verify_mod.VERSION_SOURCES:
        src_file = repo_root / rel_path
        if not src_file.is_file():
            pytest.skip(f"{rel_path} is not part of this source tree")
        dest = tmp_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src_file.read_text(encoding="utf-8-sig"), encoding="utf-8")

    # In clean state, all 12 sources must verify cleanly
    verified = verify_mod.check_version_sync(tmp_path, __version__)
    assert len(verified) == 12

    # 3. Single file mismatch -> caught with specific error
    pyproject_file = tmp_path / "pyproject.toml"
    orig_pyproject = pyproject_file.read_text(encoding="utf-8")
    pyproject_file.write_text(
        orig_pyproject.replace(f'version = "{__version__}"', 'version = "0.0.1"'),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError) as exc_info:
        verify_mod.check_version_sync(tmp_path, __version__)
    assert "Version mismatch in pyproject.toml: found '0.0.1'" in str(exc_info.value)
    pyproject_file.write_text(orig_pyproject, encoding="utf-8")

    # 4. Multiple simultaneous mismatches -> all listed
    aur_file = tmp_path / "aur" / "PKGBUILD"
    orig_aur = aur_file.read_text(encoding="utf-8")
    aur_file.write_text(orig_aur.replace(f"pkgver={__version__}", "pkgver=0.0.2"), encoding="utf-8")

    recipe_file = tmp_path / "recipe" / "recipe.yaml"
    orig_recipe = recipe_file.read_text(encoding="utf-8")
    recipe_file.write_text(
        orig_recipe.replace(f'version: "{__version__}"', 'version: "0.0.3"'),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError) as exc_info:
        verify_mod.check_version_sync(tmp_path, __version__)
    err_msg = str(exc_info.value)
    assert "Version mismatch in aur/PKGBUILD: found '0.0.2'" in err_msg
    assert "Version mismatch in recipe/recipe.yaml: found '0.0.3'" in err_msg

    # Restore clean state
    aur_file.write_text(orig_aur, encoding="utf-8")
    recipe_file.write_text(orig_recipe, encoding="utf-8")

    # 5. Stale installed version scenario: all files agree on current version but expected differs
    with pytest.raises(AssertionError) as exc_info:
        verify_mod.check_version_sync(tmp_path, "0.9.9")
    assert "reinstall it (pip install -e .) and re-run" in str(exc_info.value)


# The tests below pin the behaviour of `check_version_sync` that the test above does not reach:
# each source is enforced on its own, a missing source or version line must stay loud in a
# checkout (and only there), and harmless file quirks must not raise a false alarm.


def _rewrite(path: Path, pattern: str, replacement: str) -> None:
    """Replace exactly one match of `pattern` in `path` (fails loudly if the file format drifts)."""
    text = path.read_text(encoding="utf-8")
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    assert count == 1, f"{path.name}: {pattern!r} not found"
    path.write_text(text, encoding="utf-8")


@pytest.fixture(scope="module")
def verify_mod():
    """scripts/verify_release.py as a module; the tests are skipped where it is not shipped."""
    module = _load_verify_release()
    if module is None:
        pytest.skip("scripts/verify_release.py not present in sdist")
    return module


@pytest.fixture
def tree(tmp_path: Path, verify_mod, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake checkout: a ``.git`` marker plus a copy of every version-carrying file of this repo.

    GITHUB_ACTIONS is removed so that the ``.git`` marker alone decides between strict and lenient.
    """
    root = Path(__file__).resolve().parent.parent
    sources = [rel_path for _label, rel_path, _pattern in verify_mod.VERSION_SOURCES]
    if not all((root / rel_path).is_file() for rel_path in sources):
        pytest.skip("not a full source checkout")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    (tmp_path / ".git").mkdir()
    for rel_path in sources:
        (tmp_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel_path).write_bytes((root / rel_path).read_bytes())
    return tmp_path


def test_every_missing_source_is_reported_in_a_checkout(
    tmp_path: Path, verify_mod, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    (tmp_path / ".git").mkdir()
    with pytest.raises(AssertionError) as exc_info:
        verify_mod.check_version_sync(tmp_path, __version__)
    for label, rel_path, _pattern in verify_mod.VERSION_SOURCES:
        assert f"Version source missing: {label} (expected at {rel_path})" in str(exc_info.value)


def test_missing_source_is_tolerated_only_outside_a_checkout(
    tree: Path, verify_mod, monkeypatch: pytest.MonkeyPatch
):
    """An sdist ships only some of the files, but a source checkout must never silently lose one."""
    (tree / "aur" / "PKGBUILD").unlink()
    with pytest.raises(AssertionError, match="Version source missing: aur/PKGBUILD"):
        verify_mod.check_version_sync(tree, __version__)  # strict: the .git marker

    (tree / ".git").rmdir()
    verified = verify_mod.check_version_sync(tree, __version__)  # lenient: skipped, not an error
    assert len(verified) == len(verify_mod.VERSION_SOURCES) - 1
    assert "aur/PKGBUILD" not in verified

    monkeypatch.setenv("GITHUB_ACTIONS", "true")  # a CI job is strict even without a .git entry
    with pytest.raises(AssertionError, match="Version source missing: aur/PKGBUILD"):
        verify_mod.check_version_sync(tree, __version__)


def test_git_file_of_a_worktree_marks_a_checkout(tree: Path, verify_mod):
    """In a git worktree or a submodule `.git` is a file, not a directory."""
    (tree / ".git").rmdir()
    (tree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    (tree / "aur" / "PKGBUILD").unlink()
    with pytest.raises(AssertionError, match="Version source missing: aur/PKGBUILD"):
        verify_mod.check_version_sync(tree, __version__)


def test_version_line_gone_from_a_source_is_reported(tree: Path, verify_mod):
    _rewrite(tree / "aur" / "PKGBUILD", r"^pkgver=.*$", "# pkgver removed")
    with pytest.raises(AssertionError, match="Version pattern not found in aur/PKGBUILD"):
        verify_mod.check_version_sync(tree, __version__)


def test_each_source_is_enforced_individually(tree: Path, verify_mod):
    """A stale version in any single source, the .SRCINFO download URL included, is reported."""
    for label, rel_path, pattern in verify_mod.VERSION_SOURCES:
        path = tree / rel_path
        original = path.read_text(encoding="utf-8")
        match = re.search(pattern, original)
        assert match is not None, f"{label}: pattern does not match the file"
        start, end = match.span(1)
        path.write_text(original[:start] + "0.0.1" + original[end:], encoding="utf-8")
        expected = re.escape(f"Version mismatch in {label}: found '0.0.1'")
        with pytest.raises(AssertionError, match=expected):
            verify_mod.check_version_sync(tree, __version__)
        path.write_text(original, encoding="utf-8")


def test_reinstall_hint_is_shown_only_when_all_files_agree(tree: Path, verify_mod):
    """A stale installed package is blamed only if every file carries the same version."""
    _rewrite(tree / "pyproject.toml", r'^version\s*=\s*".*"$', 'version = "0.0.1"')
    with pytest.raises(AssertionError) as exc_info:
        verify_mod.check_version_sync(tree, "9.9.9")
    message = str(exc_info.value)
    assert "reinstall" not in message
    assert "found '0.0.1'" in message
    assert f"found '{__version__}'" in message


@pytest.mark.parametrize(
    ("rel_path", "transform"),
    [
        ("CHANGELOG.md", lambda raw: b"## [Unreleased]\n\n" + raw),
        ("debian/changelog", lambda raw: b"\xef\xbb\xbf" + raw),
        ("debian/changelog", lambda raw: b"\n" + raw),
    ],
    ids=["unreleased-section", "utf8-bom", "leading-blank-line"],
)
def test_harmless_file_quirks_do_not_raise_a_false_alarm(
    tree: Path, verify_mod, rel_path, transform
):
    path = tree / rel_path
    path.write_bytes(transform(path.read_bytes()))
    assert len(verify_mod.check_version_sync(tree, __version__)) == len(verify_mod.VERSION_SOURCES)


def test_crlf_checkout_is_accepted(tree: Path, verify_mod):
    """Windows checkouts with core.autocrlf (the Windows CI jobs) verify like LF ones."""
    for _label, rel_path, _pattern in verify_mod.VERSION_SOURCES:
        path = tree / rel_path
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert len(verify_mod.check_version_sync(tree, __version__)) == len(verify_mod.VERSION_SOURCES)


def test_all_public_symbols_exported():
    """Verify all symbols in __all__ are accessible and not None."""
    expected_exports = [
        "ALLOWED_ASSET_TYPES",
        "FOREX_56_PAIRS",
        "FOREX_MAJOR_CURRENCIES",
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
        "validate_asset_type",
    ]
    for sym in expected_exports:
        assert hasattr(yfinance_ta_patterns, sym), f"Missing export '{sym}' in yfinance_ta_patterns"
        val = getattr(yfinance_ta_patterns, sym)
        assert val is not None, f"Export '{sym}' is None"


# ---------------------------------------------------------------------------
# 2. CLI Execution & Help
# ---------------------------------------------------------------------------


def test_cli_module_version():
    """python -m yfinance_ta_patterns.cli --version must succeed and print version."""
    proc = subprocess.run(
        [sys.executable, "-m", "yfinance_ta_patterns.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"CLI --version failed: {proc.stderr}"
    assert __version__ in proc.stdout or __version__ in proc.stderr


def test_cli_module_help():
    """python -m yfinance_ta_patterns.cli --help must output options and exit 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "yfinance_ta_patterns.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"CLI --help failed: {proc.stderr}"
    assert "--pattern" in proc.stdout
    assert "--symbol" in proc.stdout
    assert "--timeframe" in proc.stdout


# ---------------------------------------------------------------------------
# 3. Runtime & Cross-Platform Compatibility (PyPy, Timedelta, Lazy Imports)
# ---------------------------------------------------------------------------


def test_interval_deltas_use_datetime_timedelta():
    """Regression test: all INTERVAL_DELTAS must be datetime.timedelta to prevent PyPy 3.8 crashes."""
    assert len(INTERVAL_DELTAS) > 0
    now = datetime.datetime(2025, 1, 1, 12, 0, 0)

    for interval_key, delta in INTERVAL_DELTAS.items():
        assert isinstance(delta, datetime.timedelta), (
            f"INTERVAL_DELTAS['{interval_key}'] is not datetime.timedelta: {type(delta)}"
        )
        # Arithmetic must not raise
        future = now + delta
        past = now - delta
        assert future > past


def test_lazy_yfinance_loading():
    """Verify that importing yfinance_ta_patterns.data does not eagerly load yfinance."""
    import yfinance_ta_patterns.data as ydata

    if hasattr(ydata, "yf") and ydata.yf is None:
        code = (
            "import sys; "
            "import yfinance_ta_patterns.data; "
            "assert 'yfinance' not in sys.modules, 'yfinance was eagerly imported!'"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, f"Lazy import failed: {proc.stderr}"


# ---------------------------------------------------------------------------
# 4. Candlestick Pattern Recognition on Synthetic Data
# ---------------------------------------------------------------------------


def test_candlestick_patterns_recognized(synthetic_market_candles: pd.DataFrame):
    """Test pattern recognition on synthetic data for all supported patterns."""
    analyzer = PatternAnalyzer(synthetic_market_candles)

    if HAS_NATIVE_TALIB:
        patterns_to_test = PatternRankingTester.get_all_patterns()
        assert len(patterns_to_test) >= 61
    else:
        patterns_to_test = sorted(SUPPORTED_FALLBACK_PATTERNS)
        assert len(patterns_to_test) == 61

    for pattern in patterns_to_test:
        signals = analyzer.get_signals(pattern)
        assert isinstance(signals, pd.Series), f"get_signals({pattern}) did not return Series"
        unique_vals = set(signals.unique())
        assert unique_vals.issubset({-100, 0, 100}), (
            f"Pattern {pattern} returned unexpected values: {unique_vals}"
        )


def test_pattern_analyzer_date_range_filtering(synthetic_market_candles: pd.DataFrame):
    """Test date and range filtering on PatternAnalyzer."""
    analyzer = PatternAnalyzer(synthetic_market_candles)

    # Filter by specific date that has a Doji
    sig_date = analyzer.get_signals("CDLDOJI", date="2025-01-21")
    assert isinstance(sig_date, pd.Series)

    # Filter by date range
    sig_range = analyzer.get_signals("CDLDOJI", start_date="2025-01-10", end_date="2025-01-25")
    assert isinstance(sig_range, pd.Series)
    for ts in sig_range.index:
        ts_str = str(ts)[:10]
        assert "2025-01-10" <= ts_str <= "2025-01-25"


# ---------------------------------------------------------------------------
# 5. Pattern Backtesting Pipeline & Metrics
# ---------------------------------------------------------------------------


def test_pattern_ranking_tester_end_to_end(synthetic_market_candles: pd.DataFrame):
    """Test full backtesting pipeline, calculation of trades, Sharpe, and report generation."""
    tester = PatternRankingTester(
        synthetic_market_candles,
        holding_period=5,
        initial_capital=10000.0,
        position_size=100.0,
    )
    results = tester.test_all_patterns(filter_news=False)
    assert isinstance(results, list)
    assert len(results) > 0

    # Verify report export
    report = tester.get_comparison_report()
    assert isinstance(report, pd.DataFrame)
    assert not report.empty
    col_names_lower = [c.lower() for c in report.columns]
    assert any("win rate" in c for c in col_names_lower)
    assert any("pnl" in c for c in col_names_lower)


# ---------------------------------------------------------------------------
# 6. AI Pattern Scorer & Trade Setup Generation
# ---------------------------------------------------------------------------


def test_ai_scorer_trade_setup_end_to_end(synthetic_market_candles: pd.DataFrame):
    """Test indicator enrichment, confluence scoring, and trade setup calculation."""
    scorer = AIPatternScorer(synthetic_market_candles)
    assert "_EMA20" in scorer.df.columns
    assert "_RSI14" in scorer.df.columns
    assert "_ATR14" in scorer.df.columns

    # Evaluate pattern on the last candle
    last_idx = synthetic_market_candles.index[-1]
    result = scorer.score_signal("HAMMER", last_idx, raw_signal=100)
    assert isinstance(result, PatternConfidenceResult)
    assert 0.0 <= result.confidence_score <= 1.0
    assert isinstance(result.grade, SignalGrade)

    # Trade setup
    setup = result.trade_setup
    assert isinstance(setup, TradeSetup)
    assert setup.entry_price > 0.0
    assert setup.stop_loss > 0.0
    assert setup.take_profit_1 > 0.0
    assert setup.risk_reward_ratio >= 0.0


def test_ai_market_analyst_synthesis(synthetic_market_candles: pd.DataFrame):
    """Test AIMarketAnalyst brief, JSON, and LLM prompt generation."""
    scorer = AIPatternScorer(synthetic_market_candles)
    last_idx = synthetic_market_candles.index[-1]
    result = scorer.score_signal("HAMMER", last_idx, raw_signal=100)

    analyst = AIMarketAnalyst(
        synthetic_market_candles,
        scored_results=[result],
        symbol="AAPL",
        timeframe="1d",
        asset_type="stock",
    )
    brief = analyst.generate_brief()
    assert isinstance(brief, str)
    assert len(brief) > 0
    assert "AAPL" in brief

    prompt = analyst.to_llm_prompt()
    assert isinstance(prompt, str)
    assert "Portfolio Manager" in prompt or "Technical Analyst" in prompt

    payload = analyst.to_dict()
    assert isinstance(payload, dict)
    assert payload.get("symbol") == "AAPL"
    assert "patterns" in payload


# ---------------------------------------------------------------------------
# 7. Concurrency & Free-Threaded (PEP 703) Safety
# ---------------------------------------------------------------------------


def test_concurrent_multithreaded_pattern_analysis(synthetic_market_candles: pd.DataFrame):
    """Ensure running multiple analyzers in parallel threads does not deadlock or crash."""
    patterns = ["CDLDOJI", "CDLHAMMER", "CDLENGULFING", "CDLHARAMI"]

    def _analyze(pat: str) -> int:
        analyzer = PatternAnalyzer(synthetic_market_candles)
        res = analyzer.get_signals(pat)
        return int((res != 0).sum())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_analyze, p): p for p in patterns}
        results = {}
        for fut in concurrent.futures.as_completed(futures):
            pat = futures[fut]
            results[pat] = fut.result()

    assert len(results) == len(patterns)
    for _pat, count in results.items():
        assert isinstance(count, int)
