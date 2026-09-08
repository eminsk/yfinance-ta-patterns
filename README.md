# Market Candlestick & AI Pattern Scanner (`yfinance-ta-patterns`)

[![PyPI](https://img.shields.io/pypi/v/yfinance-ta-patterns?color=blue)](https://pypi.org/project/yfinance-ta-patterns/)
[![Python](https://img.shields.io/pypi/pyversions/yfinance-ta-patterns)](https://pypi.org/project/yfinance-ta-patterns/)
[![CI](https://github.com/eminsk/yfinance-ta-patterns/actions/workflows/ci.yml/badge.svg)](https://github.com/eminsk/yfinance-ta-patterns/actions)
[![Downloads](https://static.pepy.tech/badge/yfinance-ta-patterns)](https://pepy.tech/project/yfinance-ta-patterns)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

High-performance Python library and CLI that downloads multi-asset market data via `yfinance`, detects TA-Lib candlestick patterns, and enriches raw signals using an **AI/Quant Confluence Engine** to generate probabilistic confidence scores, trade setups, and LLM-ready market briefs.

Compatible with **Python 3.12, 3.13, 3.14, and Python 3.15 (Release Candidate & Preview)**.

---

## Key Features

- **Multi-Asset Data Loader**: Universal fetching and candle normalization for stocks (`AAPL`, `NVDA`), crypto (`BTC-USD`), commodities (`GC=F`), indices (`^GSPC`), and forex pairs (`EURUSD`). Supports custom date ranges (`start`, `end`), timezone conversion, UTC-anchored 4h resampling, and raw OHLC integrity validation.
- **TA-Lib Pattern Detection**: Full recognition engine across 60+ classic candlestick patterns (via native TA-Lib) with built-in zero-dependency pure-NumPy fallback engine for 11 core patterns.
- **AI Pattern Confidence Scorer**: Probabilistic score ($0.0 - 1.0$) evaluating multi-factor confluence:
  - Multi-EMA trend alignment (20, 50, 200 EMA) with safe warm-up handling
  - Zero-lookahead Relative Volume surge (RVOL)
  - Canonical Wilder's RSI (14) momentum exhaustion & divergence
  - Canonical Wilder's ATR (14) volatility expansion & candle body dominance
- **Automated Trade Setups**: Computes entry price, ATR-based invalidation stop-loss, and multi-tier take-profit targets (1.5x / 3.0x risk/reward).
- **AI Market Analyst & LLM Integration**: Generates executive markdown briefs, JSON payloads, and engineered prompts tailored for external AI agents (GPT-4o, Claude 3.5, Gemini, Ollama).
- **Quantitative Pattern Ranking & Backtesting**: Unbiased next-open execution (`Open[i+1]`), symmetric Long/Short trading, universal FX quote-currency conversion, transaction cost modeling (commissions & slippage), overfitting filters (`min_signals`), and periodic Sharpe ratio accounting for non-trading hold periods.

---

## Quick start

### 1. Installation

```bash
# Recommended with uv:
uv add yfinance-ta-patterns

# Or standard pip:
pip install yfinance-ta-patterns

# Or run instantly via uvx without installing:
uvx --from yfinance-ta-patterns yftp --all-patterns --symbol AAPL --timeframe 1h --ai
```

#### ⚡ Python 3.15 Ready (Early Adopters)
`yfinance-ta-patterns` is tested and **100% verified (68/68 tests passing)** on upcoming **Python 3.15** (`cpython-3.15.0rc2`), **Python 3.14**, **Python 3.13**, and **Python 3.12**.

While upstream wheels for C-dependencies (`ta-lib` and `pandas`) are pending official PyPI release for 3.15 and Free-Threaded No-GIL, pre-compiled native Windows x64 binary wheels are provided in our [Release Assets](https://github.com/eminsk/yfinance-ta-patterns/releases/tag/v0.2.0):
* `ta_lib-0.7.1-cp313-cp313t-win_amd64.whl` (Python 3.13 Free-Threaded No-GIL)
* `ta_lib-0.7.1-cp314-cp314t-win_amd64.whl` (Python 3.14 Free-Threaded No-GIL)
* `ta_lib-0.7.1-cp315-cp315t-win_amd64.whl` (Python 3.15 Free-Threaded No-GIL)
* `ta_lib-0.7.1-cp315-cp315-win_amd64.whl` (Python 3.15 Standard)
* `pandas-3.0.5-cp315-cp315-win_amd64.whl` (MSVC x64 binary)

To install cleanly on a Python 3.15 / Free-Threaded project:
```bash
uv add yfinance-ta-patterns --find-links https://github.com/eminsk/yfinance-ta-patterns/releases/expanded_assets/v0.2.0
```

#### 🧵 Free-Threaded (No-GIL / PEP 703) & Zero-Dependency Execution
`yfinance-ta-patterns` is **100% verified on Python 3.13t, 3.14t, and 3.15t Free-Threaded without GIL** (`-X gil=0`). 
Includes dual-mode execution:
1. **Native C Acceleration**: Verified with pre-compiled No-GIL wheels (`ta_lib-0.7.1-cp313t`, `cp314t`, and `cp315t`) for full 60+ pattern detection.
2. **Zero-Dependency Fallback Engine (`talib_compat`)**: Built-in vectorized pure-NumPy engine providing thread-safe detection for the 11 primary candlestick patterns without C compilers or system TA-Lib binaries:

| Supported Fallback Patterns (`SUPPORTED_FALLBACK_PATTERNS`) |
|------------------------------------------------------------|
| `CDLDOJI`, `CDLHAMMER`, `CDLINVERTEDHAMMER`, `CDLENGULFING`, `CDLSHOOTINGSTAR`, `CDLHANGINGMAN`, `CDLMORNINGSTAR`, `CDLEVENINGSTAR`, `CDLMARUBOZU`, `CDLBELTHOLD`, `CDLKICKING` |

> [!NOTE]
> For patterns outside the core 11 (e.g. `CDLPIERCING`, `CDLHARAMI`), the fallback raises `NotImplementedError` with clear instructions to install native TA-Lib.

Or configure your project's `pyproject.toml`:
```toml
[tool.uv]
find-links = ["https://github.com/eminsk/yfinance-ta-patterns/releases/expanded_assets/v0.2.0"]
```

### 2. Run CLI

```bash
# Detect a single pattern with classic output
yftp --pattern HAMMER --symbol AAPL --timeframe 1h --period 60d

# Scan all patterns on crypto with AI Confluence Scoring
yftp --all-patterns --symbol BTC-USD --timeframe 4h --period 60d --ai --min-confidence 0.65

# Generate an Executive AI Analyst Brief in Markdown
yftp --all-patterns --symbol NVDA --timeframe 15m --period 10d --ai-analyst --format markdown

# Generate an LLM-ready prompt template for GPT-4o / Claude
yftp --all-patterns --symbol EURUSD --timeframe 1h --period 60d --prompt
```

---

## CLI Reference

```text
yfinance-ta-patterns [-h] [-v] (--pattern PATTERN | --all-patterns)
                     [--symbol SYMBOL] [--period PERIOD] [--timeframe TIMEFRAME]
                     [--date YYYY-MM-DD] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
                     [--execution {next_open,close}] [--min-signals MIN_SIGNALS]
                     [--commission COMMISSION] [--slippage SLIPPAGE] [--auto-adjust]
                     [--ai] [--min-confidence MIN_CONFIDENCE]
                     [--ai-analyst] [--prompt] [--format {text,json,markdown}]
```

### Options

| Flag | Description |
|------|-------------|
| `-v`, `--version` | Show package version. |
| `--pattern` | Single candlestick pattern (e.g. `HAMMER`, `DOJI`, `CDLKICKING`). |
| `--all-patterns` | Scan and display signals for all available candlestick patterns. |
| `--symbol` | Ticker symbol (e.g. `AAPL`, `BTC-USD`, `GC=F`, `EURUSD`). |
| `--period` | History period (`5d`, `60d`, `1y`, `max`). |
| `--timeframe` | Interval alias (`M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`) or `yfinance` interval (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`). |
| `--date` | Filter signals for a specific date (`YYYY-MM-DD`). |
| `--start-date` / `--end-date` | Date range filter (`YYYY-MM-DD`). |
| `--execution` | Execution timing: `next_open` (unbiased, enters on bar i+1) or `close` (legacy). |
| `--min-signals` | Minimum signal count required for ranking (reduces overfitting from 1-trade samples). |
| `--commission` | Fixed transaction cost per round-trip trade in backtester. |
| `--slippage` | Slippage in price units applied adversely to entries and exits. |
| `--auto-adjust` | Enable dividend and split adjustments (recommended for long multi-year stock backtests). |
| `--ai` | Enrich detected patterns with AI confidence scoring, signal grade, and trade setups. |
| `--min-confidence` | Minimum confidence threshold for AI scoring ($0.0$ to $1.0$, default: $0.0$). |
| `--ai-analyst` | Run executive AI market analysis with synthesis and trade setups. |
| `--prompt` | Generate an LLM prompt ready to pass to ChatGPT, Claude, or local LLMs. |
| `--format` | Output format: `text` (default), `json`, or `markdown`. |

---

## Python API

### 1. Universal Multi-Asset Data Loader (`MarketDataLoader`)

Fetches and normalizes OHLC data across equities, crypto, forex, and commodities:

```python
from yfinance_ta_patterns import MarketDataLoader

# Stocks
loader = MarketDataLoader(symbol="NVDA", period="60d", interval="1h")
df = loader.get_data()

# Crypto
crypto_loader = MarketDataLoader(symbol="BTC-USD", period="30d", interval="15m")
crypto_df = crypto_loader.get_data()

# Forex (ForexDataLoader is fully compatible alias)
from yfinance_ta_patterns import ForexDataLoader

forex_loader = ForexDataLoader(symbol="EURUSD", period="60d", interval="1h")
forex_df = forex_loader.get_data()
```

### 2. AI Pattern Confidence Scorer (`AIPatternScorer`)

Evaluates technical confluence (trend, momentum, volume, volatility) and builds complete risk-managed trade setups:

```python
from yfinance_ta_patterns import MarketDataLoader, AIPatternScorer

data = MarketDataLoader("AAPL", period="60d", interval="1d").get_data()

scorer = AIPatternScorer(data)
scored_signals = scorer.score_all_active(min_confidence=0.60)

for sig in scored_signals:
    print(f"Pattern: {sig.pattern_name}")
    print(f"Confidence: {sig.confidence * 100:.1f}% ({sig.grade.value})")
    print(f"Action: {sig.action}")
    if sig.setup:
        print(f"Entry: {sig.setup.entry_price:.2f}")
        print(f"Stop Loss: {sig.setup.stop_loss:.2f}")
        print(f"Target 1: {sig.setup.take_profit_1:.2f} (R:R {sig.setup.risk_reward_ratio:.1f})")
    print("Confluences:", ", ".join(sig.confluences))
    print("-" * 40)
```

### 3. AI Market Analyst & LLM Prompting (`AIMarketAnalyst`)

Generates structured briefs and prompt templates for external LLM reasoning agents:

```python
from yfinance_ta_patterns import MarketDataLoader, AIMarketAnalyst

data = MarketDataLoader("BTC-USD", period="30d", interval="4h").get_data()

analyst = AIMarketAnalyst(data, symbol="BTC-USD")
results = analyst.analyze(min_confidence=0.65)

# 1. Executive Markdown Brief
brief = analyst.generate_brief(results)
print(brief)

# 2. Prompt for GPT-4o / Claude / Local LLM
llm_prompt = analyst.to_llm_prompt(results)

# 3. JSON Payload for APIs / Microservices
json_data = analyst.to_json(results)
```

### 4. Quantitative Backtesting & Pattern Ranking (`PatternRankingTester`)

Tests all candlestick patterns and ranks them by quantitative performance metrics:

```python
from yfinance_ta_patterns import MarketDataLoader, PatternRankingTester

data = MarketDataLoader("EURUSD", period="60d", interval="1h").get_data()

tester = PatternRankingTester(
    data,
    symbol="EURUSD",
    initial_capital=10000.0,
    position_size=1000.0,
    execution="next_open",      # Unbiased: enters on Open of bar i+1
    allow_short=True,           # Full symmetric short trades on bearish signals
    commission=1.50,            # Transaction fee per trade
    slippage=0.0001,            # Execution slippage in price units
    min_signals=5,              # Exclude patterns with < 5 signals (reduces overfitting)
    sharpe_mode="periodic",     # Accounts for 0% returns during idle hold periods
)
results = tester.test_all_patterns(min_signals=5)

top_patterns = tester.get_top_patterns(5)
for r in top_patterns:
    print(
        f"{r.pattern_name}: Win Rate={r.win_rate:.1f}%, PnL=${r.total_pnl:.2f}, "
        f"Periodic Sharpe={r.periodic_sharpe:.2f}, MaxDD=${r.max_drawdown:.2f}, "
        f"Strength={r.avg_strength:.0f}"
    )

# Export comparison report across news filters
report_df = tester.get_comparison_report()
print(report_df)
```

---

## Development & Testing

Run the test suite and quality checks:

```bash
# Run pytest across test suite
uv run --extra dev pytest -v

# Linter and formatting check
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .

# Static type check
uv run --extra dev mypy yfinance_ta_patterns

# Build source distribution and binary wheel
uv build
```

---
 
 ## 🌐 High-Performance Systems Ecosystem

`yfinance-ta-patterns` is developed by [**@eminsk**](https://github.com/eminsk) as part of an open-source engineering ecosystem:

* ⚡ [**NanoGEMM**](https://github.com/eminsk/nanogemm) — Minimalist, bare-metal AVX2+FMA SIMD matrix multiplication engine in ~100KB for sub-microsecond CPU neural network inference (`pip install nanogemm`).
* 🎥 [**screenvideo**](https://github.com/eminsk/screenvideo) — Lightweight desktop screen recorder with WASAPI audio and a standalone pure x64 Flat Assembler (FASM) native edition.
* 📊 [**xlsx_vievers**](https://github.com/eminsk/xlsx_vievers) — Desktop spreadsheet processor with 80+ formula functions, Chart Wizard, and hardware-accelerated SIMD SSE2 math engine.
* 🔍 [**StackOverflowAPI**](https://github.com/eminsk/StackOverflowAPI) — Bilingual desktop client for Stack Overflow built with CustomTkinter and native FASM x64 search client.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
