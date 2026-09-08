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

`yfinance-ta-patterns` installs in seconds with **zero C build requirements** by default, using an internal vectorized pure-Python/NumPy engine for candlestick pattern recognition.

#### Option A: Using `uv` (Fastest & Recommended)

```bash
# 1. Standard Python (with GIL: 3.12, 3.13, 3.14):
uv add yfinance-ta-patterns

# 2. Free-Threaded (No-GIL / PEP 703: 3.14t, 3.15t):
uv python pin 3.14t
uv add yfinance-ta-patterns

# 3. Instant execution without installing into environment:
uvx --from yfinance-ta-patterns yftp --all-patterns --symbol AAPL --timeframe 1h --ai
```

#### Option B: Using Standard `pip`

```bash
# Standard install (pure-Python fallback by default):
pip install yfinance-ta-patterns

# Optional: With native C TA-Lib acceleration (requires ta-lib C headers):
pip install "yfinance-ta-patterns[talib]"
```

#### 🐍 Python Version Compatibility Matrix

| Python Version | Execution Mode | Installation Status | Recommendation |
|:---:|:---:|:---:|---|
| **Python 3.14t** | **Free-Threaded (No-GIL)** | ✅ **100% Supported** | **Recommended No-GIL release**. All dependencies (`numpy`, `pandas`, `cffi`) provide official wheels on PyPI. |
| **Python 3.15t** | **Free-Threaded (No-GIL)** | ✅ **Supported** | Next-generation No-GIL preview. Fully functional with cached/built wheels. |
| **Python 3.13** | **Standard (GIL)** | ✅ **100% Supported** | Current stable Python release. Full support for native TA-Lib and pre-built wheels. |
| **Python 3.12** | **Standard (GIL)** | ✅ **100% Supported** | Long-Term Support release with instant sub-second wheel installation. |
| **Python 3.13t** | **Experimental No-GIL** | ❌ *Blocked Upstream* | Blocked by upstream `cffi` (`RuntimeError: CFFI does not support 3.13t`). **Use 3.14t for No-GIL instead.** |

---

## 🚀 Ready-to-Use Examples (`examples/`)

The repository includes production-ready backtesting and real-time scanning scripts in the [`examples/`](examples/) directory:

### 1. Multi-Pair Historical Backtesting (`examples/multi_pair_backtest.py`)

A rigorous multi-pair portfolio backtest across 6 major and cross Forex pairs (`EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `EURGBP`, `EURJPY`) over 2 years on hourly (`1h`) candles.

**Key capabilities demonstrated:**
- **Unbiased Execution**: Fills trades at `Open[i+1]` (next candle open) to eliminate lookahead bias.
- **Fixed Holding Period (`holding_period=5`)**: Automatically closes trades after 5 bars. Essential for single-direction candlestick patterns (Hammer, Inverted Hammer, 3 White Soldiers) which otherwise hold positions indefinitely without reverse exit signals.
- **Universal Forex PnL & Position Sizing**: Automatic base-currency normalization for USD-base pairs (`USDJPY`) and cross-rates (`EURGBP`), applying commissions in account currency ($0.05 ECN micro-lot).
- **Composite Scoring**: Ranks combinations by combining Win Rate, Profit Factor, Sharpe Ratio, and logarithmic trade volume.

```bash
# Run with Python:
python examples/multi_pair_backtest.py

# Or run with uv:
uv run examples/multi_pair_backtest.py
```

**Sample Output:**
```text
==================================================================================================
 TOP-10 BEST COMBINATIONS (PAIR + PATTERN) OVER 2y:
==================================================================================================
  Pair        Pattern  Signals WinRate TotalPnL  ProfitFactor  Sharpe  MaxDD MaxDD_%
EURJPY INVERTEDHAMMER       82   63.0%  $+55.34          2.55    0.38  $7.11   0.07%
EURJPY    MORNINGSTAR       46   63.0%  $+21.49          2.66    0.34  $2.87   0.03%
EURUSD         INNECK       12   75.0%   $+6.31          3.81    0.25  $1.14   0.01%
EURUSD STALLEDPATTERN       26   65.4%  $+23.44          3.09    0.29  $5.17   0.05%
EURUSD  STICKSANDWICH        8   62.5%   $+8.13          5.01    0.22  $1.00   0.01%
EURUSD         ONNECK       11   63.6%  $+11.53          4.44    0.17  $1.97   0.02%
USDJPY    MATCHINGLOW      195   55.6%  $+50.36          1.40    0.24 $15.09   0.15%
EURJPY GRAVESTONEDOJI      226   52.4%  $+58.74          1.44    0.24 $20.81   0.21%
EURJPY        HIKKAKE     1604   52.7%  $+50.97          1.06    0.09 $75.80   0.76%
USDJPY       BELTHOLD     1596   47.7%  $+78.87          1.09    0.13 $83.89   0.84%

Full ranked report saved to: all_pairs_ranked.csv
```

### 2. Live Multi-Asset Signal Scanner (`examples/live_signals_scanner.py`)

Scans a 12-asset watchlist across Forex and Cryptocurrencies (`BTC-USD`, `ETH-USD`) for active candlestick patterns over recent candles, scoring each opportunity with the AI Confluence Engine.

**Key capabilities demonstrated:**
- **AI Confluence Scoring**: Quantifies trend alignment (EMA 20/50/200), volume expansion, and Wilder RSI (14) momentum into a 0–100% confidence score.
- **Automated Trade Setups**: Calculates exact Entry Price, ATR-based Stop Loss, and Take Profit targets (1:1.5 Risk/Reward ratio).
- **Automated Opportunity Ranking**: Sorts all live setups and highlights the **TOP-1 Highest Confidence Trade Setup**.

```bash
# Run with Python:
python examples/live_signals_scanner.py

# Or run with uv:
uv run examples/live_signals_scanner.py
```

**Sample Output:**
```text
=========================================================================================================
 SIGNALS FOUND: 26 (Ranked by AI Confidence)
=========================================================================================================
       Time  Symbol Direction        Pattern Confidence     Grade        Entry     StopLoss  TakeProfit_1 Risk_Reward          Trend
08.09 19:00 ETH-USD       BUY 3WHITESOLDIERS      80.0% EXCELLENT  2499.389893  2483.227621   2523.633300       1:1.5        BULLISH
08.09 17:00  EURUSD       BUY 3WHITESOLDIERS      70.0%    STRONG     1.163332     1.162101      1.165178       1:1.5 STRONG_BULLISH
08.09 17:00  USDCAD      SELL     HANGINGMAN      70.0%    STRONG     1.377840     1.378940      1.376190       1:1.5 STRONG_BEARISH

**********************************************************
 TOP-1 SIGNAL: ETH-USD — BUY (3WHITESOLDIERS)
 Candle Time: 08.09 19:00 | Confidence: 80.0% [EXCELLENT]
 Market Trend: BULLISH | RSI: 56.3
 Entry Price: 2499.39
 Stop Loss:   2483.23
 Take Profit: 2523.63 (R:R 1:1.5)
**********************************************************
```

---

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
    execution="next_open",  # Unbiased: enters on Open of bar i+1
    allow_short=True,  # Full symmetric short trades on bearish signals
    commission=1.50,  # Transaction fee per trade
    slippage=0.0001,  # Execution slippage in price units
    min_signals=5,  # Exclude patterns with < 5 signals (reduces overfitting)
    sharpe_mode="periodic",  # Accounts for 0% returns during idle hold periods
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
