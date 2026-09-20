# Market Candlestick & AI Pattern Scanner (`yfinance-ta-patterns`)

[![PyPI](https://img.shields.io/pypi/v/yfinance-ta-patterns?color=blue)](https://pypi.org/project/yfinance-ta-patterns/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/yfinance-ta-patterns.svg)](https://anaconda.org/conda-forge/yfinance-ta-patterns)
[![Ubuntu / Debian PPA](https://img.shields.io/badge/Ubuntu%20%2F%20Debian-APT%20PPA-E95420?logo=ubuntu&logoColor=white)](https://eminsk.github.io/ppa/)
[![MSYS2](https://img.shields.io/badge/MSYS2-MinGW64-blue?logo=windows&logoColor=white)](https://github.com/msys2/MINGW-packages/pull/31765)
[![Arch Linux](https://img.shields.io/badge/AUR-Arch%20Linux-1793D1?logo=arch-linux&logoColor=white)](https://github.com/eminsk/yfinance-ta-patterns/tree/main/aur)
[![Python](https://img.shields.io/pypi/pyversions/yfinance-ta-patterns)](https://pypi.org/project/yfinance-ta-patterns/)
[![PyPy](https://img.shields.io/badge/PyPy-3.8%20--%203.11-orange.svg)](https://www.pypy.org/)
[![No-GIL](https://img.shields.io/badge/No--GIL-3.14t%20--%203.15t-purple.svg)](https://peps.python.org/pep-0703/)
[![CI](https://github.com/eminsk/yfinance-ta-patterns/actions/workflows/ci.yml/badge.svg)](https://github.com/eminsk/yfinance-ta-patterns/actions)
[![Downloads](https://static.pepy.tech/badge/yfinance-ta-patterns)](https://pepy.tech/project/yfinance-ta-patterns)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

High-performance Python library and CLI that downloads multi-asset market data via `yfinance`, detects TA-Lib candlestick patterns, and enriches raw signals using an **AI/Quant Confluence Engine** to generate multi-factor confluence scores (deterministic quantitative confluence heuristic, not uncalibrated win-rate probability), trade setups, and LLM-ready market briefs.

Universal runtime compatibility across **CPython 3.8 to 3.14, PyPy 3.8 to 3.11 with high-speed JIT tracing, and legacy Windows 7+ support**. Free-threaded CPython 3.14t and 3.15.0rc2t are verified when the interpreter is started with its GIL disabled; standard CPython 3.15 remains preview-only until upstream Windows dependency wheels are available.

---

## Key Features

- **Multi-Asset Data Loader**: Universal fetching and candle normalization for stocks (`AAPL`, `NVDA`), crypto (`BTC-USD`), commodities (`GC=F`), indices (`^GSPC`), and forex pairs (`EURUSD`). Supports custom date ranges (`start`, `end`), timezone conversion, UTC-anchored 4h resampling, and raw OHLC integrity validation.
- **TA-Lib Pattern Detection**: Full recognition engine across 60+ classic candlestick patterns (via native TA-Lib) with built-in zero-dependency pure-NumPy fallback engine for 11 core patterns.
- **AI Pattern Confluence Scorer**: Multi-factor confluence score ($0.0 - 1.0$) evaluating quantitative confluence (deterministic heuristic, not win-rate probability):
  - Multi-EMA trend alignment (20, 50, 200 EMA) with safe warm-up handling
  - Zero-lookahead Relative Volume surge (RVOL)
  - Canonical Wilder's RSI (14) momentum exhaustion & divergence
  - Canonical Wilder's ATR (14) volatility expansion & candle body dominance
- **Automated Trade Setups**: Computes entry price, ATR-based invalidation stop-loss, and multi-tier take-profit targets (1.5x / 3.0x risk/reward).
- **AI Market Analyst & LLM Integration**: Generates executive markdown briefs, JSON payloads, and engineered prompts tailored for external AI agents (GPT-4o, Claude 3.5, Gemini, Ollama).
- **Quantitative Pattern Ranking & Backtesting**: Unbiased next-open execution (`Open[i+1]`), symmetric Long/Short trading, universal FX quote-currency conversion, transaction cost modeling (commissions & slippage), overfitting filters (`min_signals`), and periodic Sharpe ratio accounting for non-trading hold periods.

---

## 🧩 Universal Compatibility Matrix

| Runtime / Implementation | Supported Versions | Execution Mode | Status | Pre-built Wheels |
|:---|:---|:---|:---:|:---:|
| **CPython (Standard)** | 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14 | Standard bytecode + GIL | ✅ Fully Supported | PyPI wheels |
| **CPython (Standard preview)** | 3.15.0rc2 | Standard bytecode + GIL | ⚠️ Preview | Windows needs a pandas source build until upstream ships a cp315 wheel |
| **CPython (Free-Threaded)** | 3.14t, 3.15.0rc2t | Multi-core No-GIL (PEP 703) | ✅ Verified with GIL disabled | Windows release wheelhouse for pandas, curl_cffi, and TA-Lib |
| **CPython (Free-Threaded, legacy)** | 3.13t | Multi-core No-GIL (PEP 703) | Best effort | Depends on third-party wheel availability |
| **PyPy (JIT Accelerated)** | 3.8, 3.9, 3.10, 3.11 | High-speed JIT tracing | ✅ Fully Supported | Included in Release |
| **Operating Systems** | Windows (7, 8, 10, 11), Linux, macOS (Intel & Apple Silicon) | x86_64, ARM64 | ✅ Fully Supported | Universal & Native |

---

## Quick start

### 1. Installation

`yfinance-ta-patterns` installs in seconds with **zero C build requirements** by default, using an internal vectorized pure-Python/NumPy engine for candlestick pattern recognition.

#### 🚀 Quick Installation by Platform

| Platform / Tool | Command | Description |
|:---|:---|:---|
| **Conda / Mamba** | `conda install -c conda-forge yfinance-ta-patterns` | Recommended for Quant, Finance & Data Science |
| **Conda + Native TA-Lib** | `conda install -c conda-forge yfinance-ta-patterns ta-lib` | Full C-accelerated TA-Lib with **zero compilation** |
| **Pixi** | `pixi add yfinance-ta-patterns` | Modern high-speed reproducible Conda workflow |
| **Ubuntu / Debian (APT PPA)** | `curl -sS https://eminsk.github.io/ppa/setup.sh \| sudo bash`<br>`sudo apt install python3-yfinance-ta-patterns` | Official PPA repository (pure apt, zero URLs) |
| **Ubuntu / Debian (.deb)** | `sudo apt install ./python3-yfinance-ta-patterns_0.3.35-1_all.deb` | Native `.deb` package from [Releases](https://github.com/eminsk/yfinance-ta-patterns/releases) |
| **Ubuntu / Debian (pipx)** | `sudo apt install pipx && pipx install yfinance-ta-patterns` | Isolated global CLI tool install (PEP 668 compliant) |
| **Windows (MSYS2)** | `pacman -S mingw-w64-x86_64-python-yfinance-ta-patterns` | Native MSYS2 MinGW package (zero C compiler needed) |
| **uv (Python)** | `uv add yfinance-ta-patterns` | Sub-second pure Python installation |
| **pip (Python)** | `pip install yfinance-ta-patterns` | Universal PyPI installation |

#### Option A: Using `uv` (Fastest & Recommended)

```bash
# 1. Full institutional setup with all extras (TA-Lib + Scikit-Learn repair + AI):
uv add "yfinance-ta-patterns[all]" --no-config

# 2. Standard install (pure-Python fallback, zero C compiler required):
uv add yfinance-ta-patterns --no-config

# 3. Free-Threaded (No-GIL / PEP 703: 3.14t, 3.15t):
uv python pin 3.14t
# PowerShell: use the matching release wheelhouse and keep the GIL disabled.
$env:PYTHON_GIL = "0"
uv add "yfinance-ta-patterns[all]" --find-links https://github.com/eminsk/yfinance-ta-patterns/releases/expanded_assets/v0.3.26 --no-config

# 4. High-Performance PyPy JIT (PyPy 3.8, 3.9, 3.10, 3.11):
uv python pin pypy-3.8
uv add "yfinance-ta-patterns[all]" --no-config

# 5. Instant execution without installing into environment:
uvx --from yfinance-ta-patterns yftp --all-patterns --symbol AAPL --timeframe 1h --ai
```

> [!TIP]
> **Why `--no-config`?** Passing `--no-config` tells `uv` to ignore any local or parent `uv.toml` settings (such as local wheel registries or find-links overrides), ensuring a clean, isolated, and reproducible installation directly from PyPI in any project directory.

> [!IMPORTANT]
> ### ⚡ Free-Threaded Python / No-GIL Guide (PEP 703: 3.14t, 3.15t)
>
> Python 3.13+ introduces experimental free-threaded (No-GIL) builds. `yfinance-ta-patterns` is verified on Windows, Linux, and macOS under free-threaded CPython. Here is what you need to know for a smooth No-GIL setup on Windows:
>
> #### 1. Python Version Selection & Wheel Availability
> - **Python 3.14t & 3.15t (Recommended)**: PyPI provides official precompiled `cp314t` and `cp315t` Windows wheels for `lxml 6.1.3`, `numpy 2.5.3`, and `scipy 1.18.1`. Combining PyPI with our release wheelhouse (`pandas 3.0.5`, `curl_cffi`, and `ta-lib 0.7.1`) allows a 100% binary install with zero C compiler required.
> - **Python 3.13t (Legacy / Experimental)**: PyPI lacks precompiled `cp313t-win_amd64` wheels for `lxml`, `numpy`, and `scipy`. Building `lxml` from source on Windows requires MSVC and `libxml2`/`libxslt` headers. If using Python 3.13t on Windows, install without the `[all]` extra to use the pure-Python vectorized engine: `uv add yfinance-ta-patterns`.
>
> #### 2. Understanding Automatic GIL Re-Enablement
> In free-threaded interpreters, `sys._is_gil_enabled()` may return `True` for two common reasons:
> 1. **Global Environment Variable**: A Windows user/system variable `PYTHON_GIL=1` forces the GIL on at startup. Check in PowerShell:
>    ```powershell
>    [System.Environment]::GetEnvironmentVariable('PYTHON_GIL', 'User')
>    ```
> 2. **PEP 703 C-Extension Protection**: When loading C extensions (such as `talib._ta_lib`) that have not yet declared `Py_MOD_GIL_NOT_USED`, CPython automatically re-enables the GIL to protect thread safety and emits a `RuntimeWarning`.
>
> #### 3. Running in True No-GIL Mode
> To run at full multi-core performance with the GIL disabled:
> - **Command Line Flag**:
>   ```bash
>   python -X gil=0 my_scanner.py
>   ```
> - **PowerShell**:
>   ```powershell
>   $env:PYTHON_GIL = "0"
>   python my_scanner.py
>   ```
> - **CMD**:
>   ```cmd
>   set PYTHON_GIL=0
>   python my_scanner.py
>   ```
> - **Pure-Python Engine**: When native TA-Lib is not installed, the built-in NumPy fallback engine keeps the GIL disabled out-of-the-box without extra flags.
>
> #### 4. Verification
> Run `yftp --check-talib` to inspect the runtime environment:
> ```bash
> yftp --check-talib
> ```
> Or verify programmatically:
> ```python
> from yfinance_ta_patterns import is_freethreaded, is_gil_enabled
> print("Free-Threaded build:", is_freethreaded())
> print("GIL active:", is_gil_enabled())
> ```

> [!IMPORTANT]
> **🪟 Windows + PyPy: One-Line Install via `--find-links` (Precompiled Wheels)**
> PyPI does not host precompiled Windows binary wheels for PyPy for `ta-lib`. Without `--find-links`, package managers (`uv` and `pip`) attempt to compile `ta-lib` from source, which may fail if a local MSVC compiler/linker is missing or incompatible.
> To install instantly with precompiled, self-contained native TA-Lib wheels (with zero compilation required):
> ```bash
> # Using uv:
> uv add "yfinance-ta-patterns[all]" --find-links https://github.com/eminsk/yfinance-ta-patterns/releases/expanded_assets/v0.3.30
>
> # Using standard pip:
> pip install "yfinance-ta-patterns[all]" --find-links https://github.com/eminsk/yfinance-ta-patterns/releases/expanded_assets/v0.3.30
> ```
> Or declare `find-links` directly in your `pyproject.toml`:
> ```toml
> [tool.uv]
> find-links = [
>     "https://github.com/eminsk/yfinance-ta-patterns/releases/expanded_assets/v0.3.30",
> ]
> ```

#### Option B: Using Standard `pip`

```bash
# Standard install (pure-Python fallback by default):
pip install yfinance-ta-patterns

# Optional: With native C TA-Lib acceleration (requires ta-lib C headers):
pip install "yfinance-ta-patterns[talib]"
```

#### Option C: Native TA-Lib & Dependencies (Pre-built Wheels on Release v0.3.26)

All 37 binary wheels are pre-compiled and attached to **[Release v0.3.26](https://github.com/eminsk/yfinance-ta-patterns/releases/tag/v0.3.26)**:

##### PyPy (High-Speed JIT 3.8 – 3.11):
```bash
# PyPy 3.8 (Windows 7+ compatible):
uv pip install https://github.com/eminsk/yfinance-ta-patterns/releases/download/v0.3.26/ta_lib-0.7.1-pp38-pypy38_pp73-win_amd64.whl
uv pip install https://github.com/eminsk/yfinance-ta-patterns/releases/download/v0.3.26/pandas-2.0.3-pp38-pypy38_pp73-win_amd64.whl
uv pip install https://github.com/eminsk/yfinance-ta-patterns/releases/download/v0.3.26/curl_cffi-0.16.3-pp38-pypy38_pp73-win_amd64.whl

# PyPy 3.9:
uv pip install https://github.com/eminsk/yfinance-ta-patterns/releases/download/v0.3.26/ta_lib-0.7.1-pp39-pypy39_pp73-win_amd64.whl
uv pip install https://github.com/eminsk/yfinance-ta-patterns/releases/download/v0.3.26/pandas-2.3.3-pp39-pypy39_pp73-win_amd64.whl

# PyPy 3.10:
uv pip install https://github.com/eminsk/yfinance-ta-patterns/releases/download/v0.3.26/ta_lib-0.7.1-pp310-pypy310_pp73-win_amd64.whl
uv pip install https://github.com/eminsk/yfinance-ta-patterns/releases/download/v0.3.26/pandas-2.3.3-pp310-pypy310_pp73-win_amd64.whl

# PyPy 3.11:
uv pip install https://github.com/eminsk/yfinance-ta-patterns/releases/download/v0.3.26/ta_lib-0.7.1-pp311-pypy311_pp73-win_amd64.whl
uv pip install https://github.com/eminsk/yfinance-ta-patterns/releases/download/v0.3.26/pandas-3.0.5-pp311-pypy311_pp73-win_amd64.whl
```

##### CPython Free-Threaded (No-GIL 3.14t – 3.15t):

Use the `uv add ... --find-links` command from Quick Start. It selects the matching `pandas`, `curl_cffi`, and TA-Lib wheels together, rather than mixing a release wheel with incompatible PyPI dependencies.

#### Option D: Native C TA-Lib on Linux & macOS (Optional)

> [!NOTE]
> **Native C TA-Lib is 100% OPTIONAL**: `yfinance-ta-patterns` already includes a built-in high-speed vectorized NumPy pattern recognition engine that works immediately with **zero compilation**.
>
> If you want the full C-accelerated TA-Lib engine without any manual compilation:
> - **Via Conda (Recommended — Zero C builds)**:
>   ```bash
>   conda install -c conda-forge yfinance-ta-patterns ta-lib
>   ```
> - **Or manual C compiler build from source**:

- **macOS (Homebrew):**
  ```bash
  brew install ta-lib
  export TA_INCLUDE_PATH="$(brew --prefix)/include"
  export TA_LIBRARY_PATH="$(brew --prefix)/lib"
  uv pip install ta-lib
  ```

- **Linux (Ubuntu / Debian / Arch):**
  ```bash
  # Ubuntu / Debian:
  sudo apt-get update && sudo apt-get install -y libta-lib0 libta-lib-dev
  # Arch Linux:
  yay -S ta-lib

  # Then install the python wrapper:
  uv pip install ta-lib
  ```




#### 🐍 Python Version Compatibility Matrix

| Python Version | Execution Mode | Installation Status | Recommendation |
|:---:|:---:|:---:|---|
| **Python 3.14t** | **Free-Threaded (No-GIL)** | ✅ **Verified** | Start with `PYTHON_GIL=0` or `-X gil=0`; use the Windows TA-Lib release wheelhouse. |
| **Python 3.13t** | **Free-Threaded (No-GIL)** | Best effort | Use the fallback or provide compatible third-party wheels for the desired extras. |
| **Python 3.15t** | **Free-Threaded (No-GIL)** | ✅ **Verified on 3.15.0rc2** | Use the matching release wheel and revalidate after the final release. |
| **Python 3.15** | **Standard (GIL)** | Preview | On Windows, wait for an upstream `pandas` cp315 wheel or build pandas from source. |
| **Python 3.13** | **Standard (GIL)** | ✅ **100% Supported** | Current stable Python release. Full support for native TA-Lib and pre-built wheels. |
| **Python 3.12** | **Standard (GIL)** | ✅ **100% Supported** | Long-Term Support release with instant sub-second wheel installation. |

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
 SIGNALS FOUND: 26 (Ranked by AI Confluence / Quality)
=========================================================================================================
       Time  Symbol Direction        Pattern Confluence     Grade        Entry     StopLoss  TakeProfit_1 Risk_Reward          Trend
08.09 19:00 ETH-USD       BUY 3WHITESOLDIERS   80.0/100 EXCELLENT  2499.389893  2483.227621   2523.633300       1:1.5        BULLISH
08.09 17:00  EURUSD       BUY 3WHITESOLDIERS   70.0/100    STRONG     1.163332     1.162101      1.165178       1:1.5 STRONG_BULLISH
08.09 17:00  USDCAD      SELL     HANGINGMAN   70.0/100    STRONG     1.377840     1.378940      1.376190       1:1.5 STRONG_BEARISH

**********************************************************
 TOP-1 SIGNAL: ETH-USD — BUY (3WHITESOLDIERS)
 Candle Time: 08.09 19:00 | Confluence: 80.0/100 [EXCELLENT] (deterministic heuristic, not win probability)
 Market Trend: BULLISH | RSI: 56.3
 Entry Price: 2499.39
 Stop Loss:   2483.23
 Take Profit: 2523.63 (R:R 1:1.5)
**********************************************************
```

---

### 2. Run CLI

```bash
# 1. Multi-Currency Portfolio Scanner (automatically scans all 56 Forex pairs without specifying --symbol):
yftp --timeframe 5m --period 30d --all-patterns --ai-analyst

# 2. Or explicitly scan all 56 pairs with custom confluence threshold:
yftp --all-pairs --timeframe 1h --period 60d --ai --min-confidence 0.60

# 3. Scan a custom comma-separated portfolio of symbols:
yftp --symbol EURUSD,GBPUSD,USDJPY --timeframe 15m --ai

# 4. Single symbol AI-scored analysis with Executive Brief:
yftp --symbol GBPUSD --timeframe 5m --period 30d --all-patterns --ai-analyst

# 5. Detect a single pattern with classic output:
yftp --pattern HAMMER --symbol AAPL --timeframe 1h --period 60d

# 6. Generate an LLM-ready prompt template for GPT-4o / Claude:
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

# Use period="2y" (or >= 200 daily candles) to ensure complete EMA200 indicator warm-up
data = MarketDataLoader("AAPL", period="2y", interval="1d").get_data()

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
