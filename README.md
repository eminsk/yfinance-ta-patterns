# Forex Candlestick Scanner (`yfinance-ta-patterns`)

Python package and CLI that downloads market data via `yfinance` and detects TA-Lib candlestick patterns for customizable symbols, timeframes, and date ranges.

Compatible with **Python 3.12, 3.13, and 3.14**.

---

## Quick start

### 1. Installation

```bash
# Using pip
pip install -e .

# Or with dev dependencies (pytest, ruff, mypy)
pip install -e ".[dev]"

# Or using uv
uv sync --all-extras
```

### 2. Run CLI

```bash
# Detect a single pattern
yftp --pattern KICKING --symbol EURUSD --timeframe 5m --period 60d

# Scan all patterns for a specific date
yftp --all-patterns --symbol EURUSD --timeframe 5m --period 60d --date 2025-04-01

# Scan all patterns across a date range
yftp --all-patterns --symbol EURUSD --timeframe 15m --period 60d --start-date 2025-04-01 --end-date 2025-04-10
```

---

## CLI Options

```
yfinance-ta-patterns [-h] [-v] (--pattern PATTERN | --all-patterns)
                     [--symbol SYMBOL] [--period PERIOD] [--timeframe TIMEFRAME]
                     [--date YYYY-MM-DD] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
```

- `-v`, `--version`: Show package version.
- `--pattern`: Single candlestick pattern name (e.g. `HAMMER`, `DOJI`, `CDLKICKING`). `CDL` prefix is optional.
- `--all-patterns`: Scan and display signals for all available TA-Lib candlestick patterns.
- `--symbol`: Ticker symbol (default: `EURUSD`). `=X` suffix is automatically appended for forex symbols.
- `--period`: History window (e.g., `5d`, `60d`, `1y`, `max`).
- `--timeframe`: Timeframe alias (`M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`) or raw `yfinance` interval (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`, `1wk`). `H4`/`4h` is automatically resampled from 1h candles.
- `--date`: Filter signals for a specific date (`YYYY-MM-DD`).
- `--start-date` / `--end-date`: Date range filter (`YYYY-MM-DD`). Mutually exclusive with `--date`.

---

## Python API

### Data Loader (`ForexDataLoader`)

Fetches and normalizes OHLC data, handling timezones and resampling:

```python
from yfinance_ta_patterns import ForexDataLoader

loader = ForexDataLoader(symbol="EURUSD", period="60d", interval="15m", timezone="Europe/Moscow")
df = loader.get_data()
print(df.head())
```

### Pattern Analyzer (`PatternAnalyzer`)

Evaluates candlestick patterns on any OHLC `DataFrame`:

```python
from yfinance_ta_patterns import ForexDataLoader, PatternAnalyzer

loader = ForexDataLoader("EURUSD", period="60d", interval="1h")
data = loader.get_data()

analyzer = PatternAnalyzer(data)
signals = analyzer.get_signals("HAMMER", start_date="2025-01-01", end_date="2025-01-15")
print(signals)
```

### Pattern Ranking & Backtest (`PatternRankingTester`)

Tests all candlestick patterns and ranks them by performance (win rate, total PnL, Sharpe ratio):

```python
from yfinance_ta_patterns import ForexDataLoader, PatternRankingTester

loader = ForexDataLoader("EURUSD", period="60d", interval="1h")
data = loader.get_data()

tester = PatternRankingTester(data, initial_capital=10000.0, position_size=100.0)
results = tester.test_all_patterns()

top_10 = tester.get_top_patterns(10)
for r in top_10:
    print(
        f"{r.pattern_name}: Win Rate={r.win_rate:.1f}%, PnL={r.total_pnl:.2f}, Sharpe={r.sharpe_ratio:.2f}"
    )

# Export to CSV
tester.export_results("pattern_ranking.csv")
```

---

## Development & Testing

Run tests and checks locally:

```bash
# Run pytest across Python 3.12, 3.13, 3.14
uv run --python 3.12 --with pytest pytest -v
uv run --python 3.13 --with pytest pytest -v
uv run --python 3.14 --with pytest pytest -v

# Linter and formatting check
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .

# Type checking
uv run --with mypy --with pandas-stubs --with types-pytz mypy yfinance_ta_patterns
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
