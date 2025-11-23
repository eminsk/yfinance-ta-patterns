## Forex Candlestick Scanner

Python package + CLI that downloads Yahoo Finance data via `yfinance` and runs TA-Lib candlestick detectors for your symbol, timeframe, and date filters.

### Quick start
```bash
# create env
python -m venv .venv
.\.venv\Scripts\activate  # PowerShell; adjust for your shell

# install the package (editable for local dev)
pip install -e .

# run CLI (two entrypoints)
yfinance-ta-patterns --pattern KICKING --symbol EURUSD --timeframe 5m --period 60d
# or
yftp --all-patterns --symbol EURUSD --timeframe 5m --period 60d --date 2025-04-01
```
`pyproject.toml` targets Python `>=3.12`.

### CLI usage
```
yfinance-ta-patterns [--pattern NAME | --all-patterns]
                     [--symbol EURUSD] [--period 60d]
                     [--timeframe 15m] [--date YYYY-MM-DD]
                     [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
```
- `--pattern`: Single candlestick name (with or without `CDL` prefix).
- `--all-patterns`: Scan every TA-Lib candlestick detector.
- `--symbol`: Ticker without suffix; `=X` is appended automatically for Forex (default `EURUSD`).
- `--period`: History window passed to `yfinance` (e.g., `60d`, `1mo`).
- `--timeframe`: Use `M1/M5/M15/M30/H1/D1` or raw `yfinance` intervals (`1m`, `5m`, `1h`, `1d`, etc.).
- `--date`: Filter signals for a single day.
- `--start-date` / `--end-date`: Inclusive range filter (cannot be combined with `--date`).

### Examples
- All patterns for a single day:
```bash
yftp --all-patterns --symbol EURUSD --timeframe 5m --period 60d --date 2025-04-01
```
- All patterns across a range:
```bash
yftp --all-patterns --symbol EURUSD --timeframe 5m --period 60d --start-date 2025-04-01 --end-date 2025-04-10
```
- One pattern without date filter:
```bash
yftp --pattern KICKING --symbol EURUSD --timeframe 5m --period 60d
```

### Data loader
`yfinance_ta_patterns/forex_data_loader.py` fetches and normalizes OHLC data. It appends `=X` to symbols when missing and converts timestamps to UTC before shifting to the configured timezone (`Europe/Moscow` by default).

### Pattern analysis
`yfinance_ta_patterns/pattern_analyzer.py` wraps TA-Lib's `CDL*` functions, returning non-zero signals and applying optional date filters. When `--all-patterns` is used, it iterates over the full catalog and prints hits per pattern.

### Pattern ranking helper (optional)
`yfinance_ta_patterns/pattern_tester.py` contains a backtesting-style ranking tool. It depends on `utils.pattern_helper.PatternHelper` to enumerate patterns; add that helper before running comparisons or exports.

### Project layout
- `yfinance_ta_patterns/cli.py`: CLI entry point and argument parsing.
- `yfinance_ta_patterns/forex_data_loader.py`: Data download and timezone normalization.
- `yfinance_ta_patterns/pattern_analyzer.py`: Candlestick signal extraction.
- `yfinance_ta_patterns/pattern_tester.py`: Experimental ranking/backtest utilities.
- `main.py`: Thin wrapper to launch the CLI.

### Packaging and releases
- Nightly GitHub Actions workflow builds onefile Nuitka binaries for Windows/macOS/Linux (PyPI wheels for TA-Lib on all platforms) and publishes nightly prereleases.
- PyPI publish workflow runs on tags (needs secret `PYPI_API_TOKEN`); tags like `v0.1.0` will build sdist/wheel and upload.
