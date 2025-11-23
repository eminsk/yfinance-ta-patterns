## Forex Candlestick Scanner

Lightweight CLI that downloads Forex (or any Yahoo Finance ticker) data via `yfinance` and runs all TA-Lib candlestick detectors to highlight signals for your chosen symbol, timeframe, and dates.

### Key features
- Pulls OHLC data from Yahoo Finance with automatic timezone normalization.
- Supports TA-Lib's entire `CDL*` candlestick catalog or a single pattern.
- Date filtering by exact day or start/end range.
- Pulls TA-Lib from PyPI wheels (no bundled wheel needed).
- Experimental pattern ranking helper in `pattern_tester.py`.

### Suggested repo/folder names
- `forex-candlestick-scanner` (descriptive and CLI-friendly)
- `yfinance-ta-patterns`
- `forex-ta-screener`

### Quick start (uv-only)
1) Create a virtual environment and install deps with `uv` (no pip needed):
```bash
uv venv
.\.venv\Scripts\activate  # PowerShell; adjust for your shell
uv sync  # installs yfinance + TA-Lib from PyPI wheels
```
If you prefer to refresh the lockfile manually instead of using the existing `uv.lock`:
```bash
uv add ./ta_lib-0.6.7-cp314-cp314-win_amd64.whl yfinance
```
`pyproject.toml` targets Python `>=3.12`; lower versions may work but are not guaranteed.

2) Run the scanner:
```bash
python main.py --pattern KICKING --symbol EURUSD --timeframe 5m --period 60d
```

### CLI usage
```
python main.py [--pattern NAME | --all-patterns]
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
python main.py --all-patterns --symbol EURUSD --timeframe 5m --period 60d --date 2025-04-01
```
- All patterns across a range:
```bash
python main.py --all-patterns --symbol EURUSD --timeframe 5m --period 60d --start-date 2025-04-01 --end-date 2025-04-10
```
- One pattern without date filter:
```bash
python main.py --pattern KICKING --symbol EURUSD --timeframe 5m --period 60d
```

### Data loader
`forex_data_loader.py` fetches and normalizes OHLC data. It appends `=X` to symbols when missing and converts timestamps to UTC before shifting to the configured timezone (`Europe/Moscow` by default).

### Pattern analysis
`pattern_analyzer.py` wraps TA-Lib's `CDL*` functions, returning non-zero signals and applying optional date filters. When `--all-patterns` is used, it iterates over the full catalog and prints hits per pattern.

### Pattern ranking helper (optional)
`pattern_tester.py` contains a backtesting-style ranking tool. It depends on `utils.pattern_helper.PatternHelper` to enumerate patterns; add that helper before running comparisons or exports.

### Project layout
- `main.py`: CLI entry point.
- `forex_data_loader.py`: Data download and timezone normalization.
- `pattern_analyzer.py`: Candlestick signal extraction.
- `pattern_tester.py`: Experimental ranking/backtest utilities.
- TA-Lib is pulled from PyPI wheels at install time (no bundled wheel).

### Notes
- The project assumes an installed TA-Lib binary; PyPI now ships wheels with the C library embedded for Windows/macOS/Linux on CPython 3.9–3.14. If you adjust the Python version, update `pyproject.toml` accordingly.
