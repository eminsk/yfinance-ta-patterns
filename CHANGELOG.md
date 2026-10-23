# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.71] - 2026-10-22

### Fixed
- **MCP Watchlist Thread Concurrency & pandas 2.x Index Cache Isolation (`mcp_server.py`)**:
  - Fixed an elusive thread race condition where custom `data_fetcher` implementations returning `.copy()` of a cached DataFrame caused worker threads in `ta_scan_watchlist` to share pandas 2.x's lazily populated internal `Index._cache` without synchronization. Under concurrent validation, this occasionally triggered false "DataFrame index contains duplicate timestamps" errors (dropping symbols like `NVDA` in ~10% of runs on Python 3.9/3.10).
  - Introduced `_isolated_frame(df)` to ensure deeply copied indices (`out.index = out.index.copy(deep=True)`) whenever data frames are injected via custom fetchers.
- **Expanded Test Suite**:
  - Test suite expanded to **602 tests** (added index isolation, passthrough of non-frame results, and 40-iteration concurrent watchlist stress test), passing on Linux x86_64 across CPython 3.8–3.15, free-threaded 3.14t, and verified across CI matrices.

## [0.3.70] - 2026-10-22

### Added
- **MCP Event Risk Test Clock Pinning & Window Hardening (`tests/test_audit_fixes_v19.py`, `tests/test_mcp.py`)**:
  - Pinned the MCP server's clock in tests by monkeypatching `mcp_server.dt.datetime.now()` to a deterministic UTC midday instant (`2026-10-21 12:00:00 UTC`), completely eliminating the time-dependent test failure window that occurred between 00:00 and 00:16 UTC where same-day 00:01 events fell within the 15-minute recency window.
  - Added parametrized test coverage (`test_mcp_event_risk_grace_window_is_fifteen_minutes`) explicitly testing the 15-minute event risk grace window boundary (14 minutes ago kept, 16 minutes ago dropped).
- **Expanded Test Suite**:
  - Test suite expanded to **599 tests**, 100% passing across all Python versions and platforms.

## [0.3.69] - 2026-10-21

### Added
- **Inclusive `--end-date` CLI Semantics (`cli.py`)**:
  - Aligned `--end-date` with user documentation: date-only inputs without timestamp are now automatically forwarded as the next midnight (`pd.to_datetime(end) + pd.Timedelta(days=1)`), compensating for `yf.download(end=...)`'s exclusive parameter and ensuring the requested final date is actually fetched.
  - Fixed single-day range execution: previously, `yftp AAPL -t 1d --start-date 2026-10-07 --end-date 2026-10-07` collapsed into an empty window `[2026-10-07, 2026-10-07)` resulting in "No market data found". It now correctly queries `[2026-10-07, 2026-10-08)`.
  - Added strict date-order validation: passing `--start-date` later than `--end-date` proactively raises an informative `ValueError` before querying upstream data.
  - Maintained inclusive date reporting in CLI backtest ranking headers (`🏆 QUANTITATIVE PATTERN RANKING: SPY (1d, 2026-08-03 to 2026-08-07)`).
- **Comprehensive 1m Request Window Validation (`cli.py`)**:
  - Extended 1m 7-day request window validation to single-sided ranges:
    - Single `--start-date` older than 7 days is proactively caught before dispatching requests spanning up to today (> 7 days).
    - Single `--end-date` without `--start-date` is rejected up-front because yfinance defaults missing start parameters to `end - 1 month`, which invariably violates the 7-day maximum query window.
  - Corrected window span calculation: counts inclusive calendar days (`(last - first).days + 1 <= 7`), properly rejecting 8-day spans that previously passed validation.
- **Table-Driven Period Clamping (`data.py`)**:
  - Replaced hardcoded string matching in `MarketDataLoader._clamp_period_for_interval` with dynamic period parsing (`_period_days()`) and table-driven lookups against `YAHOO_INTRADAY_MAX_WINDOW_DAYS`.
  - Correctly clamps previously unhandled periods: `20d` or `2mo` for `1m` clamps to `7d`; `61d`–`89d` for `5m`–`90m` clamps to `60d`; `731d` and `3y` for `1h`/`4h` clamp to `730d`.
- **Pure-Python `fast_score_batch` Buffer Parity (`gemm.py`)**:
  - Unified `out` parameter handling across all three execution branches (native NanoGEMM SIMD, NumPy, and pure-Python): in the pure-Python branch, `out` is now only written to inplace if `isinstance(out, list) and len(out) == len(res_list)`. Mismatched lists are left untouched and a new list is returned, eliminating silent list mutation/resizing.
  - Updated `fast_score_batch` docstring to explicitly describe `out` matching requirements.
- **Expanded Test Suite**:
  - Added dedicated regression test suite `tests/test_audit_fixes_v25.py` (34 new tests, expanding total suite to **597 tests**, 100% passing across all Python versions).

## [0.3.68] - 2026-10-21

### Added
- **Yahoo Finance 1m Limits Refactor: Depth vs. Window (`data.py`, `cli.py`)**:
  - Live API validation confirmed Yahoo Finance allows 1m granularity up to **30 days** of historical depth from today, but restricts individual query spans to at most **7 days**.
  - Refactored `YAHOO_INTRADAY_MAX_DEPTH_DAYS` (1m: 30d, 2m–90m: 60d, 1h–4h: 730d) and `YAHOO_INTRADAY_MAX_WINDOW_DAYS` (1m: 7d, 2m–90m: 60d, 1h–4h: 730d), keeping `YAHOO_INTRADAY_MAX_DAYS` as the canonical depth alias.
  - Clarified loader vs CLI roles: `MarketDataLoader._clamp_period_for_interval` automatically clamps period windows to provider limits (e.g. 1m clamped to 7d), while CLI validates rolling historical depths from today and single-request query spans.
  - Added CLI span validation: `--start-date` and `--end-date` for 1m requests spanning $> 7$ days are proactively caught with an informative `ValueError` before sending invalid requests to Yahoo.
- **Economic Calendar Country Mode & Zero-Allocation Hoisting (`economic_calendar.py`)**:
  - Added missing ISO 3166-1 alpha-3 country mappings: `ARE` -> `['AED']`, `CHL` -> `['CLP']`, `COL` -> `['COP']`, `MYS` -> `['MYR']`, and `PHL` -> `['PHP']`.
  - Hoisted `_ISO3_TO_CURRENCY`, `_COMMODITY_METALS`, and `_G10_AUTO_CURRENCIES` to module-level frozensets/dicts, eliminating redundant dictionary re-creation on every symbol resolution call.
  - Cleaned up obsolete dead code `_STOCK_COLLISIONS` in auto mode since G10 bare currency filtering naturally shields all US equity tickers (`COP`, `DE`, `IT`, `NOK`, etc.).
- **Platform Clarification for Hardware SIMD Acceleration (`[gemm]`)**:
  - Documented that `[gemm]` (`nanogemm`) ships pre-compiled bare-metal AVX2+FMA binary wheels on Windows x64. On Linux, macOS, and other platforms, a standard `pip install` fetches the universal wheel containing the Windows binary and transparently runs via vectorized NumPy (backed by OpenBLAS/MKL/Accelerate) and pure-Python fallbacks; native compilation on Unix occurs only when explicitly passing `pip install --no-binary nanogemm nanogemm`.
- **Expanded Test Suite**:
  - Added dedicated regression test suite `tests/test_audit_fixes_v24.py` (10 new tests, expanding total suite to **563 tests**, 100% passing across all Python versions).

### Fixed
- **Oversized Buffer Scalar Broadcasting (`gemm.py`)**:
  - Fixed an issue in `fast_score_batch` where passing an oversized `out` buffer (e.g. `(5,)`) with a single sample `(1,)` broadcast the scalar score into all elements. Now strictly enforces `out.shape == res.shape` (and `len(out) == len(res)` for list buffers) before writing in-place on both native NanoGEMM SIMD and NumPy fallback branches.
- **CLI Backtest Hook Error Safety (`cli.py`)**:
  - Guarded `--calendar` and `--news` hooks in `--backtest` mode to only execute when `ret_code == 0`, preventing displaying irrelevant macroeconomic or news tables when backtesting fails (e.g. due to an unknown pattern or data retrieval error).

## [0.3.67] - 2026-10-21

### Added
- **Backtest CLI Hooks & Failure Isolation (`cli.py`)**:
  - Connected `--calendar` and `--news` display hooks following backtest execution (`yftp EURUSD -t 1h --backtest --calendar --news`) in text and markdown modes while preserving pristine output in `--format json`.
  - Added case-insensitive pattern normalization (`--pattern cdlengulfing` -> `CDLENGULFING`) and strict validation against `get_all_patterns()`, exiting with code 1 and listing available patterns on unknown inputs.
  - Multi-symbol failure isolation in backtest loop: wrapped data retrieval and backtest execution in `try...except`, capturing errors per-symbol in JSON mode (`{"SYM": {"error": "..."}}`) and exiting with code 1 if any failure occurs.
  - Actual date range display in backtest report headers (`🏆 QUANTITATIVE PATTERN RANKING: SPY (1d, 2026-08-01 to 2026-09-01)`).
- **Intraday Rolling Limits Unification (`data.py`, `cli.py`)**:
  - Exported canonical `YAHOO_INTRADAY_MAX_DAYS` dictionary (`{"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "90m": 60, "60m": 730, "1h": 730, "4h": 730}`).
  - Fixed boundary check to `>=` (preventing `start == end` edge cases), aligned `2m` to 60 days, added `90m` support, and validated `--start-date` and `--end-date` against intraday rolling windows.
- **Economic Calendar & Currency Resolution (`economic_calendar.py`)**:
  - Fixed delimiter stripping for precious metals (`XAU-EUR` -> `['EUR']`), supporting `-`, `/`, `=`, `_` separators.
  - Restricted auto-mode bare currency codes to G10 (`USD`, `EUR`, `GBP`, `JPY`, `CHF`, `AUD`, `CAD`, `NZD`) and added US equity ticker collisions (`NOK`, `PEN`, `EGP`, `IDR`), correctly resolving them to `['USD']` in auto mode while preserving currency codes in `forex` mode or with `=X`.
  - Raised descriptive `ValueError` for unrecognized country identifiers in `asset_type="country"`, and added ISO 3166-1 alpha-3 mappings (`DEU` -> `['EUR']`, etc.).
- **Memory Aliasing & Buffer Resilience (`gemm.py`)**:
  - Added memory aliasing protection in `fast_matmul` via `np.shares_memory(out, a) or np.shares_memory(out, b)`, allocating an intermediate buffer when output overlaps with operands to prevent corrupting in-flight matrix data.
  - Supported `out=list` in `fast_score_batch` across both native NanoGEMM and NumPy fallback branches via `res.tolist()`.
- **Expanded Test Suite**:
  - Added dedicated regression test suite `tests/test_audit_fixes_v23.py` (14 new automated tests, expanding total suite to **553 tests**, 100% passing across all Python versions).

## [0.3.66] - 2026-10-20

### Added
- **Multi-Symbol & Targeted Pattern Backtesting (`cli.py`, `pattern_tester.py`)**:
  - Full multi-symbol backtesting support via `_run_backtest_cli()` (`yftp EURUSD GBPUSD --backtest`), eliminating silent fallback to scanning mode.
  - Targeted pattern ranking via `--pattern X --backtest` across single and multi-symbol portfolios.
  - Added `-t` command-line shortcut for `--timeframe` in CLI parser.
  - Added dedicated regression test suite `tests/test_audit_fixes_v22.py` (14 new automated tests, expanding total suite to **539 tests**, 100% passing).
- **Platform Clarification for Optional SIMD Acceleration**:
  - Documented that pre-compiled binary wheels of optional extra `[gemm]` (`nanogemm`) provide bare-metal AVX2+FMA SIMD acceleration on Windows x64 out of the box, while on Unix platforms a standard installation utilizes vectorized NumPy and pure-Python fallbacks (or compiles natively from sdist via `pip install --no-binary nanogemm nanogemm`).

### Fixed
- **Redundant Backtesting Pass Elimination**: Removed 3x redundant backtesting passes in text/markdown mode by directly calling `tester.get_comparison_report(patterns=...)` rather than discarding an unused initial pass.
- **Strict MCP Schema Parameter Validation**: Changed `clamp=False` by default in `_parse_int_param` and `_parse_float_param`, returning structured JSON-RPC `isError=True` errors for out-of-bounds parameters (e.g. `min_confidence=70`, `days=-5`, `lookback_bars=100000`) rather than silently clamping and masking issues.
- **Yahoo Finance Intraday Limits & `--date` Clamping**:
  - Enforced provider intraday rolling window limits (1m: 7d; 5m/15m/30m: 60d; 60m/1h/4h: 730d), throwing descriptive validation errors when target `--date` is outside the historical window.
  - Clamped warm-up lookback start date to provider cutoff so historical intraday requests get maximum warm-up bars without triggering HTTP 400 errors.
- **`gemm.py` Memory Layouts & Fallback Resilience**:
  - Supported caller-supplied `out` buffers in `fast_matmul` via `np.copyto` and in `fast_score_batch` on native NanoGEMM SIMD branch.
  - Handled `(K, 1)` and `(1, K)` column/row weight vectors in pure-Python fallback.
  - Handled empty batch `features=[]` in pure-Python fallback without raising `IndexError`.
- **Macroeconomic Calendar Currency Resolution**:
  - Corrected currency resolution for precious metals (`XAU`, `XAG`, `XPT`, `XPD`) and currency pairs (`XAUUSD` -> `['USD']`, `XAUEUR` -> `['EUR']`).
  - Handled `asset_type="country"` directly without raising `Unsupported asset_type: country`.
  - Expanded auto-mode ISO currency code recognition (`MXN`, `NOK`, `SEK`, `ZAR`, `CNY`, `INR`, `TRY`) while shielding against US equity ticker collisions (`COP`, `DE`, `IT`, `ES`, `FR`).
- **Lookback Warmup Documentation in `talib_compat.py`**:
  - Documented canonical lookback behavior for micro-candle series ($n \le \text{lookback}$) versus full signal parity on market data.

---

## [0.3.65] - 2026-10-20

### Fixed
- **CLI Positional Symbol Deduplication**: Fixed double-merging in `cli.run_cli` when positional tickers are supplied (`parsed.positional_symbols` is now cleared once merged into `parsed.symbol`), preventing single-symbol scans like `yftp SPY` from turning into multi-instrument scans with 0.55 confidence thresholds.
- **Dependency Cleanliness (`nanogemm` & `numpy`)**: Returned `nanogemm` from base `dependencies` to optional extra `[gemm]` (`pip install yfinance-ta-patterns[gemm]`). This completely avoids pulling in `nanogemm`'s `numpy<2.3` constraint and prevents downstream resolver conflicts with `numpy>=2.3`.
- **Fast Score Batch Float32 Sanitization**: Explicitly cast and ensure C-contiguous `float32` memory layout in `fast_score_batch` before invoking native NanoGEMM SIMD kernels, preventing garbage scores when passed standard `float64` pandas/numpy matrices.
- **`cdl_hikkakemod` Warmup Parity**: Restored setup scanning across warmup bars 7..9 so confirmations at bar 10 match canonical native C TA-Lib 100%.
- **Forex 4h Friday Weekly Close Timing**: Fixed secondary closed-only candle filtering in `MarketDataLoader.process` for resampled series, ensuring the completed 20:00 UTC Friday candle is immediately available upon the weekly close.
- **Empty `--symbol ""` Validation**: Enforced strict `args.symbol is not None` checks in CLI runner to raise a clean validation error instead of silently falling back to scanning all 56 Forex pairs.
- **Economic Calendar Commodity Metal Handling**: Excluded commodity metal codes (`XAU`, `XAG`, `XPT`, `XPD`) from macroeconomic calendar currency extraction so precious metals like `XAUUSD` properly resolve to their quote fiat currency (`USD`).
- **Packaging & Lockfile Cleanup**: Removed all local machine paths (`C:/proekts/talib-build`) from `uv.lock` and refreshed to the latest upstream packages.

---

## [0.3.64] - 2026-10-19

### Changed
- **CLI Output Modernization**: Replaced verbose native TA-Lib compilation prompts in standard market scanning commands with a clean, confident status banner:
  - Displays built-in autonomous engine status (`61/61 patterns active`).
  - Displays the active hardware matrix SIMD backend (`NanoGEMM AVX2+FMA`).
  - Retains full manual C-binary installation guides and environment diagnostics strictly for `yftp --check-talib`.

---

## [0.3.63] - 2026-10-19

### Changed
- **Default Core Dependency**: Moved `nanogemm>=0.3.13` from optional extras (`[gemm]`) directly into primary `dependencies` in `pyproject.toml`.
  - Standard installations (`pip install yfinance-ta-patterns`, `uv add yfinance-ta-patterns`) now automatically install `nanogemm` and activate hardware AVX2+FMA SIMD acceleration out of the box without requiring extra brackets.
  - Environment markers (`implementation_name != 'pypy' and python_version < '3.16'`) preserve autonomous zero-binary fallback on PyPy and Python 3.16.
- **Recipe & Packaging Updates**: Added `nanogemm >=0.3.13` to run requirements in Conda recipe (`recipe/recipe.yaml` and `conda-recipe/recipe.yaml`) and Debian changelog.

---

## [0.3.62] - 2026-10-19

### Added
- **NanoGEMM Bare-Metal SIMD Acceleration Engine (`yfinance_ta_patterns.gemm`)**:
  - Integrated hybrid linear algebra and matrix scoring engine with dynamic dispatch between bare-metal AVX2+FMA SIMD (`nanogemm`), OpenBLAS vectorized fallback (`numpy`), and scalar Pure-Python fallback.
  - Implemented `fast_matmul(left, right, out=None)` for sub-microsecond 2D matrix multiplication ($M \le 64$, $K \le 64$).
  - Implemented `fast_score_batch(features, weights, bias=0.0, activation="sigmoid", out=None)` supporting scalar and multi-vector bias arrays, multi-column weights, and stable activations (`sigmoid`, `relu`, `linear`).
  - Implemented `cosine_similarity_matrix(query_vectors, pattern_prototypes)` for rapid candlestick vector matching with zero-division protections (`norm == 0` guards).
  - Added `AIPatternScorer.score_batch_matrix()` and `AIPatternScorer.gemm_backend` properties.
  - Added CLI diagnostics reporting the active linear algebra and SIMD ISA backend via `yftp --check-talib`.
- **Ecosystem Integration & Packaging**:
  - Added optional dependency `gemm = ["nanogemm>=0.3.13"]` in `pyproject.toml`.
  - Added automated PyPI and Conda-Forge download counter badges in `README.md`.
  - Exported `HAS_NANOGEMM`, `get_gemm_backend`, `fast_matmul`, `fast_score_batch`, and `cosine_similarity_matrix` in `__init__.py`.
- **Extended Test Suite**:
  - Added 17 unit and integration tests in `tests/test_nanogemm_integration.py` covering dimension parity, memory continuity (Fortran order), numerical precision, zero vectors, dimension mismatch errors, and fallback behaviors.
  - Total test suite expanded to **512 automated tests** (100% pass rate).

### Performance & Benchmarks (v0.3.61 vs v0.3.62)
- **Matrix Multiplication Latency**: NanoGEMM achieves **1.11 – 1.44 µs** per operation on small matrices (e.g., $(8 \times 16) \times (16 \times 8)$) vs NumPy's **3.29 – 4.07 µs** (**2.5x – 2.9x faster**).
- **Mathematical Precision**: Maximum absolute deviation $\Delta \le 1.91 \times 10^{-6}$ (within `float32` machine epsilon $\varepsilon$).
- **FMA3 Accuracy**: Leverages hardware Fused Multiply-Add (single rounding step) eliminating intermediate rounding errors.
- **Concurrency & Free-Threading**: NanoGEMM executes directly in the calling thread without contending for OpenBLAS global thread pool locks under Python 3.13t/3.14t No-GIL free-threading.

---

## [0.3.61] - 2026-10-19

### Fixed & Hardened
- **CLI MCP Stdio Server**: Fixed `yftp --mcp` and `uvx yfinance-ta-patterns --mcp` argument parsing conflict.
- **Forex Weekend Candle Retention**: Preserves completed Friday 17:00 NY weekly close candles throughout weekends for 4h and clean forex daily intervals.
- **Intraday Yahoo Clamp**: Hardened `clean_forex_daily=True` to clamp download history to 730 days, matching Yahoo Finance 1h interval constraints.
- **Pure JSON CLI Output**: Routed scanning banners and progress indicators to `stderr`, guaranteeing pure parseable JSON on `stdout` and returning `[]` for empty results.
- **CLI Backtester Flags**: Connected `--execution`, `--commission`, `--slippage`, and `--min-signals` to `PatternRankingTester` via `--backtest`.
- **Precious Metals Normalization**: Extended currency code registry to normalize `XAUUSD`, `XAGUSD`, and `XAU/USD` to `=X` forex commodity pairs.
- **Macroeconomic Calendar Equity Shield**: Fixed equity tickers (`COP`, `DE`, `IT`, etc.) from conflicting with ISO country/currency resolutions.
- **TA-Lib Warmup Parity**: Aligned pure-Python warmup indexing with native TA-Lib for `CDL3OUTSIDE` (index 3) and `CDLHIKKAKEMOD` (index 10).
- **CLI Positional Symbols**: Added support for intermixed arguments (`yftp SPY --timeframe 1d NVDA`) and multi-arg `--symbol`.
- **MCP JSON-RPC 2.0 Batch Support**: Full support for batch arrays and strict timeframe enum schema validation.
- **Caller DataFrame Safety**: `MarketDataLoader.process()` now enforces caller DataFrame column immutability.
- **Two-Way Pattern Integrity**: Pruned 23 strictly one-directional patterns from `TWO_WAY_PATTERNS`.

---

## [0.3.60] - 2026-10-13

### Added
- **Positional Symbol Arguments**: Support for running `yftp SPY NVDA AAPL` without `--symbol` flags.
- **Native CLI Flag `--mcp`**: Direct launching of MCP server without external wrapper scripts.
- **AI Agent Registry**: Added `smithery.yaml` for Smithery AI registry discovery.
- **Reasoning Model Prompt Templates**: Structured prompt engineering for frontier LLMs (Claude 3.7 Sonnet, o3-mini, Gemini 2.5 Pro).

---

## [0.3.59] - 2026-10-12

### Changed
- **Autonomous Zero-Binary Packaging**: Complete separation from mandatory binary TA-Lib and scikit-learn installations, enabling instant installation on bare Linux, macOS, and Windows environments.

---

## [0.3.58] - 2026-10-11

### Added
- **PatternRankingTester API**: Added `PatternRankingTester.get_all_patterns()` public inspection interface.
- **Python 3.16 Support**: Confirmed standalone compatibility with Python 3.16 pre-releases.

---

## [0.3.57] - 2026-10-10

### Added
- **PyPy 3.11 & 3.12 Acceleration**: Added pure-Python vector shims removing strict NumPy compilation requirements on PyPy.

---

## [0.3.56] - 2026-10-09

### Added
- **Universal PyPy Fallback**: Cross-platform wheelhouse support and pure-Python pattern fallback.

---

## [0.3.55] - 2026-10-08

### Fixed
- **Windows dateutil ctypes Handle**: Fixed handle leak when parsing date intervals under Windows CRT.

---

## [0.3.50] - 2026-10-07

### Added
- **Free-Threaded No-GIL Support**: Verified execution under CPython 3.13t/3.14t/3.16t (PEP 703) with zero race conditions.

---

## [0.3.49] - 2026-10-06

### Added
- **Conda-Forge Rattler-Build**: Automated recipe and noarch packaging on Conda-Forge.

---

## [0.3.48] - 2026-10-06

### Added
- **MCP Protocol 2025-11-25**: Lifecycle version negotiation, outputSchema typing, UTF-8 raw token encoding, and compact watchlist modes.

---

## [0.3.45] - 2026-09-28

### Added
- **100.000% TA-Lib Parity**: Mathematical alignment with canonical C TA-Lib across all 61 candlestick patterns, Greg Morris threshold matching ($\pm 80$), and rolling sum lookbacks.
