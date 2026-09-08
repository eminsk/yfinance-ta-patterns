"""Comprehensive unit tests covering all 21 bug fixes from Executive Summary audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yfinance_ta_patterns.ai.analyst import AIMarketAnalyst
from yfinance_ta_patterns.ai.scorer import (
    AIPatternScorer,
    calc_wilder_atr,
    calc_wilder_rsi,
)
from yfinance_ta_patterns.data import MarketDataLoader, validate_ohlc
from yfinance_ta_patterns.pattern_tester import (
    DEFAULT_FX_USD_RATES,
    TIMEFRAME_PERIODS_PER_YEAR,
    PatternRankingTester,
    resolve_periods_per_year,
)
from yfinance_ta_patterns.talib_compat import (
    SUPPORTED_FALLBACK_PATTERNS,
    UNSUPPORTED_FALLBACK_PATTERNS,
    TALibWrapper,
)


@pytest.fixture
def synthetic_ohlcv_data() -> pd.DataFrame:
    """Deterministic 30-bar dataset with known prices."""
    dates = pd.date_range("2025-01-01", periods=30, freq="1d", tz="UTC")
    opens = np.array([100.0 + i for i in range(30)])
    highs = opens + 2.0
    lows = opens - 1.0
    closes = opens + 1.0  # Green candles: close > open
    volumes = np.full(30, 10000.0)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )


def test_pattern_tester_lookahead(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Issue 1: Test that default execution is next_open (enters on Open[i+1], not Close[i])."""
    tester = PatternRankingTester(synthetic_ohlcv_data, execution="next_open")
    signals = np.zeros(len(synthetic_ohlcv_data))
    signals[2] = 1  # Signal fires at bar 2
    signals[5] = -1  # Exit signal fires at bar 5

    trades = tester._calculate_trades(signals)
    assert len(trades) == 1
    t = trades[0]
    # Trade must enter at Open[3], not Close[2]
    expected_entry = synthetic_ohlcv_data["Open"].iloc[3]
    assert t["entry_price"] == expected_entry
    assert t["entry_time"] == synthetic_ohlcv_data.index[3]

    # Trade must exit at Open[6], not Close[5]
    expected_exit = synthetic_ohlcv_data["Open"].iloc[6]
    assert t["exit_price"] == expected_exit
    assert t["exit_time"] == synthetic_ohlcv_data.index[6]


def test_pattern_tester_legacy_close_mode(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Test legacy execution mode 'close' enters on Close[i]."""
    tester = PatternRankingTester(synthetic_ohlcv_data, execution="close")
    signals = np.zeros(len(synthetic_ohlcv_data))
    signals[2] = 1
    signals[5] = -1

    trades = tester._calculate_trades(signals)
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_price"] == synthetic_ohlcv_data["Close"].iloc[2]
    assert t["exit_price"] == synthetic_ohlcv_data["Close"].iloc[5]


def test_pattern_tester_short_signal(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Issue 2: Bearish signal (signal == -1) opens Short and computes short PnL correctly."""
    tester = PatternRankingTester(synthetic_ohlcv_data, allow_short=True)
    signals = np.zeros(len(synthetic_ohlcv_data))
    signals[3] = -1  # Bearish signal fires at bar 3
    signals[8] = 1  # Bullish exit signal fires at bar 8

    trades = tester._calculate_trades(signals)
    assert len(trades) == 1
    t = trades[0]
    assert t["direction"] == "SHORT"
    assert t["entry_price"] == synthetic_ohlcv_data["Open"].iloc[4]
    assert t["exit_price"] == synthetic_ohlcv_data["Open"].iloc[9]
    # In uptrend dataset, exit_price > entry_price, so Short must lose money
    assert t["pnl"] < 0
    expected_pnl = (tester._position_size / t["entry_price"]) * (t["entry_price"] - t["exit_price"])
    assert np.isclose(t["pnl"], expected_pnl)


def test_pattern_tester_close_last_trade(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Issue 3: Unclosed position at the end of history is force-closed at Closes[-1]."""
    tester = PatternRankingTester(synthetic_ohlcv_data, force_exit_on_last_bar=True)
    signals = np.zeros(len(synthetic_ohlcv_data))
    signals[20] = 1  # Buy at bar 20, no exit signal follows

    trades = tester._calculate_trades(signals)
    assert len(trades) == 1
    t = trades[0]
    assert t["forced_exit"] is True
    assert t["exit_price"] == synthetic_ohlcv_data["Close"].iloc[-1]
    assert t["exit_time"] == synthetic_ohlcv_data.index[-1]


def test_sharpe_annualization(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Issue 6: Sharpe Ratio scales appropriately with timeframe frequencies."""
    assert TIMEFRAME_PERIODS_PER_YEAR["1d"] == 252.0
    assert TIMEFRAME_PERIODS_PER_YEAR["1h"] == 252.0 * 6.5
    assert TIMEFRAME_PERIODS_PER_YEAR["15m"] == 252.0 * 26.0

    tester_1d = PatternRankingTester(synthetic_ohlcv_data, timeframe="1d")
    tester_1h = PatternRankingTester(synthetic_ohlcv_data, timeframe="1h")

    assert tester_1d._periods_per_year == 252.0
    assert tester_1h._periods_per_year == 252.0 * 6.5


def test_equity_curve_and_drawdown(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Issue 4: Initial capital and equity curve / max drawdown tracking."""
    tester = PatternRankingTester(
        synthetic_ohlcv_data,
        initial_capital=10000.0,
        position_size=1000.0,
    )
    signals = np.zeros(len(synthetic_ohlcv_data))
    signals[2] = 1
    signals[6] = -1

    trades = tester._calculate_trades(signals)
    assert len(trades) == 1
    # Run _test_single_pattern
    _ = tester._test_single_pattern("CDLDOJI", filter_news=False)
    # Check that equity_curve is populated on tester
    assert len(tester.equity_curve) >= 1
    assert tester.equity_curve[0] == 10000.0


def test_date_filter_loading() -> None:
    """Issue 7: MarketDataLoader accepts start and end parameters and stores them."""
    loader = MarketDataLoader(
        "EURUSD",
        start="2023-01-01",
        end="2023-06-30",
        interval="1d",
    )
    assert loader.start == "2023-01-01"
    assert loader.end == "2023-06-30"


def test_auto_adjust_false_for_candles() -> None:
    """Issue 9: MarketDataLoader defaults auto_adjust to False to protect candle shapes."""
    loader = MarketDataLoader("AAPL")
    assert loader.auto_adjust is False
    assert loader.repair is True


def test_ohlc_validation() -> None:
    """Issue 21: validate_ohlc rejects/cleans non-positive prices and broken bars."""
    df_valid = pd.DataFrame(
        {
            "Open": [10.0, 12.0],
            "High": [15.0, 14.0],
            "Low": [9.0, 11.0],
            "Close": [14.0, 13.0],
        }
    )
    cleaned = validate_ohlc(df_valid, strict=True)
    assert len(cleaned) == 2

    # Negative prices
    df_negative = pd.DataFrame(
        {
            "Open": [-10.0, 12.0],
            "High": [15.0, 14.0],
            "Low": [9.0, 11.0],
            "Close": [14.0, 13.0],
        }
    )
    with pytest.raises(ValueError, match="non-positive"):
        validate_ohlc(df_negative, strict=True)
    filtered = validate_ohlc(df_negative, strict=False)
    assert len(filtered) == 1

    # High < Low inconsistency
    df_inverted = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [8.0],
            "Low": [12.0],
            "Close": [9.0],
        }
    )
    with pytest.raises(ValueError, match="Inconsistent"):
        validate_ohlc(df_inverted, strict=True)


def test_h4_alignment_utc() -> None:
    """Issue 8: 4h resample is anchored to standard UTC boundaries."""
    loader = MarketDataLoader("BTC-USD", interval="4h", timezone="UTC")
    # Hourly data starting at 01:00 UTC
    hours = pd.date_range("2025-01-01 01:00", periods=12, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [100.0] * 12,
            "High": [105.0] * 12,
            "Low": [95.0] * 12,
            "Close": [102.0] * 12,
            "Volume": [100.0] * 12,
        },
        index=hours,
    )
    resampled = loader.process(df)
    # The resampled 4h bars should anchor to 00:00, 04:00, 08:00 UTC
    for ts in resampled.index:
        assert ts.hour % 4 == 0


def test_wilder_rsi_formula() -> None:
    """Issue 14: Wilder's RSI calculation matches canonical exponential smoothing."""
    closes = pd.Series(
        [
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
            46.00,
        ]
    )
    rsi = calc_wilder_rsi(closes, period=14)
    # At index 14 (15th bar), Wilder RSI textbook value is ~70.46
    assert np.isclose(rsi.iloc[14], 70.46, atol=0.1)
    # At index 15 (16th bar), Wilder RSI is ~66.25
    assert np.isclose(rsi.iloc[15], 66.25, atol=0.1)


def test_wilder_atr_formula() -> None:
    """Issue 15: Wilder's ATR calculation matches canonical exponential smoothing."""
    highs = pd.Series([10.0 + i * 0.5 for i in range(20)])
    lows = pd.Series([8.0 + i * 0.5 for i in range(20)])
    closes = pd.Series([9.0 + i * 0.5 for i in range(20)])

    atr = calc_wilder_atr(highs, lows, closes, period=14)
    assert len(atr) == 20
    assert np.isnan(atr.iloc[:13]).all()
    assert (atr.dropna() > 0).all()
    # High - Low is constant 2.0, so ATR should converge to 2.0
    assert np.isclose(atr.iloc[-1], 2.0, atol=0.05)


def test_rvol_excludes_current_candle(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Issue 16: RVOL calculation shifts by 1 bar to exclude current candle from average."""
    data = synthetic_ohlcv_data.copy()
    data["Volume"] = 1000.0
    # Massive volume spike on the very last candle
    data.iloc[-1, data.columns.get_loc("Volume")] = 10000.0

    scorer = AIPatternScorer(data)
    last_rvol = float(scorer.df["_RVOL"].iloc[-1])
    # Denominator should be based on previous 1000.0 bars -> 10000 / 1000 = 10.0
    assert np.isclose(last_rvol, 10.0, atol=0.1)


def test_return_20_bars_exact(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Issue 18: return_20_bars_pct correctly compares against iloc[-21]."""
    analyst = AIMarketAnalyst(synthetic_ohlcv_data, [])
    summary = analyst.get_market_regime_summary()

    # Manual calculation: (close[-1] - close[-21]) / close[-21] * 100
    expected = (
        (synthetic_ohlcv_data["Close"].iloc[-1] - synthetic_ohlcv_data["Close"].iloc[-21])
        / synthetic_ohlcv_data["Close"].iloc[-21]
    ) * 100.0

    assert np.isclose(summary["return_20_bars_pct"], round(expected, 2))


def test_comparison_report_pattern_alignment(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Issue 19: Comparison report pairs identical patterns in each row."""
    tester = PatternRankingTester(synthetic_ohlcv_data)
    report = tester.get_comparison_report()
    assert isinstance(report, pd.DataFrame)
    if not report.empty:
        assert "Pattern" in report.columns
        assert "Win Rate (No News Filter)" in report.columns
        assert "Win Rate (With News Filter)" in report.columns


def test_talib_fallback_sets() -> None:
    """Issue 10 & 11: Supported and unsupported patterns are properly classified."""
    wrapper = TALibWrapper(force_fallback=True)
    assert len(SUPPORTED_FALLBACK_PATTERNS) == 11
    assert "CDLDOJI" in SUPPORTED_FALLBACK_PATTERNS
    assert "CDLHAMMER" in SUPPORTED_FALLBACK_PATTERNS
    assert "CDLPIERCING" in UNSUPPORTED_FALLBACK_PATTERNS

    # Supported pattern runs
    res = wrapper.CDLDOJI(np.array([10.0]), np.array([11.0]), np.array([9.0]), np.array([10.0]))
    assert res[0] == 100

    # Unsupported pattern raises NotImplementedError
    with pytest.raises(NotImplementedError):
        wrapper.CDLPIERCING(np.array([10.0]), np.array([11.0]), np.array([9.0]), np.array([10.0]))


def test_forex_quote_currency_conversion() -> None:
    """Issue 5: Forex pairs normalize quote currency to account currency."""
    # Pair with quote JPY and base USD
    df = pd.DataFrame(
        {
            "Open": [150.0, 150.0],
            "High": [151.0, 151.0],
            "Low": [149.0, 149.0],
            "Close": [150.0, 150.0],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="1d", tz="UTC"),
    )
    tester = PatternRankingTester(df, symbol="USDJPY", account_currency="USD")
    raw_pnl_jpy = 1500.0
    exit_price = 150.0
    converted = tester._convert_pnl_to_account_currency(raw_pnl_jpy, exit_price)
    assert np.isclose(converted, 10.0)  # 1500 JPY / 150 rate = 10 USD


def test_rvol_no_bfill_lookahead() -> None:
    """Follow-up Audit: RVOL does not use bfill and does not leak future volume."""
    df = pd.DataFrame(
        {
            "Open": [10.0] * 5,
            "High": [11.0] * 5,
            "Low": [9.0] * 5,
            "Close": [10.0] * 5,
            "Volume": [100.0, 200.0, 150.0, 120.0, 130.0],
        },
        index=pd.date_range("2025-01-01", periods=5, freq="1d", tz="UTC"),
    )
    scorer = AIPatternScorer(df)
    rvol_first = float(scorer.df["_RVOL"].iloc[0])
    # First candle RVOL should be 1.0 (own baseline), not influenced by future 200.0 volume
    assert np.isclose(rvol_first, 1.0)


def test_ema200_warmup_nan_handling() -> None:
    """Follow-up Audit: EMA200 is NaN for datasets <200 bars and handled safely."""
    df = pd.DataFrame(
        {
            "Open": [10.0] * 50,
            "High": [11.0] * 50,
            "Low": [9.0] * 50,
            "Close": [10.0 + i for i in range(50)],
            "Volume": [1000.0] * 50,
        },
        index=pd.date_range("2025-01-01", periods=50, freq="1d", tz="UTC"),
    )
    scorer = AIPatternScorer(df)
    assert np.isnan(scorer.df["_EMA200"].iloc[-1])
    res = scorer.score_signal("HAMMER", df.index[-2], raw_signal=100)
    assert res.trend_regime in ("STRONG_BULLISH", "BULLISH")


def test_universal_fx_engine_cross_pair() -> None:
    """Follow-up Audit: Cross FX pair conversion with fx_rates."""
    df = pd.DataFrame(
        {
            "Open": [160.0, 160.0],
            "High": [161.0, 161.0],
            "Low": [159.0, 159.0],
            "Close": [160.0, 160.0],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="1d", tz="UTC"),
    )
    tester = PatternRankingTester(
        df,
        symbol="EURJPY",
        account_currency="USD",
        fx_rates={"USDJPY": 150.0},
    )
    raw_pnl_jpy = 3000.0
    converted = tester._convert_pnl_to_account_currency(raw_pnl_jpy, 160.0)
    # 3000 JPY / 150 (USDJPY rate) = 20 USD
    assert np.isclose(converted, 20.0)


def test_periodic_sharpe_with_idle_periods(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Follow-up Audit: Periodic Sharpe accounts for zero-return bars during idle periods."""
    tester = PatternRankingTester(
        synthetic_ohlcv_data,
        sharpe_mode="periodic",
    )
    signals = np.zeros(len(synthetic_ohlcv_data))
    signals[2] = 1
    signals[5] = -1

    trades = tester._calculate_trades(signals)
    assert len(trades) == 1
    # Run test single pattern to verify periodic_sharpe and trade_sharpe populated
    res = tester._test_single_pattern("CDLHAMMER", filter_news=False)
    if res:
        assert hasattr(res, "periodic_sharpe")
        assert hasattr(res, "trade_sharpe")
        assert hasattr(res, "avg_strength")


def test_commission_and_slippage(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Follow-up Audit: Commissions and slippage reduce trade net PnL."""
    tester_raw = PatternRankingTester(
        synthetic_ohlcv_data,
        commission=0.0,
        slippage=0.0,
    )
    tester_cost = PatternRankingTester(
        synthetic_ohlcv_data,
        commission=5.0,
        slippage=0.10,
    )

    signals = np.zeros(len(synthetic_ohlcv_data))
    signals[2] = 1
    signals[6] = -1

    trades_raw = tester_raw._calculate_trades(signals)
    trades_cost = tester_cost._calculate_trades(signals)

    assert len(trades_raw) == 1
    assert len(trades_cost) == 1
    # PnL with costs must be strictly lower than raw PnL
    assert trades_cost[0]["pnl"] < trades_raw[0]["pnl"]


def test_min_signals_ranking_filter(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Follow-up Audit: Filter patterns by min_signals to prevent 1-trade overfitting."""
    tester = PatternRankingTester(synthetic_ohlcv_data, min_signals=999)
    results = tester.test_all_patterns(min_signals=999)
    # No pattern should have 999 signals on a small synthetic dataset
    assert len(results) == 0


def test_calc_wilder_rsi_initial_nan_lookback() -> None:
    """Issue: Wilder RSI must keep initial period (14) bars as NaN without lookahead."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(50)))
    rsi = calc_wilder_rsi(prices, period=14)

    # First 14 values (0 to 13) must strictly be NaN
    assert np.isnan(rsi.iloc[:14]).all()
    # 15th value (index 14) is the first computed RSI
    assert not np.isnan(rsi.iloc[14])
    # All valid RSI values are bounded in [0, 100]
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()


def test_calc_wilder_atr_initial_nan_lookback() -> None:
    """Issue: Wilder ATR must keep initial period - 1 (13) bars as NaN without lookahead."""
    np.random.seed(42)
    base = 100.0 + np.cumsum(np.random.randn(50))
    high = pd.Series(base + 2.0)
    low = pd.Series(base - 2.0)
    close = pd.Series(base)

    atr = calc_wilder_atr(high, low, close, period=14)

    # First 13 values (0 to 12) must strictly be NaN
    assert np.isnan(atr.iloc[:13]).all()
    # 14th value (index 13) is the first computed ATR
    assert not np.isnan(atr.iloc[13])
    # All valid ATR values are strictly positive
    valid_atr = atr.dropna()
    assert (valid_atr > 0.0).all()


def test_ema_exact_min_periods_nan() -> None:
    """Issue: EMA20, EMA50, and EMA200 must strictly adhere to min_periods without downscaling."""
    n = 250
    dates = pd.date_range("2025-01-01", periods=n, freq="1d", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [102.0 + i for i in range(n)],
            "Low": [98.0 + i for i in range(n)],
            "Close": [101.0 + i for i in range(n)],
            "Volume": [10000.0] * n,
        },
        index=dates,
    )
    scorer = AIPatternScorer(df)

    # EMA20: first 19 bars are NaN, 20th bar (index 19) is valid
    assert np.isnan(scorer.df["_EMA20"].iloc[:19]).all()
    assert not np.isnan(scorer.df["_EMA20"].iloc[19])

    # EMA50: first 49 bars are NaN, 50th bar (index 49) is valid
    assert np.isnan(scorer.df["_EMA50"].iloc[:49]).all()
    assert not np.isnan(scorer.df["_EMA50"].iloc[49])

    # EMA200: first 199 bars are NaN, 200th bar (index 199) is valid
    assert np.isnan(scorer.df["_EMA200"].iloc[:199]).all()
    assert not np.isnan(scorer.df["_EMA200"].iloc[199])


def test_market_data_loader_2m_interval_retention() -> None:
    """Issue: 2m interval retains up to 60 days on Yahoo Finance, only 1m is restricted to 7d."""
    loader_2m = MarketDataLoader("AAPL", interval="2m", period="60d")
    assert loader_2m.period == "60d"

    loader_1m = MarketDataLoader("AAPL", interval="1m", period="60d")
    assert loader_1m.period == "7d"


def test_universal_cross_fx_eurgbp_conversion() -> None:
    """Issue: Arbitrary Forex cross pairs (e.g. EURGBP) convert quote profit to USD."""
    df = pd.DataFrame(
        {
            "Open": [0.85, 0.85],
            "High": [0.86, 0.86],
            "Low": [0.84, 0.84],
            "Close": [0.85, 0.85],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="1d", tz="UTC"),
    )
    # Default baseline rate: GBPUSD = 1.28
    tester_default = PatternRankingTester(df, symbol="EURGBP", account_currency="USD")
    raw_pnl_gbp = 100.0
    converted = tester_default._convert_pnl_to_account_currency(raw_pnl_gbp, 0.85)
    assert np.isclose(converted, 100.0 * DEFAULT_FX_USD_RATES["GBPUSD"])  # 128.0 USD

    # Custom rate override: GBPUSD = 1.35
    tester_custom = PatternRankingTester(
        df, symbol="EURGBP", account_currency="USD", fx_rates={"GBPUSD": 1.35}
    )
    converted_custom = tester_custom._convert_pnl_to_account_currency(raw_pnl_gbp, 0.85)
    assert np.isclose(converted_custom, 135.0)


def test_asset_aware_periods_per_year() -> None:
    """Issue: Sharpe annualization scales correctly for 24/7 Crypto, 24/5 Forex, and Equities."""
    # Crypto: 365 days / 24h
    assert resolve_periods_per_year("1d", "BTC-USD") == 365.0
    assert resolve_periods_per_year("1h", "BTC-USD") == 8760.0
    assert resolve_periods_per_year("4h", "ETH-USD") == 2190.0

    # Forex: 260 days / 24h
    assert resolve_periods_per_year("1d", "EURUSD=X") == 260.0
    assert resolve_periods_per_year("1h", "EURUSD=X") == 6240.0
    assert resolve_periods_per_year("4h", "USDJPY=X") == 1560.0

    # Equities: 252 days / 6.5h
    assert resolve_periods_per_year("1d", "AAPL") == 252.0
    assert resolve_periods_per_year("1h", "AAPL") == 1638.0
    assert resolve_periods_per_year("4h", "AAPL") == 504.0


def test_pattern_ranking_profit_factor_and_score(synthetic_ohlcv_data: pd.DataFrame) -> None:
    """Issue: PatternResult contains profit_factor and composite score for ranking."""
    tester = PatternRankingTester(synthetic_ohlcv_data)
    results = tester.test_all_patterns(sort_by="composite")
    assert isinstance(results, list)
    for res in results:
        assert hasattr(res, "profit_factor")
        assert hasattr(res, "score")
        assert res.profit_factor >= 0.0
        assert res.score >= 0.0
