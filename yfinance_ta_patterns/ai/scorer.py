"""AI-driven pattern confidence scorer and trade setup calculator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd


class SignalGrade(StrEnum):
    """Categorical evaluation grade for pattern quality."""

    EXCELLENT = "EXCELLENT"  # >= 0.80
    STRONG = "STRONG"  # >= 0.70
    MODERATE = "MODERATE"  # >= 0.55
    WEAK = "WEAK"  # >= 0.40
    FALSE_SIGNAL = "FALSE"  # < 0.40


@dataclass(slots=True, frozen=True)
class TradeSetup:
    """Actionable trade setup parameters based on ATR and pattern invalidation."""

    direction: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    risk_per_unit: float
    rr_tp1: float = 1.5
    rr_tp2: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        """Convert setup to plain dictionary."""
        return {
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "rr_tp1": round(self.rr_tp1, 2),
            "rr_tp2": round(self.rr_tp2, 2),
            "risk_per_unit": self.risk_per_unit,
        }


@dataclass(slots=True, frozen=True)
class PatternConfidenceResult:
    """Comprehensive AI evaluation of a detected candlestick pattern."""

    pattern_name: str
    timestamp: pd.Timestamp
    raw_signal: int
    confidence_score: float
    grade: SignalGrade
    trend_regime: str
    rvol: float
    rsi: float
    atr: float
    confluence_factors: list[str]
    risk_factors: list[str]
    trade_setup: TradeSetup | None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a structured dictionary for JSON / LLM consumption."""
        return {
            "pattern": self.pattern_name,
            "timestamp": str(self.timestamp),
            "raw_signal": self.raw_signal,
            "confidence_score": round(self.confidence_score, 4),
            "grade": self.grade.value,
            "trend_regime": self.trend_regime,
            "metrics": {
                "rvol": round(self.rvol, 2),
                "rsi": round(self.rsi, 2),
                "atr": round(self.atr, 5),
            },
            "confluence_factors": self.confluence_factors,
            "risk_factors": self.risk_factors,
            "trade_setup": self.trade_setup.to_dict() if self.trade_setup else None,
        }

    @property
    def confidence(self) -> float:
        """Convenience alias for confidence_score."""
        return self.confidence_score

    @property
    def setup(self) -> TradeSetup | None:
        """Convenience alias for trade_setup."""
        return self.trade_setup

    @property
    def confluences(self) -> list[str]:
        """Convenience alias for confluence_factors."""
        return self.confluence_factors

    @property
    def risks(self) -> list[str]:
        """Convenience alias for risk_factors."""
        return self.risk_factors

    @property
    def action(self) -> str:
        """Trading action direction ('BUY', 'SELL', or 'HOLD')."""
        if self.trade_setup:
            return self.trade_setup.direction
        return "BUY" if self.raw_signal > 0 else ("SELL" if self.raw_signal < 0 else "HOLD")


def calc_wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using canonical Wilder's exponential smoothing.

    Initial bars before period remain NaN in accordance with TA-Lib lookback invariants.
    """
    n = len(close)
    if n <= 1:
        return pd.Series(np.nan, index=close.index)

    delta = close.diff().to_numpy()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    rsi = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return pd.Series(rsi, index=close.index)

    eff_period = period

    # First value at eff_period using SMA of initial gains/losses
    avg_g = float(np.mean(gain[1 : eff_period + 1]))
    avg_l = float(np.mean(loss[1 : eff_period + 1]))

    if avg_l == 0.0:
        rsi[eff_period] = 100.0 if avg_g > 0 else 50.0
    else:
        rs = avg_g / avg_l
        rsi[eff_period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(eff_period + 1, n):
        avg_g = (avg_g * (period - 1) + gain[i]) / period
        avg_l = (avg_l * (period - 1) + loss[i]) / period
        if avg_l == 0.0:
            rsi[i] = 100.0 if avg_g > 0 else 50.0
        else:
            rs = avg_g / avg_l
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    # Initial bars before eff_period remain NaN (zero future lookahead)
    return pd.Series(rsi, index=close.index)


def calc_wilder_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate Average True Range (ATR) using canonical Wilder's exponential smoothing.

    Initial bars before index period remain NaN in accordance with TA-Lib lookback invariants.
    """
    n = len(close)
    if n == 0:
        return pd.Series(dtype=np.float64)

    h = high.to_numpy()
    lo = low.to_numpy()
    c = close.to_numpy()

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = h[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))

    atr = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return pd.Series(atr, index=close.index)

    # First value at index period using mean of TR[1 : period + 1] matching TA-Lib
    atr[period] = float(np.mean(tr[1 : period + 1]))
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Initial bars before index period remain NaN (zero future lookahead)
    return pd.Series(atr, index=close.index)


class AIPatternScorer:
    """Probabilistic multi-factor technical scoring engine for candlestick patterns.

    Evaluates market context (Trend regime, RVOL volume expansion, Wilder RSI momentum,
    and ATR volatility) to transform discrete TA-Lib signals (+100/-100) into calibrated
    confidence scores (0.0 to 1.0) and actionable risk-managed trade setups.

    Note: This is an interpretable, deterministic quantitative confluence scoring engine,
    not a black-box deep learning model.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize scorer with OHLCV market data."""
        if data.empty or len(data) < 5:
            raise ValueError("Data must contain at least 5 candles for technical context.")

        self.df = data.copy()
        self._calculate_technical_indicators()

    def _calculate_technical_indicators(self) -> None:
        """Vectorized computation of EMA, Wilder RSI, Wilder ATR, and shifted RVOL indicators."""
        close = self.df["Close"]
        high = self.df["High"]
        low = self.df["Low"]

        # EMAs for Trend Regime
        # Standard lookbacks: 20, 50, and 200 bars. Initial bars before min_periods remain NaN.
        self.df["_EMA20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
        self.df["_EMA50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
        self.df["_EMA200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()

        # Average True Range (Wilder's ATR 14 - Issue 15)
        self.df["_ATR14"] = calc_wilder_atr(high, low, close, period=14)

        # Relative Strength Index (Wilder's RSI 14 - Issue 14)
        self.df["_RSI14"] = calc_wilder_rsi(close, period=14)

        # Relative Volume (RVOL 20 - strictly historical, excludes current candle from baseline)
        if "Volume" in self.df.columns and self.df["Volume"].sum() > 0:
            vol = self.df["Volume"].astype(float)
            prev_vol = vol.shift(1).copy()
            if len(prev_vol) > 0:
                prev_vol.iloc[0] = vol.iloc[0]
            prev_vol = prev_vol.ffill().fillna(vol)
            avg_vol = prev_vol.rolling(window=min(20, len(self.df)), min_periods=1).mean()
            self.df["_RVOL"] = (vol / avg_vol.replace(0, np.nan)).fillna(1.0)
        else:
            self.df["_RVOL"] = 1.0

    def _determine_trend_regime(
        self, close: float, ema20: float, ema50: float, ema200: float
    ) -> str:
        """Classify prevailing trend regime."""
        has_200 = not np.isnan(ema200)
        has_50 = not np.isnan(ema50)
        has_20 = not np.isnan(ema20)

        if has_200 and has_50 and has_20:
            if close > ema20 > ema50 > ema200:
                return "STRONG_BULLISH"
            elif close > ema50 and ema50 >= ema200:
                return "BULLISH"
            elif close < ema20 < ema50 < ema200:
                return "STRONG_BEARISH"
            elif close < ema50 and ema50 <= ema200:
                return "BEARISH"
            return "NEUTRAL"
        elif has_50 and has_20:
            if close > ema20 > ema50:
                return "STRONG_BULLISH"
            elif close > ema50:
                return "BULLISH"
            elif close < ema20 < ema50:
                return "STRONG_BEARISH"
            elif close < ema50:
                return "BEARISH"
            return "NEUTRAL"
        elif has_20:
            if close > ema20:
                return "BULLISH"
            elif close < ema20:
                return "BEARISH"
            return "NEUTRAL"
        return "NEUTRAL"

    def score_signal(
        self,
        pattern_name: str,
        timestamp: pd.Timestamp,
        raw_signal: int,
    ) -> PatternConfidenceResult:
        """Compute multi-factor AI confidence score and trade setup for a specific signal."""
        if raw_signal == 0:
            raise ValueError(
                "Cannot score an inactive signal: raw_signal must be non-zero (+100/-100 or +1/-1)."
            )

        if timestamp not in self.df.index:
            raise KeyError(f"Timestamp {timestamp} not found in market data.")

        row = self.df.loc[timestamp]
        close = float(cast(Any, row["Close"]))
        high = float(cast(Any, row["High"]))
        low = float(cast(Any, row["Low"]))
        open_val = float(cast(Any, row["Open"]))
        ema20 = float(cast(Any, row["_EMA20"]))
        ema50 = float(cast(Any, row["_EMA50"]))
        ema200 = float(cast(Any, row["_EMA200"]))

        raw_atr = float(cast(Any, row["_ATR14"]))
        min_atr = close * 1e-4
        atr = max(raw_atr, min_atr) if not np.isnan(raw_atr) else (close * 0.01)

        raw_rsi = float(cast(Any, row["_RSI14"]))
        rsi = raw_rsi if not np.isnan(raw_rsi) else 50.0

        rvol = float(cast(Any, row["_RVOL"]))

        trend = self._determine_trend_regime(close, ema20, ema50, ema200)
        confluence_factors: list[str] = []
        risk_factors: list[str] = []

        # Baseline prior probability
        confidence = 0.50
        is_bullish = raw_signal > 0

        # Factor 1: Trend Alignment
        if is_bullish:
            if trend in ("STRONG_BULLISH", "BULLISH"):
                confidence += 0.15
                confluence_factors.append(
                    f"Trend Alignment: Bullish pattern aligned with {trend} market"
                )
            elif trend in ("STRONG_BEARISH", "BEARISH"):
                confidence -= 0.15
                risk_factors.append(
                    f"Counter-Trend Warning: Bullish signal against prevailing {trend} trend"
                )
            else:
                confluence_factors.append("Market Regime: Neutral consolidation")
        else:
            if trend in ("STRONG_BEARISH", "BEARISH"):
                confidence += 0.15
                confluence_factors.append(
                    f"Trend Alignment: Bearish pattern aligned with {trend} market"
                )
            elif trend in ("STRONG_BULLISH", "BULLISH"):
                confidence -= 0.15
                risk_factors.append(
                    f"Counter-Trend Warning: Bearish signal against prevailing {trend} trend"
                )
            else:
                confluence_factors.append("Market Regime: Neutral consolidation")

        # Factor 2: Volume Confirmation (RVOL)
        if rvol >= 1.5:
            confidence += 0.15
            confluence_factors.append(
                f"Volume Surge: Relative volume {rvol:.2f}x signals strong institutional interest"
            )
        elif rvol >= 1.1:
            confidence += 0.05
            confluence_factors.append(f"Volume Confirmation: Above average volume ({rvol:.2f}x)")
        elif rvol < 0.75:
            confidence -= 0.10
            risk_factors.append(
                f"Low Volume Warning: Subdued volume ({rvol:.2f}x) suggests lack of conviction"
            )

        # Factor 3: Momentum & Exhaustion (RSI 14)
        if not np.isnan(raw_rsi):
            if is_bullish:
                if 30 <= rsi <= 45:
                    confidence += 0.10
                    confluence_factors.append(
                        f"Momentum Reset: RSI at {rsi:.1f} indicates prime dip-buying territory"
                    )
                elif rsi < 30:
                    confidence += 0.08
                    confluence_factors.append(
                        f"Oversold Rebound: RSI at {rsi:.1f} supports potential reversal"
                    )
                elif rsi > 75:
                    confidence -= 0.15
                    risk_factors.append(
                        f"Overbought Warning: RSI at {rsi:.1f} signals exhaustion risk"
                    )
            else:
                if 55 <= rsi <= 70:
                    confidence += 0.10
                    confluence_factors.append(
                        f"Bearish Continuation: RSI at {rsi:.1f} shows sustained selling momentum"
                    )
                elif rsi > 70:
                    confidence += 0.08
                    confluence_factors.append(
                        f"Overbought Rejection: RSI at {rsi:.1f} supports bearish reversal"
                    )
                elif rsi < 25:
                    confidence -= 0.15
                    risk_factors.append(
                        f"Oversold Risk: RSI at {rsi:.1f} warns of potential snapback"
                    )

        # Factor 4: Candle Geometry & Range Significance
        candle_range = high - low
        body_size = abs(close - open_val)
        if candle_range > 1.2 * atr:
            confidence += 0.05
            confluence_factors.append(
                f"High Volatility Expansion: Candle range ({candle_range:.4f}) exceeds 1.2x ATR"
            )
        elif candle_range < 0.4 * atr:
            confidence -= 0.05
            risk_factors.append("Low Range Anomaly: Candle range compressed below 0.4x ATR")

        if candle_range > 0 and (body_size / candle_range) >= 0.60:
            confidence += 0.03
            confluence_factors.append(
                f"Decisive Candle Body: Body covers {(body_size / candle_range) * 100:.1f}% of range"
            )

        # Clamp confidence to [0.05, 0.98]
        final_confidence = max(0.05, min(0.98, confidence))

        # Categorize Signal Grade
        if final_confidence >= 0.80:
            grade = SignalGrade.EXCELLENT
        elif final_confidence >= 0.70:
            grade = SignalGrade.STRONG
        elif final_confidence >= 0.55:
            grade = SignalGrade.MODERATE
        elif final_confidence >= 0.40:
            grade = SignalGrade.WEAK
        else:
            grade = SignalGrade.FALSE_SIGNAL

        # Calculate Actionable Trade Setup
        trade_setup = self._build_trade_setup(is_bullish, close, high, low, atr)

        return PatternConfidenceResult(
            pattern_name=pattern_name.replace("CDL", ""),
            timestamp=timestamp,
            raw_signal=raw_signal,
            confidence_score=final_confidence,
            grade=grade,
            trend_regime=trend,
            rvol=rvol,
            rsi=rsi,
            atr=atr,
            confluence_factors=confluence_factors,
            risk_factors=risk_factors,
            trade_setup=trade_setup,
        )

    def _build_trade_setup(
        self,
        is_bullish: bool,
        close: float,
        high: float,
        low: float,
        atr: float,
    ) -> TradeSetup:
        """Construct ATR-governed entry, stop loss, and tiered profit targets."""
        buffer = 0.2 * atr
        rr_tp1 = 1.5
        rr_tp2 = 3.0
        rrr = 1.5  # Primary target risk-reward ratio matching TP1

        if is_bullish:
            direction = "BUY"
            entry = close
            # Stop below the low minus ATR buffer, strictly positive
            stop_loss = max(low - buffer, close * 0.001)
            risk = max(entry - stop_loss, close * 1e-5)
            tp1 = entry + (rr_tp1 * risk)
            tp2 = entry + (rr_tp2 * risk)
        else:
            direction = "SELL"
            entry = close
            # Stop above the high plus ATR buffer
            stop_loss = high + buffer
            risk = max(stop_loss - entry, close * 1e-5)
            # Ensure profit targets remain strictly positive
            tp1 = max(entry - (rr_tp1 * risk), close * 0.001)
            tp2 = max(entry - (rr_tp2 * risk), close * 0.0005)

        return TradeSetup(
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=rrr,
            risk_per_unit=risk,
            rr_tp1=rr_tp1,
            rr_tp2=rr_tp2,
        )

    def score_all_signals(
        self,
        signals: pd.Series,
        pattern_name: str,
        min_confidence: float = 0.0,
    ) -> list[PatternConfidenceResult]:
        """Evaluate a series of TA-Lib signals and return sorted scored results."""
        results: list[PatternConfidenceResult] = []
        for ts, raw_sig in signals.items():
            if raw_sig == 0:
                continue
            scored = self.score_signal(pattern_name, pd.to_datetime(cast(Any, ts)), int(raw_sig))
            if scored.confidence_score >= min_confidence:
                results.append(scored)

        results.sort(key=lambda r: r.confidence_score, reverse=True)
        return results

    def score_all_active(
        self,
        min_confidence: float = 0.5,
        patterns: list[str] | None = None,
        date: str | None = None,
    ) -> list[PatternConfidenceResult]:
        """Scan data for active patterns and return scored results above min_confidence.

        Args:
            min_confidence: Threshold between 0.0 and 1.0 to filter low-conviction signals.
            patterns: Optional list of pattern names (e.g. ['CDLHAMMER', 'CDLENGULFING']).
                      If None, scans all available patterns.
            date: Optional date filter string.

        Returns:
            List of PatternConfidenceResult sorted by confidence score descending.
        """
        from ..pattern_analyzer import PatternAnalyzer

        analyzer = PatternAnalyzer(self.df)
        patterns_to_scan = patterns if patterns is not None else sorted(analyzer.pattern_functions)

        all_scored: list[PatternConfidenceResult] = []
        for pat in patterns_to_scan:
            try:
                signals = analyzer.get_signals(pat, date=date)
            except NotImplementedError:
                continue
            if signals.empty:
                continue
            clean_name = pat.replace("CDL", "")
            scored = self.score_all_signals(signals, clean_name, min_confidence=min_confidence)
            all_scored.extend(scored)

        all_scored.sort(key=lambda r: r.confidence_score, reverse=True)
        return all_scored
