"""AI Market Analyst: Synthesizes technical patterns into structured intelligence briefs and LLM prompts."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .scorer import PatternConfidenceResult, SignalGrade


class AIMarketAnalyst:
    """Institutional-grade market intelligence synthesizer.

    Aggregates scored candlestick patterns and macro-technical context to generate:
    1. Human-readable executive market briefs.
    2. Structured JSON payloads for automated trading systems.
    3. Context-rich prompt templates optimized for LLMs (GPT-4o, Claude 3.5, Gemini).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        scored_results: list[PatternConfidenceResult] | None = None,
        *,
        symbol: str = "ASSET",
        timeframe: str = "1d",
    ) -> None:
        """Initialize analyst with market data and optional evaluated patterns.

        Args:
            data: OHLCV market DataFrame.
            scored_results: Optional pre-computed list of PatternConfidenceResult.
            symbol: Ticker symbol (e.g. 'BTC-USD', 'EURUSD'). Default is 'ASSET'.
            timeframe: Timeframe or interval (e.g. '4h', '1d'). Default is '1d'.
        """
        self.data = data
        self.symbol = symbol
        self.timeframe = timeframe
        self.scored_results: list[PatternConfidenceResult] = (
            sorted(scored_results, key=lambda r: r.confidence_score, reverse=True)
            if scored_results is not None
            else []
        )

    def analyze(
        self,
        min_confidence: float = 0.5,
        patterns: list[str] | None = None,
        date: str | None = None,
    ) -> list[PatternConfidenceResult]:
        """Automatically scan and score candlestick patterns across the dataset.

        Args:
            min_confidence: Threshold between 0.0 and 1.0 to filter low-conviction signals.
            patterns: Optional list of specific pattern names to scan. If None, scans all patterns.
            date: Optional single-date filter string.

        Returns:
            List of PatternConfidenceResult sorted by confidence score descending.
        """
        from ..pattern_analyzer import PatternAnalyzer
        from .scorer import AIPatternScorer

        analyzer = PatternAnalyzer(self.data)
        scorer = AIPatternScorer(self.data)

        patterns_to_scan = (
            patterns if patterns is not None else sorted(analyzer.pattern_functions)
        )

        all_scored: list[PatternConfidenceResult] = []
        for pat in patterns_to_scan:
            try:
                signals = analyzer.get_signals(pat, date=date)
            except NotImplementedError:
                continue
            if signals.empty:
                continue
            clean_name = pat.replace("CDL", "")
            scored = scorer.score_all_signals(
                signals, clean_name, min_confidence=min_confidence
            )
            all_scored.extend(scored)

        self.scored_results = sorted(all_scored, key=lambda r: r.confidence_score, reverse=True)
        return self.scored_results

    def get_market_regime_summary(self) -> dict[str, Any]:
        """Extract current volatility, volume, and price trends from data."""
        if self.data.empty:
            return {"status": "NO_DATA"}

        last_row = self.data.iloc[-1]
        close = float(last_row["Close"])
        high = float(last_row["High"])
        low = float(last_row["Low"])
        vol = float(last_row["Volume"]) if "Volume" in last_row else 0.0

        ret_20 = 0.0
        if len(self.data) >= 21:
            first_20 = float(self.data.iloc[-21]["Close"])
            ret_20 = ((close - first_20) / first_20) * 100.0
        elif len(self.data) > 1:
            first_20 = float(self.data.iloc[0]["Close"])
            ret_20 = ((close - first_20) / first_20) * 100.0

        return {
            "current_price": close,
            "last_high": high,
            "last_low": low,
            "last_volume": vol,
            "return_20_bars_pct": round(ret_20, 2),
            "total_signals_detected": len(self.scored_results),
            "high_conviction_signals": sum(
                1
                for r in self.scored_results
                if r.grade in (SignalGrade.EXCELLENT, SignalGrade.STRONG)
            ),
        }

    def _resolve_context(
        self,
        symbol_or_results: str | list[PatternConfidenceResult] | None = None,
        timeframe: str | None = None,
    ) -> tuple[str, str, list[PatternConfidenceResult]]:
        """Flexibly resolve (symbol, timeframe, results) across overloaded method calls."""
        if isinstance(symbol_or_results, list):
            results = symbol_or_results
            symbol = self.symbol
            tf = timeframe or self.timeframe
        elif isinstance(symbol_or_results, str):
            results = self.scored_results
            symbol = symbol_or_results
            tf = timeframe or self.timeframe
        else:
            results = self.scored_results
            symbol = self.symbol
            tf = timeframe or self.timeframe
        return symbol, tf, results

    def generate_brief(
        self,
        symbol_or_results: str | list[PatternConfidenceResult] | None = None,
        timeframe: str | None = None,
    ) -> str:
        """Format an executive markdown brief of technical findings."""
        symbol, tf, results = self._resolve_context(symbol_or_results, timeframe)
        regime = self.get_market_regime_summary()
        high_conv = sum(
            1 for r in results if r.grade in (SignalGrade.EXCELLENT, SignalGrade.STRONG)
        )
        lines: list[str] = [
            f"# AI Technical Intelligence Brief: {symbol} ({tf})",
            f"**Current Price:** `{regime['current_price']}` | **20-Bar Return:** `{regime['return_20_bars_pct']:+.2f}%`",
            f"**Signals Analyzed:** `{len(results)}` (High Conviction: `{high_conv}`)",
            "",
            "---",
            "## Key Pattern Setups",
        ]

        if not results:
            lines.append("No active patterns identified matching the selected criteria.")
            return "\n".join(lines)

        for i, res in enumerate(results[:5], 1):
            grade_badge = f"[{res.grade.value}]"
            conf_pct = f"{res.confidence_score * 100:.1f}%"
            lines.append(f"### {i}. {res.pattern_name} - {grade_badge} (Confidence: {conf_pct})")
            lines.append(f"- **Timestamp:** `{res.timestamp}` | **Regime:** `{res.trend_regime}`")
            lines.append(
                f"- **Metrics:** RVOL: `{res.rvol:.2f}x` | RSI(14): `{res.rsi:.1f}` | ATR: `{res.atr:.5f}`"
            )

            if res.confluence_factors:
                lines.append("- **Confluence Drivers:**")
                for c in res.confluence_factors:
                    lines.append(f"  * {c}")

            if res.risk_factors:
                lines.append("- **Risk & Caveats:**")
                for rf in res.risk_factors:
                    lines.append(f"  * {rf}")

            if res.trade_setup:
                ts = res.trade_setup
                lines.append(
                    f"- **Trade Setup:** `{ts.direction}` @ `{ts.entry_price}` | "
                    f"Stop: `{ts.stop_loss}` | TP1: `{ts.take_profit_1}` | TP2: `{ts.take_profit_2}` "
                    f"(R/R: `{ts.risk_reward_ratio}:1`)"
                )
            lines.append("")

        return "\n".join(lines)

    def to_dict(
        self,
        symbol_or_results: str | list[PatternConfidenceResult] | None = None,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        """Convert intelligence report to a complete dictionary."""
        symbol, tf, results = self._resolve_context(symbol_or_results, timeframe)
        return {
            "symbol": symbol,
            "timeframe": tf,
            "market_summary": self.get_market_regime_summary(),
            "patterns": [r.to_dict() for r in results],
        }

    def to_json(
        self,
        symbol_or_results: str | list[PatternConfidenceResult] | None = None,
        timeframe: str | None = None,
        indent: int = 2,
    ) -> str:
        """Serialize intelligence report to JSON string."""
        return json.dumps(
            self.to_dict(symbol_or_results, timeframe),
            indent=indent,
            default=str,
        )

    def to_llm_prompt(
        self,
        symbol_or_results: str | list[PatternConfidenceResult] | None = None,
        timeframe: str | None = None,
    ) -> str:
        """Create a tailored prompt for Large Language Models to provide second-opinion analysis."""
        brief = self.generate_brief(symbol_or_results, timeframe)
        return (
            "You are a Senior Quantitative Portfolio Manager and Technical Analyst. "
            "Review the following algorithmic pattern detection and market intelligence report. "
            "Evaluate whether the identified setups offer positive expected value (+EV), "
            "assess potential macroeconomic or correlation risks, and advise on optimal position sizing.\n\n"
            f"{brief}\n\n"
            "Provide your structured investment committee assessment."
        )
