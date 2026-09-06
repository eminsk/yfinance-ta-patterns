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

    def __init__(self, data: pd.DataFrame, scored_results: list[PatternConfidenceResult]) -> None:
        """Initialize analyst with market data and evaluated patterns."""
        self.data = data
        self.scored_results = sorted(scored_results, key=lambda r: r.confidence_score, reverse=True)

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
        if len(self.data) >= 20:
            first_20 = float(self.data.iloc[-20]["Close"])
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

    def generate_brief(self, symbol: str, timeframe: str) -> str:
        """Format an executive markdown brief of technical findings."""
        regime = self.get_market_regime_summary()
        lines: list[str] = [
            f"# AI Technical Intelligence Brief: {symbol} ({timeframe})",
            f"**Current Price:** `{regime['current_price']}` | **20-Bar Return:** `{regime['return_20_bars_pct']:+.2f}%`",
            f"**Signals Analyzed:** `{regime['total_signals_detected']}` (High Conviction: `{regime['high_conviction_signals']}`)",
            "",
            "---",
            "## Key Pattern Setups",
        ]

        if not self.scored_results:
            lines.append("No active patterns identified matching the selected criteria.")
            return "\n".join(lines)

        for i, res in enumerate(self.scored_results[:5], 1):
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

    def to_dict(self, symbol: str, timeframe: str) -> dict[str, Any]:
        """Convert intelligence report to a complete dictionary."""
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "market_summary": self.get_market_regime_summary(),
            "patterns": [r.to_dict() for r in self.scored_results],
        }

    def to_json(self, symbol: str, timeframe: str, indent: int = 2) -> str:
        """Serialize intelligence report to JSON string."""
        return json.dumps(self.to_dict(symbol, timeframe), indent=indent, default=str)

    def to_llm_prompt(self, symbol: str, timeframe: str) -> str:
        """Create a tailored prompt for Large Language Models to provide second-opinion analysis."""
        brief = self.generate_brief(symbol, timeframe)
        return (
            "You are a Senior Quantitative Portfolio Manager and Technical Analyst. "
            "Review the following algorithmic pattern detection and market intelligence report. "
            "Evaluate whether the identified setups offer positive expected value (+EV), "
            "assess potential macroeconomic or correlation risks, and advise on optimal position sizing.\n\n"
            f"{brief}\n\n"
            "Provide your structured investment committee assessment."
        )
