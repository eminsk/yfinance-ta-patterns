"""Pattern ranking tester - find best performing candlestick patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from yfinance_ta_patterns.data import normalize_interval
from yfinance_ta_patterns.talib_compat import talib

# Standard annual periods for timeframe-aware Sharpe Ratio calculation
TIMEFRAME_PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 252.0 * 390.0,   # 98,280 periods/year
    "2m": 252.0 * 195.0,   # 49,140 periods/year
    "5m": 252.0 * 78.0,    # 19,656 periods/year
    "15m": 252.0 * 26.0,   # 6,552 periods/year
    "30m": 252.0 * 13.0,   # 3,276 periods/year
    "60m": 252.0 * 6.5,    # 1,638 periods/year
    "1h": 252.0 * 6.5,     # 1,638 periods/year
    "4h": 252.0 * 2.0,     # 504 periods/year
    "1d": 252.0,           # 252 trading days/year
    "1wk": 52.0,           # 52 weeks/year
    "1mo": 12.0,           # 12 months/year
}


@dataclass
class PatternResult:
    """Result for a single pattern test."""

    pattern_name: str
    total_signals: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_profit: float
    max_loss: float
    sharpe_ratio: float
    max_drawdown: float = 0.0
    equity_curve: list[float] | None = None


class PatternRankingTester:
    """Test all candlestick patterns and rank them by performance.

    Can filter by economic news events.
    """

    __slots__ = (
        "_account_currency",
        "_allow_short",
        "_data",
        "_execution",
        "_force_exit_on_last_bar",
        "_initial_capital",
        "_news_dates",
        "_periods_per_year",
        "_position_size",
        "_results",
        "_symbol",
        "_timeframe",
        "equity_curve",
        "trades",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        position_size: float = 100.0,
        news_dates: list[str] | None = None,
        execution: str = "next_open",
        timeframe: str = "1d",
        periods_per_year: float | None = None,
        symbol: str = "",
        account_currency: str = "USD",
        force_exit_on_last_bar: bool = True,
        allow_short: bool = True,
    ) -> None:
        """Initialize pattern tester.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        initial_capital : float
            Initial capital
        position_size : float
            Position size per trade (in base/quote units)
        news_dates : list[str], optional
            List of dates with important news (YYYY-MM-DD)
        execution : str
            Execution mode: 'next_open' (unbiased, enters on open of bar i+1)
            or 'close' (legacy, enters on close of bar i)
        timeframe : str
            Timeframe notation (e.g. '1d', '1h', '15m', '4h') for Sharpe scaling
        periods_per_year : float, optional
            Custom periods per year override for annualization
        symbol : str
            Ticker symbol for currency/asset conversion
        account_currency : str
            Account base currency (default 'USD')
        force_exit_on_last_bar : bool
            Whether to close active position at final bar close
        allow_short : bool
            Whether to execute short trades on bearish signals
        """
        self._data: pd.DataFrame = data
        self._initial_capital: float = initial_capital
        self._position_size: float = position_size
        self._results: list[PatternResult] = []
        self._news_dates: set[str] = set(news_dates) if news_dates else set()
        self._execution: str = execution
        self._timeframe: str = normalize_interval(timeframe) if timeframe else "1d"
        self._symbol: str = symbol
        self._account_currency: str = account_currency
        self._force_exit_on_last_bar: bool = force_exit_on_last_bar
        self._allow_short: bool = allow_short

        if periods_per_year is not None:
            self._periods_per_year: float = periods_per_year
        else:
            self._periods_per_year = TIMEFRAME_PERIODS_PER_YEAR.get(self._timeframe, 252.0)

        self.equity_curve: list[float] = [initial_capital]
        self.trades: list[dict[str, Any]] = []

    @staticmethod
    def get_all_patterns() -> list[str]:
        """Get all TA-Lib candlestick pattern function names."""
        return [f for f in dir(talib) if f.startswith("CDL")]

    def _convert_pnl_to_account_currency(self, raw_pnl: float, exit_price: float) -> float:
        """Normalize Forex / CFD PnL into account base currency."""
        if not self._symbol or exit_price <= 0:
            return raw_pnl
        sym = self._symbol.upper().replace("=X", "").replace("-USD", "")
        # Common 6-letter FX pair e.g. USDJPY where base=USD, quote=JPY
        if len(sym) == 6 and sym.isalpha():
            base, quote = sym[:3], sym[3:6]
            if self._account_currency == "USD" and quote == "JPY" and base == "USD":
                return raw_pnl / exit_price
        return raw_pnl


    def test_all_patterns(self, filter_news: bool = False) -> list[PatternResult]:
        """Test all patterns and return ranked results.

        Parameters:
        -----------
        filter_news : bool
            If True, exclude trades during news events

        Returns:
        --------
        list[PatternResult] : Ranked pattern results
        """
        self._results = []
        all_patterns = self.get_all_patterns()

        for pattern_name in all_patterns:
            try:
                result = self._test_single_pattern(pattern_name, filter_news)
                if result and result.total_signals > 0:
                    self._results.append(result)
            except NotImplementedError:
                # Silently skip patterns not implemented in pure-Python fallback
                continue
            except Exception as exc:
                print(f"Error testing {pattern_name}: {exc}")
                continue

        # Sort by win rate, then by total PnL
        self._results.sort(key=lambda x: (x.win_rate, x.total_pnl), reverse=True)
        return self._results

    def _test_single_pattern(
        self,
        pattern_name: str,
        filter_news: bool,
    ) -> PatternResult | None:
        """Test a single pattern."""
        pattern_func = getattr(talib, pattern_name, None)
        if not pattern_func:
            print(f"Pattern function not found: {pattern_name}")
            return None

        try:
            pattern_values = pattern_func(
                self._data["Open"].values,
                self._data["High"].values,
                self._data["Low"].values,
                self._data["Close"].values,
            )
        except NotImplementedError:
            # Propagate up so test_all_patterns can skip
            raise
        except Exception as exc:
            print(f"Error detecting pattern {pattern_name}: {exc}")
            return None

        # Generate signals and preserve pattern strength
        signals = np.zeros(len(self._data))
        strengths = np.zeros(len(self._data))
        for i in range(len(pattern_values)):
            val = pattern_values[i]
            if val > 0:  # Bullish pattern
                signals[i] = 1
                strengths[i] = abs(val)
            elif val < 0:  # Bearish pattern
                signals[i] = -1
                strengths[i] = abs(val)

        # Filter by news if requested
        if filter_news and self._news_dates:
            for i in range(len(signals)):
                date_str = self._data.index[i].strftime("%Y-%m-%d")
                if date_str in self._news_dates:
                    signals[i] = 0

        # Calculate trades
        trades = self._calculate_trades(signals)

        # Return None if no trades or too few signals
        if not trades:
            total_signals = int(np.sum(signals != 0))
            if total_signals == 0:
                print(f"{pattern_name}: No signals generated")
            else:
                print(f"{pattern_name}: {total_signals} signals but no completed trades")
            return None

        # Calculate statistics
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0.0
        win_rate = len(winning_trades) / len(trades) * 100.0 if trades else 0.0

        max_profit = max(t["pnl"] for t in trades) if trades else 0.0
        max_loss = min(t["pnl"] for t in trades) if trades else 0.0

        # Build equity curve and calculate maximum drawdown
        equity = self._initial_capital
        curve = [equity]
        peak = equity
        max_drawdown = 0.0
        for t in trades:
            equity += t["pnl"]
            curve.append(equity)
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_drawdown:
                max_drawdown = dd

        self.equity_curve = curve
        self.trades = trades

        # Calculate Sharpe ratio scaled by timeframe frequency
        pnls = [t["pnl"] for t in trades]
        std_pnl = float(np.std(pnls))
        sharpe = (
            (float(np.mean(pnls)) / std_pnl) * float(np.sqrt(self._periods_per_year))
            if len(pnls) > 1 and std_pnl > 0
            else 0.0
        )

        return PatternResult(
            pattern_name=pattern_name.replace("CDL", ""),
            total_signals=int(np.sum(signals != 0)),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            max_profit=max_profit,
            max_loss=max_loss,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            equity_curve=curve,
        )

    def _calculate_trades(self, signals: np.ndarray) -> list[dict[str, Any]]:
        """Calculate trades from signals with next_open execution and symmetric Short trading."""
        trades: list[dict[str, Any]] = []
        position: float = 0.0
        entry_idx: int | None = None
        entry_price: float = 0.0

        closes = self._data["Close"].values
        opens = self._data["Open"].values
        times = self._data.index.to_numpy()
        n = len(signals)

        is_next_open = (self._execution == "next_open")
        loop_limit = (n - 1) if is_next_open else n

        for i in range(loop_limit):
            signal = signals[i]
            if signal == 0:
                continue

            exec_price = opens[i + 1] if is_next_open else closes[i]
            exec_idx = (i + 1) if is_next_open else i

            if signal == 1:
                # Close Short if currently short
                if position < 0.0 and entry_idx is not None:
                    raw_pnl = (-position) * (entry_price - exec_price)
                    pnl = self._convert_pnl_to_account_currency(raw_pnl, exec_price)
                    trades.append(
                        {
                            "entry_time": times[entry_idx],
                            "exit_time": times[exec_idx],
                            "direction": "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exec_price,
                            "pnl": pnl,
                        }
                    )
                    position = 0.0
                    entry_idx = None
                # Open Long if flat
                elif position == 0.0:
                    entry_idx = exec_idx
                    entry_price = exec_price
                    position = self._position_size / exec_price

            elif signal == -1:
                # Close Long if currently long
                if position > 0.0 and entry_idx is not None:
                    raw_pnl = position * (exec_price - entry_price)
                    pnl = self._convert_pnl_to_account_currency(raw_pnl, exec_price)
                    trades.append(
                        {
                            "entry_time": times[entry_idx],
                            "exit_time": times[exec_idx],
                            "direction": "LONG",
                            "entry_price": entry_price,
                            "exit_price": exec_price,
                            "pnl": pnl,
                        }
                    )
                    position = 0.0
                    entry_idx = None
                # Open Short if flat and shorts are allowed
                elif position == 0.0 and self._allow_short:
                    entry_idx = exec_idx
                    entry_price = exec_price
                    position = -self._position_size / exec_price


        # Force-close position on the last bar if still open
        if self._force_exit_on_last_bar and position != 0.0 and entry_idx is not None:
            last_exit_price = closes[-1]
            if position > 0.0:
                raw_pnl = position * (last_exit_price - entry_price)
                direction = "LONG"
            else:
                raw_pnl = (-position) * (entry_price - last_exit_price)
                direction = "SHORT"
            pnl = self._convert_pnl_to_account_currency(raw_pnl, last_exit_price)
            trades.append(
                {
                    "entry_time": times[entry_idx],
                    "exit_time": times[-1],
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": last_exit_price,
                    "pnl": pnl,
                    "forced_exit": True,
                }
            )

        return trades

    def get_top_patterns(self, n: int = 10) -> list[PatternResult]:
        """Get top N patterns by performance."""
        return self._results[:n]

    def get_comparison_report(self) -> pd.DataFrame:
        """Get comparison report with/without news filter, correctly matched by pattern."""
        results_no_filter = self.test_all_patterns(filter_news=False)
        results_with_filter = self.test_all_patterns(filter_news=True)

        map_with_filter = {r.pattern_name: r for r in results_with_filter}

        data: list[dict[str, Any]] = []
        for r_no in results_no_filter[:20]:
            r_yes = map_with_filter.get(r_no.pattern_name)
            data.append(
                {
                    "Pattern": r_no.pattern_name,
                    "Win Rate (No News Filter)": f"{r_no.win_rate:.1f}%",
                    "Win Rate (With News Filter)": f"{r_yes.win_rate:.1f}%" if r_yes else "N/A",
                    "Total PnL (No Filter)": f"${r_no.total_pnl:.2f}",
                    "Total PnL (With Filter)": f"${r_yes.total_pnl:.2f}" if r_yes else "N/A",
                    "Signals (No Filter)": r_no.total_signals,
                    "Signals (With Filter)": r_yes.total_signals if r_yes else 0,
                    "Sharpe (No Filter)": f"{r_no.sharpe_ratio:.2f}",
                    "Sharpe (With Filter)": f"{r_yes.sharpe_ratio:.2f}" if r_yes else "N/A",
                }
            )

        return pd.DataFrame(data)

    def export_results(self, filename: str = "pattern_ranking.csv") -> None:
        """Export results to CSV."""
        data: list[dict[str, Any]] = []
        for result in self._results:
            data.append(
                {
                    "Rank": len(data) + 1,
                    "Pattern": result.pattern_name,
                    "Total Signals": result.total_signals,
                    "Winning Trades": result.winning_trades,
                    "Losing Trades": result.losing_trades,
                    "Win Rate %": f"{result.win_rate:.2f}",
                    "Total PnL": f"{result.total_pnl:.2f}",
                    "Avg PnL": f"{result.avg_pnl:.2f}",
                    "Max Profit": f"{result.max_profit:.2f}",
                    "Max Loss": f"{result.max_loss:.2f}",
                    "Max Drawdown": f"{result.max_drawdown:.2f}",
                    "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

