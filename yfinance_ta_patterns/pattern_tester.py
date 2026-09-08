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
    periodic_sharpe: float = 0.0
    trade_sharpe: float = 0.0
    avg_strength: float = 100.0


class PatternRankingTester:
    """Test all candlestick patterns and rank them by performance.

    Can filter by economic news events.
    """

    __slots__ = (
        "_account_currency",
        "_allow_short",
        "_commission",
        "_data",
        "_execution",
        "_force_exit_on_last_bar",
        "_fx_rates",
        "_initial_capital",
        "_min_signals",
        "_news_dates",
        "_periods_per_year",
        "_position_size",
        "_results",
        "_sharpe_mode",
        "_slippage",
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
        commission: float = 0.0,
        slippage: float = 0.0,
        min_signals: int = 1,
        fx_rates: dict[str, float] | None = None,
        sharpe_mode: str = "periodic",
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
        commission : float
            Fixed commission cost deducted per completed trade (default 0.0)
        slippage : float
            Slippage in price units applied adversely to entries and exits (default 0.0)
        min_signals : int
            Minimum signals required for ranking (default 1)
        fx_rates : dict[str, float], optional
            Dictionary of cross-currency exchange rates for Forex conversion
        sharpe_mode : str
            Sharpe mode: 'periodic' (returns per bar, with 0% on idle bars) or 'trades'
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
        self._commission: float = max(commission, 0.0)
        self._slippage: float = max(slippage, 0.0)
        self._min_signals: int = max(min_signals, 1)
        self._fx_rates: dict[str, float] = fx_rates or {}
        self._sharpe_mode: str = sharpe_mode

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
        """Universal FX conversion: normalize Forex / CFD PnL into account base currency."""
        if not self._symbol or exit_price <= 0:
            return raw_pnl
        sym = self._symbol.upper().replace("=X", "").replace("-USD", "").replace("/", "")
        # Common 6-letter FX pair e.g. USDJPY, EURUSD, EURJPY
        if len(sym) == 6 and sym.isalpha():
            base, quote = sym[:3], sym[3:6]
            # 1. Quote matches account currency (e.g. EURUSD with USD account) -> already in USD
            if quote == self._account_currency:
                return raw_pnl
            # 2. Base matches account currency (e.g. USDJPY with USD account) -> raw_pnl is in quote, divide by exit_price
            if base == self._account_currency:
                return raw_pnl / exit_price
            # 3. Cross pair (e.g. EURJPY, GBPJPY with USD account) -> raw_pnl in quote (JPY)
            if self._fx_rates:
                direct_pair = f"{self._account_currency}{quote}"
                inv_pair = f"{quote}{self._account_currency}"
                if direct_pair in self._fx_rates and self._fx_rates[direct_pair] > 0:
                    return raw_pnl / self._fx_rates[direct_pair]
                elif inv_pair in self._fx_rates and self._fx_rates[inv_pair] > 0:
                    return raw_pnl * self._fx_rates[inv_pair]
            # Fallback for USD account when quote is JPY (e.g. EURJPY): raw_pnl in JPY
            if self._account_currency == "USD" and quote == "JPY" and exit_price > 50:
                # If pair is EURJPY, exit_price ~ 160. EURUSD ~ 1.08 => USDJPY ~ 160/1.08 ~ 148
                return raw_pnl / (exit_price / 1.08 if base == "EUR" else exit_price)
        return raw_pnl


    def test_all_patterns(
        self, filter_news: bool = False, min_signals: int | None = None
    ) -> list[PatternResult]:
        """Test all patterns and return ranked results.

        Parameters:
        -----------
        filter_news : bool
            If True, exclude trades during news events
        min_signals : int, optional
            Minimum total signals required to include in ranked results (default from __init__)

        Returns:
        --------
        list[PatternResult] : Ranked pattern results
        """
        self._results = []
        all_patterns = self.get_all_patterns()
        thresh = min_signals if min_signals is not None else self._min_signals

        for pattern_name in all_patterns:
            try:
                result = self._test_single_pattern(pattern_name, filter_news)
                if result and result.total_signals >= thresh:
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

        # Build trade equity curve and calculate maximum drawdown
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

        # Trade-level Sharpe ratio
        pnls = [t["pnl"] for t in trades]
        std_pnl = float(np.std(pnls))
        trade_sharpe = (
            (float(np.mean(pnls)) / std_pnl) * float(np.sqrt(self._periods_per_year))
            if len(pnls) > 1 and std_pnl > 0
            else 0.0
        )

        # Bar-level periodic returns & Periodic Sharpe (accounting for 0% return during idle holding periods)
        n_bars = len(self._data)
        bar_pnl = np.zeros(n_bars, dtype=float)
        for t in trades:
            idx = t.get("exit_idx")
            if idx is not None and 0 <= idx < n_bars:
                bar_pnl[idx] += t["pnl"]

        bar_equity = np.full(n_bars, self._initial_capital, dtype=float)
        curr_eq = self._initial_capital
        for b in range(n_bars):
            curr_eq += bar_pnl[b]
            bar_equity[b] = curr_eq

        bar_returns = np.zeros(max(n_bars - 1, 1), dtype=float)
        if n_bars > 1:
            prev_eq = bar_equity[:-1]
            non_zero = prev_eq > 0
            bar_returns[non_zero] = (bar_equity[1:][non_zero] - prev_eq[non_zero]) / prev_eq[non_zero]

        std_bar_ret = float(np.std(bar_returns))
        periodic_sharpe = (
            (float(np.mean(bar_returns)) / std_bar_ret) * float(np.sqrt(self._periods_per_year))
            if std_bar_ret > 0
            else 0.0
        )

        primary_sharpe = periodic_sharpe if self._sharpe_mode == "periodic" else trade_sharpe

        # Pattern strength tracking
        active_strengths = [s for s in strengths if s > 0]
        avg_strength = float(np.mean(active_strengths)) if active_strengths else 100.0

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
            sharpe_ratio=primary_sharpe,
            max_drawdown=max_drawdown,
            equity_curve=curve,
            periodic_sharpe=periodic_sharpe,
            trade_sharpe=trade_sharpe,
            avg_strength=avg_strength,
        )

    def _calculate_trades(self, signals: np.ndarray) -> list[dict[str, Any]]:
        """Calculate trades from signals with short support, slippage, and commissions."""
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
                    eff_exit = exec_price + self._slippage
                    raw_pnl = (-position) * (entry_price - eff_exit) - self._commission
                    pnl = self._convert_pnl_to_account_currency(raw_pnl, exec_price)
                    trades.append(
                        {
                            "entry_time": times[entry_idx],
                            "exit_time": times[exec_idx],
                            "exit_idx": exec_idx,
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
                    entry_price = exec_price + self._slippage
                    position = self._position_size / exec_price

            elif signal == -1:
                # Close Long if currently long
                if position > 0.0 and entry_idx is not None:
                    eff_exit = exec_price - self._slippage
                    raw_pnl = position * (eff_exit - entry_price) - self._commission
                    pnl = self._convert_pnl_to_account_currency(raw_pnl, exec_price)
                    trades.append(
                        {
                            "entry_time": times[entry_idx],
                            "exit_time": times[exec_idx],
                            "exit_idx": exec_idx,
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
                    entry_price = exec_price - self._slippage
                    position = -self._position_size / exec_price

        # Force-close position on the last bar if still open
        if self._force_exit_on_last_bar and position != 0.0 and entry_idx is not None:
            last_exit_price = closes[-1]
            if position > 0.0:
                eff_exit = last_exit_price - self._slippage
                raw_pnl = position * (eff_exit - entry_price) - self._commission
                direction = "LONG"
            else:
                eff_exit = last_exit_price + self._slippage
                raw_pnl = (-position) * (entry_price - eff_exit) - self._commission
                direction = "SHORT"
            pnl = self._convert_pnl_to_account_currency(raw_pnl, last_exit_price)
            trades.append(
                {
                    "entry_time": times[entry_idx],
                    "exit_time": times[-1],
                    "exit_idx": n - 1,
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

