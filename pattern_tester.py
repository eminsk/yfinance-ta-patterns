"""Pattern ranking tester - find best performing candlestick patterns."""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from utils.pattern_helper import PatternHelper


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


class PatternRankingTester:
    """
    Test all candlestick patterns and rank them by performance.
    Can filter by economic news events.
    """
    
    __slots__ = ('_data', '_initial_capital', '_position_size', '_results', '_news_dates')
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        position_size: float = 100.0,
        news_dates: Optional[List[str]] = None
    ):
        """
        Initialize pattern tester.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        initial_capital : float
            Initial capital
        position_size : float
            Position size per trade
        news_dates : list, optional
            List of dates with important news (YYYY-MM-DD)
        """
        self._data = data
        self._initial_capital = initial_capital
        self._position_size = position_size
        self._results: List[PatternResult] = []
        self._news_dates = set(news_dates) if news_dates else set()
    
    def test_all_patterns(self, filter_news: bool = False) -> List[PatternResult]:
        """
        Test all patterns and return ranked results.
        
        Parameters:
        -----------
        filter_news : bool
            If True, exclude trades during news events
        
        Returns:
        --------
        list : Ranked pattern results
        """
        self._results = []
        
        # Get all patterns dynamically from TA-Lib
        all_patterns = PatternHelper.get_all_patterns()
        
        for pattern_name in all_patterns:
            try:
                result = self._test_single_pattern(pattern_name, filter_news)
                if result and result.total_signals > 0:
                    self._results.append(result)
            except Exception as e:
                print(f"Error testing {pattern_name}: {e}")
                continue
        
        # Sort by win rate, then by total PnL
        self._results.sort(key=lambda x: (x.win_rate, x.total_pnl), reverse=True)
        
        return self._results
    
    def _test_single_pattern(
        self,
        pattern_name: str,
        filter_news: bool
    ) -> Optional[PatternResult]:
        """Test a single pattern."""
        # Detect pattern
        pattern_func = getattr(talib, pattern_name, None)
        if not pattern_func:
            print(f"Pattern function not found: {pattern_name}")
            return None
        
        try:
            pattern_values = pattern_func(
                self._data['Open'].values,
                self._data['High'].values,
                self._data['Low'].values,
                self._data['Close'].values
            )
        except Exception as e:
            print(f"Error detecting pattern {pattern_name}: {e}")
            return None
        
        # Generate signals
        signals = np.zeros(len(self._data))
        for i in range(len(pattern_values)):
            if pattern_values[i] > 0:  # Bullish pattern
                signals[i] = 1
            elif pattern_values[i] < 0:  # Bearish pattern
                signals[i] = -1
        
        # Filter by news if requested
        if filter_news and self._news_dates:
            for i in range(len(signals)):
                date_str = self._data.index[i].strftime('%Y-%m-%d')
                if date_str in self._news_dates:
                    signals[i] = 0
        
        # Calculate trades
        trades = self._calculate_trades(signals)
        
        # Return None if no trades or too few signals
        if not trades:
            # Check if there were any signals at all
            total_signals = int(np.sum(signals != 0))
            if total_signals == 0:
                print(f"{pattern_name}: No signals generated")
            else:
                print(f"{pattern_name}: {total_signals} signals but no completed trades")
            return None
        
        # Calculate statistics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        max_profit = max([t['pnl'] for t in trades]) if trades else 0
        max_loss = min([t['pnl'] for t in trades]) if trades else 0
        
        # Calculate Sharpe ratio
        pnls = [t['pnl'] for t in trades]
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) > 0 else 0
        
        return PatternResult(
            pattern_name=pattern_name.replace('CDL', ''),
            total_signals=int(np.sum(signals != 0)),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            max_profit=max_profit,
            max_loss=max_loss,
            sharpe_ratio=sharpe
        )
    
    def _calculate_trades(self, signals: np.ndarray) -> List[Dict]:
        """Calculate trades from signals."""
        trades = []
        position = 0
        entry_idx = None
        
        closes = self._data['Close'].values
        times = self._data.index.to_numpy()
        
        for i in range(len(signals)):
            signal = signals[i]
            
            # BUY signal
            if signal == 1 and position == 0:
                entry_idx = i
                position = self._position_size / closes[i]
            
            # SELL signal
            elif signal == -1 and position > 0 and entry_idx is not None:
                exit_price = closes[i]
                entry_price = closes[entry_idx]
                pnl = position * (exit_price - entry_price)
                
                trades.append({
                    'entry_time': times[entry_idx],
                    'exit_time': times[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                
                position = 0
                entry_idx = None
        
        return trades
    
    def get_top_patterns(self, n: int = 10) -> List[PatternResult]:
        """Get top N patterns by performance."""
        return self._results[:n]
    
    def get_comparison_report(self) -> pd.DataFrame:
        """Get comparison report with/without news filter."""
        # Test without news filter
        results_no_filter = self.test_all_patterns(filter_news=False)
        
        # Test with news filter
        results_with_filter = self.test_all_patterns(filter_news=True)
        
        # Create comparison DataFrame
        data = []
        for r_no, r_yes in zip(results_no_filter[:20], results_with_filter[:20]):
            data.append({
                'Pattern': r_no.pattern_name,
                'Win Rate (No News Filter)': f"{r_no.win_rate:.1f}%",
                'Win Rate (With News Filter)': f"{r_yes.win_rate:.1f}%",
                'Total PnL (No Filter)': f"${r_no.total_pnl:.2f}",
                'Total PnL (With Filter)': f"${r_yes.total_pnl:.2f}",
                'Signals (No Filter)': r_no.total_signals,
                'Signals (With Filter)': r_yes.total_signals,
                'Sharpe (No Filter)': f"{r_no.sharpe_ratio:.2f}",
                'Sharpe (With Filter)': f"{r_yes.sharpe_ratio:.2f}"
            })
        
        return pd.DataFrame(data)
    
    def export_results(self, filename: str = 'pattern_ranking.csv') -> None:
        """Export results to CSV."""
        data = []
        for result in self._results:
            data.append({
                'Rank': len(data) + 1,
                'Pattern': result.pattern_name,
                'Total Signals': result.total_signals,
                'Winning Trades': result.winning_trades,
                'Losing Trades': result.losing_trades,
                'Win Rate %': f"{result.win_rate:.2f}",
                'Total PnL': f"{result.total_pnl:.2f}",
                'Avg PnL': f"{result.avg_pnl:.2f}",
                'Max Profit': f"{result.max_profit:.2f}",
                'Max Loss': f"{result.max_loss:.2f}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
