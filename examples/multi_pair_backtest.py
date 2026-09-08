"""
Multi-Pair Historical Candlestick Backtesting Example.
Пример глобального бэктеста по набору валютных пар за 2 года.

Features demonstrated:
- MarketDataLoader with interval and repair=False
- PatternRankingTester with unbiased next-open execution
- holding_period=5 (realistic time-exit for single-direction candlestick patterns)
- Universal Forex cross-currency conversion and base-USD position sizing
- Export to ranked CSV (all_pairs_ranked.csv)

Usage:
    python examples/multi_pair_backtest.py
    # or
    uv run examples/multi_pair_backtest.py
"""

import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from yfinance_ta_patterns import MarketDataLoader, PatternRankingTester

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "EURGBP=X", "EURJPY=X"]
TIMEFRAME = "1h"
PERIOD = "2y"

global_results = []

print(f"Запуск глобального бэктеста по {len(PAIRS)} парам за {PERIOD}...")

for sym in PAIRS:
    try:
        # repair=False отключает требование sklearn в yfinance
        loader = MarketDataLoader(sym, interval=TIMEFRAME, period=PERIOD, repair=False)
        data = loader.get_data()

        if data.empty or len(data) < 50:
            print(f"[{sym}] Недостаточно данных, пропуск.")
            continue

        tester = PatternRankingTester(
            data=data,
            symbol=sym,
            account_currency="USD",
            initial_capital=10000.0,
            position_size=1000.0,
            execution="next_open",  # Честный вход на следующей свече
            commission=0.05,  # Реалистичная ECN-комиссия $0.05 на микро-лот (0.01 лота / $1000)
            slippage=0.0001,  # 1 пипс спреда
            min_signals=5,  # Мин. 5 сделок
            holding_period=5,  # Фиксированный горизонт удержания 5 свечей
        )

        # Тестируем все паттерны
        results = tester.test_all_patterns(sort_by="composite")

        for r in results:
            dd_pct = (r.max_drawdown / 10000.0) * 100.0 if r.max_drawdown > 0 else 0.0

            global_results.append(
                {
                    "Pair": sym.replace("=X", ""),
                    "Pattern": r.pattern_name,
                    "Signals": r.total_signals,
                    "WinRate": f"{r.win_rate:.1f}%",
                    "TotalPnL": f"${r.total_pnl:+.2f}",
                    "ProfitFactor": round(r.profit_factor, 2)
                    if r.profit_factor != float("inf")
                    else 99.0,
                    "Sharpe": round(r.sharpe_ratio, 2),
                    "MaxDD": f"${r.max_drawdown:.2f}",
                    "MaxDD_%": f"{dd_pct:.2f}%",
                    "Score": round(r.score, 2),
                }
            )

    except Exception as e:
        print(f"Ошибка при тестировании {sym}: {e}")

if global_results:
    df_all = pd.DataFrame(global_results).sort_values(by="Score", ascending=False)

    print("\n" + "=" * 98)
    print(" ТОП-10 ЛУЧШИХ СВЯЗОК (ПАРА + ПАТТЕРН) ЗА 2 ГОДА:")
    print("=" * 98)
    print(
        df_all[
            [
                "Pair",
                "Pattern",
                "Signals",
                "WinRate",
                "TotalPnL",
                "ProfitFactor",
                "Sharpe",
                "MaxDD",
                "MaxDD_%",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    df_all.to_csv("all_pairs_ranked.csv", index=False)
    print("\nСводный отчет сохранен в: all_pairs_ranked.csv")
else:
    print("Нет данных для отчета.")
