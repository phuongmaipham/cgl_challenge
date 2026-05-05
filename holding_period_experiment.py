"""Experiment 14: Comprehensive holding period tests.

Tests multiple holding periods (2, 3, 5, 10 days) with two methods
(staggered holds, skip-rebalance) on both the static candidate and
walk-forward strategies.
"""

from __future__ import print_function

import sys
import numpy as np
import pandas as pd

from grain_futures_strategy import (
    load_train_set,
    build_feature_panels,
    build_improved_model_signals,
    build_walk_forward_model_signals,
    model_predictions_to_positions,
    edge_filtered_positions,
    backtest_positions,
    split_performance,
)

SPLIT_DATE = "2018-01-01"
COST_PER_LOT = 0.0
HOLDING_PERIODS = [2, 3, 5, 10]


def staggered_hold_positions(positions, n_days):
    """Build N offset sleeves, each refreshed every N days, then average."""
    sleeves = []
    idx = positions.index
    for offset in range(n_days):
        sleeve = positions.copy() * 0.0
        for i in range(offset, len(idx), n_days):
            end = min(i + n_days, len(idx))
            sleeve.iloc[i:end] = positions.iloc[i]
        sleeves.append(sleeve)
    return sum(sleeves) / float(n_days)


def skip_rebalance_positions(positions, n_days):
    """Only update positions every N days; hold between updates."""
    idx = positions.index
    out = positions.copy() * 0.0
    for i in range(0, len(idx), n_days):
        end = min(i + n_days, len(idx))
        out.iloc[i:end] = positions.iloc[i]
    return out


def run_all_experiments():
    print("Loading data...")
    data = load_train_set("train_set")
    feature_panels, futures_pnl = build_feature_panels(data)

    print("Building static candidate signals...")
    predictions, _, _, _ = build_improved_model_signals(feature_panels, futures_pnl, SPLIT_DATE)
    static_positions, _, _ = edge_filtered_positions(predictions, futures_pnl, quantile=0.50)
    static_unfiltered = model_predictions_to_positions(predictions, futures_pnl)

    print("Building walk-forward signals...")
    wf_predictions, _ = build_walk_forward_model_signals(feature_panels, futures_pnl)
    wf_positions = model_predictions_to_positions(wf_predictions, futures_pnl)

    results = []

    # Baselines
    for name, pos in [
        ("Static edge-filtered (daily)", static_positions),
        ("Static unfiltered (daily)", static_unfiltered),
        ("Walk-forward (daily)", wf_positions),
    ]:
        bt, _ = backtest_positions(pos, futures_pnl, COST_PER_LOT)
        metrics = split_performance(bt, SPLIT_DATE)
        results.append({
            "strategy": name,
            "method": "daily",
            "hold_days": 1,
            "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
            "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
            "full_sharpe": metrics.loc["sharpe", "full_period"],
            "full_pnl": metrics.loc["total_pnl", "full_period"],
            "max_dd": metrics.loc["max_drawdown", "full_period"],
            "turnover": metrics.loc["avg_daily_turnover", "full_period"],
        })
        print(f"  {name}: OOS Sharpe={results[-1]['oos_sharpe']:.3f}, OOS PnL=${results[-1]['oos_pnl']:,.0f}")

    # Test holding periods on each strategy
    strategy_configs = [
        ("Static edge-filtered", static_positions),
        ("Static unfiltered", static_unfiltered),
        ("Walk-forward", wf_positions),
    ]

    for strat_name, base_pos in strategy_configs:
        for n_days in HOLDING_PERIODS:
            for method_name, method_fn in [("staggered", staggered_hold_positions), ("skip-rebalance", skip_rebalance_positions)]:
                try:
                    held_pos = method_fn(base_pos, n_days)
                    bt, _ = backtest_positions(held_pos, futures_pnl, COST_PER_LOT)
                    metrics = split_performance(bt, SPLIT_DATE)
                    row = {
                        "strategy": strat_name,
                        "method": method_name,
                        "hold_days": n_days,
                        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
                        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
                        "full_sharpe": metrics.loc["sharpe", "full_period"],
                        "full_pnl": metrics.loc["total_pnl", "full_period"],
                        "max_dd": metrics.loc["max_drawdown", "full_period"],
                        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
                        "status": "ok",
                    }
                    results.append(row)
                    print(f"  {strat_name} | {method_name} {n_days}d: OOS Sharpe={row['oos_sharpe']:.3f}, OOS PnL=${row['oos_pnl']:,.0f}, Turnover={row['turnover']:.3f}")
                except Exception as e:
                    row = {
                        "strategy": strat_name,
                        "method": method_name,
                        "hold_days": n_days,
                        "oos_sharpe": np.nan,
                        "oos_pnl": np.nan,
                        "full_sharpe": np.nan,
                        "full_pnl": np.nan,
                        "max_dd": np.nan,
                        "turnover": np.nan,
                        "status": "FAILED: " + str(e),
                    }
                    results.append(row)
                    print(f"  FAILED: {strat_name} | {method_name} {n_days}d: {e}")

    # Also test walk-forward with edge filter + holding periods
    print("\nTesting walk-forward + edge filter + holding periods...")
    wf_edge_pos, _, _ = edge_filtered_positions(wf_predictions, futures_pnl, quantile=0.50)
    bt, _ = backtest_positions(wf_edge_pos, futures_pnl, COST_PER_LOT)
    metrics = split_performance(bt, SPLIT_DATE)
    results.append({
        "strategy": "Walk-forward edge-filtered",
        "method": "daily",
        "hold_days": 1,
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "full_pnl": metrics.loc["total_pnl", "full_period"],
        "max_dd": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
    })
    print(f"  Walk-forward edge-filtered (daily): OOS Sharpe={results[-1]['oos_sharpe']:.3f}, OOS PnL=${results[-1]['oos_pnl']:,.0f}")

    for n_days in HOLDING_PERIODS:
        for method_name, method_fn in [("staggered", staggered_hold_positions), ("skip-rebalance", skip_rebalance_positions)]:
            try:
                held_pos = method_fn(wf_edge_pos, n_days)
                bt, _ = backtest_positions(held_pos, futures_pnl, COST_PER_LOT)
                metrics = split_performance(bt, SPLIT_DATE)
                row = {
                    "strategy": "Walk-forward edge-filtered",
                    "method": method_name,
                    "hold_days": n_days,
                    "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
                    "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
                    "full_sharpe": metrics.loc["sharpe", "full_period"],
                    "full_pnl": metrics.loc["total_pnl", "full_period"],
                    "max_dd": metrics.loc["max_drawdown", "full_period"],
                    "turnover": metrics.loc["avg_daily_turnover", "full_period"],
                    "status": "ok",
                }
                results.append(row)
                print(f"  WF edge-filtered | {method_name} {n_days}d: OOS Sharpe={row['oos_sharpe']:.3f}, OOS PnL=${row['oos_pnl']:,.0f}, Turnover={row['turnover']:.3f}")
            except Exception as e:
                row = {
                    "strategy": "Walk-forward edge-filtered",
                    "method": method_name,
                    "hold_days": n_days,
                    "oos_sharpe": np.nan,
                    "oos_pnl": np.nan,
                    "full_sharpe": np.nan,
                    "full_pnl": np.nan,
                    "max_dd": np.nan,
                    "turnover": np.nan,
                    "status": "FAILED: " + str(e),
                }
                results.append(row)
                print(f"  FAILED: WF edge-filtered | {method_name} {n_days}d: {e}")

    df = pd.DataFrame(results)
    print("\n\n===== FULL RESULTS TABLE =====")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    df = run_all_experiments()
