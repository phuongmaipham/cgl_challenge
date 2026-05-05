"""Corn abundant-supply risk-control experiment.

The current best corn strategy is regime_ic_vol from
regime_ic_sleeve_experiment.py. It performs well in event/shock regimes but
loses in the long low-price abundant-supply period.

This script rebuilds the same base strategy and tests fixed, observable
abundant-supply controls. No fitted coefficients, Ridge, OLS, Kalman, or
date-specific switches are used.
"""

from __future__ import print_function

import os

import numpy as np
import pandas as pd

from grain_futures_strategy import (
    backtest_positions,
    backtest_positions_with_costs,
    build_feature_panels,
    load_train_set,
    performance_metrics,
    period_performance,
    split_performance,
)
from ic_threshold_sleeve_experiment import (
    _clean_signal,
    _fetch_external_signals,
    _given_signal_universe,
    _positions_from_signal,
)
import regime_ic_sleeve_experiment as regime_ic


COMMODITY = "CORN"
TEST_START = "2018-01-01"


def _rebuild_regime_ic_vol(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    given = _given_signal_universe(feature_panels, COMMODITY)
    external, errors, _ = _fetch_external_signals(COMMODITY, futures_pnl)
    signals = dict(given)
    signals.update(external)
    signals = {name: _clean_signal(signal, futures_pnl.index) for name, signal in signals.items()}
    regimes = regime_ic._regime_masks(feature_panels, futures_pnl, COMMODITY)
    signal, selected_table, signal_ics, candidate_tables = regime_ic._combined_regime_signal(
        COMMODITY,
        signals,
        futures_pnl,
        regimes["vol"],
    )
    positions = _positions_from_signal(signal, futures_pnl, COMMODITY, mode="long_short")
    return {
        "data": data,
        "feature_panels": feature_panels,
        "futures_pnl": futures_pnl,
        "signals": signals,
        "positions": positions,
        "selected_table": selected_table,
        "signal_ics": signal_ics,
        "candidate_tables": candidate_tables,
        "errors": errors,
    }


def _abundant_supply_masks(data, feature_panels, futures_pnl):
    index = futures_pnl.index
    price = data["adj1"][COMMODITY].reindex(index).ffill()
    below_ma = price < price.rolling(252, min_periods=120).mean().shift(1)
    mom60_negative = feature_panels[COMMODITY]["mom_60"].reindex(index).fillna(0.0) < 0.0
    pnl = futures_pnl[COMMODITY].fillna(0.0)
    vol = pnl.rolling(60, min_periods=20).std().shift(1)
    lt_vol = vol.expanding(min_periods=252).median().shift(1)
    low_or_normal_vol = (vol <= 1.05 * lt_vol).fillna(False)
    low_vol = (vol < 0.80 * lt_vol).fillna(False)
    curve_weak = feature_panels[COMMODITY]["curve_spread"].reindex(index).fillna(0.0) <= 0.0

    return {
        "below_ma_and_negative_mom": (below_ma & mom60_negative).fillna(False),
        "below_ma_or_negative_mom": (below_ma | mom60_negative).fillna(False),
        "abundant_low_or_normal": (below_ma & mom60_negative & low_or_normal_vol).fillna(False),
        "abundant_low_vol": (below_ma & mom60_negative & low_vol).fillna(False),
        "abundant_curve_confirmed": (below_ma & mom60_negative & curve_weak).fillna(False),
    }


def _scale_when(positions, condition, scale):
    mask = pd.Series(condition, index=positions.index).fillna(False).astype(bool)
    out = positions.copy()
    out.loc[mask, COMMODITY] = float(scale) * out.loc[mask, COMMODITY]
    return out.fillna(0.0)


def _metrics(bt):
    table = split_performance(bt, TEST_START)
    low_price = performance_metrics(bt.loc[(bt.index >= "2014-06-01") & (bt.index <= "2017-12-31")])
    covid = performance_metrics(bt.loc[(bt.index >= "2020-02-24") & (bt.index <= "2020-06-30")])
    return {
        "oos_sharpe": table.loc["sharpe", "out_of_sample"],
        "oos_pnl": table.loc["total_pnl", "out_of_sample"],
        "oos_dd": table.loc["max_drawdown", "out_of_sample"],
        "full_sharpe": table.loc["sharpe", "full_period"],
        "full_pnl": table.loc["total_pnl", "full_period"],
        "full_dd": table.loc["max_drawdown", "full_period"],
        "turnover": table.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": table.loc["avg_gross_exposure", "full_period"],
        "low_price_sharpe": low_price.get("sharpe", np.nan),
        "low_price_pnl": low_price.get("total_pnl", np.nan),
        "low_price_dd": low_price.get("max_drawdown", np.nan),
        "covid_sharpe": covid.get("sharpe", np.nan),
        "covid_pnl": covid.get("total_pnl", np.nan),
        "covid_dd": covid.get("max_drawdown", np.nan),
    }


def _evaluate(name, positions, futures_pnl):
    rows = []
    backtests = {}
    for cost_adjusted in [False, True]:
        if cost_adjusted:
            bt, _ = backtest_positions_with_costs(positions, futures_pnl[[COMMODITY]], 8.75, 0.05)
        else:
            bt, _ = backtest_positions(positions, futures_pnl[[COMMODITY]], 0.0)
        row = {"strategy": name, "cost_adjusted": bool(cost_adjusted)}
        row.update(_metrics(bt))
        rows.append(row)
        backtests[name + ("_cost" if cost_adjusted else "_zero")] = bt
    return rows, backtests


def _write_log(results, period_tables, selected_table, path="notes/corn_abundant_supply_improvement.txt"):
    lines = []
    lines.append("Corn abundant-supply risk-control experiment")
    lines.append("Date: 2026-05-02")
    lines.append("")
    lines.append("Base strategy")
    lines.append("-------------")
    lines.append("regime_ic_vol rebuilt from regime_ic_sleeve_experiment.py.")
    lines.append("The base strategy selects IC-passing families inside observable volatility buckets.")
    lines.append("")
    lines.append("Selected base regime table")
    lines.append("--------------------------")
    lines.append(selected_table.to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    lines.append("")
    lines.append("Risk controls")
    lines.append("-------------")
    lines.append("- below_ma_and_negative_mom: corn price below 252-day MA and mom_60 < 0.")
    lines.append("- abundant_low_or_normal: same, but only when 60d vol <= 1.05 * long-run median vol.")
    lines.append("- abundant_curve_confirmed: same, but curve_spread <= 0 confirms no nearby tightness.")
    lines.append("- Action tested: half exposure or flat while condition is true.")
    lines.append("")
    lines.append("Results")
    lines.append("-------")
    cols = [
        "strategy",
        "cost_adjusted",
        "oos_sharpe",
        "oos_pnl",
        "oos_dd",
        "full_sharpe",
        "full_dd",
        "low_price_sharpe",
        "low_price_pnl",
        "low_price_dd",
        "covid_sharpe",
        "covid_pnl",
        "covid_dd",
        "turnover",
    ]
    lines.append(results[cols].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    lines.append("")
    lines.append("Key period performance")
    lines.append("----------------------")
    for name, table in period_tables.items():
        lines.append("")
        lines.append(name)
        lines.append(table.to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def run_corn_abundant_supply_improvement(data_dir="train_set"):
    rebuilt = _rebuild_regime_ic_vol(data_dir=data_dir)
    data = rebuilt["data"]
    feature_panels = rebuilt["feature_panels"]
    futures_pnl = rebuilt["futures_pnl"]
    base = rebuilt["positions"]
    masks = _abundant_supply_masks(data, feature_panels, futures_pnl)

    positions = {"base_regime_ic_vol": base}
    for mask_name, mask in masks.items():
        positions[mask_name + "_half"] = _scale_when(base, mask, 0.50)
        positions[mask_name + "_flat"] = _scale_when(base, mask, 0.0)

    rows = []
    backtests = {}
    for name, pos in positions.items():
        new_rows, new_bt = _evaluate(name, pos, futures_pnl)
        rows.extend(new_rows)
        backtests.update(new_bt)

    results = pd.DataFrame(rows).sort_values(["cost_adjusted", "oos_sharpe"], ascending=[True, False])
    selected_keys = [
        "base_regime_ic_vol_cost",
        "abundant_low_or_normal_half_cost",
        "abundant_low_or_normal_flat_cost",
        "abundant_curve_confirmed_half_cost",
    ]
    period_tables = {}
    for key in selected_keys:
        if key in backtests:
            period_tables[key] = period_performance(backtests[key])[
                ["period", "total_pnl", "sharpe", "max_drawdown", "hit_rate", "days"]
            ]
    _write_log(results, period_tables, rebuilt["selected_table"])
    return {
        "results": results.reset_index(drop=True),
        "positions": positions,
        "backtests": backtests,
        "period_tables": period_tables,
        "selected_table": rebuilt["selected_table"],
        "errors": rebuilt["errors"],
    }


if __name__ == "__main__":
    out = run_corn_abundant_supply_improvement()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("Errors:", out["errors"])
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
