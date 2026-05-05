"""Soybean low-volatility switch experiment.

The existing soybean drawdown-priority strategy is strong overall but weak in
quiet abundant-supply periods. This experiment tests fixed, low-overfit switches:

- keep the drawdown-priority strategy outside low volatility;
- in low volatility, either reduce/zero exposure or use a simple no-fit
  soybean mean-reversion sleeve.

No regression coefficients, IC selection, or optimized switch weights are used.
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
from soybean_external_signal_experiment import (
    COMMODITY,
    _positions_from_signal as _soy_external_positions_from_signal,
    _soybean_regime_frames,
    _weighted_sum,
    run_soybean_external_signal_experiment,
)
from soybean_no_fit_experiment import build_soybean_signal, signal_to_soybean_positions


TEST_START = "2018-01-01"


def _base_drawdown_signal(given, external):
    weights = {
        "given_physical_family": 0.40,
        "external_fx_export_family": 0.20,
        "external_crush_family": 0.20,
        "external_weather_hdd_cdd_family": 0.20,
    }
    families = dict(external)
    families["given_physical_family"] = given["given_physical_family"]
    return _weighted_sum(families, weights).clip(-5.0, 5.0).fillna(0.0)


def _positions_from_signal(signal, futures_pnl, mode):
    return _soy_external_positions_from_signal(signal, futures_pnl, mode=mode)


def _switch_positions(base_positions, low_vol_positions, low_vol_mask, low_vol_scale=1.0):
    mask = pd.Series(low_vol_mask, index=base_positions.index).fillna(False).astype(bool)
    out = base_positions.copy()
    out.loc[mask, COMMODITY] = float(low_vol_scale) * low_vol_positions.loc[mask, COMMODITY]
    return out.fillna(0.0)


def _conditional_scale_positions(base_positions, condition, scale_when_true):
    mask = pd.Series(condition, index=base_positions.index).fillna(False).astype(bool)
    out = base_positions.copy()
    out.loc[mask, COMMODITY] = float(scale_when_true) * out.loc[mask, COMMODITY]
    return out.fillna(0.0)


def _metrics(bt, low_vol_mask):
    table = split_performance(bt, TEST_START)
    low_mask = pd.Series(low_vol_mask, index=bt.index).fillna(False).astype(bool)
    high_mask = ~low_mask
    low = performance_metrics(bt.loc[low_mask])
    non_low = performance_metrics(bt.loc[high_mask])
    low_price = performance_metrics(bt.loc[(bt.index >= "2014-06-01") & (bt.index <= "2017-12-31")])
    return {
        "train_sharpe": table.loc["sharpe", "in_sample"],
        "oos_sharpe": table.loc["sharpe", "out_of_sample"],
        "oos_pnl": table.loc["total_pnl", "out_of_sample"],
        "oos_dd": table.loc["max_drawdown", "out_of_sample"],
        "full_sharpe": table.loc["sharpe", "full_period"],
        "full_dd": table.loc["max_drawdown", "full_period"],
        "turnover": table.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": table.loc["avg_gross_exposure", "full_period"],
        "low_vol_sharpe": low.get("sharpe", np.nan),
        "low_vol_pnl": low.get("total_pnl", np.nan),
        "low_vol_dd": low.get("max_drawdown", np.nan),
        "non_low_vol_sharpe": non_low.get("sharpe", np.nan),
        "low_price_sharpe": low_price.get("sharpe", np.nan),
        "low_price_pnl": low_price.get("total_pnl", np.nan),
        "low_price_dd": low_price.get("max_drawdown", np.nan),
    }


def _evaluate(name, positions, futures_pnl, low_vol_mask):
    rows = []
    backtests = {}
    for cost_adjusted in [False, True]:
        if cost_adjusted:
            bt, _ = backtest_positions_with_costs(positions, futures_pnl[[COMMODITY]], 8.75, 0.05)
        else:
            bt, _ = backtest_positions(positions, futures_pnl[[COMMODITY]], 0.0)
        row = {"strategy": name, "cost_adjusted": bool(cost_adjusted)}
        row.update(_metrics(bt, low_vol_mask))
        rows.append(row)
        backtests[name + ("_cost" if cost_adjusted else "_zero")] = bt
    return rows, backtests


def _write_log(results, period_tables, path="notes/soybean_low_vol_switch.txt"):
    lines = []
    lines.append("Soybean low-volatility switch experiment")
    lines.append("Date: 2026-05-02")
    lines.append("")
    lines.append("Method")
    lines.append("------")
    lines.append("Low-vol regime = 60-day soybean PnL volatility < 0.80 * expanding median volatility.")
    lines.append("Base strategy = drawdown-priority physical 40 / FX 20 / external crush 20 / weather 20.")
    lines.append("Switches are fixed position-level rules; no fitted coefficients or optimized weights.")
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
        "low_vol_sharpe",
        "low_vol_pnl",
        "low_vol_dd",
        "low_price_sharpe",
        "low_price_pnl",
        "low_price_dd",
        "turnover",
    ]
    lines.append(results[cols].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    lines.append("")
    lines.append("Key period performance for selected rows")
    lines.append("----------------------------------------")
    for name, table in period_tables.items():
        lines.append("")
        lines.append(name)
        lines.append(table.to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def run_soybean_low_vol_switch_experiment(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    external_out = run_soybean_external_signal_experiment(data_dir=data_dir)
    given = external_out["given_signals"]
    external = external_out["external_signals"]
    if not {"external_fx_export_family", "external_crush_family", "external_weather_hdd_cdd_family"}.issubset(external):
        raise RuntimeError("Missing required external families: {}".format(external_out["errors"]))

    base_signal = _base_drawdown_signal(given, external)
    regimes = _soybean_regime_frames(given, external, futures_pnl)
    low_vol_mask = regimes["low_vol"] > 0.0
    weak_trend = given["given_trend"].reindex(futures_pnl.index).fillna(0.0) <= 0.0
    weak_physical = given["given_physical_family"].reindex(futures_pnl.index).fillna(0.0) <= 0.0
    soy_price = data["adj1"][COMMODITY].reindex(futures_pnl.index).ffill()
    below_long_ma = soy_price < soy_price.rolling(252, min_periods=120).mean().shift(1)
    negative_medium_mom = feature_panels[COMMODITY]["mom_60"].reindex(futures_pnl.index).fillna(0.0) < 0.0
    abundant_supply_proxy = low_vol_mask & below_long_ma.fillna(False) & negative_medium_mom
    weak_low_vol_proxy = low_vol_mask & weak_trend
    weak_physical_low_vol_proxy = low_vol_mask & weak_physical
    weak_non_abundant_proxy = weak_low_vol_proxy & ~abundant_supply_proxy
    combined_abundant_or_weak_non_abundant = abundant_supply_proxy | weak_non_abundant_proxy

    base_pos = _positions_from_signal(base_signal, futures_pnl, mode="long_only")
    flat_pos = base_pos * 0.0
    half_base_pos = 0.50 * base_pos
    rev5_pos = _positions_from_signal(build_soybean_signal(feature_panels, "rev5"), futures_pnl, mode="long_short")
    defensive_pos = _positions_from_signal(
        build_soybean_signal(feature_panels, "soy_defensive_blend"),
        futures_pnl,
        mode="long_short",
    )
    curve_crush_pos = _positions_from_signal(
        build_soybean_signal(feature_panels, "soy_curve_crush"),
        futures_pnl,
        mode="long_short",
    )

    positions = {
        "base_drawdown_priority": base_pos,
        "low_vol_flat_else_base": _switch_positions(base_pos, flat_pos, low_vol_mask),
        "low_vol_half_base_else_base": _switch_positions(base_pos, half_base_pos, low_vol_mask),
        "low_vol_weak_trend_flat_else_base": _conditional_scale_positions(base_pos, weak_low_vol_proxy, 0.0),
        "low_vol_weak_trend_half_else_base": _conditional_scale_positions(base_pos, weak_low_vol_proxy, 0.50),
        "low_vol_weak_physical_flat_else_base": _conditional_scale_positions(
            base_pos,
            weak_physical_low_vol_proxy,
            0.0,
        ),
        "low_vol_abundant_proxy_flat_else_base": _conditional_scale_positions(
            base_pos,
            abundant_supply_proxy,
            0.0,
        ),
        "low_vol_abundant_proxy_half_else_base": _conditional_scale_positions(
            base_pos,
            abundant_supply_proxy,
            0.50,
        ),
        "low_vol_abundant_flat_weak_nonabundant_flat_else_base": _conditional_scale_positions(
            base_pos,
            combined_abundant_or_weak_non_abundant,
            0.0,
        ),
        "low_vol_abundant_half_weak_nonabundant_flat_else_base": _conditional_scale_positions(
            _conditional_scale_positions(base_pos, abundant_supply_proxy, 0.50),
            weak_non_abundant_proxy,
            0.0,
        ),
        "low_vol_rev5_ls_else_base": _switch_positions(base_pos, rev5_pos, low_vol_mask),
        "low_vol_defensive_mr_else_base": _switch_positions(base_pos, defensive_pos, low_vol_mask),
        "low_vol_curve_crush_ls_else_base": _switch_positions(base_pos, curve_crush_pos, low_vol_mask),
    }

    rows = []
    backtests = {}
    for name, pos in positions.items():
        new_rows, new_bt = _evaluate(name, pos, futures_pnl, low_vol_mask)
        rows.extend(new_rows)
        backtests.update(new_bt)

    results = pd.DataFrame(rows).sort_values(["cost_adjusted", "oos_sharpe"], ascending=[True, False])
    selected = [
        "base_drawdown_priority_cost",
        "low_vol_flat_else_base_cost",
        "low_vol_half_base_else_base_cost",
        "low_vol_weak_trend_flat_else_base_cost",
        "low_vol_abundant_proxy_flat_else_base_cost",
        "low_vol_abundant_half_weak_nonabundant_flat_else_base_cost",
        "low_vol_rev5_ls_else_base_cost",
    ]
    period_tables = {}
    for key in selected:
        if key in backtests:
            table = period_performance(backtests[key])
            cols = ["period", "total_pnl", "sharpe", "max_drawdown", "hit_rate", "days"]
            period_tables[key] = table[cols]
    _write_log(results, period_tables)
    return {
        "results": results.reset_index(drop=True),
        "positions": positions,
        "backtests": backtests,
        "low_vol_mask": low_vol_mask,
        "period_tables": period_tables,
        "external_errors": external_out["errors"],
    }


if __name__ == "__main__":
    out = run_soybean_low_vol_switch_experiment()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    if out["external_errors"]:
        print("External errors:", out["external_errors"])
