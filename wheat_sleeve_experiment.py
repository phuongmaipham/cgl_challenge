"""Standalone SRW and HRW wheat sleeve backtests.

One fixed no-fit model is tested for each wheat product:
- WHEAT_SRW: short-term reversion + trend/curve/COT + inventory pressure.
- WHEAT_HRW: short-term reversion + trend/curve/COT + inventory pressure.

No fitted coefficients and no external data are used. The recipes are the wheat
legs from the previously tested commodity-specific no-fit model, isolated as
standalone sleeves.
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
    split_performance,
)


SPLIT_DATE = "2018-01-01"
COMMODITIES = ["WHEAT_SRW", "WHEAT_HRW"]


def _tanh_signal(series, divisor=2.0):
    return pd.Series(np.tanh(series.astype(float) / float(divisor)), index=series.index)


def _wheat_components(panel):
    mr = panel["rev_5"]
    trend = (
        panel["mom_20"]
        + panel["mom_60"]
        + panel["curve_spread"]
        + panel["cot_pm_oi_level"]
    ) / 4.0
    inventory_pressure = (
        -panel["public_inventory_change"]
        - panel["receipts_change"]
        - panel["cgl_inventory_change"]
    ) / 3.0
    curve_tightness = (panel["curve_spread"] + panel["curve_ratio"]) / 2.0
    cot_flow = panel["cot_mm_change"]
    return {
        "mr": mr.fillna(0.0),
        "trend": trend.fillna(0.0),
        "inventory_pressure": inventory_pressure.fillna(0.0),
        "curve_tightness": curve_tightness.fillna(0.0),
        "cot_flow": cot_flow.fillna(0.0),
    }


def build_wheat_signal(feature_panels, commodity, recipe="mr_trend_physical"):
    components = _wheat_components(feature_panels[commodity])
    if recipe == "mr_trend_physical":
        raw = (
            0.55 * components["mr"]
            + 0.25 * components["trend"]
            + 0.15 * components["inventory_pressure"]
            + 0.05 * components["cot_flow"]
        )
    elif recipe == "trend_curve_physical":
        raw = (
            0.45 * components["trend"]
            + 0.25 * components["curve_tightness"]
            + 0.20 * components["inventory_pressure"]
            + 0.10 * components["cot_flow"]
        )
    elif recipe == "defensive_long_trend":
        raw = (
            0.55 * components["trend"]
            + 0.20 * components["inventory_pressure"]
            + 0.15 * components["curve_tightness"]
            + 0.10 * components["cot_flow"]
        )
    else:
        raise ValueError("Unknown recipe: {}".format(recipe))
    return raw.clip(lower=-5.0, upper=5.0).fillna(0.0)


def _positions_from_signal(signal, futures_pnl, commodity, mode="long_short", target_daily_pnl_vol=60.0, max_lot=0.50):
    signal = _tanh_signal(signal.reindex(futures_pnl.index).fillna(0.0))
    signal = signal.ewm(halflife=2.0, adjust=False, min_periods=1).mean()
    signal[signal.abs() < 0.05] = 0.0
    if mode == "long_only":
        signal = signal.clip(lower=0.0)
    elif mode == "short_only":
        signal = signal.clip(upper=0.0)
    elif mode != "long_short":
        raise ValueError("Unknown mode: {}".format(mode))

    pnl = futures_pnl[[commodity]]
    asset_vol = pnl[commodity].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    pos = signal * (float(target_daily_pnl_vol) / asset_vol)
    out = pd.DataFrame(0.0, index=futures_pnl.index, columns=[commodity])
    out[commodity] = pos.clip(lower=-float(max_lot), upper=float(max_lot)).fillna(0.0)
    return out


def _metrics(bt):
    table = split_performance(bt, SPLIT_DATE)
    return {
        "is_sharpe": table.loc["sharpe", "in_sample"],
        "oos_sharpe": table.loc["sharpe", "out_of_sample"],
        "oos_pnl": table.loc["total_pnl", "out_of_sample"],
        "oos_max_drawdown": table.loc["max_drawdown", "out_of_sample"],
        "full_sharpe": table.loc["sharpe", "full_period"],
        "full_pnl": table.loc["total_pnl", "full_period"],
        "max_drawdown": table.loc["max_drawdown", "full_period"],
        "turnover": table.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": table.loc["avg_gross_exposure", "full_period"],
    }


def _evaluate(commodity, recipe, signal, futures_pnl, mode):
    positions = _positions_from_signal(signal, futures_pnl, commodity, mode=mode)
    rows = []
    backtests = {}
    for cost_adjusted in [False, True]:
        if cost_adjusted:
            bt, _ = backtest_positions_with_costs(positions, futures_pnl[[commodity]], 8.75, 0.05)
        else:
            bt, _ = backtest_positions(positions, futures_pnl[[commodity]], 0.0)
        row = {
            "commodity": commodity,
            "strategy": recipe,
            "mode": mode,
            "cost_adjusted": cost_adjusted,
        }
        row.update(_metrics(bt))
        rows.append(row)
        backtests[commodity + "_" + mode + ("_cost" if cost_adjusted else "_zero")] = bt
    return rows, backtests, positions


def _write_log(results, path="notes/wheat_sleeves.txt"):
    lines = []
    lines.append("Standalone wheat sleeve backtests")
    lines.append("Date: 2026-05-02")
    lines.append("")
    lines.append("Model")
    lines.append("-----")
    lines.append("Three fixed no-fit wheat recipes were tested, then one model is recommended for WHEAT_SRW and one for WHEAT_HRW.")
    lines.append("")
    lines.append("Primary recipe:")
    lines.append("signal = 0.55 * rev_5")
    lines.append("       + 0.25 * trend")
    lines.append("       + 0.15 * inventory_pressure")
    lines.append("       + 0.05 * cot_mm_change")
    lines.append("")
    lines.append("Alternates tested:")
    lines.append("- trend_curve_physical = 0.45 * trend + 0.25 * curve_tightness + 0.20 * inventory_pressure + 0.10 * cot_mm_change")
    lines.append("- defensive_long_trend = 0.55 * trend + 0.20 * inventory_pressure + 0.15 * curve_tightness + 0.10 * cot_mm_change")
    lines.append("")
    lines.append("trend = mean(mom_20, mom_60, curve_spread, cot_pm_oi_level)")
    lines.append("inventory_pressure = mean(-public_inventory_change, -receipts_change, -cgl_inventory_change)")
    lines.append("")
    lines.append("Controls")
    lines.append("--------")
    lines.append("- No fitted coefficients.")
    lines.append("- No external data.")
    lines.append("- Same fixed formula for SRW and HRW.")
    lines.append("- Tested both long/short and long-only; the long/short version is the primary wheat hypothesis.")
    lines.append("- Cost-adjusted rows include 8.75 USD per one-way lot plus 5% annual margin funding.")
    lines.append("")
    lines.append("Results")
    lines.append("-------")
    cols = [
        "commodity",
        "strategy",
        "mode",
        "cost_adjusted",
        "is_sharpe",
        "oos_sharpe",
        "oos_pnl",
        "oos_max_drawdown",
        "full_sharpe",
        "full_pnl",
        "max_drawdown",
        "turnover",
        "avg_gross_exposure",
    ]
    lines.append(results[cols].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    cost = results.loc[results["cost_adjusted"]].copy()
    for commodity in COMMODITIES:
        best = cost.loc[cost["commodity"] == commodity].sort_values("oos_sharpe", ascending=False).iloc[0]
        lines.append(
            "- {} recommended row: {} / {} mode, OOS Sharpe {:.3f}, OOS DD {:.3f}, full Sharpe {:.3f}.".format(
                commodity,
                best["strategy"],
                best["mode"],
                best["oos_sharpe"],
                best["oos_max_drawdown"],
                best["full_sharpe"],
            )
        )
        if best["oos_sharpe"] <= 0.0:
            lines.append("  Reject as standalone sleeve.")
        elif best["oos_sharpe"] < 0.75:
            lines.append("  Positive but not strong enough to compete with the corn/soybean sleeves.")
        else:
            lines.append("  Worth keeping as a candidate, pending overfit review.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def run_wheat_sleeve_experiment(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    rows = []
    backtests = {}
    positions = {}
    signals = {}
    for commodity in COMMODITIES:
        for recipe in ["mr_trend_physical", "trend_curve_physical", "defensive_long_trend"]:
            signal = build_wheat_signal(feature_panels, commodity, recipe=recipe)
            signals[commodity + "_" + recipe] = signal
            for mode in ["long_short", "long_only"]:
                new_rows, new_bt, pos = _evaluate(commodity, recipe, signal, futures_pnl, mode)
                rows.extend(new_rows)
                backtests.update(new_bt)
                positions[commodity + "_" + recipe + "_" + mode] = pos
    results = pd.DataFrame(rows).sort_values(["commodity", "cost_adjusted", "oos_sharpe"], ascending=[True, True, False])
    _write_log(results)
    return {
        "results": results.reset_index(drop=True),
        "signals": signals,
        "positions": positions,
        "backtests": backtests,
    }


if __name__ == "__main__":
    out = run_wheat_sleeve_experiment()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
