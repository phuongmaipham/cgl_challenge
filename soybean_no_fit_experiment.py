"""Soybean-only lower-overfit strategy experiment.

The experiment uses fixed economic recipes and no regression coefficients.
Candidate selection is done on a pre-2018 validation slice, then 2018-2020 is
reported as the out-of-sample test.
"""

from __future__ import print_function

import numpy as np
import pandas as pd

from grain_futures_strategy import (
    backtest_positions,
    backtest_positions_with_costs,
    build_feature_panels,
    load_train_set,
    split_performance,
)


COMMODITY = "SOYABEAN"
TRAIN_END = "2016-01-01"
VALIDATION_END = "2018-01-01"
TEST_START = "2018-01-01"


def _z_tanh(series, divisor=2.0):
    return np.tanh(series.astype(float) / float(divisor))


def _smooth(series, halflife=2):
    return series.ewm(halflife=float(halflife), adjust=False, min_periods=1).mean()


def _threshold(series, threshold=0.05):
    out = series.copy()
    out[out.abs() < float(threshold)] = 0.0
    return out


def _soybean_components(feature_panels):
    soy = feature_panels[COMMODITY]
    inventory_pressure = (
        -soy["public_inventory_change"] - soy["receipts_change"] - soy["cgl_inventory_change"]
    ) / 3.0
    crush_pressure = (soy["crush_surprise"] + soy["crush_utilization"]) / 2.0
    trend = (soy["mom_20"] + soy["mom_60"] + soy["curve_spread"] + soy["cot_pm_oi_level"]) / 4.0
    slow_mr = (-soy["mom_20"] - soy["mom_60"]) / 2.0
    curve_tightness = (soy["curve_spread"] + soy["curve_ratio"]) / 2.0
    components = {
        "rev_5": soy["rev_5"],
        "inventory_pressure": inventory_pressure,
        "crush_pressure": crush_pressure,
        "trend": trend,
        "slow_mr": slow_mr,
        "curve_tightness": curve_tightness,
        "cot_pressure": soy["cot_pm_oi_level"],
    }
    return {name: values.fillna(0.0) for name, values in components.items()}


def build_soybean_signal(feature_panels, recipe):
    c = _soybean_components(feature_panels)
    if recipe == "rev5":
        raw = c["rev_5"]
    elif recipe == "rev5_plus_physical_10":
        raw = c["rev_5"] + 0.10 * ((c["inventory_pressure"] + c["crush_pressure"]) / 2.0)
    elif recipe == "soy_physical_balanced":
        raw = 0.60 * c["rev_5"] + 0.20 * c["inventory_pressure"] + 0.20 * c["crush_pressure"]
    elif recipe == "soy_crush_inventory":
        raw = 0.50 * c["rev_5"] + 0.25 * c["inventory_pressure"] + 0.25 * c["crush_pressure"]
    elif recipe == "soy_slow_mr_physical":
        raw = 0.45 * c["slow_mr"] + 0.25 * c["rev_5"] + 0.15 * c["inventory_pressure"] + 0.15 * c["crush_pressure"]
    elif recipe == "soy_trend_physical":
        raw = 0.50 * c["trend"] + 0.25 * c["inventory_pressure"] + 0.25 * c["crush_pressure"]
    elif recipe == "soy_curve_crush":
        raw = 0.45 * c["curve_tightness"] + 0.35 * c["crush_pressure"] + 0.20 * c["rev_5"]
    elif recipe == "soy_defensive_blend":
        raw = 0.50 * c["rev_5"] + 0.20 * c["slow_mr"] + 0.15 * c["inventory_pressure"] + 0.15 * c["crush_pressure"]
    elif recipe == "soy_conservative_long_blend":
        trend_physical = 0.50 * c["trend"] + 0.25 * c["inventory_pressure"] + 0.25 * c["crush_pressure"]
        curve_crush = 0.45 * c["curve_tightness"] + 0.35 * c["crush_pressure"] + 0.20 * c["rev_5"]
        raw = 0.50 * trend_physical + 0.50 * curve_crush
    else:
        raise ValueError("Unknown soybean recipe: {}".format(recipe))
    return raw.clip(lower=-5.0, upper=5.0)


def signal_to_soybean_positions(signal, futures_pnl, target_daily_pnl_vol=75.0, max_lot=0.50, mode="long_short"):
    soy_pnl = futures_pnl[[COMMODITY]]
    raw = _threshold(_smooth(_z_tanh(signal), halflife=2), threshold=0.05)
    if mode == "long_only":
        raw = raw.clip(lower=0.0)
    elif mode == "short_only":
        raw = raw.clip(upper=0.0)
    elif mode != "long_short":
        raise ValueError("Unknown position mode: {}".format(mode))
    asset_vol = soy_pnl[COMMODITY].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    pos = raw.reindex(soy_pnl.index).fillna(0.0) * (float(target_daily_pnl_vol) / asset_vol)
    out = pd.DataFrame(0.0, index=soy_pnl.index, columns=[COMMODITY])
    out[COMMODITY] = pos.clip(lower=-float(max_lot), upper=float(max_lot)).fillna(0.0)
    return out


def _metric_columns(bt):
    by_split = split_performance(bt, TEST_START)
    train_val = split_performance(bt.loc[bt.index < TEST_START], TRAIN_END)
    return {
        "train_sharpe": train_val.loc["sharpe", "in_sample"],
        "validation_sharpe": train_val.loc["sharpe", "out_of_sample"],
        "validation_max_drawdown": train_val.loc["max_drawdown", "out_of_sample"],
        "test_sharpe": by_split.loc["sharpe", "out_of_sample"],
        "test_pnl": by_split.loc["total_pnl", "out_of_sample"],
        "test_max_drawdown": by_split.loc["max_drawdown", "out_of_sample"],
        "full_sharpe": by_split.loc["sharpe", "full_period"],
        "full_pnl": by_split.loc["total_pnl", "full_period"],
        "max_drawdown": by_split.loc["max_drawdown", "full_period"],
        "turnover": by_split.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": by_split.loc["avg_gross_exposure", "full_period"],
    }


def _row(recipe, mode, positions, futures_pnl, cost_adjusted):
    if cost_adjusted:
        bt, _ = backtest_positions_with_costs(positions, futures_pnl[[COMMODITY]], 8.75, 0.05)
    else:
        bt, _ = backtest_positions(positions, futures_pnl[[COMMODITY]], 0.0)
    row = {"recipe": recipe, "mode": mode, "cost_adjusted": bool(cost_adjusted)}
    row.update(_metric_columns(bt))
    return row, bt


def _choose_by_validation(results):
    candidates = results[
        (results["cost_adjusted"] == False)
        & (results["train_sharpe"] > 0.0)
        & (results["validation_sharpe"] > 0.0)
    ].copy()
    if candidates.empty:
        return None
    best_validation = candidates["validation_sharpe"].max()
    near_best = candidates[candidates["validation_sharpe"] >= 0.90 * best_validation].copy()
    near_best["score"] = (
        near_best["validation_sharpe"]
        + 0.25 * near_best["train_sharpe"]
        + 0.001 * near_best["validation_max_drawdown"]
    )
    return near_best.sort_values(["score", "validation_max_drawdown"], ascending=[False, False]).iloc[0]


def run_soybean_no_fit_experiment(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    candidates = [
        ("rev5", "long_short"),
        ("rev5_plus_physical_10", "long_short"),
        ("soy_physical_balanced", "long_short"),
        ("soy_crush_inventory", "long_short"),
        ("soy_slow_mr_physical", "long_short"),
        ("soy_trend_physical", "long_short"),
        ("soy_curve_crush", "long_short"),
        ("soy_defensive_blend", "long_short"),
        ("soy_trend_physical", "long_only"),
        ("soy_curve_crush", "long_only"),
        ("soy_physical_balanced", "long_only"),
        ("soy_conservative_long_blend", "long_only"),
    ]
    rows = []
    backtests = {}
    positions = {}

    for recipe, mode in candidates:
        signal = build_soybean_signal(feature_panels, recipe)
        pos = signal_to_soybean_positions(signal, futures_pnl, mode=mode)
        key = recipe + "_" + mode
        positions[key] = pos
        for cost_adjusted in [False, True]:
            row, bt = _row(recipe, mode, pos, futures_pnl, cost_adjusted)
            rows.append(row)
            suffix = "cost_adjusted" if cost_adjusted else "zero_cost"
            backtests[key + "_" + suffix] = bt

    results = pd.DataFrame(rows).sort_values(["cost_adjusted", "test_sharpe"], ascending=[True, False])
    selected = _choose_by_validation(results)
    return {
        "results": results,
        "selected_by_validation": selected,
        "positions": positions,
        "backtests": backtests,
    }


if __name__ == "__main__":
    out = run_soybean_no_fit_experiment()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("Soybean-only no-fit candidate results")
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    print("\nSelected by train/validation rule")
    if out["selected_by_validation"] is None:
        print("No candidate passed train/validation filter.")
    else:
        print(out["selected_by_validation"].to_string(float_format=lambda value: "{:.3f}".format(value)))
