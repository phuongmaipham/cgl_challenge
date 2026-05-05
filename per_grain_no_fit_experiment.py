"""Per-grain no-fit strategy experiment.

Builds commodity-specific fixed rules instead of one fitted formula for all
grains. The rules use only existing train_set features and fixed economic signs.
"""

from __future__ import print_function

import numpy as np
import pandas as pd

from grain_futures_strategy import (
    COMMODITIES,
    backtest_positions,
    backtest_positions_with_costs,
    build_feature_panels,
    load_train_set,
    model_predictions_to_positions,
    split_performance,
    volatility_target_positions,
    _feature_frame,
)


SPLIT_DATE = "2018-01-01"


def _tanh_signal(frame, divisor=2.0):
    return np.tanh(frame.astype(float) / float(divisor))


def _independent_positions(signal, futures_pnl, target_daily_pnl_vol=60.0, max_lot=0.50):
    """Convert per-asset signals to independently volatility-targeted positions."""
    asset_vol = futures_pnl.rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    raw = _tanh_signal(signal).reindex(futures_pnl.index).fillna(0.0)
    positions = raw * (float(target_daily_pnl_vol) / asset_vol)
    return positions.clip(lower=-float(max_lot), upper=float(max_lot)).fillna(0.0)


def _portfolio_row(name, positions, futures_pnl, cost=False):
    if cost:
        bt, _ = backtest_positions_with_costs(positions, futures_pnl, 8.75, 0.05)
    else:
        bt, _ = backtest_positions(positions, futures_pnl, 0.0)
    metrics = split_performance(bt, SPLIT_DATE)
    return {
        "experiment": name,
        "cost_adjusted": bool(cost),
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "full_pnl": metrics.loc["total_pnl", "full_period"],
        "max_drawdown": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
    }, bt


def _asset_rows(name, positions, futures_pnl):
    rows = []
    held = positions.shift(1).fillna(0.0)
    for commodity in COMMODITIES:
        one_bt = pd.DataFrame(index=futures_pnl.index)
        one_bt["net_pnl"] = held[commodity] * futures_pnl[commodity]
        one_bt["gross_pnl"] = one_bt["net_pnl"]
        one_bt["costs"] = 0.0
        one_bt["turnover"] = positions[commodity].diff().abs().fillna(0.0)
        one_bt["gross_exposure"] = positions[commodity].abs()
        one_bt["held_gross_exposure"] = held[commodity].abs()
        one_bt["cum_pnl"] = one_bt["net_pnl"].cumsum()
        metrics = split_performance(one_bt, SPLIT_DATE)
        rows.append(
            {
                "experiment": name,
                "commodity": commodity,
                "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
                "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
                "full_sharpe": metrics.loc["sharpe", "full_period"],
                "full_pnl": metrics.loc["total_pnl", "full_period"],
                "max_drawdown": metrics.loc["max_drawdown", "full_period"],
                "turnover": metrics.loc["avg_daily_turnover", "full_period"],
            }
        )
    return rows


def build_per_grain_no_fit_signal(feature_panels, recipe="commodity_specific"):
    """Build fixed per-grain signals with no fitted coefficients."""
    mr = _feature_frame(feature_panels, "rev_5")
    trend = (
        _feature_frame(feature_panels, "mom_20")
        + _feature_frame(feature_panels, "mom_60")
        + _feature_frame(feature_panels, "curve_spread")
        + _feature_frame(feature_panels, "cot_pm_oi_level")
    ) / 4.0
    inv_pressure = (
        -_feature_frame(feature_panels, "public_inventory_change")
        - _feature_frame(feature_panels, "receipts_change")
        - _feature_frame(feature_panels, "cgl_inventory_change")
    ) / 3.0
    crush = pd.DataFrame(0.0, index=mr.index, columns=mr.columns)
    crush["SOYABEAN"] = (
        _feature_frame(feature_panels, "crush_surprise")["SOYABEAN"]
        + _feature_frame(feature_panels, "crush_utilization")["SOYABEAN"]
    ) / 2.0

    signal = pd.DataFrame(0.0, index=mr.index, columns=mr.columns)
    if recipe == "mr_only":
        return mr.clip(lower=-5.0, upper=5.0)
    if recipe == "trend_only":
        return trend.clip(lower=-5.0, upper=5.0)
    if recipe == "mr_plus_physical_10":
        return (mr + 0.10 * ((inv_pressure + crush) / 2.0)).clip(lower=-5.0, upper=5.0)
    if recipe == "commodity_specific":
        signal["CORN"] = 0.80 * mr["CORN"] + 0.20 * inv_pressure["CORN"]
        signal["SOYABEAN"] = 0.70 * mr["SOYABEAN"] + 0.20 * inv_pressure["SOYABEAN"] + 0.10 * crush["SOYABEAN"]
        signal["WHEAT_SRW"] = 0.60 * mr["WHEAT_SRW"] + 0.25 * trend["WHEAT_SRW"] + 0.15 * inv_pressure["WHEAT_SRW"]
        signal["WHEAT_HRW"] = 0.60 * mr["WHEAT_HRW"] + 0.25 * trend["WHEAT_HRW"] + 0.15 * inv_pressure["WHEAT_HRW"]
        return signal.clip(lower=-5.0, upper=5.0)
    raise ValueError("Unknown recipe: {}".format(recipe))


def run_per_grain_no_fit_experiment(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    rows = []
    asset_rows = []
    positions_by_recipe = {}
    backtests = {}

    for recipe in ["mr_only", "mr_plus_physical_10", "commodity_specific", "trend_only"]:
        signal = build_per_grain_no_fit_signal(feature_panels, recipe=recipe)
        independent_positions = _independent_positions(signal, futures_pnl, target_daily_pnl_vol=60.0, max_lot=0.50)
        positions_by_recipe[recipe + "_independent"] = independent_positions
        row, bt = _portfolio_row(recipe + "_independent", independent_positions, futures_pnl, cost=False)
        rows.append(row)
        backtests[recipe + "_independent_zero_cost"] = bt
        row, bt = _portfolio_row(recipe + "_independent", independent_positions, futures_pnl, cost=True)
        rows.append(row)
        backtests[recipe + "_independent_cost_adjusted"] = bt
        asset_rows.extend(_asset_rows(recipe + "_independent", independent_positions, futures_pnl))

        relative_positions = model_predictions_to_positions(signal, futures_pnl)
        relative_positions, _ = volatility_target_positions(
            relative_positions,
            futures_pnl,
            target_daily_pnl_vol=120.0,
            max_scale=1.0,
        )
        positions_by_recipe[recipe + "_relative_value"] = relative_positions
        row, bt = _portfolio_row(recipe + "_relative_value", relative_positions, futures_pnl, cost=False)
        rows.append(row)
        backtests[recipe + "_relative_value_zero_cost"] = bt
        row, bt = _portfolio_row(recipe + "_relative_value", relative_positions, futures_pnl, cost=True)
        rows.append(row)
        backtests[recipe + "_relative_value_cost_adjusted"] = bt
        asset_rows.extend(_asset_rows(recipe + "_relative_value", relative_positions, futures_pnl))

    return {
        "results": pd.DataFrame(rows).sort_values(["cost_adjusted", "oos_sharpe"], ascending=[True, False]),
        "asset_results": pd.DataFrame(asset_rows).sort_values(["experiment", "commodity"]),
        "positions": positions_by_recipe,
        "backtests": backtests,
    }


if __name__ == "__main__":
    out = run_per_grain_no_fit_experiment()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("Portfolio results")
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    print("\nPer-grain results")
    print(out["asset_results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
