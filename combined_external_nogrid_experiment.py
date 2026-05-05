"""No-grid Macro + EIA ethanol overlay tests.

These variants avoid choosing overlay weights from a full-sample grid. The
weights are either fixed by a simple prior or computed from lagged realized
information only.
"""

from __future__ import print_function

import numpy as np
import pandas as pd

from grain_futures_strategy import (
    backtest_positions,
    build_feature_panels,
    build_improved_model_signals,
    edge_filtered_positions,
    load_train_set,
    split_performance,
    trailing_performance_ensemble_positions,
)
from macro_yfinance_experiment import (
    _download_yfinance_prices,
    build_macro_feature_block,
    build_macro_predictions,
)
from eia_ethanol_experiment import (
    build_ethanol_feature_panel,
    build_ethanol_predictions,
    fetch_eia_ethanol,
)


SPLIT_DATE = "2018-01-01"


def _metrics_from_positions(name, positions, futures_pnl, rule):
    bt, _ = backtest_positions(positions, futures_pnl, 0.0)
    metrics = split_performance(bt, SPLIT_DATE)
    return {
        "experiment": name,
        "rule": rule,
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "full_pnl": metrics.loc["total_pnl", "full_period"],
        "max_drawdown": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
    }, bt


def _combine_sleeves(sleeves, weights):
    out = None
    for name, weight in weights.items():
        weighted = sleeves[name] * float(weight)
        out = weighted if out is None else out.add(weighted, fill_value=0.0)
    return out.fillna(0.0)


def _inverse_vol_external_positions(selected_positions, macro_positions, ethanol_positions, futures_pnl):
    """Keep 80% selected book, split 20% external budget by lagged inverse vol."""
    macro_bt, _ = backtest_positions(macro_positions, futures_pnl, 0.0)
    ethanol_bt, _ = backtest_positions(ethanol_positions, futures_pnl, 0.0)
    pnl = pd.DataFrame({"macro": macro_bt["net_pnl"], "ethanol": ethanol_bt["net_pnl"]}, index=futures_pnl.index)
    vol = pnl.rolling(252, min_periods=60).std().shift(1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    external_weights = inv.div(inv.sum(axis=1), axis=0).fillna(0.5)
    macro_weight = 0.20 * external_weights["macro"]
    ethanol_weight = 0.20 * external_weights["ethanol"]
    positions = (
        selected_positions * 0.80
        + macro_positions.mul(macro_weight, axis=0)
        + ethanol_positions.mul(ethanol_weight, axis=0)
    )
    diagnostics = pd.DataFrame(
        {"selected_weight": 0.80, "macro_weight": macro_weight, "ethanol_weight": ethanol_weight},
        index=futures_pnl.index,
    )
    return positions.fillna(0.0), diagnostics


def _lagged_drawdown_scale(positions, futures_pnl, trigger=-3500.0, scale=0.75, lookback=126):
    """A fixed, lagged drawdown throttle; not selected by grid."""
    bt, _ = backtest_positions(positions, futures_pnl, 0.0)
    lagged_cum = bt["net_pnl"].cumsum().shift(1)
    trailing_peak = lagged_cum.rolling(int(lookback), min_periods=20).max()
    trailing_drawdown = lagged_cum - trailing_peak
    exposure_scale = pd.Series(1.0, index=positions.index)
    exposure_scale.loc[trailing_drawdown < float(trigger)] = float(scale)
    return positions.mul(exposure_scale.fillna(1.0), axis=0), exposure_scale


def run_combined_external_nogrid_experiment():
    data = load_train_set("train_set")
    feature_panels, futures_pnl = build_feature_panels(data)
    selected_predictions, _, _, _ = build_improved_model_signals(feature_panels, futures_pnl, SPLIT_DATE)

    macro_prices = _download_yfinance_prices(futures_pnl.index.min(), futures_pnl.index.max())
    macro_features = build_macro_feature_block(macro_prices, futures_pnl.index)
    macro_predictions, macro_coefficients = build_macro_predictions(macro_features, futures_pnl)

    ethanol = fetch_eia_ethanol()
    ethanol_features = build_ethanol_feature_panel(ethanol, futures_pnl.index)
    ethanol_predictions, ethanol_coefficients = build_ethanol_predictions(ethanol_features, futures_pnl)

    selected_positions, _, _ = edge_filtered_positions(selected_predictions, futures_pnl, quantile=0.50)
    macro_positions, _, _ = edge_filtered_positions(macro_predictions, futures_pnl, quantile=0.50)
    ethanol_positions, _, _ = edge_filtered_positions(ethanol_predictions, futures_pnl, quantile=0.50)

    rows = []
    baseline_row, baseline_bt = _metrics_from_positions(
        "Current selected core + physical | edge-filtered",
        selected_positions,
        futures_pnl,
        "baseline",
    )
    rows.append(baseline_row)

    fixed_prediction_positions, _, _ = edge_filtered_positions(
        selected_predictions.fillna(0.0)
        + 0.10 * macro_predictions.fillna(0.0)
        + 0.10 * ethanol_predictions.fillna(0.0),
        futures_pnl,
        quantile=0.50,
    )
    row, fixed_prediction_bt = _metrics_from_positions(
        "No-grid fixed small prediction overlay | macro 0.10 + ethanol 0.10",
        fixed_prediction_positions,
        futures_pnl,
        "fixed_prediction_overlay",
    )
    rows.append(row)

    fixed_sleeve_positions = _combine_sleeves(
        {"selected": selected_positions, "macro": macro_positions, "ethanol": ethanol_positions},
        {"selected": 0.80, "macro": 0.10, "ethanol": 0.10},
    )
    row, fixed_sleeve_bt = _metrics_from_positions(
        "No-grid fixed sleeve budget | selected 80% + macro 10% + ethanol 10%",
        fixed_sleeve_positions,
        futures_pnl,
        "fixed_sleeve_budget",
    )
    rows.append(row)

    inv_vol_positions, inv_vol_weights = _inverse_vol_external_positions(
        selected_positions,
        macro_positions,
        ethanol_positions,
        futures_pnl,
    )
    row, inv_vol_bt = _metrics_from_positions(
        "No-grid inverse-vol external split | selected 80% + external 20%",
        inv_vol_positions,
        futures_pnl,
        "lagged_inverse_vol_split",
    )
    row["avg_macro_weight"] = inv_vol_weights["macro_weight"].mean()
    row["avg_ethanol_weight"] = inv_vol_weights["ethanol_weight"].mean()
    rows.append(row)

    trailing_positions, trailing_weights, trailing_scores = trailing_performance_ensemble_positions(
        {"selected": selected_positions, "macro": macro_positions, "ethanol": ethanol_positions},
        futures_pnl,
        lookback=252,
        temperature=1.0,
    )
    row, trailing_bt = _metrics_from_positions(
        "No-grid lagged trailing-performance ensemble",
        trailing_positions,
        futures_pnl,
        "lagged_trailing_performance",
    )
    for column in trailing_weights.columns:
        row["avg_weight_" + column] = trailing_weights[column].mean()
    rows.append(row)

    throttled_positions, throttle_scale = _lagged_drawdown_scale(fixed_prediction_positions, futures_pnl)
    row, throttled_bt = _metrics_from_positions(
        "No-grid fixed small overlay + lagged drawdown throttle",
        throttled_positions,
        futures_pnl,
        "fixed_overlay_lagged_drawdown_throttle",
    )
    row["avg_scale"] = throttle_scale.mean()
    rows.append(row)

    results = pd.DataFrame(rows).sort_values(["oos_sharpe", "max_drawdown"], ascending=[False, False])
    return {
        "results": results.reset_index(drop=True),
        "baseline_bt": baseline_bt,
        "fixed_prediction_bt": fixed_prediction_bt,
        "fixed_sleeve_bt": fixed_sleeve_bt,
        "inverse_vol_bt": inv_vol_bt,
        "trailing_bt": trailing_bt,
        "throttled_bt": throttled_bt,
        "inverse_vol_weights": inv_vol_weights,
        "trailing_weights": trailing_weights,
        "trailing_scores": trailing_scores,
        "macro_coefficients": macro_coefficients,
        "ethanol_coefficients": ethanol_coefficients,
    }


if __name__ == "__main__":
    out = run_combined_external_nogrid_experiment()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 24)
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
