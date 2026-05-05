"""Experiment 15: targeted new signal tests.

This script tests a small set of interpretable feature blocks suggested after
the first research pass:
- seasonal inventory normalisation
- Cargill-vs-public inventory divergence
- soybean-specific Cargill crush
- inventory/use pressure proxies

The tests keep the existing core and physical model intact, then add each new
block as a strongly regularised overlay.
"""

from __future__ import print_function

import numpy as np
import pandas as pd

from grain_futures_strategy import (
    COMMODITIES,
    OUTRIGHT_CORE_FEATURES,
    OUTRIGHT_PHYSICAL_FEATURES,
    backtest_positions,
    build_feature_panels,
    build_improved_model_signals,
    edge_filtered_positions,
    fit_ridge_predict,
    load_train_set,
    model_predictions_to_positions,
    rolling_zscore,
    split_performance,
    to_available_calendar,
)


SPLIT_DATE = "2018-01-01"
ALPHA_NEW_BLOCK = 1000.0
OVERLAY_WEIGHTS = [0.25, 0.50, 1.00]


def seasonal_zscore(df, min_periods=3):
    """Expanding same-week z-score using only prior observations."""
    week = pd.Series(df.index.isocalendar().week.astype(int), index=df.index)
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for column in df.columns:
        series = df[column]
        pieces = []
        for _, values in series.groupby(week):
            mean = values.expanding(min_periods=min_periods).mean().shift(1)
            std = values.expanding(min_periods=min_periods).std().shift(1)
            pieces.append((values - mean) / std.replace(0.0, np.nan))
        out[column] = pd.concat(pieces).sort_index()

    return out.clip(lower=-5.0, upper=5.0)


def build_new_signal_feature_panels(data, feature_panels):
    """Return feature panels for each proposed new signal block."""
    trading_index = data["adj1"].index
    inventories = to_available_calendar(data["inventories"], trading_index, 2)
    receipts = to_available_calendar(data["receipts"], trading_index, 2)
    cgl_inv = to_available_calendar(data["cgl_inv"], trading_index, 1)
    cgl_crush = to_available_calendar(data["cgl_crush"], trading_index, 1)

    public_inv_seasonal = seasonal_zscore(inventories)
    public_inv_change_seasonal = seasonal_zscore(inventories.diff(5))
    cgl_inv_seasonal = seasonal_zscore(cgl_inv)
    cgl_inv_change_seasonal = seasonal_zscore(cgl_inv.diff(5))
    receipts_change = rolling_zscore(receipts.diff(5), 126, 30)

    crush = pd.DataFrame(index=trading_index)
    crush["crush_processed"] = cgl_crush["processed"]
    crush["crush_planned"] = cgl_crush["planned"]
    crush["crush_surprise"] = cgl_crush["processed"] - cgl_crush["planned"]
    crush["crush_utilization"] = cgl_crush["processed"] / cgl_crush["planned"].replace(0.0, np.nan) - 1.0
    crush_features = rolling_zscore(crush, 252, 60)

    blocks = {
        "seasonal_inventory": {},
        "cargill_public_divergence": {},
        "soybean_crush_only": {},
        "inventory_use_pressure": {},
        "combined_new_signals": {},
    }

    for commodity in COMMODITIES:
        seasonal = pd.DataFrame(index=trading_index)
        seasonal["public_inventory_seasonal"] = public_inv_seasonal[commodity]
        seasonal["public_inventory_change_seasonal"] = public_inv_change_seasonal[commodity]
        seasonal["cgl_inventory_seasonal"] = cgl_inv_seasonal[commodity]
        seasonal["cgl_inventory_change_seasonal"] = cgl_inv_change_seasonal[commodity]

        divergence = pd.DataFrame(index=trading_index)
        divergence["cgl_minus_public_inventory"] = cgl_inv_seasonal[commodity] - public_inv_seasonal[commodity]
        divergence["cgl_minus_public_inventory_change"] = (
            cgl_inv_change_seasonal[commodity] - public_inv_change_seasonal[commodity]
        )

        soybean_crush = pd.DataFrame(index=trading_index)
        for column in ["crush_processed", "crush_planned", "crush_surprise", "crush_utilization"]:
            soybean_crush[column + "_soy_only"] = crush_features[column] if commodity == "SOYABEAN" else 0.0

        pressure = pd.DataFrame(index=trading_index)
        pressure["public_inventory_pressure"] = -public_inv_seasonal[commodity] + receipts_change[commodity]
        pressure["cgl_inventory_pressure"] = -cgl_inv_seasonal[commodity]
        pressure["cgl_draw_vs_public_draw"] = (
            -cgl_inv_change_seasonal[commodity] + public_inv_change_seasonal[commodity]
        )
        pressure["soy_crush_demand_pressure"] = (
            crush_features["crush_surprise"] + crush_features["crush_utilization"]
            if commodity == "SOYABEAN"
            else 0.0
        )

        combined = pd.concat([seasonal, divergence, soybean_crush, pressure], axis=1)

        blocks["seasonal_inventory"][commodity] = seasonal.clip(lower=-5.0, upper=5.0)
        blocks["cargill_public_divergence"][commodity] = divergence.clip(lower=-5.0, upper=5.0)
        blocks["soybean_crush_only"][commodity] = soybean_crush.clip(lower=-5.0, upper=5.0)
        blocks["inventory_use_pressure"][commodity] = pressure.clip(lower=-5.0, upper=5.0)
        blocks["combined_new_signals"][commodity] = combined.clip(lower=-5.0, upper=5.0)

    return blocks


def build_overlay_predictions(feature_block, futures_pnl, split_date=SPLIT_DATE, alpha=ALPHA_NEW_BLOCK):
    split_date = pd.Timestamp(split_date)
    train_mask = futures_pnl.index < split_date
    predictions = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float)
    coefficients = {}

    for commodity in COMMODITIES:
        target = futures_pnl[commodity].shift(-1).rolling(5, min_periods=5).sum().shift(-4)
        pred, coef = fit_ridge_predict(feature_block[commodity], target, train_mask, alpha=alpha)
        predictions[commodity] = pred
        coefficients[commodity] = coef

    return predictions.fillna(0.0), pd.DataFrame(coefficients)


def metrics_row(name, predictions, futures_pnl, split_date=SPLIT_DATE, edge_filter=False):
    if edge_filter:
        positions, _, _ = edge_filtered_positions(predictions, futures_pnl, quantile=0.50)
        label = name + " | edge-filtered"
    else:
        positions = model_predictions_to_positions(predictions, futures_pnl)
        label = name + " | unfiltered"

    bt, _ = backtest_positions(positions, futures_pnl, 0.0)
    metrics = split_performance(bt, split_date)
    return {
        "experiment": label,
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "full_pnl": metrics.loc["total_pnl", "full_period"],
        "max_drawdown": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
    }


def run_experiment():
    data = load_train_set("train_set")
    feature_panels, futures_pnl = build_feature_panels(data)

    selected_predictions, _, core_predictions, physical_predictions = build_improved_model_signals(
        feature_panels, futures_pnl, SPLIT_DATE
    )
    new_blocks = build_new_signal_feature_panels(data, feature_panels)

    rows = []
    rows.append(metrics_row("Current selected core + physical", selected_predictions, futures_pnl, edge_filter=False))
    rows.append(metrics_row("Current selected core + physical", selected_predictions, futures_pnl, edge_filter=True))

    overlay_predictions = {}
    for block_name, block in new_blocks.items():
        block_pred, _ = build_overlay_predictions(block, futures_pnl)
        overlay_predictions[block_name] = block_pred
        rows.append(metrics_row(block_name + " only", block_pred, futures_pnl, edge_filter=False))

        for weight in OVERLAY_WEIGHTS:
            combined = selected_predictions.fillna(0.0) + float(weight) * block_pred.fillna(0.0)
            rows.append(
                metrics_row(
                    "Current selected + " + block_name + " w=" + str(weight),
                    combined,
                    futures_pnl,
                    edge_filter=False,
                )
            )

    all_new = sum(overlay_predictions.values())
    for weight in OVERLAY_WEIGHTS:
        combined = selected_predictions.fillna(0.0) + float(weight) * all_new.fillna(0.0)
        rows.append(metrics_row("Current selected + all new overlays w=" + str(weight), combined, futures_pnl))

    expanded_static_rows = []
    train_mask = futures_pnl.index < pd.Timestamp(SPLIT_DATE)
    expanded_predictions = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float)
    expanded_features = {}
    for commodity in COMMODITIES:
        target = futures_pnl[commodity].shift(-1).rolling(5, min_periods=5).sum().shift(-4)
        columns = OUTRIGHT_PHYSICAL_FEATURES
        expanded_features[commodity] = pd.concat(
            [feature_panels[commodity][columns], new_blocks["combined_new_signals"][commodity]], axis=1
        )
        pred, _ = fit_ridge_predict(expanded_features[commodity], target, train_mask, alpha=1000.0)
        expanded_predictions[commodity] = pred

    expanded_combined = core_predictions.fillna(0.0) + expanded_predictions.fillna(0.0)
    expanded_static_rows.append(
        metrics_row("Core + expanded physical/new block", expanded_combined, futures_pnl, edge_filter=False)
    )
    expanded_static_rows.append(
        metrics_row("Core + expanded physical/new block", expanded_combined, futures_pnl, edge_filter=True)
    )
    rows.extend(expanded_static_rows)

    result = pd.DataFrame(rows)
    result = result.sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False]).reset_index(drop=True)
    return result


if __name__ == "__main__":
    results = run_experiment()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(results.to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
