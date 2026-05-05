"""Optional combined external overlay experiment.

Combines the yfinance macro overlay and EIA ethanol overlay on top of the
selected Cargill-aware model.
"""

from __future__ import print_function

import pandas as pd

from grain_futures_strategy import (
    backtest_positions,
    build_feature_panels,
    build_improved_model_signals,
    edge_filtered_positions,
    load_train_set,
    model_predictions_to_positions,
    split_performance,
)
from macro_yfinance_experiment import (
    _download_yfinance_prices,
    build_macro_feature_block,
    build_macro_predictions,
)
from eia_ethanol_experiment import (
    fetch_eia_ethanol,
    build_ethanol_feature_panel,
    build_ethanol_predictions,
)


SPLIT_DATE = "2018-01-01"
WEIGHTS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]


def _metrics_row(name, predictions, futures_pnl, edge_filter=False, macro_weight=None, ethanol_weight=None):
    if edge_filter:
        positions, _, _ = edge_filtered_positions(predictions, futures_pnl, quantile=0.50)
        label = name + " | edge-filtered"
    else:
        positions = model_predictions_to_positions(predictions, futures_pnl)
        label = name + " | unfiltered"

    bt, _ = backtest_positions(positions, futures_pnl, 0.0)
    metrics = split_performance(bt, SPLIT_DATE)
    return {
        "experiment": label,
        "macro_weight": macro_weight,
        "ethanol_weight": ethanol_weight,
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "full_pnl": metrics.loc["total_pnl", "full_period"],
        "max_drawdown": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
    }


def run_combined_external_experiment():
    data = load_train_set("train_set")
    feature_panels, futures_pnl = build_feature_panels(data)
    selected_predictions, _, _, _ = build_improved_model_signals(feature_panels, futures_pnl, SPLIT_DATE)

    macro_prices = _download_yfinance_prices(futures_pnl.index.min(), futures_pnl.index.max())
    macro_features = build_macro_feature_block(macro_prices, futures_pnl.index)
    macro_predictions, macro_coefficients = build_macro_predictions(macro_features, futures_pnl)

    ethanol = fetch_eia_ethanol()
    ethanol_features = build_ethanol_feature_panel(ethanol, futures_pnl.index)
    ethanol_predictions, ethanol_coefficients = build_ethanol_predictions(ethanol_features, futures_pnl)

    rows = []
    rows.append(_metrics_row("Current selected core + physical", selected_predictions, futures_pnl, False, 0.0, 0.0))
    rows.append(_metrics_row("Current selected core + physical", selected_predictions, futures_pnl, True, 0.0, 0.0))

    for macro_weight in WEIGHTS:
        for ethanol_weight in WEIGHTS:
            if macro_weight == 0.0 and ethanol_weight == 0.0:
                continue
            combined = (
                selected_predictions.fillna(0.0)
                + float(macro_weight) * macro_predictions.fillna(0.0)
                + float(ethanol_weight) * ethanol_predictions.fillna(0.0)
            )
            name = "Selected + macro w={} + ethanol w={}".format(macro_weight, ethanol_weight)
            rows.append(_metrics_row(name, combined, futures_pnl, False, macro_weight, ethanol_weight))
            rows.append(_metrics_row(name, combined, futures_pnl, True, macro_weight, ethanol_weight))

    results = pd.DataFrame(rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False])
    return {
        "macro_prices": macro_prices,
        "macro_features": macro_features,
        "macro_predictions": macro_predictions,
        "macro_coefficients": macro_coefficients,
        "ethanol": ethanol,
        "ethanol_features": ethanol_features,
        "ethanol_predictions": ethanol_predictions,
        "ethanol_coefficients": ethanol_coefficients,
        "results": results.reset_index(drop=True),
    }


if __name__ == "__main__":
    out = run_combined_external_experiment()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    print(out["results"].head(30).to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
