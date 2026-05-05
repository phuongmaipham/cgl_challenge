"""Fixed-weight external overlay ablation.

Tests equal-weight core/macro/EIA/weather sleeves, one-block removals, and
core-heavy fixed budgets. No weights are chosen from a search objective.
"""

from __future__ import print_function

import pandas as pd

from grain_futures_strategy import (
    backtest_positions,
    build_feature_panels,
    build_improved_model_signals,
    edge_filtered_positions,
    load_train_set,
    split_performance,
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
from meteostat_experiment import (
    build_meteostat_feature_panels,
    build_meteostat_predictions,
    fetch_meteostat_weather,
)


SPLIT_DATE = "2018-01-01"


def _metrics(name, positions, pnl, weights):
    bt, _ = backtest_positions(positions, pnl, 0.0)
    perf = split_performance(bt, SPLIT_DATE)
    row = {
        "experiment": name,
        "is_sharpe": perf.loc["sharpe", "in_sample"],
        "oos_sharpe": perf.loc["sharpe", "out_of_sample"],
        "oos_pnl": perf.loc["total_pnl", "out_of_sample"],
        "full_sharpe": perf.loc["sharpe", "full_period"],
        "full_pnl": perf.loc["total_pnl", "full_period"],
        "max_drawdown": perf.loc["max_drawdown", "full_period"],
        "turnover": perf.loc["avg_daily_turnover", "full_period"],
    }
    for key in ["core", "macro", "eia", "weather"]:
        row["w_" + key] = float(weights.get(key, 0.0))
    return row


def _combine(sleeves, weights):
    out = None
    for name, weight in weights.items():
        if float(weight) == 0.0:
            continue
        weighted = sleeves[name] * float(weight)
        out = weighted if out is None else out.add(weighted, fill_value=0.0)
    return out.fillna(0.0)


def run_external_equal_weight_ablation():
    data = load_train_set("train_set")
    feature_panels, futures_pnl = build_feature_panels(data)
    core_predictions, _, _, _ = build_improved_model_signals(feature_panels, futures_pnl, SPLIT_DATE)

    macro_prices = _download_yfinance_prices(futures_pnl.index.min(), futures_pnl.index.max())
    macro_features = build_macro_feature_block(macro_prices, futures_pnl.index)
    macro_predictions, macro_coefficients = build_macro_predictions(macro_features, futures_pnl)

    ethanol = fetch_eia_ethanol()
    ethanol_features = build_ethanol_feature_panel(ethanol, futures_pnl.index)
    eia_predictions, eia_coefficients = build_ethanol_predictions(ethanol_features, futures_pnl)

    weather = fetch_meteostat_weather(futures_pnl.index.min(), futures_pnl.index.max())
    weather_panels = build_meteostat_feature_panels(weather, futures_pnl.index, mode="commodity_basic")
    weather_predictions, weather_coefficients = build_meteostat_predictions(
        weather_panels,
        futures_pnl,
        alpha=1000.0,
        horizon=20,
    )

    sleeves = {}
    for name, predictions in [
        ("core", core_predictions),
        ("macro", macro_predictions),
        ("eia", eia_predictions),
        ("weather", weather_predictions),
    ]:
        positions, _, _ = edge_filtered_positions(predictions, futures_pnl, quantile=0.50)
        sleeves[name] = positions

    rules = [
        ("Core only", {"core": 1.0}),
        ("Equal all sleeves | core/macro/EIA/weather 25% each", {"core": 0.25, "macro": 0.25, "eia": 0.25, "weather": 0.25}),
        ("Equal minus Macro | core/EIA/weather 33.3% each", {"core": 1.0 / 3.0, "eia": 1.0 / 3.0, "weather": 1.0 / 3.0}),
        ("Equal minus EIA | core/macro/weather 33.3% each", {"core": 1.0 / 3.0, "macro": 1.0 / 3.0, "weather": 1.0 / 3.0}),
        ("Equal minus Weather | core/macro/EIA 33.3% each", {"core": 1.0 / 3.0, "macro": 1.0 / 3.0, "eia": 1.0 / 3.0}),
        ("Core-heavy 70/10/10/10", {"core": 0.70, "macro": 0.10, "eia": 0.10, "weather": 0.10}),
        ("Core-heavy 80/equal external 20", {"core": 0.80, "macro": 1.0 / 15.0, "eia": 1.0 / 15.0, "weather": 1.0 / 15.0}),
        ("Core-heavy 90/equal external 10", {"core": 0.90, "macro": 1.0 / 30.0, "eia": 1.0 / 30.0, "weather": 1.0 / 30.0}),
        ("Core-heavy no EIA | core 80 + macro/weather 10/10", {"core": 0.80, "macro": 0.10, "weather": 0.10}),
        ("Core-heavy no Weather | core 80 + macro/EIA 10/10", {"core": 0.80, "macro": 0.10, "eia": 0.10}),
        ("Core-heavy no Macro | core 80 + EIA/weather 10/10", {"core": 0.80, "eia": 0.10, "weather": 0.10}),
    ]

    rows = []
    for label, weights in rules:
        rows.append(_metrics(label, _combine(sleeves, weights), futures_pnl, weights))

    results = pd.DataFrame(rows).sort_values(["oos_sharpe", "max_drawdown"], ascending=[False, False])
    return {
        "results": results.reset_index(drop=True),
        "macro_coefficients": macro_coefficients,
        "eia_coefficients": eia_coefficients,
        "weather_coefficients": weather_coefficients,
    }


if __name__ == "__main__":
    out = run_external_equal_weight_ablation()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 24)
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
