"""Optional EIA ethanol production/stocks overlay experiment.

Requires an EIA API key via EIA_API_KEY/EIA_KEY or the natgas config file.
The key is never stored in this file or printed by the script.
"""

from __future__ import print_function

import os
import re

import pandas as pd
import requests

from grain_futures_strategy import (
    COMMODITIES,
    backtest_positions,
    build_feature_panels,
    build_improved_model_signals,
    edge_filtered_positions,
    fit_ridge_predict,
    load_train_set,
    model_predictions_to_positions,
    rolling_zscore,
    split_performance,
)


SPLIT_DATE = "2018-01-01"
OVERLAY_WEIGHTS = [0.25, 0.50, 1.00]
NATGAS_CONFIG_PATH = "/Users/phuongpham/natgas_trading/live/config.yaml"
EIA_SERIES = {
    "ethanol_production": "PET.W_EPOOXE_YOP_NUS_MBBLD.W",
    "ethanol_stocks": "PET.W_EPOOXE_SAE_NUS_MBBL.W",
}


def read_eia_api_key(config_path=NATGAS_CONFIG_PATH):
    key = os.environ.get("EIA_API_KEY") or os.environ.get("EIA_KEY")
    if key:
        return key.strip()
    if config_path and os.path.exists(config_path):
        text = open(config_path).read()
        match = re.search(r"eia_api_key:\s*[\"']?([^\"'\n]+)", text)
        if match:
            return match.group(1).strip()
    raise RuntimeError("Missing EIA API key. Set EIA_API_KEY or provide natgas config.")


def fetch_eia_series(series_id, api_key):
    url = "https://api.eia.gov/v2/seriesid/" + series_id
    response = requests.get(url, params={"api_key": api_key}, timeout=60)
    if response.status_code != 200:
        raise RuntimeError("EIA request failed for {}".format(series_id))
    payload = response.json()
    rows = ((payload.get("response") or {}).get("data")) or []
    if not rows:
        raise RuntimeError("EIA returned no rows for {}".format(series_id))
    frame = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["period"], errors="coerce"),
            "value": pd.to_numeric(frame["value"], errors="coerce"),
        }
    ).dropna()
    return out.sort_values("date").drop_duplicates("date", keep="last").set_index("date")


def fetch_eia_ethanol(api_key=None):
    api_key = api_key or read_eia_api_key()
    frames = []
    for name, series_id in EIA_SERIES.items():
        frame = fetch_eia_series(series_id, api_key).rename(columns={"value": name})
        frames.append(frame)
    out = pd.concat(frames, axis=1).sort_index()
    return out


def build_ethanol_feature_panel(ethanol, trading_index):
    """Build lagged weekly ethanol features aligned to grain trading dates."""
    available = ethanol.copy()
    available.index = available.index + pd.DateOffset(days=7)
    aligned = available.reindex(trading_index).ffill().shift(1)

    features = pd.DataFrame(index=trading_index)
    features["ethanol_production_level"] = rolling_zscore(aligned["ethanol_production"], 156, 40)
    features["ethanol_production_change_4w"] = rolling_zscore(aligned["ethanol_production"].diff(20), 156, 40)
    features["ethanol_stocks_level"] = rolling_zscore(aligned["ethanol_stocks"], 156, 40)
    features["ethanol_stocks_change_4w"] = rolling_zscore(aligned["ethanol_stocks"].diff(20), 156, 40)
    ratio = aligned["ethanol_production"] / aligned["ethanol_stocks"].replace(0.0, pd.NA)
    features["ethanol_prod_to_stocks"] = rolling_zscore(ratio, 156, 40)
    pressure = aligned["ethanol_production"].diff(20) - aligned["ethanol_stocks"].diff(20)
    features["ethanol_demand_pressure"] = rolling_zscore(pressure, 156, 40)
    return features.clip(lower=-5.0, upper=5.0)


def build_ethanol_predictions(ethanol_features, futures_pnl, split_date=SPLIT_DATE, alpha=1000.0):
    """Predict only CORN from ethanol features; other commodities get zero overlay."""
    train_mask = futures_pnl.index < pd.Timestamp(split_date)
    predictions = pd.DataFrame(0.0, index=futures_pnl.index, columns=COMMODITIES, dtype=float)
    target = futures_pnl["CORN"].shift(-1).rolling(5, min_periods=5).sum().shift(-4)
    pred, coef = fit_ridge_predict(ethanol_features, target, train_mask, alpha=alpha)
    predictions["CORN"] = pred.fillna(0.0)
    return predictions, coef


def _metrics_row(name, predictions, futures_pnl, edge_filter=False):
    if edge_filter:
        positions, _, _ = edge_filtered_positions(predictions, futures_pnl, quantile=0.50)
        name = name + " | edge-filtered"
    else:
        positions = model_predictions_to_positions(predictions, futures_pnl)
        name = name + " | unfiltered"
    bt, _ = backtest_positions(positions, futures_pnl, 0.0)
    metrics = split_performance(bt, SPLIT_DATE)
    return {
        "experiment": name,
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "full_pnl": metrics.loc["total_pnl", "full_period"],
        "max_drawdown": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
    }


def run_eia_ethanol_experiment(api_key=None):
    data = load_train_set("train_set")
    feature_panels, futures_pnl = build_feature_panels(data)
    selected_predictions, _, _, _ = build_improved_model_signals(feature_panels, futures_pnl, SPLIT_DATE)

    ethanol = fetch_eia_ethanol(api_key=api_key)
    ethanol_features = build_ethanol_feature_panel(ethanol, futures_pnl.index)
    ethanol_predictions, ethanol_coefficients = build_ethanol_predictions(ethanol_features, futures_pnl)

    rows = []
    rows.append(_metrics_row("Current selected core + physical", selected_predictions, futures_pnl, False))
    rows.append(_metrics_row("Current selected core + physical", selected_predictions, futures_pnl, True))
    rows.append(_metrics_row("EIA ethanol only", ethanol_predictions, futures_pnl, False))
    rows.append(_metrics_row("EIA ethanol only", ethanol_predictions, futures_pnl, True))
    for weight in OVERLAY_WEIGHTS:
        combined = selected_predictions.fillna(0.0) + float(weight) * ethanol_predictions.fillna(0.0)
        rows.append(_metrics_row("Current selected + EIA ethanol overlay w=" + str(weight), combined, futures_pnl, False))
        rows.append(_metrics_row("Current selected + EIA ethanol overlay w=" + str(weight), combined, futures_pnl, True))

    results = pd.DataFrame(rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False])
    return {
        "ethanol": ethanol,
        "ethanol_features": ethanol_features,
        "ethanol_predictions": ethanol_predictions,
        "ethanol_coefficients": ethanol_coefficients,
        "results": results.reset_index(drop=True),
    }


if __name__ == "__main__":
    out = run_eia_ethanol_experiment()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
