"""Optional experiment: FX, crude, and rates from yfinance.

This is intentionally separate from the main strategy. It downloads external
macro markets, converts them into lagged z-score features, and tests them as a
strongly regularized overlay on the selected Cargill-aware model.
"""

from __future__ import print_function

import numpy as np
import pandas as pd

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
MACRO_TICKERS = {
    "usd_index": "DX-Y.NYB",
    "crude_oil": "CL=F",
    "ten_year_yield": "^TNX",
    "short_rate": "^IRX",
    "eurusd": "EURUSD=X",
    "brlusd": "BRL=X",
}
OVERLAY_WEIGHTS = [0.25, 0.50, 1.00]


def _download_yfinance_prices(start, end):
    import yfinance as yf

    tickers = list(MACRO_TICKERS.values())
    raw = yf.download(
        tickers,
        start=str(start.date()),
        end=str((pd.Timestamp(end) + pd.DateOffset(days=7)).date()),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"]
        else:
            close = raw.xs(raw.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        close = raw.to_frame(tickers[0]) if isinstance(raw, pd.Series) else raw

    rename = {ticker: name for name, ticker in MACRO_TICKERS.items()}
    close = close.rename(columns=rename)
    close = close[[name for name in MACRO_TICKERS if name in close.columns]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def build_macro_feature_block(macro_prices, trading_index):
    aligned = macro_prices.reindex(trading_index).ffill()
    aligned = aligned.shift(1)
    returns = aligned.pct_change(fill_method=None)

    features = pd.DataFrame(index=trading_index)
    for column in aligned.columns:
        features[column + "_ret_5"] = rolling_zscore(aligned[column].pct_change(5, fill_method=None), 252, 60)
        features[column + "_ret_20"] = rolling_zscore(aligned[column].pct_change(20, fill_method=None), 252, 60)
        features[column + "_ret_60"] = rolling_zscore(aligned[column].pct_change(60, fill_method=None), 252, 80)
        features[column + "_vol_20"] = rolling_zscore(returns[column].rolling(20, min_periods=10).std(), 252, 60)

    for column in ["ten_year_yield", "short_rate", "usd_index", "crude_oil"]:
        if column in aligned.columns:
            features[column + "_level"] = rolling_zscore(aligned[column], 252, 60)

    if "ten_year_yield" in aligned.columns and "short_rate" in aligned.columns:
        features["yield_curve_10y_3m"] = rolling_zscore(aligned["ten_year_yield"] - aligned["short_rate"], 252, 60)

    return features.clip(lower=-5.0, upper=5.0)


def build_macro_predictions(macro_features, futures_pnl, split_date=SPLIT_DATE, alpha=1000.0):
    if macro_features.dropna(how="all").empty:
        raise RuntimeError("No yfinance macro data was downloaded; cannot build macro features.")

    train_mask = futures_pnl.index < pd.Timestamp(split_date)
    predictions = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float)
    coefficients = {}
    for commodity in COMMODITIES:
        target = futures_pnl[commodity].shift(-1).rolling(5, min_periods=5).sum().shift(-4)
        pred, coef = fit_ridge_predict(macro_features, target, train_mask, alpha=alpha)
        predictions[commodity] = pred
        coefficients[commodity] = coef
    return predictions.fillna(0.0), pd.DataFrame(coefficients)


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


def run_macro_yfinance_experiment():
    data = load_train_set("train_set")
    feature_panels, futures_pnl = build_feature_panels(data)
    selected_predictions, _, _, _ = build_improved_model_signals(feature_panels, futures_pnl, SPLIT_DATE)

    macro_prices = _download_yfinance_prices(futures_pnl.index.min(), futures_pnl.index.max())
    macro_features = build_macro_feature_block(macro_prices, futures_pnl.index)
    macro_predictions, macro_coefficients = build_macro_predictions(macro_features, futures_pnl)

    rows = []
    rows.append(_metrics_row("Current selected core + physical", selected_predictions, futures_pnl, False))
    rows.append(_metrics_row("Current selected core + physical", selected_predictions, futures_pnl, True))
    rows.append(_metrics_row("Macro FX/crude/rates only", macro_predictions, futures_pnl, False))
    rows.append(_metrics_row("Macro FX/crude/rates only", macro_predictions, futures_pnl, True))
    for weight in OVERLAY_WEIGHTS:
        combined = selected_predictions.fillna(0.0) + float(weight) * macro_predictions.fillna(0.0)
        rows.append(_metrics_row("Current selected + macro overlay w=" + str(weight), combined, futures_pnl, False))
        rows.append(_metrics_row("Current selected + macro overlay w=" + str(weight), combined, futures_pnl, True))

    results = pd.DataFrame(rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False])
    return {
        "macro_prices": macro_prices,
        "macro_features": macro_features,
        "macro_predictions": macro_predictions,
        "macro_coefficients": macro_coefficients,
        "results": results.reset_index(drop=True),
    }


if __name__ == "__main__":
    out = run_macro_yfinance_experiment()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
