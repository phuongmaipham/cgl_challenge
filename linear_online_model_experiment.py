"""Walk-forward OLS/Ridge/RLS/Kalman tests for corn and soybeans.

The goal is to test whether simple fitted linear models improve the
commodity-specific sleeves without using future information.

Controls:
- no full-sample fitting;
- no OOS hyperparameter search;
- expanding walk-forward OLS/Ridge with fixed alpha;
- online RLS/Kalman updates with fixed parameters;
- predictions at date t use only data available before date t;
- prediction signals are converted to rolling z-scores using lagged history.
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
from ic_threshold_sleeve_experiment import (
    TEST_START,
    TRAIN_END,
    _clean_signal,
    _fetch_external_signals,
    _format_table,
    _given_signal_universe,
    _positions_from_signal,
)


COMMODITIES = ["CORN", "SOYABEAN"]
MIN_TRAIN_OBS = 504
REFIT_EVERY = 21
RIDGE_ALPHA = 100.0
RLS_LAMBDA = 0.995
RLS_DELTA = 100.0
KALMAN_Q = 1.0e-5


def _feature_matrix(feature_panels, futures_pnl, commodity):
    given = _given_signal_universe(feature_panels, commodity)
    external, errors, _ = _fetch_external_signals(commodity, futures_pnl)
    signals = dict(given)
    signals.update(external)
    cleaned = {
        name: _clean_signal(signal, futures_pnl.index)
        for name, signal in signals.items()
    }
    x = pd.DataFrame(cleaned, index=futures_pnl.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Drop constant or near-empty columns so linear algebra stays well behaved.
    keep = [col for col in x.columns if x[col].std() > 1.0e-8 and x[col].abs().sum() > 0.0]
    return x[keep], errors


def _standardize_train_apply(x_train, x_row):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0).replace(0.0, np.nan).fillna(1.0)
    return (x_train - mean) / std, (x_row - mean) / std


def _fit_linear_beta(x_train, y_train, alpha=0.0):
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    xtx = x.T.dot(x)
    if alpha > 0.0:
        penalty = np.eye(xtx.shape[0]) * float(alpha)
        penalty[0, 0] = 0.0
        xtx = xtx + penalty
    try:
        return np.linalg.solve(xtx, x.T.dot(y))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx).dot(x.T).dot(y)


def _expanding_predictions(x, y, alpha=0.0):
    preds = pd.Series(np.nan, index=x.index)
    beta = None
    last_fit = None
    for i, date in enumerate(x.index):
        train_mask = (x.index < date) & y.notna()
        if train_mask.sum() < MIN_TRAIN_OBS:
            continue
        if beta is None or last_fit is None or (i - last_fit) >= REFIT_EVERY:
            x_train_raw = x.loc[train_mask]
            y_train = y.loc[train_mask]
            x_train, x_row = _standardize_train_apply(x_train_raw, x.loc[date])
            beta = _fit_linear_beta(x_train, y_train, alpha=alpha)
            last_fit = i
            preds.loc[date] = np.r_[1.0, np.asarray(x_row, dtype=float)].dot(beta)
        else:
            x_train_raw = x.loc[train_mask]
            _, x_row = _standardize_train_apply(x_train_raw, x.loc[date])
            preds.loc[date] = np.r_[1.0, np.asarray(x_row, dtype=float)].dot(beta)
    return preds


def _rls_predictions(x, y, lambda_=RLS_LAMBDA, delta=RLS_DELTA):
    columns = list(x.columns)
    beta = np.zeros(len(columns) + 1)
    p = np.eye(len(beta)) * float(delta)
    preds = pd.Series(np.nan, index=x.index)
    mean = pd.Series(0.0, index=columns)
    var = pd.Series(1.0, index=columns)
    n = 0
    for date in x.index:
        row = x.loc[date]
        if n > 20:
            std = np.sqrt(var.clip(lower=1.0e-8))
            z = ((row - mean) / std).clip(lower=-5.0, upper=5.0)
            phi = np.r_[1.0, np.asarray(z, dtype=float)]
            preds.loc[date] = phi.dot(beta)
        y_value = y.loc[date]
        if pd.notnull(y_value):
            n += 1
            old_mean = mean.copy()
            mean = mean + (row - mean) / float(n)
            var = ((n - 2.0) / max(n - 1.0, 1.0)) * var + ((row - old_mean) * (row - mean)) / max(n - 1.0, 1.0)
            if n > MIN_TRAIN_OBS:
                std = np.sqrt(var.clip(lower=1.0e-8))
                z = ((row - mean) / std).clip(lower=-5.0, upper=5.0)
                phi = np.r_[1.0, np.asarray(z, dtype=float)]
                denom = float(lambda_ + phi.dot(p).dot(phi))
                gain = p.dot(phi) / denom
                err = float(y_value - phi.dot(beta))
                beta = beta + gain * err
                p = (p - np.outer(gain, phi).dot(p)) / float(lambda_)
    return preds


def _kalman_predictions(x, y, q=KALMAN_Q):
    columns = list(x.columns)
    beta = np.zeros(len(columns) + 1)
    p = np.eye(len(beta)) * 10.0
    preds = pd.Series(np.nan, index=x.index)
    mean = pd.Series(0.0, index=columns)
    var = pd.Series(1.0, index=columns)
    target_var = 1.0
    n = 0
    for date in x.index:
        row = x.loc[date]
        if n > MIN_TRAIN_OBS:
            std = np.sqrt(var.clip(lower=1.0e-8))
            z = ((row - mean) / std).clip(lower=-5.0, upper=5.0)
            phi = np.r_[1.0, np.asarray(z, dtype=float)]
            preds.loc[date] = phi.dot(beta)
        y_value = y.loc[date]
        if pd.notnull(y_value):
            n += 1
            old_mean = mean.copy()
            mean = mean + (row - mean) / float(n)
            var = ((n - 2.0) / max(n - 1.0, 1.0)) * var + ((row - old_mean) * (row - mean)) / max(n - 1.0, 1.0)
            target_var = target_var + (float(y_value) ** 2 - target_var) / float(n)
            if n > MIN_TRAIN_OBS:
                std = np.sqrt(var.clip(lower=1.0e-8))
                z = ((row - mean) / std).clip(lower=-5.0, upper=5.0)
                phi = np.r_[1.0, np.asarray(z, dtype=float)]
                p = p + np.eye(len(beta)) * float(q)
                r = max(target_var, 1.0)
                innovation_var = float(phi.dot(p).dot(phi) + r)
                gain = p.dot(phi) / innovation_var
                err = float(y_value - phi.dot(beta))
                beta = beta + gain * err
                p = p - np.outer(gain, phi).dot(p)
    return preds


def _prediction_to_signal(pred):
    pred = pred.replace([np.inf, -np.inf], np.nan)
    mean = pred.rolling(252, min_periods=60).mean().shift(1)
    std = pred.rolling(252, min_periods=60).std().shift(1).replace(0.0, np.nan)
    signal = ((pred - mean) / std).clip(lower=-5.0, upper=5.0).fillna(0.0)
    return signal


def _metrics(bt):
    table = split_performance(bt, TEST_START)
    train_val = split_performance(bt.loc[bt.index < TEST_START], TRAIN_END)
    return {
        "train_sharpe": train_val.loc["sharpe", "in_sample"],
        "validation_sharpe": train_val.loc["sharpe", "out_of_sample"],
        "validation_max_drawdown": train_val.loc["max_drawdown", "out_of_sample"],
        "test_sharpe": table.loc["sharpe", "out_of_sample"],
        "test_pnl": table.loc["total_pnl", "out_of_sample"],
        "test_max_drawdown": table.loc["max_drawdown", "out_of_sample"],
        "full_sharpe": table.loc["sharpe", "full_period"],
        "full_pnl": table.loc["total_pnl", "full_period"],
        "max_drawdown": table.loc["max_drawdown", "full_period"],
        "turnover": table.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": table.loc["avg_gross_exposure", "full_period"],
    }


def _evaluate_model_signal(model, commodity, signal, futures_pnl, mode="long_short"):
    positions = _positions_from_signal(signal, futures_pnl, commodity, mode=mode)
    rows = []
    for cost_adjusted in [False, True]:
        if cost_adjusted:
            bt, _ = backtest_positions_with_costs(positions, futures_pnl[[commodity]], 8.75, 0.05)
        else:
            bt, _ = backtest_positions(positions, futures_pnl[[commodity]], 0.0)
        row = {"commodity": commodity, "model": model, "mode": mode, "cost_adjusted": cost_adjusted}
        row.update(_metrics(bt))
        rows.append(row)
    return rows


def _run_one(commodity, feature_panels, futures_pnl):
    x, errors = _feature_matrix(feature_panels, futures_pnl, commodity)
    y = futures_pnl[commodity].shift(-1)
    raw_predictions = {
        "ols_expanding": _expanding_predictions(x, y, alpha=0.0),
        "ridge_expanding_alpha100": _expanding_predictions(x, y, alpha=RIDGE_ALPHA),
        "rls_lambda0995": _rls_predictions(x, y),
        "kalman_dynamic_linear": _kalman_predictions(x, y),
    }
    rows = []
    signals = {}
    for model, pred in raw_predictions.items():
        signal = _prediction_to_signal(pred)
        signals[model] = signal
        for mode in ["long_only", "long_short"]:
            rows.extend(_evaluate_model_signal(model, commodity, signal, futures_pnl, mode=mode))
    results = pd.DataFrame(rows).sort_values(["cost_adjusted", "test_sharpe"], ascending=[True, False]).reset_index(drop=True)
    return {
        "commodity": commodity,
        "errors": errors,
        "features": list(x.columns),
        "signals": signals,
        "raw_predictions": raw_predictions,
        "results": results,
    }


def _write_log(outputs, path="notes/linear_online_models_corn_soybean.txt"):
    lines = []
    lines.append("OLS / Ridge / RLS / Kalman commodity-sleeve test")
    lines.append("Date: 2026-05-02")
    lines.append("")
    lines.append("Controls")
    lines.append("--------")
    lines.append("- Expanding OLS and Ridge refit every {} trading days.".format(REFIT_EVERY))
    lines.append("- Ridge alpha is fixed at {:.1f}; it is not tuned on OOS.".format(RIDGE_ALPHA))
    lines.append("- RLS uses fixed lambda {:.3f} and delta {:.1f}.".format(RLS_LAMBDA, RLS_DELTA))
    lines.append("- Kalman uses fixed process noise q {:.1e}.".format(KALMAN_Q))
    lines.append("- Prediction at date t uses only observations before or at date t, then positions trade next-day PnL via the standard backtester.")
    lines.append("- Predictions are converted to lagged rolling z-score signals before volatility targeting.")
    lines.append("")
    for label, out in outputs.items():
        lines.append("")
        lines.append("{} results".format(label))
        lines.append("=" * (len(label) + 8))
        lines.append("External data warnings: {}".format("; ".join(out["errors"]) if out["errors"] else "none"))
        lines.append("Feature count: {}".format(len(out["features"])))
        lines.append("Features: {}".format(", ".join(out["features"])))
        lines.append("")
        cols = [
            "model",
            "mode",
            "cost_adjusted",
            "train_sharpe",
            "validation_sharpe",
            "validation_max_drawdown",
            "test_sharpe",
            "test_pnl",
            "test_max_drawdown",
            "full_sharpe",
            "max_drawdown",
            "turnover",
            "avg_gross_exposure",
        ]
        lines.append("All model rows")
        lines.append("--------------")
        lines.append(_format_table(out["results"][cols]))
        lines.append("")
        cost = out["results"].loc[out["results"]["cost_adjusted"]].copy()
        best = cost.sort_values("test_sharpe", ascending=False).iloc[0]
        lines.append("Best cost-adjusted OOS row")
        lines.append("--------------------------")
        lines.append(
            "- {} {}: OOS Sharpe {:.3f}, PnL {:.3f}, DD {:.3f}, full Sharpe {:.3f}, full DD {:.3f}".format(
                best["model"],
                best["mode"],
                best["test_sharpe"],
                best["test_pnl"],
                best["test_max_drawdown"],
                best["full_sharpe"],
                best["max_drawdown"],
            )
        )
        lines.append("")
        lines.append("Overfit read")
        lines.append("------------")
        if best["test_sharpe"] > 0.5 and best["validation_sharpe"] > 0.0:
            lines.append("- Model has positive validation and OOS behavior, but still carries coefficient-estimation risk.")
        elif best["test_sharpe"] > 0.5:
            lines.append("- Model looks good OOS but did not have clean validation support; treat as research luck until further holdout.")
        else:
            lines.append("- No model is strong enough to replace the simpler fixed-family sleeve.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def run_linear_online_model_experiment(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    outputs = {
        "Corn": _run_one("CORN", feature_panels, futures_pnl),
        "Soybeans": _run_one("SOYABEAN", feature_panels, futures_pnl),
    }
    _write_log(outputs)
    return outputs


if __name__ == "__main__":
    out = run_linear_online_model_experiment()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for label, result in out.items():
        print("\n", label)
        print("errors:", result["errors"])
        print(result["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
