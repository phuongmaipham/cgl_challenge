"""Lag-aware grain futures research utilities used by the standalone notebook."""

from __future__ import print_function

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from grain_backtest_core import (
    backtest_positions_with_costs,
    build_feature_panels,
    performance_metrics,
    split_performance,
)


def _load_config():
    with open(Path(__file__).resolve().parent / "grain_futures_strategy_config.json") as f:
        config = json.load(f)
    for case in config["COST_CASES"]:  # JSON has no infinity literal
        if case["margin_budget"] is None:
            case["margin_budget"] = np.inf
    return config


_CONFIG = _load_config()
COMMODITIES = _CONFIG["COMMODITIES"]
OUTRIGHT_CORE_FEATURES = _CONFIG["OUTRIGHT_CORE_FEATURES"]
OUTRIGHT_PHYSICAL_FEATURES = _CONFIG["OUTRIGHT_PHYSICAL_FEATURES"]
COST_CASES = _CONFIG["COST_CASES"]
FINAL_BLEND_WEIGHTS = _CONFIG["FINAL_BLEND_WEIGHTS"]
FINAL_OPPORTUNITY_QUANTILES = _CONFIG["FINAL_OPPORTUNITY_QUANTILES"]

_TRAIN_SET_FILES = {
    "adj1": "train_adjPrices1.csv", "adj2": "train_adjPrices2.csv",
    "unadj1": "train_unadjPrices1.csv", "unadj2": "train_unadjPrices2.csv",
    "cot_mm": "train_cot_mm.csv", "cot_pm_oi": "train_cot_pm_oi.csv",
    "inventories": "train_inventories.csv", "receipts": "train_receipts.csv",
    "cgl_inv": "train_cgl_inv.csv", "cgl_crush": "train_cgl_crush.csv",
}


def load_train_set(data_dir="train_set"):
    """Load all expected training CSVs into a dict of DataFrames."""
    return {k: pd.read_csv(os.path.join(data_dir, fn), index_col=0, parse_dates=True)
                .sort_index().apply(pd.to_numeric, errors="coerce")
            for k, fn in _TRAIN_SET_FILES.items()}


def fit_ridge_predict(features, target, train_mask, alpha=10.0):
    """Fit a standardised Ridge model and return (predictions, scaled coefficients)."""
    valid = features.notnull().all(axis=1) & target.notnull()
    train = valid & train_mask
    if int(train.sum()) < max(40, features.shape[1] * 3):
        return pd.Series(np.nan, index=features.index), pd.Series(np.nan, index=features.columns)
    x_train = features.loc[train].values.astype(float)
    y_train = target.loc[train].values.astype(float)
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    y_mean = y_train.mean()
    x_scaled = (x_train - x_mean) / x_std
    beta = np.linalg.solve(x_scaled.T @ x_scaled + alpha * np.eye(x_scaled.shape[1]),
                           x_scaled.T @ (y_train - y_mean))
    all_valid = features.notnull().all(axis=1)
    pred = pd.Series(np.nan, index=features.index)
    pred.loc[all_valid] = y_mean + ((features.loc[all_valid].values.astype(float) - x_mean) / x_std) @ beta
    return pred, pd.Series(beta / x_std, index=features.columns)


def _forward_target(pnl, h):
    return pnl.shift(-1) if int(h) <= 1 else pnl.shift(-1).rolling(int(h), min_periods=int(h)).sum().shift(-(int(h) - 1))


def _per_commodity(feature_panels, futures_pnl, train_mask, horizon, alpha, columns=None):
    preds, coefs = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float), {}
    for c in COMMODITIES:
        feats = feature_panels[c] if columns is None else feature_panels[c][columns]
        preds[c], coefs[c] = fit_ridge_predict(feats, _forward_target(futures_pnl[c], horizon), train_mask, alpha=alpha)
    return preds, pd.DataFrame(coefs)


def build_improved_model_signals(feature_panels, futures_pnl, split_date="2018-01-01"):
    """Two-block 5-day Ridge: core + physical overlay."""
    train = futures_pnl.index < pd.Timestamp(split_date)
    cp, cc = _per_commodity(feature_panels, futures_pnl, train, 5, 25.0, OUTRIGHT_CORE_FEATURES)
    pp, pc = _per_commodity(feature_panels, futures_pnl, train, 5, 1000.0, OUTRIGHT_PHYSICAL_FEATURES)
    return cp.fillna(0.0) + pp.fillna(0.0), {"core": cc, "physical": pc}


def build_walk_forward_model_signals(feature_panels, futures_pnl, start_date="2014-01-01",
                                     retrain_frequency="A-DEC", min_train_days=756, horizon=20):
    """Annual walk-forward two-block Ridge."""
    index = futures_pnl.index
    schedule = sorted(set(index[index >= d][0] for d in
                          pd.date_range(pd.Timestamp(start_date), index.max(), freq=retrain_frequency)
                          if len(index[index >= d]) > 0))
    preds = pd.DataFrame(0.0, index=index, columns=COMMODITIES)
    for i, train_end in enumerate(schedule):
        next_end = schedule[i + 1] if i + 1 < len(schedule) else index.max() + pd.DateOffset(days=1)
        apply_mask, train_mask = (index >= train_end) & (index < next_end), index < train_end
        if int(train_mask.sum()) < int(min_train_days):
            continue
        for c in COMMODITIES:
            target = _forward_target(futures_pnl[c], int(horizon))
            for cols, alpha in [(OUTRIGHT_CORE_FEATURES, 25.0), (OUTRIGHT_PHYSICAL_FEATURES, 1000.0)]:
                pred, _ = fit_ridge_predict(feature_panels[c][cols], target, train_mask, alpha=alpha)
                preds.loc[apply_mask, c] += pred.loc[apply_mask].fillna(0.0)
    return preds


def model_predictions_to_positions(predictions, futures_pnl, gross_lots=1.0):
    """Convert predictions to market-neutral cross-sectional positions."""
    risk = futures_pnl.rolling(60, min_periods=20).std().shift(1)
    risk_adj = predictions / risk.replace(0.0, np.nan)
    demeaned = risk_adj.sub(risk_adj.mean(axis=1), axis=0)
    return (demeaned.div(demeaned.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
            * float(gross_lots)).fillna(0.0).clip(-1.0, 1.0)


def edge_filtered_positions(predictions, futures_pnl, quantile=0.50, min_periods=252, gross_lots=1.0):
    """Trade only when cross-sectional prediction-edge is above its expanding quantile."""
    base = model_predictions_to_positions(predictions, futures_pnl, gross_lots)
    edge = (predictions / futures_pnl.rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)).std(axis=1)
    threshold = edge.expanding(min_periods=int(min_periods)).quantile(float(quantile)).shift(1)
    return base.mul((edge > threshold).astype(float).reindex(base.index).fillna(0.0), axis=0)


def backtest_positions(positions, futures_pnl, cost_per_lot=0.0):
    """Backtest close-to-close PnL from positions known at prior close (no margin)."""
    positions = positions.reindex(futures_pnl.index).fillna(0.0)
    held = positions.shift(1).fillna(0.0)
    turnover = positions.diff().abs().fillna(0.0)
    net = held * futures_pnl - turnover * float(cost_per_lot)
    out = pd.DataFrame({
        "net_pnl": net.sum(axis=1), "turnover": turnover.sum(axis=1),
        "gross_exposure": positions.abs().sum(axis=1),
        "held_gross_exposure": held.abs().sum(axis=1),
    }, index=futures_pnl.index)
    out["cum_pnl"] = out["net_pnl"].cumsum()
    return out


def _hold(positions, n_days, stagger):
    """Skip-rebalance (stagger=False) or staggered (stagger=True) hold transformation."""
    n = int(n_days)
    if n <= 1:
        return positions.copy()
    sleeves = []
    for offset in (range(n) if stagger else [0]):
        s = positions.copy() * 0.0
        for i in range(offset, len(positions.index), n):
            s.iloc[i:min(i + n, len(positions.index))] = positions.iloc[i]
        sleeves.append(s)
    return sum(sleeves) / float(n) if stagger else sleeves[0]


def staggered_hold_positions(positions, n_days):
    return _hold(positions, n_days, stagger=True)


def skip_rebalance_positions(positions, n_days):
    return _hold(positions, n_days, stagger=False)


_HOLDING_COLS = [
    ("is_sharpe", "sharpe", "in_sample"),
    ("oos_sharpe", "sharpe", "out_of_sample"),
    ("full_sharpe", "sharpe", "full_period"),
    ("oos_pnl", "total_pnl", "out_of_sample"),
    ("max_drawdown", "max_drawdown", "full_period"),
    ("turnover", "avg_daily_turnover", "full_period"),
]
_COST_COLS = [
    ("is_sharpe", "sharpe", "in_sample"),
    ("oos_sharpe", "sharpe", "out_of_sample"),
    ("oos_pnl", "total_pnl", "out_of_sample"),
    ("max_dd", "max_drawdown", "full_period"),
    ("turnover", "avg_daily_turnover", "full_period"),
]


def _split_cols(bt, split_date, cols):
    m = split_performance(bt, split_date)
    return {alias: m.loc[metric, window] for alias, metric, window in cols}


def multi_condition_filter_positions(predictions, futures_pnl, feature_panels,
                                     prediction_quantile=0.40, curve_quantile=0.40, momentum_quantile=0.40):
    """Trade only when prediction, curve, and momentum dispersion are all high."""
    risk = futures_pnl.rolling(60, min_periods=20).std().shift(1)
    pred_d = (predictions / risk.replace(0.0, np.nan)).std(axis=1)
    curve_d = pd.DataFrame({c: feature_panels[c]["curve_spread"] + feature_panels[c]["curve_ratio"]
                            for c in COMMODITIES}).std(axis=1) / 2.0
    mom_d = pd.DataFrame({c: feature_panels[c]["mom_60"] for c in COMMODITIES}).std(axis=1)

    def _active(score, q):
        threshold = score.expanding(min_periods=252).quantile(float(q)).shift(1)
        return (score > threshold).astype(float).fillna(0.0)

    active = _active(pred_d, prediction_quantile) * _active(curve_d, curve_quantile) * _active(mom_d, momentum_quantile)
    return model_predictions_to_positions(predictions, futures_pnl).mul(active, axis=0), active


def run_research_pipeline(data_dir="train_set", split_date="2018-01-01", alpha=25.0, cost_per_lot=0.0):
    """Build features, fit broad/two-block/walk-forward Ridge models, backtest and split."""
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    train = futures_pnl.index < pd.Timestamp(split_date)

    broad_preds, _ = _per_commodity(feature_panels, futures_pnl, train, 1, alpha)
    broad_bt = backtest_positions(model_predictions_to_positions(broad_preds, futures_pnl), futures_pnl, cost_per_lot)

    predictions, coefficients = build_improved_model_signals(feature_panels, futures_pnl, split_date)
    unfiltered_positions = model_predictions_to_positions(predictions, futures_pnl)
    model_positions = edge_filtered_positions(predictions, futures_pnl, quantile=0.50)
    model_bt = backtest_positions(model_positions, futures_pnl, cost_per_lot)

    wf_predictions = build_walk_forward_model_signals(feature_panels, futures_pnl)
    wf_positions = model_predictions_to_positions(wf_predictions, futures_pnl)
    wf_bt = backtest_positions(wf_positions, futures_pnl, cost_per_lot)

    return {
        "summary": pd.DataFrame([
            {"dataset": k, "rows": len(d), "columns": len(d.columns),
             "start": d.index.min(), "end": d.index.max(),
             "missing_cells": int(d.isnull().sum().sum()),
             "columns_list": ", ".join(str(c) for c in d.columns)}
            for k, d in sorted(data.items())
        ]),
        "feature_panels": feature_panels, "futures_pnl": futures_pnl,
        "predictions": predictions, "coefficients": coefficients,
        "broad_metrics": split_performance(broad_bt, split_date),
        "model_positions": model_positions,
        "model_metrics": split_performance(model_bt, split_date),
        "walk_forward_positions": wf_positions, "walk_forward_predictions": wf_predictions,
        "walk_forward_metrics": split_performance(wf_bt, split_date),
        "unfiltered_positions": unfiltered_positions,
    }


def run_holding_period_experiment(data_dir="train_set", split_date="2018-01-01", hold_periods=None, cost_per_lot=0.0):
    """Daily / staggered / skip-rebalance hold across the four main strategy variants."""
    if hold_periods is None:
        hold_periods = [2, 3, 5, 10]
    results = run_research_pipeline(data_dir=data_dir, split_date=split_date, cost_per_lot=cost_per_lot)
    pnl = results["futures_pnl"]
    variants = [
        ("Static edge-filtered", results["model_positions"]),
        ("Static unfiltered", results["unfiltered_positions"]),
        ("Walk-forward", results["walk_forward_positions"]),
        ("WF edge-filtered", edge_filtered_positions(results["walk_forward_predictions"], pnl, quantile=0.50)),
    ]
    rows = []
    for label, positions in variants:
        for method, n, transformed in (
            [("daily", 1, positions)]
            + [(m, n, _hold(positions, n, m == "staggered"))
               for n in hold_periods for m in ("staggered", "skip-rebalance")]
        ):
            bt = backtest_positions(transformed, pnl, cost_per_lot)
            rows.append({"strategy": label, "method": method, "hold_days": int(n),
                         **_split_cols(bt, split_date, _HOLDING_COLS)})
    return {"holding_period_table": pd.DataFrame(rows).sort_values(["strategy", "method", "hold_days"]).reset_index(drop=True)}


def run_filter_sleeve_experiment(data_dir="train_set", split_date="2018-01-01"):
    """Multi-condition opportunity filter: validation-period and OOS metrics."""
    results = run_research_pipeline(data_dir=data_dir, split_date=split_date)
    pnl, panels, preds = results["futures_pnl"], results["feature_panels"], results["predictions"]
    multi, active = multi_condition_filter_positions(preds, pnl, panels)
    bt = backtest_positions(multi, pnl, 0.0)
    m = split_performance(bt, split_date)
    val = performance_metrics(bt.loc[(bt.index >= "2016-01-01") & (bt.index <= "2017-12-31")])
    return {
        "filter_sleeve_table": pd.DataFrame([{
            "experiment": "Multi-condition opportunity filter",
            "validation_sharpe": val.get("sharpe", np.nan),
            "oos_sharpe": m.loc["sharpe", "out_of_sample"],
            "oos_pnl": m.loc["total_pnl", "out_of_sample"],
            "max_dd": m.loc["max_drawdown", "full_period"],
        }], index=[2]),
        "multi_condition_active": active,
    }


def run_cost_margin_experiment(data_dir="train_set", split_date="2018-01-01", cost_cases=None):
    """Apply cost cases to four strategies and return one row per (strategy, case) pair."""
    if cost_cases is None:
        cost_cases = COST_CASES
    results = run_research_pipeline(data_dir=data_dir, split_date=split_date)
    pnl = results["futures_pnl"]
    multi, _ = multi_condition_filter_positions(results["predictions"], pnl, results["feature_panels"])
    strategies = [
        ("Static edge-filtered", results["model_positions"]),
        ("2-day skip-rebalance", skip_rebalance_positions(results["model_positions"], 2)),
        ("Multi-condition filter", multi),
        ("Annual walk-forward Ridge", results["walk_forward_positions"]),
    ]
    rows = []
    for strategy, positions in strategies:
        for case in cost_cases:
            bt, _ = backtest_positions_with_costs(
                positions, pnl,
                trade_cost_per_lot=case["trade_cost_per_lot"],
                holding_cost_rate=case["holding_cost_rate"],
                margin_budget=case["margin_budget"],
            )
            active = bt.loc[bt["held_gross_exposure"] > 1.0e-12]
            tc = float(active["trade_cost"].sum()) if len(active) else 0.0
            hc = float(active["holding_cost"].sum()) if len(active) else 0.0
            rows.append({"strategy": strategy, "case": case["case"],
                         "trade_cost": tc, "holding_cost": hc, "total_cost": tc + hc,
                         **_split_cols(bt, split_date, _COST_COLS)})
    return {"cost_margin_table": pd.DataFrame(rows)}


def run_final_strategy_selection(data_dir="train_set", split_date="2018-01-01",
                                 trade_cost_per_lot=8.75, holding_cost_rate=0.05, margin_budget=np.inf,
                                 skip_weight=FINAL_BLEND_WEIGHTS["skip_rebalance"],
                                 multi_condition_weight=FINAL_BLEND_WEIGHTS["multi_condition"]):
    """Fixed 50/50 blend of 2-day skip-rebalance and multi-condition filter, with cost-aware metrics."""
    results = run_research_pipeline(data_dir=data_dir, split_date=split_date)
    pnl = results["futures_pnl"]
    skip_positions = skip_rebalance_positions(results["model_positions"], 2)
    multi_positions, _ = multi_condition_filter_positions(
        results["predictions"], pnl, results["feature_panels"],
        FINAL_OPPORTUNITY_QUANTILES["prediction"],
        FINAL_OPPORTUNITY_QUANTILES["curve"],
        FINAL_OPPORTUNITY_QUANTILES["momentum"],
    )
    bt, _ = backtest_positions_with_costs(
        skip_positions * float(skip_weight) + multi_positions * float(multi_condition_weight), pnl,
        trade_cost_per_lot=trade_cost_per_lot, holding_cost_rate=holding_cost_rate, margin_budget=margin_budget,
    )

    split = pd.Timestamp(split_date)
    rows = []
    for label, w in [("in_sample", bt.loc[bt.index < split]),
                     ("out_of_sample", bt.loc[bt.index >= split]),
                     ("full_period", bt)]:
        active = w.loc[w["held_gross_exposure"] > 1.0e-12]
        m = performance_metrics(w)
        if len(active) == 0 or len(m) == 0:
            rows.append({"window": label, "sharpe": np.nan, "total_pnl": 0.0,
                         "max_drawdown": np.nan, "hit_rate": np.nan, "total_cost": 0.0})
        else:
            tc = float(active["trade_cost"].sum()) if "trade_cost" in active else 0.0
            hc = float(active["holding_cost"].sum()) if "holding_cost" in active else 0.0
            rows.append({"window": label, "sharpe": m["sharpe"], "total_pnl": m["total_pnl"],
                         "max_drawdown": m["max_drawdown"], "hit_rate": m["hit_rate"], "total_cost": tc + hc})
    return {"final_strategy_metrics": pd.DataFrame(rows), "final_backtest": bt}
