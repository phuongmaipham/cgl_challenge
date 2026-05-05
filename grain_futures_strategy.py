"""Lag-aware grain futures research utilities.

The module keeps the research pipeline in plain pandas/numpy so the notebooks
remain portable and easy to audit.
"""

from __future__ import print_function

import os

import numpy as np
import pandas as pd


COMMODITIES = ["CORN", "SOYABEAN", "WHEAT_SRW", "WHEAT_HRW"]
CONTRACT_MULTIPLIER = 5000.0
INTERCOMMODITY_SPREAD_PAIRS = [
    ("CORN", "SOYABEAN"),
    ("CORN", "WHEAT_SRW"),
    ("SOYABEAN", "WHEAT_SRW"),
    ("WHEAT_SRW", "WHEAT_HRW"),
]
OUTRIGHT_CORE_FEATURES = [
    "mom_60",
    "rev_5",
    "curve_spread",
    "curve_ratio",
    "cot_mm_level",
    "cot_pm_oi_level",
]
OUTRIGHT_PHYSICAL_FEATURES = [
    "public_inventory_change",
    "receipts_change",
    "cgl_inventory_change",
    "crush_surprise",
    "crush_utilization",
]
CALENDAR_SPREAD_FEATURES = [
    "curve_spread",
    "curve_ratio",
    "curve_change_20",
    "mom_20",
    "mom_60",
    "vol_20",
    "cot_mm_level",
    "cot_pm_oi_level",
    "public_inventory_level",
    "public_inventory_change",
    "cgl_inventory_change",
]
INTERCOMMODITY_RELATIVE_FEATURES = [
    "mom_20",
    "mom_60",
    "rev_5",
    "vol_20",
    "curve_spread",
    "curve_ratio",
    "curve_change_20",
    "cot_mm_level",
    "cot_pm_oi_level",
    "public_inventory_level",
    "public_inventory_change",
    "cgl_inventory_change",
]
DEFAULT_MARGIN_PER_LOT = {
    "CORN": 1500.0,
    "SOYABEAN": 3500.0,
    "WHEAT_SRW": 2500.0,
    "WHEAT_HRW": 2500.0,
}
COST_CASES = [
    {
        "case": "zero_cost_no_margin_cap",
        "trade_cost_per_lot": 0.0,
        "holding_cost_rate": 0.0,
        "margin_budget": np.inf,
        "description": "Research baseline with no transaction or funding costs.",
    },
    {
        "case": "market_assumption",
        "trade_cost_per_lot": 8.75,
        "holding_cost_rate": 0.05,
        "margin_budget": np.inf,
        "description": "Approx. 0.5 tick bid/ask plus commissions/fees, 5% annual margin funding.",
    },
    {
        "case": "market_assumption_margin_cap",
        "trade_cost_per_lot": 8.75,
        "holding_cost_rate": 0.05,
        "margin_budget": 2500.0,
        "description": "Market cost assumption plus a 2,500 USD margin budget per aggregate book.",
    },
    {
        "case": "stress_cost_margin_cap",
        "trade_cost_per_lot": 15.00,
        "holding_cost_rate": 0.08,
        "margin_budget": 2500.0,
        "description": "Stress case: wider execution cost, higher funding rate, same margin budget.",
    },
]
FINAL_BLEND_WEIGHTS = {
    "skip_rebalance": 0.50,
    "multi_condition": 0.50,
}
FINAL_OPPORTUNITY_QUANTILES = {
    "prediction": 0.40,
    "curve": 0.40,
    "momentum": 0.40,
}
REGIME_PERIODS = [
    {
        "period": "Russian drought/export ban shock",
        "start": "2010-07-01",
        "end": "2011-06-30",
        "reason": "Russian heat wave, drought, and grain export ban lifted in mid-2011.",
    },
    {
        "period": "US drought rally/retrace",
        "start": "2012-06-01",
        "end": "2013-05-31",
        "reason": "Historic US drought drove corn/soybean/wheat price shock and later retrace.",
    },
    {
        "period": "Crimea/Black Sea shock",
        "start": "2014-02-15",
        "end": "2014-05-31",
        "reason": "Ukraine/Crimea crisis raised Black Sea wheat and corn export risk.",
    },
    {
        "period": "Low-price abundant supply",
        "start": "2014-06-01",
        "end": "2017-12-31",
        "reason": "Post-drought supply rebuild and generally lower grain price regime.",
    },
    {
        "period": "US-China trade war",
        "start": "2018-07-06",
        "end": "2020-01-15",
        "reason": "Tariff escalation hit US soybean demand until the Phase One agreement.",
    },
    {
        "period": "2019 prevented planting floods",
        "start": "2019-05-01",
        "end": "2019-07-31",
        "reason": "Wet spring and Midwest flooding delayed corn and soybean planting.",
    },
    {
        "period": "COVID demand shock",
        "start": "2020-02-24",
        "end": "2020-06-30",
        "reason": "COVID restrictions reduced gasoline/ethanol demand and changed food demand.",
    },
    {
        "period": "COVID recovery/China buying",
        "start": "2020-07-01",
        "end": "2020-12-31",
        "reason": "Recovery phase with stronger Chinese buying and post-shock grain repricing.",
    },
]


def load_train_set(data_dir="train_set"):
    """Load all expected training CSVs into a dictionary of DataFrames."""
    names = {
        "adj1": "train_adjPrices1.csv",
        "adj2": "train_adjPrices2.csv",
        "unadj1": "train_unadjPrices1.csv",
        "unadj2": "train_unadjPrices2.csv",
        "cot_mm": "train_cot_mm.csv",
        "cot_pm_oi": "train_cot_pm_oi.csv",
        "inventories": "train_inventories.csv",
        "receipts": "train_receipts.csv",
        "cgl_inv": "train_cgl_inv.csv",
        "cgl_crush": "train_cgl_crush.csv",
    }
    data = {}
    for key, filename in names.items():
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        data[key] = df
    return data


def dataset_summary(data):
    """Return a compact table describing each loaded DataFrame."""
    rows = []
    for key in sorted(data):
        df = data[key]
        rows.append(
            {
                "dataset": key,
                "rows": len(df),
                "columns": len(df.columns),
                "start": df.index.min(),
                "end": df.index.max(),
                "missing_cells": int(df.isnull().sum().sum()),
                "columns_list": ", ".join([str(c) for c in df.columns]),
            }
        )
    return pd.DataFrame(rows)


def to_available_calendar(df, trading_index, lag_days):
    """Shift observations to their first usable date, then forward-fill.

    The input timestamp is treated as the observation date. A lag of 1 means
    the value can first be used on timestamp + 1 calendar day. The shifted
    series is then aligned to the adjusted-price trading calendar.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index) + pd.DateOffset(days=int(lag_days))
    out = out.sort_index()
    out = out.groupby(out.index).last()
    return out.reindex(trading_index).ffill()


def rolling_zscore(df, window=252, min_periods=40):
    mean = df.rolling(window=window, min_periods=min_periods).mean()
    std = df.rolling(window=window, min_periods=min_periods).std()
    return (df - mean) / std.replace(0.0, np.nan)


def _signed_clip(df, limit=5.0):
    return df.clip(lower=-limit, upper=limit)


def build_feature_panels(data):
    """Build one feature DataFrame per commodity.

    Feature timing:
    - Price and curve features are observed on the adjusted-price calendar.
    - Public inventories and receipts are shifted by T+2.
    - Cargill inventory and crush are shifted by T+1.
    - COT data is shifted by 3 calendar days as a conservative weekly release
      assumption.
    """
    trading_index = data["adj1"].index
    adj1 = data["adj1"].reindex(trading_index).ffill()
    unadj1 = data["unadj1"].reindex(trading_index).ffill()
    unadj2 = data["unadj2"].reindex(trading_index).ffill()

    daily_price_change = adj1.diff()
    pct_change = adj1.pct_change()
    futures_pnl = daily_price_change * CONTRACT_MULTIPLIER

    cot_mm = to_available_calendar(data["cot_mm"], trading_index, 3)
    cot_pm_oi = to_available_calendar(data["cot_pm_oi"], trading_index, 3)
    inventories = to_available_calendar(data["inventories"], trading_index, 2)
    receipts = to_available_calendar(data["receipts"], trading_index, 2)
    cgl_inv = to_available_calendar(data["cgl_inv"], trading_index, 1)
    cgl_crush = to_available_calendar(data["cgl_crush"], trading_index, 1)

    curve_spread = unadj1 - unadj2
    curve_ratio = unadj1 / unadj2.replace(0.0, np.nan) - 1.0

    base_feature_blocks = {
        "mom_20": rolling_zscore(adj1.pct_change(20), 252, 60),
        "mom_60": rolling_zscore(adj1.pct_change(60), 252, 80),
        "rev_5": -rolling_zscore(adj1.pct_change(5), 126, 30),
        "vol_20": rolling_zscore(pct_change.rolling(20, min_periods=10).std(), 252, 60),
        "curve_spread": rolling_zscore(curve_spread, 252, 60),
        "curve_ratio": rolling_zscore(curve_ratio, 252, 60),
        "curve_change_20": rolling_zscore(curve_spread.diff(20), 252, 60),
        "cot_mm_level": rolling_zscore(cot_mm, 156, 40),
        "cot_mm_change": rolling_zscore(cot_mm.diff(5), 156, 40),
        "cot_pm_oi_level": rolling_zscore(cot_pm_oi, 156, 40),
        "cot_pm_oi_change": rolling_zscore(cot_pm_oi.diff(5), 156, 40),
        "public_inventory_level": rolling_zscore(inventories, 156, 40),
        "public_inventory_change": rolling_zscore(inventories.diff(5), 156, 40),
        "receipts_level": rolling_zscore(receipts, 126, 30),
        "receipts_change": rolling_zscore(receipts.diff(5), 126, 30),
        "cgl_inventory_level": rolling_zscore(cgl_inv, 252, 60),
        "cgl_inventory_change": rolling_zscore(cgl_inv.diff(5), 252, 60),
    }

    crush = pd.DataFrame(index=trading_index)
    crush["crush_processed"] = cgl_crush["processed"]
    crush["crush_planned"] = cgl_crush["planned"]
    crush["crush_surprise"] = cgl_crush["processed"] - cgl_crush["planned"]
    crush["crush_utilization"] = cgl_crush["processed"] / cgl_crush["planned"].replace(0.0, np.nan) - 1.0
    crush_features = rolling_zscore(crush, 252, 60)

    panels = {}
    for commodity in COMMODITIES:
        frame = pd.DataFrame(index=trading_index)
        for feature_name, block in base_feature_blocks.items():
            frame[feature_name] = block[commodity]
        for feature_name in crush_features.columns:
            frame[feature_name] = crush_features[feature_name]
        frame = _signed_clip(frame)
        panels[commodity] = frame

    return panels, futures_pnl


def fit_ridge_predict(features, target, train_mask, alpha=10.0):
    """Fit a standardised Ridge model and predict all rows."""
    valid = features.notnull().all(axis=1) & target.notnull()
    train = valid & train_mask
    if int(train.sum()) < max(40, features.shape[1] * 3):
        return pd.Series(np.nan, index=features.index), pd.Series(np.nan, index=features.columns)

    x_train = features.loc[train].values.astype(float)
    y_train = target.loc[train].values.astype(float)
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    y_mean = y_train.mean()

    x_scaled = (x_train - x_mean) / x_std
    y_centered = y_train - y_mean
    penalty = alpha * np.eye(x_scaled.shape[1])
    beta = np.linalg.solve(np.dot(x_scaled.T, x_scaled) + penalty, np.dot(x_scaled.T, y_centered))

    all_valid = features.notnull().all(axis=1)
    pred = pd.Series(np.nan, index=features.index)
    x_all = features.loc[all_valid].values.astype(float)
    pred.loc[all_valid] = y_mean + np.dot((x_all - x_mean) / x_std, beta)
    coef = pd.Series(beta / x_std, index=features.columns)
    return pred, coef


def build_model_signals(feature_panels, futures_pnl, split_date="2018-01-01", alpha=25.0):
    """Fit one Ridge model per commodity to predict next-day dollar PnL."""
    split_date = pd.Timestamp(split_date)
    train_mask = futures_pnl.index < split_date
    predictions = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float)
    coefficients = {}

    for commodity in COMMODITIES:
        target = futures_pnl[commodity].shift(-1)
        pred, coef = fit_ridge_predict(feature_panels[commodity], target, train_mask, alpha=alpha)
        predictions[commodity] = pred
        coefficients[commodity] = coef

    return predictions, pd.DataFrame(coefficients)


def _forward_pnl_target(pnl_series, horizon):
    if int(horizon) <= 1:
        return pnl_series.shift(-1)
    return pnl_series.shift(-1).rolling(int(horizon), min_periods=int(horizon)).sum().shift(-(int(horizon) - 1))


def build_improved_model_signals(feature_panels, futures_pnl, split_date="2018-01-01"):
    """Build the improved two-block Ridge signal.

    The broad one-day Ridge model overfit the training data. This variant uses a
    slower five-day target and two deliberately small feature blocks:
    - core: price, curve, and COT features
    - physical overlay: public flow plus Cargill inventory/crush features

    The overlay is strongly regularised and added at full weight because it was
    the best choice on the 2016-2017 validation window while still improving
    2018-2020 out-of-sample performance.
    """
    split_date = pd.Timestamp(split_date)
    train_mask = futures_pnl.index < split_date
    core_predictions = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float)
    physical_predictions = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float)
    core_coefficients = {}
    physical_coefficients = {}

    for commodity in COMMODITIES:
        target = _forward_pnl_target(futures_pnl[commodity], 5)
        core_pred, core_coef = fit_ridge_predict(
            feature_panels[commodity][OUTRIGHT_CORE_FEATURES], target, train_mask, alpha=25.0
        )
        phys_pred, phys_coef = fit_ridge_predict(
            feature_panels[commodity][OUTRIGHT_PHYSICAL_FEATURES], target, train_mask, alpha=1000.0
        )
        core_predictions[commodity] = core_pred
        physical_predictions[commodity] = phys_pred
        core_coefficients[commodity] = core_coef
        physical_coefficients[commodity] = phys_coef

    predictions = core_predictions.fillna(0.0) + physical_predictions.fillna(0.0)
    coefficients = {
        "core": pd.DataFrame(core_coefficients),
        "physical": pd.DataFrame(physical_coefficients),
    }
    return predictions, coefficients, core_predictions, physical_predictions


def build_walk_forward_model_signals(
    feature_panels,
    futures_pnl,
    start_date="2014-01-01",
    retrain_frequency="YE",
    min_train_days=756,
    horizon=20,
):
    """Build expanding walk-forward signals.

    At each retraining date, coefficients are fit using only data strictly before
    that date. Those coefficients are then used until the next retraining date.
    This is slower than the static model but is a cleaner anti-overfit check.
    """
    start_date = pd.Timestamp(start_date)
    index = futures_pnl.index
    schedule = pd.date_range(start=start_date, end=index.max(), freq=retrain_frequency)
    schedule = [index[index >= date][0] for date in schedule if len(index[index >= date]) > 0]
    schedule = sorted(set(schedule))

    predictions = pd.DataFrame(0.0, index=index, columns=COMMODITIES)
    coefficient_rows = []

    for i, train_end in enumerate(schedule):
        next_end = schedule[i + 1] if i + 1 < len(schedule) else index.max() + pd.DateOffset(days=1)
        apply_mask = (index >= train_end) & (index < next_end)
        train_mask = index < train_end
        if int(train_mask.sum()) < int(min_train_days):
            continue

        for commodity in COMMODITIES:
            target = _forward_pnl_target(futures_pnl[commodity], int(horizon))
            core_pred, core_coef = fit_ridge_predict(
                feature_panels[commodity][OUTRIGHT_CORE_FEATURES], target, train_mask, alpha=25.0
            )
            phys_pred, phys_coef = fit_ridge_predict(
                feature_panels[commodity][OUTRIGHT_PHYSICAL_FEATURES], target, train_mask, alpha=1000.0
            )
            predictions.loc[apply_mask, commodity] = (
                core_pred.loc[apply_mask].fillna(0.0) + phys_pred.loc[apply_mask].fillna(0.0)
            )
            row = {"retrain_date": train_end, "commodity": commodity, "block": "core"}
            row.update(core_coef.to_dict())
            coefficient_rows.append(row)
            row = {"retrain_date": train_end, "commodity": commodity, "block": "physical"}
            row.update(phys_coef.to_dict())
            coefficient_rows.append(row)

    coefficients = pd.DataFrame(coefficient_rows)
    return predictions, coefficients


def build_calendar_spread_pnl(data):
    """Build front-vs-second calendar-spread PnL by commodity.

    A positive unit is long the front adjusted futures series and short the
    second adjusted futures series. Unadjusted front/second prices are still
    used as predictive curve features; adjusted prices are used for PnL to avoid
    roll jumps dominating the backtest.
    """
    adj1 = data["adj1"].ffill()
    adj2 = data["adj2"].reindex(adj1.index).ffill()
    return (adj1.diff() - adj2.diff()) * CONTRACT_MULTIPLIER


def build_calendar_spread_feature_panels(feature_panels):
    """Use commodity-specific curve/storage features for calendar-spread models."""
    panels = {}
    for commodity in COMMODITIES:
        cols = [name for name in CALENDAR_SPREAD_FEATURES if name in feature_panels[commodity].columns]
        panels[commodity] = feature_panels[commodity][cols].copy()
    return panels


def _spread_name(first, second):
    return str(first) + "_VS_" + str(second)


def build_intercommodity_spread_pnl(futures_pnl, pairs=None):
    """Build volatility-hedged inter-commodity spread PnL.

    A positive unit is long the first commodity and short a rolling-volatility
    hedge ratio of the second commodity. The hedge ratio is shifted one day, so
    it only uses information available before the traded close.
    """
    if pairs is None:
        pairs = INTERCOMMODITY_SPREAD_PAIRS
    vol = futures_pnl.rolling(60, min_periods=20).std().shift(1)
    out = pd.DataFrame(index=futures_pnl.index)
    hedge_ratios = pd.DataFrame(index=futures_pnl.index)
    for first, second in pairs:
        name = _spread_name(first, second)
        ratio = (vol[first] / vol[second].replace(0.0, np.nan)).clip(lower=0.25, upper=4.0)
        out[name] = futures_pnl[first] - ratio * futures_pnl[second]
        hedge_ratios[name] = ratio
    return out, hedge_ratios


def build_intercommodity_feature_panels(feature_panels, pairs=None):
    """Build relative feature panels for synthetic inter-commodity spreads."""
    if pairs is None:
        pairs = INTERCOMMODITY_SPREAD_PAIRS
    panels = {}
    for first, second in pairs:
        name = _spread_name(first, second)
        frame = pd.DataFrame(index=feature_panels[first].index)
        for feature in INTERCOMMODITY_RELATIVE_FEATURES:
            if feature in feature_panels[first].columns and feature in feature_panels[second].columns:
                frame[feature + "_rel"] = feature_panels[first][feature] - feature_panels[second][feature]
        panels[name] = _signed_clip(frame)
    return panels


def build_walk_forward_spread_signals(
    spread_feature_panels,
    spread_pnl,
    start_date="2014-01-01",
    retrain_frequency="YE",
    min_train_days=756,
    horizon=20,
    alpha=100.0,
):
    """Build annual walk-forward Ridge predictions for spread instruments."""
    start_date = pd.Timestamp(start_date)
    index = spread_pnl.index
    schedule = pd.date_range(start=start_date, end=index.max(), freq=retrain_frequency)
    schedule = [index[index >= date][0] for date in schedule if len(index[index >= date]) > 0]
    schedule = sorted(set(schedule))

    instruments = list(spread_pnl.columns)
    predictions = pd.DataFrame(0.0, index=index, columns=instruments)
    coefficient_rows = []

    for i, train_end in enumerate(schedule):
        next_end = schedule[i + 1] if i + 1 < len(schedule) else index.max() + pd.DateOffset(days=1)
        apply_mask = (index >= train_end) & (index < next_end)
        train_mask = index < train_end
        if int(train_mask.sum()) < int(min_train_days):
            continue

        for instrument in instruments:
            features = spread_feature_panels[instrument]
            target = _forward_pnl_target(spread_pnl[instrument], int(horizon))
            pred, coef = fit_ridge_predict(features, target, train_mask, alpha=float(alpha))
            predictions.loc[apply_mask, instrument] = pred.loc[apply_mask].fillna(0.0)
            row = {"retrain_date": train_end, "instrument": instrument}
            row.update(coef.to_dict())
            coefficient_rows.append(row)

    return predictions, pd.DataFrame(coefficient_rows)


def run_spread_experiment(data_dir="train_set", split_date="2018-01-01", cost_per_lot=0.0):
    """Test outright, calendar-spread, inter-commodity-spread, and combined books."""
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)

    outright_predictions, _ = build_walk_forward_model_signals(feature_panels, futures_pnl)
    outright_positions = model_predictions_to_positions(outright_predictions, futures_pnl)
    outright_bt, _ = backtest_positions(outright_positions, futures_pnl, cost_per_lot)

    calendar_pnl = build_calendar_spread_pnl(data)
    calendar_panels = build_calendar_spread_feature_panels(feature_panels)
    calendar_predictions, calendar_coefficients = build_walk_forward_spread_signals(
        calendar_panels, calendar_pnl, alpha=100.0
    )
    calendar_positions = model_predictions_to_positions(calendar_predictions, calendar_pnl)
    calendar_bt, _ = backtest_positions(calendar_positions, calendar_pnl, cost_per_lot)

    inter_pnl, hedge_ratios = build_intercommodity_spread_pnl(futures_pnl)
    inter_panels = build_intercommodity_feature_panels(feature_panels)
    inter_predictions, inter_coefficients = build_walk_forward_spread_signals(
        inter_panels, inter_pnl, alpha=100.0
    )
    inter_positions = model_predictions_to_positions(inter_predictions, inter_pnl)
    inter_bt, _ = backtest_positions(inter_positions, inter_pnl, cost_per_lot)

    spread_pnl = pd.concat([calendar_pnl.add_prefix("CAL_"), inter_pnl.add_prefix("PAIR_")], axis=1)
    spread_predictions = pd.concat(
        [calendar_predictions.add_prefix("CAL_"), inter_predictions.add_prefix("PAIR_")], axis=1
    )
    spread_positions = model_predictions_to_positions(spread_predictions, spread_pnl)
    spread_bt, _ = backtest_positions(spread_positions, spread_pnl, cost_per_lot)

    all_pnl = pd.concat([futures_pnl.add_prefix("OUT_"), spread_pnl], axis=1)
    all_predictions = pd.concat([outright_predictions.add_prefix("OUT_"), spread_predictions], axis=1)
    all_positions = model_predictions_to_positions(all_predictions, all_pnl)
    all_bt, _ = backtest_positions(all_positions, all_pnl, cost_per_lot)

    inter_overlay_20_bt = _backtest_weighted_sleeves(
        [(outright_positions, futures_pnl), (inter_positions, inter_pnl)],
        [0.80, 0.20],
        cost_per_lot,
    )
    inter_overlay_30_bt = _backtest_weighted_sleeves(
        [(outright_positions, futures_pnl), (inter_positions, inter_pnl)],
        [0.70, 0.30],
        cost_per_lot,
    )
    spread_overlay_50_bt = _backtest_weighted_sleeves(
        [(outright_positions, futures_pnl), (spread_positions, spread_pnl)],
        [0.50, 0.50],
        cost_per_lot,
    )

    return {
        "outright_predictions": outright_predictions,
        "outright_positions": outright_positions,
        "outright_metrics": split_performance(outright_bt, split_date),
        "calendar_pnl": calendar_pnl,
        "calendar_predictions": calendar_predictions,
        "calendar_coefficients": calendar_coefficients,
        "calendar_positions": calendar_positions,
        "calendar_metrics": split_performance(calendar_bt, split_date),
        "intercommodity_pnl": inter_pnl,
        "intercommodity_hedge_ratios": hedge_ratios,
        "intercommodity_predictions": inter_predictions,
        "intercommodity_coefficients": inter_coefficients,
        "intercommodity_positions": inter_positions,
        "intercommodity_metrics": split_performance(inter_bt, split_date),
        "spread_pnl": spread_pnl,
        "spread_predictions": spread_predictions,
        "spread_positions": spread_positions,
        "spread_metrics": split_performance(spread_bt, split_date),
        "combined_pnl": all_pnl,
        "combined_predictions": all_predictions,
        "combined_positions": all_positions,
        "combined_metrics": split_performance(all_bt, split_date),
        "inter_overlay_20_metrics": split_performance(inter_overlay_20_bt, split_date),
        "inter_overlay_30_metrics": split_performance(inter_overlay_30_bt, split_date),
        "spread_overlay_50_metrics": split_performance(spread_overlay_50_bt, split_date),
    }


def _backtest_weighted_sleeves(sleeves, weights, cost_per_lot=0.0):
    """Combine independently sized sleeves into one portfolio-level backtest."""
    backtests = []
    for sleeve, weight in zip(sleeves, weights):
        positions, pnl = sleeve
        bt, _ = backtest_positions(positions * float(weight), pnl, cost_per_lot)
        backtests.append(bt)

    out = pd.DataFrame(index=backtests[0].index)
    for column in ["gross_pnl", "costs", "net_pnl", "turnover", "gross_exposure", "held_gross_exposure"]:
        out[column] = sum(bt[column].reindex(out.index).fillna(0.0) for bt in backtests)
    out["cum_pnl"] = out["net_pnl"].cumsum()
    return out


def model_predictions_to_positions(predictions, futures_pnl, gross_lots=1.0):
    """Convert predictions to market-neutral cross-sectional positions."""
    risk = futures_pnl.rolling(60, min_periods=20).std().shift(1)
    risk_adjusted = predictions / risk.replace(0.0, np.nan)
    demeaned = risk_adjusted.sub(risk_adjusted.mean(axis=1), axis=0)
    denom = demeaned.abs().sum(axis=1).replace(0.0, np.nan)
    positions = demeaned.div(denom, axis=0) * float(gross_lots)
    return positions.fillna(0.0).clip(lower=-1.0, upper=1.0)


def edge_filtered_positions(predictions, futures_pnl, quantile=0.50, min_periods=252, gross_lots=1.0):
    """Build positions only when model cross-sectional edge is above its past quantile.

    The edge score is the daily cross-sectional standard deviation of risk
    adjusted predictions. It is compared with an expanding, one-day-lagged
    quantile so the filter uses only information available at the time.
    """
    base_positions = model_predictions_to_positions(predictions, futures_pnl, gross_lots)
    risk = futures_pnl.rolling(60, min_periods=20).std().shift(1)
    risk_adjusted = predictions / risk.replace(0.0, np.nan)
    edge = risk_adjusted.std(axis=1)
    threshold = edge.expanding(min_periods=int(min_periods)).quantile(float(quantile)).shift(1)
    active = (edge > threshold).astype(float).reindex(base_positions.index).fillna(0.0)
    return base_positions.mul(active, axis=0), edge, threshold


def baseline_momentum_positions(data, gross_lots=1.0):
    """Simple cross-sectional 60-day momentum benchmark."""
    adj1 = data["adj1"].ffill()
    signal = rolling_zscore(adj1.pct_change(60), 252, 80)
    demeaned = signal.sub(signal.mean(axis=1), axis=0)
    denom = demeaned.abs().sum(axis=1).replace(0.0, np.nan)
    positions = demeaned.div(denom, axis=0) * float(gross_lots)
    return positions.fillna(0.0).clip(lower=-1.0, upper=1.0)


def backtest_positions(positions, futures_pnl, cost_per_lot=0.0):
    """Backtest close-to-close PnL from positions known at prior close."""
    positions = positions.reindex(futures_pnl.index).fillna(0.0)
    held_positions = positions.shift(1).fillna(0.0)
    gross_pnl_by_asset = held_positions * futures_pnl
    turnover_by_asset = positions.diff().abs().fillna(0.0)
    costs = turnover_by_asset * float(cost_per_lot)
    net_pnl_by_asset = gross_pnl_by_asset - costs

    result = pd.DataFrame(index=futures_pnl.index)
    result["gross_pnl"] = gross_pnl_by_asset.sum(axis=1)
    result["costs"] = costs.sum(axis=1)
    result["net_pnl"] = net_pnl_by_asset.sum(axis=1)
    result["turnover"] = turnover_by_asset.sum(axis=1)
    result["gross_exposure"] = positions.abs().sum(axis=1)
    result["held_gross_exposure"] = held_positions.abs().sum(axis=1)
    result["cum_pnl"] = result["net_pnl"].cumsum()
    return result, net_pnl_by_asset


def _margin_frame(columns, margin_per_lot=None):
    if margin_per_lot is None:
        margin_per_lot = DEFAULT_MARGIN_PER_LOT
    values = {}
    for column in columns:
        key = str(column)
        if key.startswith("OUT_"):
            key = key[4:]
        if key.startswith("CAL_"):
            key = key[4:]
        values[column] = float(margin_per_lot.get(key, np.nan))
    fallback = np.nanmedian([v for v in values.values() if pd.notnull(v)])
    if not pd.notnull(fallback):
        fallback = 2500.0
    return pd.Series({k: (fallback if pd.isnull(v) else v) for k, v in values.items()})


def apply_margin_budget(positions, margin_per_lot=None, margin_budget=np.inf):
    """Scale positions down if estimated margin use exceeds a book-level budget."""
    if margin_budget is None or not np.isfinite(float(margin_budget)):
        return positions.copy(), pd.Series(1.0, index=positions.index)

    margin = _margin_frame(positions.columns, margin_per_lot)
    margin_use = positions.abs().mul(margin, axis=1).sum(axis=1)
    scale = (float(margin_budget) / margin_use.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    return positions.mul(scale, axis=0), scale


def backtest_positions_with_costs(
    positions,
    futures_pnl,
    trade_cost_per_lot=0.0,
    holding_cost_rate=0.0,
    margin_per_lot=None,
    margin_budget=np.inf,
):
    """Backtest with execution costs, margin funding costs, and optional margin cap."""
    adjusted_positions, margin_scale = apply_margin_budget(positions, margin_per_lot, margin_budget)
    adjusted_positions = adjusted_positions.reindex(futures_pnl.index).fillna(0.0)
    held_positions = adjusted_positions.shift(1).fillna(0.0)
    pnl = futures_pnl.reindex(adjusted_positions.index).fillna(0.0)

    gross_pnl_by_asset = held_positions * pnl
    turnover_by_asset = adjusted_positions.diff().abs().fillna(0.0)
    trade_cost_by_asset = turnover_by_asset * float(trade_cost_per_lot)

    margin = _margin_frame(adjusted_positions.columns, margin_per_lot)
    margin_use_by_asset = held_positions.abs().mul(margin, axis=1)
    holding_cost_by_asset = margin_use_by_asset * (float(holding_cost_rate) / 252.0)

    net_pnl_by_asset = gross_pnl_by_asset - trade_cost_by_asset - holding_cost_by_asset
    result = pd.DataFrame(index=adjusted_positions.index)
    result["gross_pnl"] = gross_pnl_by_asset.sum(axis=1)
    result["trade_cost"] = trade_cost_by_asset.sum(axis=1)
    result["holding_cost"] = holding_cost_by_asset.sum(axis=1)
    result["costs"] = result["trade_cost"] + result["holding_cost"]
    result["net_pnl"] = net_pnl_by_asset.sum(axis=1)
    result["turnover"] = turnover_by_asset.sum(axis=1)
    result["gross_exposure"] = adjusted_positions.abs().sum(axis=1)
    result["held_gross_exposure"] = held_positions.abs().sum(axis=1)
    result["margin_used"] = margin_use_by_asset.sum(axis=1)
    result["margin_scale"] = margin_scale.reindex(result.index).fillna(1.0)
    result["cum_pnl"] = result["net_pnl"].cumsum()
    return result, net_pnl_by_asset


def performance_metrics(bt, split_date=None):
    """Compute dollar-PnL performance metrics."""
    active = bt["held_gross_exposure"] > 1.0e-12
    pnl = bt.loc[active, "net_pnl"].dropna()
    if len(pnl) == 0:
        return pd.Series(dtype=float)
    ann_factor = 252.0
    avg = pnl.mean()
    vol = pnl.std()
    sharpe = np.nan if vol == 0.0 else (avg / vol) * np.sqrt(ann_factor)
    cum = pnl.cumsum()
    drawdown = cum - cum.cummax()
    metrics = {
        "days": float(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "annualized_avg_pnl": float(avg * ann_factor),
        "annualized_vol": float(vol * np.sqrt(ann_factor)),
        "sharpe": float(sharpe) if pd.notnull(sharpe) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((pnl > 0.0).mean()),
        "avg_daily_turnover": float(bt["turnover"].reindex(pnl.index).mean()),
        "avg_gross_exposure": float(bt["gross_exposure"].reindex(pnl.index).mean()),
    }
    out = pd.Series(metrics)
    if split_date is not None:
        out.name = str(split_date)
    return out


def split_performance(bt, split_date):
    split_date = pd.Timestamp(split_date)
    before = bt.loc[bt.index < split_date]
    after = bt.loc[bt.index >= split_date]
    table = pd.DataFrame(
        {
            "in_sample": performance_metrics(before),
            "out_of_sample": performance_metrics(after),
            "full_period": performance_metrics(bt),
        }
    )
    return table


def period_performance(bt, periods=None):
    if periods is None:
        periods = REGIME_PERIODS
    rows = []
    for item in periods:
        start = pd.Timestamp(item["start"])
        end = pd.Timestamp(item["end"])
        metrics = performance_metrics(bt.loc[(bt.index >= start) & (bt.index <= end)])
        row = {
            "period": item["period"],
            "start": start,
            "end": end,
            "reason": item.get("reason", ""),
        }
        for key, value in metrics.items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run_research_pipeline(data_dir="train_set", split_date="2018-01-01", alpha=25.0, cost_per_lot=0.0):
    """Run the full load-feature-model-backtest pipeline."""
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)

    broad_predictions, broad_coefficients = build_model_signals(feature_panels, futures_pnl, split_date, alpha)
    broad_positions = model_predictions_to_positions(broad_predictions, futures_pnl)
    broad_bt, broad_pnl_by_asset = backtest_positions(broad_positions, futures_pnl, cost_per_lot)

    predictions, coefficient_blocks, core_predictions, physical_predictions = build_improved_model_signals(
        feature_panels, futures_pnl, split_date
    )
    unfiltered_positions = model_predictions_to_positions(predictions, futures_pnl)
    unfiltered_bt, unfiltered_pnl_by_asset = backtest_positions(unfiltered_positions, futures_pnl, cost_per_lot)

    model_positions, edge_score, edge_threshold = edge_filtered_positions(predictions, futures_pnl, quantile=0.50)
    model_bt, model_pnl_by_asset = backtest_positions(model_positions, futures_pnl, cost_per_lot)

    walk_forward_predictions, walk_forward_coefficients = build_walk_forward_model_signals(feature_panels, futures_pnl)
    walk_forward_positions = model_predictions_to_positions(walk_forward_predictions, futures_pnl)
    walk_forward_risk = futures_pnl.rolling(60, min_periods=20).std().shift(1)
    walk_forward_edge = (walk_forward_predictions / walk_forward_risk.replace(0.0, np.nan)).std(axis=1)
    walk_forward_threshold = pd.Series(np.nan, index=futures_pnl.index)
    walk_forward_bt, walk_forward_pnl_by_asset = backtest_positions(
        walk_forward_positions, futures_pnl, cost_per_lot
    )

    baseline_positions = baseline_momentum_positions(data)
    baseline_bt, baseline_pnl_by_asset = backtest_positions(baseline_positions, futures_pnl, cost_per_lot)

    return {
        "data": data,
        "summary": dataset_summary(data),
        "feature_panels": feature_panels,
        "futures_pnl": futures_pnl,
        "predictions": predictions,
        "coefficients": coefficient_blocks,
        "core_predictions": core_predictions,
        "physical_predictions": physical_predictions,
        "edge_score": edge_score,
        "edge_threshold": edge_threshold,
        "model_positions": model_positions,
        "model_bt": model_bt,
        "model_pnl_by_asset": model_pnl_by_asset,
        "walk_forward_predictions": walk_forward_predictions,
        "walk_forward_coefficients": walk_forward_coefficients,
        "walk_forward_positions": walk_forward_positions,
        "walk_forward_edge": walk_forward_edge,
        "walk_forward_threshold": walk_forward_threshold,
        "walk_forward_bt": walk_forward_bt,
        "walk_forward_pnl_by_asset": walk_forward_pnl_by_asset,
        "walk_forward_metrics": split_performance(walk_forward_bt, split_date),
        "unfiltered_positions": unfiltered_positions,
        "unfiltered_bt": unfiltered_bt,
        "unfiltered_pnl_by_asset": unfiltered_pnl_by_asset,
        "unfiltered_metrics": split_performance(unfiltered_bt, split_date),
        "broad_predictions": broad_predictions,
        "broad_coefficients": broad_coefficients,
        "broad_positions": broad_positions,
        "broad_bt": broad_bt,
        "broad_pnl_by_asset": broad_pnl_by_asset,
        "broad_metrics": split_performance(broad_bt, split_date),
        "baseline_positions": baseline_positions,
        "baseline_bt": baseline_bt,
        "baseline_pnl_by_asset": baseline_pnl_by_asset,
        "model_metrics": split_performance(model_bt, split_date),
        "baseline_metrics": split_performance(baseline_bt, split_date),
        "period_metrics": {
            "model": period_performance(model_bt),
            "walk_forward": period_performance(walk_forward_bt),
            "unfiltered": period_performance(unfiltered_bt),
            "broad": period_performance(broad_bt),
            "baseline": period_performance(baseline_bt),
        },
    }


def skip_rebalance_positions(positions, n_days):
    """Update positions every N trading days and hold them flat between updates."""
    n_days = int(n_days)
    if n_days <= 1:
        return positions.copy()

    out = positions.copy() * 0.0
    for i in range(0, len(positions.index), n_days):
        end = min(i + n_days, len(positions.index))
        out.iloc[i:end] = positions.iloc[i]
    return out


def staggered_hold_positions(positions, n_days):
    """Average N offset skip-rebalance sleeves to approximate overlapping holds."""
    n_days = int(n_days)
    if n_days <= 1:
        return positions.copy()

    sleeves = []
    for offset in range(n_days):
        sleeve = positions.copy() * 0.0
        for i in range(offset, len(positions.index), n_days):
            end = min(i + n_days, len(positions.index))
            sleeve.iloc[i:end] = positions.iloc[i]
        sleeves.append(sleeve)
    return sum(sleeves) / float(n_days)


def _holding_period_rows(label, base_positions, pnl, split_date, hold_periods, cost_per_lot):
    rows = []
    methods = [
        ("daily", 1, lambda pos, days: pos.copy()),
    ]
    for n_days in hold_periods:
        methods.append(("staggered", int(n_days), staggered_hold_positions))
        methods.append(("skip-rebalance", int(n_days), skip_rebalance_positions))

    for method, hold_days, transform in methods:
        held_positions = transform(base_positions, hold_days)
        bt, _ = backtest_positions(held_positions, pnl, cost_per_lot)
        metrics = split_performance(bt, split_date)
        rows.append(
            {
                "strategy": label,
                "method": method,
                "hold_days": int(hold_days),
                "is_sharpe": metrics.loc["sharpe", "in_sample"],
                "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
                "full_sharpe": metrics.loc["sharpe", "full_period"],
                "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
                "full_pnl": metrics.loc["total_pnl", "full_period"],
                "max_drawdown": metrics.loc["max_drawdown", "full_period"],
                "turnover": metrics.loc["avg_daily_turnover", "full_period"],
            }
        )
    return rows


def run_holding_period_experiment(
    data_dir="train_set",
    split_date="2018-01-01",
    hold_periods=None,
    cost_per_lot=0.0,
):
    """Run reproducible holding-period tests for the main strategy variants."""
    if hold_periods is None:
        hold_periods = [2, 3, 5, 10]

    results = run_research_pipeline(data_dir=data_dir, split_date=split_date, cost_per_lot=cost_per_lot)
    pnl = results["futures_pnl"]
    wf_edge_positions, _, _ = edge_filtered_positions(results["walk_forward_predictions"], pnl, quantile=0.50)

    variants = [
        ("Static edge-filtered", results["model_positions"]),
        ("Static unfiltered", results["unfiltered_positions"]),
        ("Walk-forward", results["walk_forward_positions"]),
        ("WF edge-filtered", wf_edge_positions),
    ]

    rows = []
    for label, positions in variants:
        rows.extend(_holding_period_rows(label, positions, pnl, split_date, hold_periods, cost_per_lot))
    table = pd.DataFrame(rows)
    table = table.sort_values(["strategy", "method", "hold_days"]).reset_index(drop=True)

    best_by_oos_sharpe = table.sort_values(["oos_sharpe", "full_sharpe"], ascending=False).head(10)
    return {
        "holding_period_table": table,
        "best_by_oos_sharpe": best_by_oos_sharpe,
    }


def validation_performance(bt, validation_start="2016-01-01", validation_end="2017-12-31"):
    """Compute metrics for a fixed validation period."""
    start = pd.Timestamp(validation_start)
    end = pd.Timestamp(validation_end)
    return performance_metrics(bt.loc[(bt.index >= start) & (bt.index <= end)])


def _experiment_score_row(name, positions, pnl, split_date, hold_days, rationale, selected_for):
    bt, _ = backtest_positions(positions, pnl, 0.0)
    metrics = split_performance(bt, split_date)
    validation = validation_performance(bt)
    us_china = performance_metrics(bt.loc[(bt.index >= "2018-07-06") & (bt.index <= "2020-01-15")])
    return {
        "experiment": name,
        "rationale": rationale,
        "selected_for": selected_for,
        "hold_days": int(hold_days),
        "validation_sharpe": validation.get("sharpe", np.nan),
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "max_dd": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
        "us_china_trade_war_sharpe": us_china.get("sharpe", np.nan),
        "us_china_trade_war_pnl": us_china.get("total_pnl", np.nan),
    }


def _feature_frame(feature_panels, feature_name, columns=None):
    if columns is None:
        columns = COMMODITIES
    return pd.DataFrame({commodity: feature_panels[commodity][feature_name] for commodity in columns})


def _expanding_active(score, quantile, min_periods=252):
    threshold = score.expanding(min_periods=int(min_periods)).quantile(float(quantile)).shift(1)
    return (score > threshold).astype(float).fillna(0.0)


def expanding_percentile_score(score, min_periods=252):
    """Lagged expanding percentile of today's score versus prior history."""
    score = pd.Series(score).astype(float)
    out = pd.Series(np.nan, index=score.index)
    values = score.values
    for i in range(len(score)):
        start = 0
        history = values[start:i]
        history = history[pd.notnull(history)]
        if len(history) < int(min_periods) or pd.isnull(values[i]):
            continue
        out.iloc[i] = float((history <= values[i]).mean())
    return out


def multi_condition_filter_positions(
    predictions,
    futures_pnl,
    feature_panels,
    prediction_quantile=0.40,
    curve_quantile=0.40,
    momentum_quantile=0.40,
):
    """Trade only when prediction, curve, and momentum dispersion are all high."""
    risk = futures_pnl.rolling(60, min_periods=20).std().shift(1)
    prediction_dispersion = (predictions / risk.replace(0.0, np.nan)).std(axis=1)
    curve_signal = (_feature_frame(feature_panels, "curve_spread") + _feature_frame(feature_panels, "curve_ratio")) / 2.0
    momentum_signal = _feature_frame(feature_panels, "mom_60")
    curve_dispersion = curve_signal.std(axis=1)
    momentum_dispersion = momentum_signal.std(axis=1)

    active = (
        _expanding_active(prediction_dispersion, prediction_quantile)
        * _expanding_active(curve_dispersion, curve_quantile)
        * _expanding_active(momentum_dispersion, momentum_quantile)
    )
    base_positions = model_predictions_to_positions(predictions, futures_pnl)
    return base_positions.mul(active, axis=0), active


def natural_opportunity_weight(
    predictions,
    futures_pnl,
    feature_panels,
    min_periods=252,
):
    """Continuous opportunity score from prediction, curve, and momentum dispersion.

    This avoids a hard regime cutoff. Each component is scored as a lagged
    expanding percentile against its own prior history, and the final score is
    the equal-weight average of the three percentiles.
    """
    risk = futures_pnl.rolling(60, min_periods=20).std().shift(1)
    prediction_dispersion = (predictions / risk.replace(0.0, np.nan)).std(axis=1)
    curve_signal = (_feature_frame(feature_panels, "curve_spread") + _feature_frame(feature_panels, "curve_ratio")) / 2.0
    momentum_signal = _feature_frame(feature_panels, "mom_60")
    curve_dispersion = curve_signal.std(axis=1)
    momentum_dispersion = momentum_signal.std(axis=1)

    parts = pd.DataFrame(
        {
            "prediction": expanding_percentile_score(prediction_dispersion, min_periods),
            "curve": expanding_percentile_score(curve_dispersion, min_periods),
            "momentum": expanding_percentile_score(momentum_dispersion, min_periods),
        }
    )
    return parts.mean(axis=1).fillna(0.0), parts


def natural_regime_weighted_positions(
    predictions,
    futures_pnl,
    feature_panels,
    low_exposure=0.35,
    high_exposure=1.00,
):
    """Continuous non-label regime allocation based on observable opportunity.

    Instead of switching by historical regime names or fixed cutoffs, exposure is
    scaled smoothly by the lagged expanding opportunity score.
    """
    opportunity, components = natural_opportunity_weight(predictions, futures_pnl, feature_panels)
    base_positions = model_predictions_to_positions(predictions, futures_pnl)
    exposure = float(low_exposure) + (float(high_exposure) - float(low_exposure)) * opportunity
    return base_positions.mul(exposure, axis=0), opportunity, components


def grain_volatility_state(futures_pnl, halflife=15):
    """Lagged realized-volatility state for the grain complex."""
    asset_vol = futures_pnl.rolling(60, min_periods=20).std().mean(axis=1)
    ewm_vol = asset_vol.ewm(halflife=float(halflife), adjust=False).mean().shift(1)
    long_vol = ewm_vol.expanding(min_periods=252).mean().shift(1)
    vol_ratio = (ewm_vol / long_vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    low_vol = (vol_ratio < 0.70).astype(float).fillna(0.0)
    high_vol = (vol_ratio > 1.30).astype(float).fillna(0.0)
    crisis_vol = (vol_ratio > 2.00).astype(float).fillna(0.0)
    return pd.DataFrame(
        {
            "ewm_vol": ewm_vol,
            "long_vol": long_vol,
            "vol_ratio": vol_ratio,
            "low_vol": low_vol,
            "high_vol": high_vol,
            "crisis_vol": crisis_vol,
        }
    )


def observable_three_sleeve_weights(predictions, futures_pnl, feature_panels):
    """Live-observable weights for static, 2-day, and opportunity sleeves.

    Inputs are lagged realized volatility and lagged expanding opportunity
    percentiles. Historical regime labels are not used.
    """
    opportunity, opportunity_parts = natural_opportunity_weight(predictions, futures_pnl, feature_panels)
    vol_state = grain_volatility_state(futures_pnl)
    vol_ratio = vol_state["vol_ratio"].clip(lower=0.0, upper=3.0).fillna(1.0)
    low_score = (1.0 - (vol_ratio / 0.70)).clip(lower=0.0, upper=1.0)
    stress_score = ((vol_ratio - 1.0) / 1.0).clip(lower=0.0, upper=1.0)
    normal_score = (1.0 - (vol_ratio - 1.0).abs()).clip(lower=0.0, upper=1.0)

    raw = pd.DataFrame(index=futures_pnl.index)
    raw["static_edge_filtered"] = 0.25 + 0.35 * normal_score + 0.20 * (1.0 - opportunity)
    raw["skip_rebalance_2d"] = 0.30 + 0.45 * low_score + 0.15 * (1.0 - stress_score)
    raw["multi_condition"] = 0.20 + 0.55 * opportunity + 0.35 * stress_score
    raw = raw.clip(lower=0.05)
    weights = raw.div(raw.sum(axis=1), axis=0).fillna(1.0 / 3.0)
    diagnostics = pd.concat(
        [
            opportunity.rename("opportunity_score"),
            opportunity_parts.add_prefix("opportunity_"),
            vol_state,
        ],
        axis=1,
    )
    return weights, diagnostics


def observable_three_sleeve_positions(
    predictions,
    futures_pnl,
    feature_panels,
    base_positions,
    apply_vol_scale=False,
    target_vol_ratio=1.0,
    min_scale=0.35,
    max_scale=1.25,
):
    """Blend three pre-defined sleeves using observable, non-label weights."""
    multi_positions, active = multi_condition_filter_positions(
        predictions,
        futures_pnl,
        feature_panels,
        FINAL_OPPORTUNITY_QUANTILES["prediction"],
        FINAL_OPPORTUNITY_QUANTILES["curve"],
        FINAL_OPPORTUNITY_QUANTILES["momentum"],
    )
    sleeves = {
        "static_edge_filtered": base_positions,
        "skip_rebalance_2d": skip_rebalance_positions(base_positions, 2),
        "multi_condition": multi_positions,
    }
    weights, diagnostics = observable_three_sleeve_weights(predictions, futures_pnl, feature_panels)
    blended = None
    for name, positions in sleeves.items():
        weighted = positions.mul(weights[name], axis=0)
        blended = weighted if blended is None else blended.add(weighted, fill_value=0.0)

    vol_scale = pd.Series(1.0, index=futures_pnl.index)
    if apply_vol_scale:
        vol_ratio = diagnostics["vol_ratio"].replace(0.0, np.nan)
        vol_scale = (float(target_vol_ratio) / vol_ratio).clip(
            lower=float(min_scale),
            upper=float(max_scale),
        )
        crisis_cap = pd.Series(1.0, index=futures_pnl.index)
        crisis_cap.loc[diagnostics["crisis_vol"] > 0.0] = 0.50
        vol_scale = (vol_scale * crisis_cap).fillna(1.0)
        blended = blended.mul(vol_scale, axis=0)

    diagnostics = diagnostics.copy()
    diagnostics["multi_condition_active"] = active
    diagnostics["vol_scale"] = vol_scale
    for column in weights.columns:
        diagnostics["weight_" + column] = weights[column]
    return blended.fillna(0.0), weights, diagnostics


def trailing_performance_ensemble_positions(strategy_positions, futures_pnl, lookback=252, temperature=1.0):
    """Blend predefined sleeves using lagged trailing Sharpe-style weights.

    This is a natural regime-weighting alternative: no historical regime labels
    are used, and weights are based only on each sleeve's trailing realized PnL.
    """
    names = list(strategy_positions.keys())
    pnl_by_strategy = pd.DataFrame(index=futures_pnl.index)
    for name in names:
        bt, _ = backtest_positions(strategy_positions[name], futures_pnl, 0.0)
        pnl_by_strategy[name] = bt["net_pnl"]

    mean = pnl_by_strategy.rolling(int(lookback), min_periods=max(60, int(lookback) // 4)).mean()
    vol = pnl_by_strategy.rolling(int(lookback), min_periods=max(60, int(lookback) // 4)).std()
    score = (mean / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    score = score.shift(1).clip(lower=-2.0, upper=2.0).fillna(0.0)
    exp_score = np.exp(score / float(temperature))
    weights = exp_score.div(exp_score.sum(axis=1), axis=0).fillna(1.0 / float(len(names)))

    out = None
    for name in names:
        weighted = strategy_positions[name].mul(weights[name], axis=0)
        out = weighted if out is None else out.add(weighted, fill_value=0.0)
    return out.fillna(0.0), weights, score


def carry_storage_positions(feature_panels, futures_pnl):
    """Simple cross-sectional carry/storage sleeve."""
    curve = (_feature_frame(feature_panels, "curve_spread") + _feature_frame(feature_panels, "curve_ratio")) / 2.0
    storage = (
        -_feature_frame(feature_panels, "public_inventory_level")
        - _feature_frame(feature_panels, "public_inventory_change")
        - _feature_frame(feature_panels, "cgl_inventory_change")
    ) / 3.0
    signal = (0.6 * curve + 0.4 * storage).clip(lower=-5.0, upper=5.0)
    return model_predictions_to_positions(signal, futures_pnl)


def smooth_positions(positions, halflife=2, rescale=False):
    """EWMA-smooth positions; optionally rescale to the original gross exposure."""
    smoothed = positions.ewm(halflife=float(halflife), adjust=False).mean().fillna(0.0)
    if rescale:
        target = positions.abs().sum(axis=1).replace(0.0, np.nan)
        current = smoothed.abs().sum(axis=1).replace(0.0, np.nan)
        smoothed = smoothed.mul((target / current).clip(upper=2.0).fillna(0.0), axis=0)
    return smoothed.clip(lower=-1.0, upper=1.0)


def run_filter_sleeve_experiment(data_dir="train_set", split_date="2018-01-01"):
    """Summarise recent filter, holding-period, smoothing, and carry/storage tests."""
    results = run_research_pipeline(data_dir=data_dir, split_date=split_date)
    pnl = results["futures_pnl"]
    rows = []

    base_positions = results["model_positions"]
    rows.append(
        _experiment_score_row(
            "Baseline static edge-filtered",
            base_positions,
            pnl,
            split_date,
            1,
            "Two-block Ridge plus prediction-dispersion edge filter.",
            "Baseline tradable candidate",
        )
    )

    rows.append(
        _experiment_score_row(
            "2-day skip-rebalance",
            skip_rebalance_positions(base_positions, 2),
            pnl,
            split_date,
            2,
            "Hold positions for two trading days to reduce churn and preserve conviction.",
            "Best OOS PnL / lower-turnover execution variant",
        )
    )

    multi_positions, active = multi_condition_filter_positions(
        results["predictions"], pnl, results["feature_panels"], 0.40, 0.40, 0.40
    )
    row = _experiment_score_row(
        "Multi-condition opportunity filter",
        multi_positions,
        pnl,
        split_date,
        1,
        "Trade only when prediction, curve, and momentum dispersion are all high.",
        "Best crisis-regime and Sharpe candidate",
    )
    row["active_day_fraction"] = float(active.mean())
    rows.append(row)

    natural_positions, opportunity, _ = natural_regime_weighted_positions(
        results["predictions"], pnl, results["feature_panels"], 0.35, 1.00
    )
    row = _experiment_score_row(
        "Natural opportunity-weighted exposure",
        natural_positions,
        pnl,
        split_date,
        1,
        "Continuously scale exposure by expanding prediction/curve/momentum opportunity percentiles.",
        "Lower-overfit regime weighting candidate",
    )
    row["avg_opportunity_weight"] = float(opportunity.mean())
    rows.append(row)

    ensemble_positions, ensemble_weights, _ = trailing_performance_ensemble_positions(
        {
            "baseline": base_positions,
            "skip_2d": skip_rebalance_positions(base_positions, 2),
            "natural_opportunity": natural_positions,
        },
        pnl,
        lookback=252,
        temperature=1.0,
    )
    row = _experiment_score_row(
        "Trailing-performance natural ensemble",
        ensemble_positions,
        pnl,
        split_date,
        1,
        "Blend baseline, 2-day, and natural-opportunity sleeves by lagged trailing performance weights.",
        "Natural regime-weighted ensemble candidate",
    )
    for column in ensemble_weights.columns:
        row["avg_weight_" + column] = float(ensemble_weights[column].mean())
    rows.append(row)

    three_sleeve_positions, three_sleeve_weights, _ = observable_three_sleeve_positions(
        results["predictions"], pnl, results["feature_panels"], base_positions, apply_vol_scale=False
    )
    row = _experiment_score_row(
        "Observable three-sleeve regime blend",
        three_sleeve_positions,
        pnl,
        split_date,
        1,
        "Blend static, 2-day, and multi-condition sleeves using lagged volatility and opportunity weights.",
        "Live-observable regime blend candidate",
    )
    for column in three_sleeve_weights.columns:
        row["avg_weight_" + column] = float(three_sleeve_weights[column].mean())
    rows.append(row)

    scaled_three_sleeve_positions, scaled_three_sleeve_weights, scaled_diagnostics = observable_three_sleeve_positions(
        results["predictions"], pnl, results["feature_panels"], base_positions, apply_vol_scale=True
    )
    row = _experiment_score_row(
        "Observable three-sleeve blend + vol scale",
        scaled_three_sleeve_positions,
        pnl,
        split_date,
        1,
        "Same three-sleeve blend with lagged realized-volatility scaling and crisis cap.",
        "Risk-managed regime blend candidate",
    )
    for column in scaled_three_sleeve_weights.columns:
        row["avg_weight_" + column] = float(scaled_three_sleeve_weights[column].mean())
    row["avg_vol_scale"] = float(scaled_diagnostics["vol_scale"].mean())
    rows.append(row)

    smooth = smooth_positions(base_positions, halflife=2, rescale=False)
    rows.append(
        _experiment_score_row(
            "Position smoothing HL2",
            smooth,
            pnl,
            split_date,
            1,
            "EWMA smooth positions to reduce turnover.",
            "Rejected: turnover reduction only",
        )
    )

    carry = carry_storage_positions(results["feature_panels"], pnl)
    carry_overlay = base_positions * 0.80 + carry * 0.20
    rows.append(
        _experiment_score_row(
            "20% carry/storage overlay",
            carry_overlay,
            pnl,
            split_date,
            1,
            "Add simple carry/backwardation and inventory-draw sleeve.",
            "Rejected: weak full-period robustness",
        )
    )

    table = pd.DataFrame(rows)
    return {"filter_sleeve_table": table, "multi_condition_active": active}


def _cost_case_row(strategy, case, positions, pnl, split_date):
    bt, _ = backtest_positions_with_costs(
        positions,
        pnl,
        trade_cost_per_lot=case["trade_cost_per_lot"],
        holding_cost_rate=case["holding_cost_rate"],
        margin_budget=case["margin_budget"],
    )
    metrics = split_performance(bt, split_date)
    active = bt["held_gross_exposure"] > 1.0e-12
    active_bt = bt.loc[active]
    gross_total = float(active_bt["gross_pnl"].sum()) if len(active_bt) else 0.0
    trade_cost = float(active_bt["trade_cost"].sum()) if len(active_bt) else 0.0
    holding_cost = float(active_bt["holding_cost"].sum()) if len(active_bt) else 0.0
    total_cost = trade_cost + holding_cost
    cdf = np.nan if abs(gross_total) < 1.0e-12 else total_cost / abs(gross_total)
    return {
        "strategy": strategy,
        "case": case["case"],
        "case_description": case["description"],
        "trade_cost_per_lot": case["trade_cost_per_lot"],
        "holding_cost_rate": case["holding_cost_rate"],
        "margin_budget": case["margin_budget"],
        "CDF": cdf,
        "trade_cost": trade_cost,
        "holding_cost": holding_cost,
        "total_cost": total_cost,
        "avg_margin_used": float(active_bt["margin_used"].mean()) if len(active_bt) else np.nan,
        "avg_margin_scale": float(active_bt["margin_scale"].mean()) if len(active_bt) else np.nan,
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "max_dd": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
    }


def run_cost_margin_experiment(data_dir="train_set", split_date="2018-01-01", cost_cases=None):
    """Run transaction-cost, holding-cost, and margin-budget stress tests."""
    if cost_cases is None:
        cost_cases = COST_CASES
    results = run_research_pipeline(data_dir=data_dir, split_date=split_date)
    pnl = results["futures_pnl"]
    multi_positions, _ = multi_condition_filter_positions(
        results["predictions"], pnl, results["feature_panels"], 0.40, 0.40, 0.40
    )
    strategies = [
        ("Static edge-filtered", results["model_positions"]),
        ("2-day skip-rebalance", skip_rebalance_positions(results["model_positions"], 2)),
        ("Multi-condition filter", multi_positions),
        ("Annual walk-forward Ridge", results["walk_forward_positions"]),
    ]
    rows = []
    for strategy, positions in strategies:
        for case in cost_cases:
            rows.append(_cost_case_row(strategy, case, positions, pnl, split_date))
    table = pd.DataFrame(rows)
    return {"cost_margin_table": table}


def _window_final_metrics(label, bt):
    empty_row = {
        "window": label,
        "trading_days": 0,
        "winning_days": 0,
        "losing_days": 0,
        "flat_days": 0,
        "hit_rate": np.nan,
        "sharpe": np.nan,
        "total_pnl": 0.0,
        "gross_pnl": 0.0,
        "trade_cost": 0.0,
        "holding_cost": 0.0,
        "total_cost": 0.0,
        "CDF": np.nan,
        "max_drawdown": np.nan,
        "avg_daily_turnover": np.nan,
        "avg_gross_exposure": np.nan,
        "avg_margin_used": np.nan,
        "avg_margin_scale": np.nan,
    }
    active_bt = bt.loc[bt["held_gross_exposure"] > 1.0e-12]
    metrics = performance_metrics(bt)
    if len(active_bt) == 0 or len(metrics) == 0:
        return empty_row

    gross_pnl = float(active_bt["gross_pnl"].sum())
    trade_cost = float(active_bt["trade_cost"].sum()) if "trade_cost" in active_bt.columns else 0.0
    holding_cost = float(active_bt["holding_cost"].sum()) if "holding_cost" in active_bt.columns else 0.0
    total_cost = trade_cost + holding_cost
    cdf = np.nan if abs(gross_pnl) < 1.0e-12 else total_cost / abs(gross_pnl)
    pnl = active_bt["net_pnl"]
    return {
        "window": label,
        "trading_days": int(len(active_bt)),
        "winning_days": int((pnl > 0.0).sum()),
        "losing_days": int((pnl < 0.0).sum()),
        "flat_days": int((pnl == 0.0).sum()),
        "hit_rate": metrics["hit_rate"],
        "sharpe": metrics["sharpe"],
        "total_pnl": metrics["total_pnl"],
        "gross_pnl": gross_pnl,
        "trade_cost": trade_cost,
        "holding_cost": holding_cost,
        "total_cost": total_cost,
        "CDF": cdf,
        "max_drawdown": metrics["max_drawdown"],
        "avg_daily_turnover": metrics["avg_daily_turnover"],
        "avg_gross_exposure": metrics["avg_gross_exposure"],
        "avg_margin_used": float(active_bt["margin_used"].mean()) if "margin_used" in active_bt else np.nan,
        "avg_margin_scale": float(active_bt["margin_scale"].mean()) if "margin_scale" in active_bt else np.nan,
    }


def run_final_strategy_selection(
    data_dir="train_set",
    split_date="2018-01-01",
    trade_cost_per_lot=8.75,
    holding_cost_rate=0.05,
    margin_budget=np.inf,
    skip_weight=FINAL_BLEND_WEIGHTS["skip_rebalance"],
    multi_condition_weight=FINAL_BLEND_WEIGHTS["multi_condition"],
):
    """Build the fixed-weight final blend and report cost-adjusted metrics.

    The blend weights are fixed before scoring: half lower-turnover 2-day
    execution sleeve, half high-Sharpe opportunity-filter sleeve.
    """
    results = run_research_pipeline(data_dir=data_dir, split_date=split_date)
    pnl = results["futures_pnl"]
    base_positions = results["model_positions"]
    skip_positions = skip_rebalance_positions(base_positions, 2)
    multi_positions, active = multi_condition_filter_positions(
        results["predictions"],
        pnl,
        results["feature_panels"],
        FINAL_OPPORTUNITY_QUANTILES["prediction"],
        FINAL_OPPORTUNITY_QUANTILES["curve"],
        FINAL_OPPORTUNITY_QUANTILES["momentum"],
    )
    final_positions = (
        skip_positions * float(skip_weight)
        + multi_positions * float(multi_condition_weight)
    )
    bt, pnl_by_asset = backtest_positions_with_costs(
        final_positions,
        pnl,
        trade_cost_per_lot=trade_cost_per_lot,
        holding_cost_rate=holding_cost_rate,
        margin_budget=margin_budget,
    )

    split = pd.Timestamp(split_date)
    windows = [
        ("in_sample", bt.loc[bt.index < split]),
        ("out_of_sample", bt.loc[bt.index >= split]),
        ("full_period", bt),
    ]
    metrics = pd.DataFrame([_window_final_metrics(label, window_bt) for label, window_bt in windows])
    assumptions = pd.DataFrame(
        [
            {
                "assumption": "final_strategy",
                "value": "Fixed 50/50 blend: 2-day skip-rebalance + multi-condition opportunity filter",
            },
            {"assumption": "skip_rebalance_weight", "value": float(skip_weight)},
            {"assumption": "multi_condition_weight", "value": float(multi_condition_weight)},
            {
                "assumption": "prediction_dispersion_quantile",
                "value": FINAL_OPPORTUNITY_QUANTILES["prediction"],
            },
            {
                "assumption": "curve_dispersion_quantile",
                "value": FINAL_OPPORTUNITY_QUANTILES["curve"],
            },
            {
                "assumption": "momentum_dispersion_quantile",
                "value": FINAL_OPPORTUNITY_QUANTILES["momentum"],
            },
            {"assumption": "threshold_policy", "value": "Expanding and one-day lagged; not retuned after holdout review"},
            {"assumption": "trade_cost_per_lot", "value": float(trade_cost_per_lot)},
            {"assumption": "holding_cost_rate_annual", "value": float(holding_cost_rate)},
            {"assumption": "margin_budget", "value": margin_budget},
            {"assumption": "selection_policy", "value": "Blend weights are fixed, not optimized on Sharpe/PnL"},
        ]
    )
    return {
        "assumptions": assumptions,
        "final_strategy_metrics": metrics,
        "final_positions": final_positions,
        "final_backtest": bt,
        "final_pnl_by_asset": pnl_by_asset,
        "multi_condition_active": active,
    }


def run_lower_overfit_strategy(
    data_dir="train_set",
    split_date="2018-01-01",
    edge_quantile=0.50,
    cost_per_lot=0.0,
):
    """Run the pre-registered lower-overfit strategy.

    This is the strategy to present when overfit control matters more than the
    best static backtest. It uses only fixed feature blocks, fixed Ridge
    penalties, annual expanding walk-forward retraining, and a fixed expanding
    median edge filter. No external-data overlays and no full-sample weight
    search are used.
    """
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    predictions, coefficients = build_walk_forward_model_signals(feature_panels, futures_pnl)
    positions, edge_score, edge_threshold = edge_filtered_positions(
        predictions,
        futures_pnl,
        quantile=float(edge_quantile),
    )
    bt, pnl_by_asset = backtest_positions(positions, futures_pnl, cost_per_lot)
    metrics = split_performance(bt, split_date)
    assumptions = pd.DataFrame(
        [
            {"assumption": "model", "value": "Annual expanding walk-forward two-block Ridge"},
            {"assumption": "core_features", "value": ", ".join(OUTRIGHT_CORE_FEATURES)},
            {"assumption": "physical_features", "value": ", ".join(OUTRIGHT_PHYSICAL_FEATURES)},
            {"assumption": "core_alpha", "value": 25.0},
            {"assumption": "physical_alpha", "value": 1000.0},
            {"assumption": "target_horizon_days", "value": 20},
            {"assumption": "retrain_frequency", "value": "annual"},
            {"assumption": "minimum_training_days", "value": 756},
            {"assumption": "edge_filter", "value": "expanding lagged prediction-dispersion quantile"},
            {"assumption": "edge_quantile", "value": float(edge_quantile)},
            {"assumption": "position_sizing", "value": "lagged 60-day volatility risk-adjusted cross-sectional ranking"},
            {"assumption": "external_data", "value": "none in final lower-overfit strategy"},
            {"assumption": "selection_policy", "value": "fixed algorithm; no full-sample weight/grid search"},
            {"assumption": "remaining_risk", "value": "Ridge coefficients are still fitted on past data; this reduces but cannot eliminate overfit"},
        ]
    )
    return {
        "assumptions": assumptions,
        "metrics": metrics,
        "predictions": predictions,
        "coefficients": coefficients,
        "positions": positions,
        "backtest": bt,
        "pnl_by_asset": pnl_by_asset,
        "edge_score": edge_score,
        "edge_threshold": edge_threshold,
    }


def no_fit_reversion_physical_signal(feature_panels, physical_weight=0.10):
    """Build a no-fit Cargill-aware short-term reversion signal.

    There are no learned coefficients. The base signal is the existing 5-day
    reversal feature. The physical tilt is deliberately small and fixed:
    public inventory pressure, receipts pressure, Cargill inventory pressure,
    and soybean-specific Cargill crush demand pressure.
    """
    reversion = _feature_frame(feature_panels, "rev_5")
    inventory_pressure = (
        -_feature_frame(feature_panels, "public_inventory_change")
        - _feature_frame(feature_panels, "receipts_change")
        - _feature_frame(feature_panels, "cgl_inventory_change")
    ) / 3.0
    crush_pressure = pd.DataFrame(0.0, index=reversion.index, columns=reversion.columns)
    crush_pressure["SOYABEAN"] = (
        _feature_frame(feature_panels, "crush_surprise")["SOYABEAN"]
        + _feature_frame(feature_panels, "crush_utilization")["SOYABEAN"]
    ) / 2.0
    physical_pressure = (inventory_pressure + crush_pressure) / 2.0
    return (reversion + float(physical_weight) * physical_pressure).clip(lower=-5.0, upper=5.0)


def volatility_target_positions(positions, futures_pnl, target_daily_pnl_vol=120.0, max_scale=1.0, lookback=60):
    """Scale positions by lagged realized strategy PnL volatility."""
    base_bt, _ = backtest_positions(positions, futures_pnl, 0.0)
    realized = base_bt["net_pnl"].rolling(int(lookback), min_periods=20).std().shift(1)
    scale = (float(target_daily_pnl_vol) / realized.replace(0.0, np.nan)).clip(upper=float(max_scale)).fillna(0.0)
    return positions.mul(scale, axis=0), scale


def run_no_fit_reversion_strategy(
    data_dir="train_set",
    split_date="2018-01-01",
    physical_weight=0.10,
    target_daily_pnl_vol=120.0,
    max_scale=1.0,
    trade_cost_per_lot=8.75,
    holding_cost_rate=0.05,
):
    """Run a no-regression, Cargill-aware reversion strategy.

    This candidate removes fitted Ridge coefficients entirely. It can still be
    overfit through research choices, but the live algorithm has no learned
    beta/weight vector.
    """
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    signal = no_fit_reversion_physical_signal(feature_panels, physical_weight=physical_weight)
    raw_positions = model_predictions_to_positions(signal, futures_pnl)
    positions, vol_scale = volatility_target_positions(
        raw_positions,
        futures_pnl,
        target_daily_pnl_vol=target_daily_pnl_vol,
        max_scale=max_scale,
    )
    zero_bt, zero_pnl_by_asset = backtest_positions(positions, futures_pnl, 0.0)
    cost_bt, cost_pnl_by_asset = backtest_positions_with_costs(
        positions,
        futures_pnl,
        trade_cost_per_lot=trade_cost_per_lot,
        holding_cost_rate=holding_cost_rate,
    )
    assumptions = pd.DataFrame(
        [
            {"assumption": "model", "value": "No-fit short-term reversion + fixed physical pressure tilt"},
            {"assumption": "base_signal", "value": "rev_5"},
            {
                "assumption": "physical_tilt",
                "value": "0.10 * average(public inventory pressure, receipts pressure, Cargill inventory pressure, soybean Cargill crush pressure)",
            },
            {"assumption": "uses_cgl_inv", "value": True},
            {"assumption": "uses_cgl_crush", "value": "soybean-specific fixed demand-pressure tilt"},
            {"assumption": "learned_coefficients", "value": "none"},
            {"assumption": "physical_weight", "value": float(physical_weight)},
            {"assumption": "target_daily_pnl_vol", "value": float(target_daily_pnl_vol)},
            {"assumption": "max_scale", "value": float(max_scale)},
            {"assumption": "vol_target_lookback", "value": 60},
            {"assumption": "trade_cost_per_lot", "value": float(trade_cost_per_lot)},
            {"assumption": "holding_cost_rate_annual", "value": float(holding_cost_rate)},
            {
                "assumption": "remaining_risk",
                "value": "No fitted Ridge coefficients, but research choices still require untouched holdout validation.",
            },
        ]
    )
    return {
        "assumptions": assumptions,
        "zero_cost_metrics": split_performance(zero_bt, split_date),
        "cost_adjusted_metrics": split_performance(cost_bt, split_date),
        "signal": signal,
        "positions": positions,
        "raw_positions": raw_positions,
        "vol_scale": vol_scale,
        "zero_cost_backtest": zero_bt,
        "zero_cost_pnl_by_asset": zero_pnl_by_asset,
        "cost_adjusted_backtest": cost_bt,
        "cost_adjusted_pnl_by_asset": cost_pnl_by_asset,
    }


def _strategy_cost_summary_row(name, positions, futures_pnl, split_date, cost_case):
    bt, _ = backtest_positions_with_costs(
        positions,
        futures_pnl,
        trade_cost_per_lot=cost_case["trade_cost_per_lot"],
        holding_cost_rate=cost_case["holding_cost_rate"],
        margin_budget=cost_case["margin_budget"],
    )
    metrics = split_performance(bt, split_date)
    active_bt = bt.loc[bt["held_gross_exposure"] > 1.0e-12]
    gross_pnl = float(active_bt["gross_pnl"].sum()) if len(active_bt) else 0.0
    trade_cost = float(active_bt["trade_cost"].sum()) if len(active_bt) else 0.0
    holding_cost = float(active_bt["holding_cost"].sum()) if len(active_bt) else 0.0
    total_cost = trade_cost + holding_cost
    total_turnover = float(active_bt["turnover"].sum()) if len(active_bt) else 0.0
    total_margin_days = float(active_bt["margin_used"].sum()) if len(active_bt) else 0.0
    expected_trade_cost = total_turnover * float(cost_case["trade_cost_per_lot"])
    expected_holding_cost = total_margin_days * float(cost_case["holding_cost_rate"]) / 252.0
    return {
        "strategy": name,
        "case": cost_case["case"],
        "trade_cost_per_lot": cost_case["trade_cost_per_lot"],
        "holding_cost_rate": cost_case["holding_cost_rate"],
        "margin_budget": cost_case["margin_budget"],
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "full_pnl": metrics.loc["total_pnl", "full_period"],
        "full_hit_rate": metrics.loc["hit_rate", "full_period"],
        "max_dd": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
        "total_turnover_lots": total_turnover,
        "margin_dollar_days": total_margin_days,
        "expected_trade_cost": expected_trade_cost,
        "expected_holding_cost": expected_holding_cost,
        "trade_cost": trade_cost,
        "holding_cost": holding_cost,
        "total_cost": total_cost,
        "CDF": np.nan if abs(gross_pnl) < 1.0e-12 else total_cost / abs(gross_pnl),
    }


def run_observable_regime_weight_experiment(
    data_dir="train_set",
    split_date="2018-01-01",
    cost_case=None,
):
    """Compare fixed and observable three-sleeve regime blends."""
    if cost_case is None:
        cost_case = COST_CASES[1]
    results = run_research_pipeline(data_dir=data_dir, split_date=split_date)
    pnl = results["futures_pnl"]
    base_positions = results["model_positions"]
    skip_positions = skip_rebalance_positions(base_positions, 2)
    multi_positions, _ = multi_condition_filter_positions(
        results["predictions"],
        pnl,
        results["feature_panels"],
        FINAL_OPPORTUNITY_QUANTILES["prediction"],
        FINAL_OPPORTUNITY_QUANTILES["curve"],
        FINAL_OPPORTUNITY_QUANTILES["momentum"],
    )
    fixed_two_sleeve = 0.50 * skip_positions + 0.50 * multi_positions
    observable_positions, weights, diagnostics = observable_three_sleeve_positions(
        results["predictions"], pnl, results["feature_panels"], base_positions, apply_vol_scale=False
    )
    scaled_positions, scaled_weights, scaled_diagnostics = observable_three_sleeve_positions(
        results["predictions"], pnl, results["feature_panels"], base_positions, apply_vol_scale=True
    )
    strategies = [
        ("Static edge-filtered", base_positions),
        ("2-day skip-rebalance", skip_positions),
        ("Multi-condition filter", multi_positions),
        ("Fixed 50/50 skip + multi", fixed_two_sleeve),
        ("Observable three-sleeve blend", observable_positions),
        ("Observable three-sleeve blend + vol scale", scaled_positions),
    ]
    table = pd.DataFrame(
        [_strategy_cost_summary_row(name, positions, pnl, split_date, cost_case) for name, positions in strategies]
    )
    weight_summary = pd.DataFrame(
        [
            {
                "strategy": "Observable three-sleeve blend",
                "avg_static_weight": float(weights["static_edge_filtered"].mean()),
                "avg_skip_weight": float(weights["skip_rebalance_2d"].mean()),
                "avg_multi_condition_weight": float(weights["multi_condition"].mean()),
                "avg_vol_scale": 1.0,
            },
            {
                "strategy": "Observable three-sleeve blend + vol scale",
                "avg_static_weight": float(scaled_weights["static_edge_filtered"].mean()),
                "avg_skip_weight": float(scaled_weights["skip_rebalance_2d"].mean()),
                "avg_multi_condition_weight": float(scaled_weights["multi_condition"].mean()),
                "avg_vol_scale": float(scaled_diagnostics["vol_scale"].mean()),
            },
        ]
    )
    assumptions = pd.DataFrame(
        [
            {"assumption": "sleeves", "value": "static edge-filtered, 2-day skip-rebalance, multi-condition filter"},
            {"assumption": "regime_inputs", "value": "lagged realized grain volatility and lagged opportunity score"},
            {"assumption": "no_named_regimes", "value": "drought/COVID/trade-war labels are diagnostics only"},
            {"assumption": "vol_low_rule", "value": "ewm_vol / expanding_long_vol < 0.70"},
            {"assumption": "vol_high_rule", "value": "ewm_vol / expanding_long_vol > 1.30"},
            {"assumption": "crisis_cap_rule", "value": "if vol_ratio > 2.00, vol-scaled variant applies 0.50 cap"},
            {"assumption": "cost_case", "value": cost_case["case"]},
            {"assumption": "trade_cost_per_lot", "value": cost_case["trade_cost_per_lot"]},
            {"assumption": "holding_cost_rate", "value": cost_case["holding_cost_rate"]},
            {
                "assumption": "cost_interpretation",
                "value": "Same unit cost assumptions for every row; realized total cost differs because turnover and margin use differ.",
            },
        ]
    )
    return {
        "assumptions": assumptions,
        "regime_weight_table": table,
        "weight_summary": weight_summary,
        "weights": weights,
        "diagnostics": diagnostics,
    }
