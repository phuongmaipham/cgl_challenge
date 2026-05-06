"""Core data, feature, and backtest utilities for the grain notebooks."""

import numpy as np
import pandas as pd
from pathlib import Path

from research_config import COMMODITIES, CONTRACT_MULTIPLIER, DEFAULT_MARGIN_PER_LOT


TRAIN_SET_FILES = {
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


def _read_indexed_numeric_csv(data_dir, filename):
    path = Path(data_dir) / filename
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index().apply(pd.to_numeric, errors="coerce")


def load_train_set(data_dir="train_set"):
    """Load every train-set CSV into a dict of DataFrames keyed by dataset name."""
    return {
        key: _read_indexed_numeric_csv(data_dir, filename)
        for key, filename in TRAIN_SET_FILES.items()
    }


def to_available_calendar(df, trading_index, lag_days):
    """Shift observations to their first usable date, then forward-fill."""
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
    """Build one feature DataFrame per commodity with conservative release lags."""
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
        panels[commodity] = _signed_clip(frame)

    return panels, futures_pnl


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
    return pd.DataFrame(
        {
            "in_sample": performance_metrics(before),
            "out_of_sample": performance_metrics(after),
            "full_period": performance_metrics(bt),
        }
    )
