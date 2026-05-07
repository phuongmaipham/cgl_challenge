"""Shared notebook helpers for the grain strategy backtest research notebooks."""

import numpy as np
import pandas as pd

from grain_backtest_core import (
    backtest_positions_with_costs,
    build_feature_panels,
    load_train_set,
    performance_metrics,
    rolling_zscore,
)
from research_config import DEFAULT_MARGIN_PER_LOT, REGIME_PERIODS


def clean_signal(series, target_index):
    return (
        pd.Series(series, index=target_index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-5.0, 5.0)
    )


def _flatten(members):
    return list(members.values()) if isinstance(members, dict) else list(members)


def _mean_signals(signals, target_index):
    values = [pd.Series(s, index=target_index) for s in signals if s is not None]
    if not values:
        return pd.Series(0.0, index=target_index)
    return clean_signal(pd.concat(values, axis=1).mean(axis=1), target_index)


def family_average(families, target_index):
    return _mean_signals([s for m in families.values() for s in _flatten(m)], target_index)


def equal_family(families, target_index):
    return _mean_signals(
        [_mean_signals(_flatten(m), target_index) for m in families.values()],
        target_index,
    )


def family_feature_frame(families, target_index):
    return pd.DataFrame(
        {name: _mean_signals(_flatten(m), target_index) for name, m in families.items()},
        index=target_index,
    ).fillna(0.0)


def make_family_tests(families, target, target_index, trend_panel):
    features = family_feature_frame(families, target_index)
    trend_source = trend_panel(families, features, target_index) if callable(trend_panel) else trend_panel
    return {
        "avg_all_signals": family_average(families, target_index),
        "equal_family": equal_family(families, target_index),
        "best_family_by_trend_mr": best_family_by_trend_mr(families, trend_source, target_index),
        "select_by_ic": select_by_ic_signal(families, target, target_index),
        "expanding_ols_family_model": prediction_to_signal(expanding_ols_prediction(features, target), target_index),
        "kalman_family_model": prediction_to_signal(kalman_prediction(features, target), target_index),
    }


def run_family_tests(
    family_sets,
    target,
    target_index,
    backtest_for_signal,
    row_for_backtest,
    trend_panel=None,
    make_tests=None,
    position_for_signal=None,
    key_mode="long_short",
    sort_columns=("signal_set", "validation_sharpe", "oos_sharpe"),
    ascending=(True, False, False),
):
    rows, backtests, positions = [], {}, {}
    for signal_set, families in family_sets.items():
        tests = (
            make_tests(families, target, target_index, signal_set)
            if make_tests is not None
            else make_family_tests(families, target, target_index, trend_panel)
        )
        for strategy, signal in tests.items():
            bt = backtest_for_signal(signal)
            rows.append(row_for_backtest(signal_set, strategy, bt))
            key = (signal_set, strategy, key_mode)
            backtests[key] = bt
            if position_for_signal is not None:
                positions[key] = position_for_signal(signal)
    results = pd.DataFrame(rows).sort_values(list(sort_columns), ascending=list(ascending))
    return {"results": results, "backtests": backtests, "positions": positions}


def _shaped_signal(signal, target_index, halflife, threshold):
    s = clean_signal(signal, target_index)
    s = pd.Series(np.tanh(s / 2.0), index=target_index).ewm(
        halflife=float(halflife), adjust=False, min_periods=1
    ).mean()
    s[s.abs() < float(threshold)] = 0.0
    return s


def positions_from_signal(
    signal,
    pnl,
    commodity,
    target_daily_vol=75.0,
    max_abs_lot=0.50,
    halflife=3.0,
    threshold=0.05,
    mode="long_short",
):
    s = _shaped_signal(signal, pnl.index, halflife, threshold)
    if mode == "long_only":
        s = s.clip(lower=0.0)
    elif mode == "short_only":
        s = s.clip(upper=0.0)
    elif mode != "long_short":
        raise ValueError(f"Unknown mode: {mode}")
    vol = pnl[commodity].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    lots = (s * float(target_daily_vol) / vol).clip(-float(max_abs_lot), float(max_abs_lot)).fillna(0.0)
    return pd.DataFrame({commodity: lots}, index=pnl.index)


def backtest_signal(
    signal,
    pnl,
    commodity,
    trade_cost_per_lot=8.75,
    holding_cost_rate=0.05,
    margin_per_lot=DEFAULT_MARGIN_PER_LOT,
    target_daily_vol=75.0,
    max_abs_lot=0.50,
    halflife=3.0,
    threshold=0.05,
    mode="long_short",
):
    positions = positions_from_signal(
        signal, pnl, commodity,
        target_daily_vol=target_daily_vol, max_abs_lot=max_abs_lot,
        halflife=halflife, threshold=threshold, mode=mode,
    )
    return backtest_positions_with_costs(
        positions, pnl,
        trade_cost_per_lot=trade_cost_per_lot,
        holding_cost_rate=holding_cost_rate,
        margin_per_lot=margin_per_lot,
    )[0]


_METRIC_INPUT_KEYS = ("days", "total_pnl", "sharpe", "max_drawdown", "hit_rate", "avg_daily_turnover", "avg_gross_exposure")
_METRIC_OUTPUT_KEYS = _METRIC_INPUT_KEYS + ("turnover",)


def active_metrics(bt, mask=None):
    sample = bt if mask is None else bt.loc[mask]
    metrics = performance_metrics(sample)
    if metrics.empty:
        return {k: np.nan for k in _METRIC_OUTPUT_KEYS}
    out = {k: metrics.get(k, np.nan) for k in _METRIC_INPUT_KEYS}
    out["turnover"] = out["avg_daily_turnover"]
    return out


def metric_row(label, bt, train_end, oos_start, dd_capital_usd, mode="long_short"):
    train = active_metrics(bt, bt.index < train_end)
    validation = active_metrics(bt, (bt.index >= train_end) & (bt.index < oos_start))
    oos = active_metrics(bt, bt.index >= oos_start)
    full = active_metrics(bt)
    return {
        "strategy": label,
        "mode": mode,
        "train_sharpe": train["sharpe"],
        "validation_sharpe": validation["sharpe"],
        "oos_sharpe": oos["sharpe"],
        "oos_pnl": oos["total_pnl"],
        "oos_dd_pct": oos["max_drawdown"] / dd_capital_usd * 100.0,
        "full_sharpe": full["sharpe"],
        "turnover": full["turnover"],
    }


def zscore_from_train(x_train, x_row):
    mean = x_train.mean()
    std = x_train.std().replace(0.0, np.nan)
    return (
        ((x_train - mean) / std).clip(-5.0, 5.0).fillna(0.0),
        ((x_row - mean) / std).clip(-5.0, 5.0).fillna(0.0),
    )


def expanding_ols_prediction(x, y, min_train_days=504, refit_every=21):
    preds = pd.Series(np.nan, index=x.index)
    beta, last_fit = None, None
    for i, date in enumerate(x.index):
        train_mask = (x.index < date) & y.notna()
        if train_mask.sum() < min_train_days:
            continue
        if beta is None or last_fit is None or (i - last_fit) >= refit_every:
            x_train, x_row = zscore_from_train(x.loc[train_mask], x.loc[date])
            design = np.column_stack([np.ones(len(x_train)), x_train.values])
            beta, *_ = np.linalg.lstsq(design, y.loc[train_mask].values.astype(float), rcond=None)
            last_fit = i
        else:
            _, x_row = zscore_from_train(x.loc[train_mask], x.loc[date])
        preds.loc[date] = np.r_[1.0, x_row.values.astype(float)].dot(beta)
    return preds


def kalman_prediction(x, y, min_train_days=504, process_noise=1.0e-5):
    columns = list(x.columns)
    beta = np.zeros(len(columns) + 1)
    covariance = np.eye(len(beta)) * 10.0
    mean = pd.Series(0.0, index=columns)
    var = pd.Series(1.0, index=columns)
    target_var = 1.0
    preds = pd.Series(np.nan, index=x.index)
    n = 0
    for date in x.index:
        row = x.loc[date]
        if n > min_train_days:
            z = ((row - mean) / np.sqrt(var.clip(lower=1.0e-8))).clip(-5.0, 5.0)
            preds.loc[date] = np.r_[1.0, z.values.astype(float)].dot(beta)
        y_value = y.loc[date]
        if pd.notnull(y_value):
            n += 1
            old_mean = mean.copy()
            mean = mean + (row - mean) / float(n)
            var = ((n - 2.0) / max(n - 1.0, 1.0)) * var + (
                (row - old_mean) * (row - mean)
            ) / max(n - 1.0, 1.0)
            target_var = target_var + (float(y_value) ** 2 - target_var) / float(n)
            if n > min_train_days:
                z = ((row - mean) / np.sqrt(var.clip(lower=1.0e-8))).clip(-5.0, 5.0)
                phi = np.r_[1.0, z.values.astype(float)]
                covariance = covariance + np.eye(len(beta)) * float(process_noise)
                innovation_var = float(phi.dot(covariance).dot(phi) + max(target_var, 1.0))
                gain = covariance.dot(phi) / innovation_var
                beta = beta + gain * float(y_value - phi.dot(beta))
                covariance = covariance - np.outer(gain, phi).dot(covariance)
    return preds


def prediction_to_signal(prediction, target_index):
    prediction = prediction.replace([np.inf, -np.inf], np.nan)
    mean = prediction.rolling(252, min_periods=60).mean().shift(1)
    std = prediction.rolling(252, min_periods=60).std().shift(1).replace(0.0, np.nan)
    return clean_signal((prediction - mean) / std, target_index)


def _period_label(item):
    start = pd.Timestamp(item["start"])
    end = pd.Timestamp(item["end"])
    years = str(start.year) if start.year == end.year else f"{start.year}-{end.year}"
    return f"{years}: {item['period']}"


def period_metrics(bt, periods=REGIME_PERIODS):
    active_index = bt.index[bt["held_gross_exposure"] > 1.0e-12]
    first_active = active_index.min() if len(active_index) else None
    last_active = active_index.max() if len(active_index) else None
    rows = []
    for item in periods:
        start = pd.Timestamp(item["start"])
        end = pd.Timestamp(item["end"])
        mask = (bt.index >= start) & (bt.index <= end)
        metrics = active_metrics(bt, mask)
        active_days = 0 if pd.isnull(metrics["days"]) else int(metrics["days"])
        if active_days:
            note = ""
        elif first_active is None:
            note = "strategy never active"
        elif end < first_active:
            note = f"before first active trade ({first_active.year})"
        elif start > last_active:
            note = f"after last active trade ({last_active.year})"
        else:
            note = "strategy flat in this period"
        rows.append({
            "period": _period_label(item),
            "start": start,
            "end": end,
            "calendar_days": int(mask.sum()),
            "active_days": active_days,
            "note": note,
            **metrics,
        })
    return pd.DataFrame(rows)


def select_by_ic_signal(families, pnl, target_index, lookback=504):
    target = pnl.shift(-1)
    family_signals = {
        name: clean_signal(pd.concat(members, axis=1).mean(axis=1), target_index)
        for name, members in families.items()
    }
    out = pd.Series(0.0, index=target_index)
    for date in target_index:
        train = target_index < date
        recent = target.loc[train].tail(lookback)
        if recent.notna().sum() < 120:
            continue
        scores = {}
        for name, signal in family_signals.items():
            aligned = pd.concat([signal.loc[recent.index], recent], axis=1).dropna()
            scores[name] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) if len(aligned) > 60 else np.nan
        scores = pd.Series(scores).dropna()
        if scores.empty:
            continue
        out.loc[date] = family_signals[scores.abs().idxmax()].loc[date]
    return clean_signal(out, target_index)


def best_family_by_trend_mr(families, panel, target_index):
    trend_strength = panel["mom_60"].abs()
    threshold = trend_strength.expanding(min_periods=252).median().shift(1)
    trend_regime = (trend_strength > threshold).fillna(False)
    price_trend = pd.concat(families["price_trend"], axis=1).mean(axis=1)
    price_mr = pd.concat(families["price_mr"], axis=1).mean(axis=1)
    return clean_signal(price_trend.where(trend_regime, price_mr), target_index)


def named_period_check(bt, dd_capital_usd, periods=REGIME_PERIODS):
    df = period_metrics(bt, periods)[
        ["period", "total_pnl", "sharpe", "max_drawdown", "hit_rate", "active_days", "note"]
    ].copy()
    df["max_dd_pct"] = df["max_drawdown"] / dd_capital_usd * 100.0
    return df[["period", "total_pnl", "sharpe", "max_dd_pct", "hit_rate", "active_days", "note"]]


def pair_components(panel):
    return {
        "price_trend": clean_signal((panel["mom_20"] + panel["mom_60"]) / 2.0, panel.index),
        "price_mr": clean_signal(panel["rev_5"], panel.index),
        "curve": clean_signal(
            (panel["curve_spread"] + panel["curve_ratio"] + panel["curve_change_20"]) / 3.0,
            panel.index,
        ),
        "cot": clean_signal(
            (
                panel["cot_mm_level"]
                + panel["cot_mm_change"]
                + panel["cot_pm_oi_level"]
                + panel["cot_pm_oi_change"]
            ) / 4.0,
            panel.index,
        ),
        "physical_public": clean_signal(
            (-panel["public_inventory_change"] - panel["receipts_change"]) / 2.0,
            panel.index,
        ),
        "physical_cargill": clean_signal(
            (-panel["cgl_inventory_change"] + panel["crush_surprise"] + panel["crush_utilization"]) / 3.0,
            panel.index,
        ),
    }


def pair_signal(component_name, srw_components, hrw_components, target_index):
    return clean_signal(srw_components[component_name] - hrw_components[component_name], target_index)


def wheat_pair_positions(
    signal,
    pnl,
    wheat=("WHEAT_SRW", "WHEAT_HRW"),
    target_daily_pair_vol=40.0,
    max_leg_lot=0.40,
    signal_threshold=0.12,
    halflife=5.0,
    rebalance_every=5,
):
    s = _shaped_signal(signal, pnl.index, halflife, signal_threshold)
    vol = pnl[list(wheat)].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    positions = pd.DataFrame(0.0, index=pnl.index, columns=list(wheat))
    positions[wheat[0]] = (s * target_daily_pair_vol / vol[wheat[0]]).clip(-max_leg_lot, max_leg_lot)
    positions[wheat[1]] = (-s * target_daily_pair_vol / vol[wheat[1]]).clip(-max_leg_lot, max_leg_lot)
    positions = positions.fillna(0.0)
    if rebalance_every > 1:
        rebalance_mask = pd.Series(False, index=positions.index)
        rebalance_mask.iloc[::rebalance_every] = True
        positions = positions.where(rebalance_mask).ffill().fillna(0.0)
    return positions


def backtest_pair(
    signal,
    futures_pnl,
    wheat=("WHEAT_SRW", "WHEAT_HRW"),
    trade_cost_per_lot=8.75,
    holding_cost_rate=0.05,
    margin_per_lot=DEFAULT_MARGIN_PER_LOT,
):
    return backtest_positions_with_costs(
        wheat_pair_positions(signal, futures_pnl, wheat=wheat),
        futures_pnl[list(wheat)],
        trade_cost_per_lot=trade_cost_per_lot,
        holding_cost_rate=holding_cost_rate,
        margin_per_lot=margin_per_lot,
    )[0]


def pair_metric_row(label, bt, train_end, oos_start, dd_capital_usd):
    return {**metric_row(label, bt, train_end, oos_start, dd_capital_usd), "book": "SRW_HRW_PAIR"}
