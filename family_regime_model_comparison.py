"""Family/regime/model comparison across grain products.

Requested strategy menu:
1. average all signals;
2. equal weight by signal family;
3. IC-threshold family filter, then validation-IC selected family;
4. trend/MR regime, select best family per regime and combine with fixed masks;
5. trend/MR regime, average families per regime and combine with fixed masks;
6. online OLS with Kalman filter;
7. expanding Ridge.

All variants use only the provided train_set feature panels. External data is
not fetched here so the comparison stays portable and audit-friendly.
"""

from __future__ import print_function

import os

import numpy as np
import pandas as pd

from grain_futures_strategy import (
    COMMODITIES,
    backtest_positions,
    backtest_positions_with_costs,
    build_feature_panels,
    load_train_set,
    split_performance,
)


TRAIN_END = pd.Timestamp("2016-01-01")
TEST_START = pd.Timestamp("2018-01-01")
IC_THRESHOLD = 0.015
RIDGE_ALPHA = 100.0
KALMAN_Q = 1.0e-5
MIN_TRAIN_OBS = 504
REFIT_EVERY = 21


def _tanh(series, divisor=2.0):
    return pd.Series(np.tanh(series.astype(float) / float(divisor)), index=series.index)


def _clean(series, index):
    return series.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)


def _signal_universe(panel):
    return {
        "price_mom_20": panel["mom_20"],
        "price_mom_60": panel["mom_60"],
        "price_rev_5": panel["rev_5"],
        "curve_spread": panel["curve_spread"],
        "curve_ratio": panel["curve_ratio"],
        "curve_change_20": panel["curve_change_20"],
        "cot_mm_level": panel["cot_mm_level"],
        "cot_mm_change": panel["cot_mm_change"],
        "cot_pm_oi_level": panel["cot_pm_oi_level"],
        "cot_pm_oi_change": panel["cot_pm_oi_change"],
        "public_inventory_pressure": -panel["public_inventory_change"],
        "receipts_pressure": -panel["receipts_change"],
        "cgl_inventory_pressure": -panel["cgl_inventory_change"],
        "crush_surprise": panel["crush_surprise"],
        "crush_utilization": panel["crush_utilization"],
    }


def _family_definitions():
    return {
        "price": ["price_mom_20", "price_mom_60", "price_rev_5"],
        "curve": ["curve_spread", "curve_ratio", "curve_change_20"],
        "cot": ["cot_mm_level", "cot_mm_change", "cot_pm_oi_level", "cot_pm_oi_change"],
        "physical_public": ["public_inventory_pressure", "receipts_pressure"],
        "physical_cargill": ["cgl_inventory_pressure", "crush_surprise", "crush_utilization"],
    }


def _build_signals(feature_panels, commodity):
    index = feature_panels[commodity].index
    signals = {name: _clean(series, index) for name, series in _signal_universe(feature_panels[commodity]).items()}
    families = {}
    members = {}
    for family, names in _family_definitions().items():
        available = [signals[name] for name in names if name in signals]
        if available:
            families[family] = sum(available) / float(len(available))
            members[family] = names
    return signals, families, members


def _ic(signal, futures_pnl, commodity, mask):
    aligned = pd.concat([signal.reindex(futures_pnl.index), futures_pnl[commodity].shift(-1)], axis=1).dropna()
    if aligned.empty:
        return np.nan
    mask = pd.Series(mask, index=futures_pnl.index).reindex(aligned.index).fillna(False).astype(bool)
    aligned = aligned.loc[mask]
    if len(aligned) < 40 or aligned.iloc[:, 0].std() == 0.0 or aligned.iloc[:, 1].std() == 0.0:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")


def _split_masks(index):
    return {
        "train": index < TRAIN_END,
        "validation": (index >= TRAIN_END) & (index < TEST_START),
        "test": index >= TEST_START,
    }


def _family_ic_table(families, futures_pnl, commodity, regime_mask=None):
    masks = _split_masks(futures_pnl.index)
    if regime_mask is None:
        regime_mask = pd.Series(True, index=futures_pnl.index)
    regime_mask = pd.Series(regime_mask, index=futures_pnl.index).fillna(False).astype(bool)
    rows = []
    for name, signal in families.items():
        row = {"family": name}
        for split_name, mask in masks.items():
            combined_mask = pd.Series(mask, index=futures_pnl.index).astype(bool) & regime_mask
            row[split_name + "_obs"] = int(combined_mask.sum())
            row[split_name + "_ic"] = _ic(signal, futures_pnl, commodity, combined_mask)
        row["passes_train_ic"] = bool(pd.notnull(row["train_ic"]) and abs(row["train_ic"]) >= IC_THRESHOLD)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["passes_train_ic", "validation_ic"], ascending=[False, False])


def _orient(signal, train_ic):
    if pd.isnull(train_ic):
        return signal
    return signal if train_ic >= 0.0 else -signal


def _regime_masks(feature_panels, futures_pnl, commodity):
    panel = feature_panels[commodity].reindex(futures_pnl.index).fillna(0.0)
    trend_strength = panel["mom_60"].abs()
    threshold = trend_strength.expanding(min_periods=252).median().shift(1)
    trend = (trend_strength > threshold).fillna(False)
    return {
        "trend": trend.astype(bool),
        "mr_or_chop": (~trend).astype(bool),
    }


def _positions_from_signal(signal, futures_pnl, commodity, mode="long_short", target_daily_pnl_vol=60.0, max_lot=0.50):
    clean = _tanh(signal.reindex(futures_pnl.index).fillna(0.0)).ewm(halflife=2.0, adjust=False, min_periods=1).mean()
    clean[clean.abs() < 0.05] = 0.0
    if mode == "long_only":
        clean = clean.clip(lower=0.0)
    elif mode != "long_short":
        raise ValueError("Unknown mode: {}".format(mode))
    asset_vol = futures_pnl[commodity].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    pos = clean * (float(target_daily_pnl_vol) / asset_vol)
    out = pd.DataFrame(0.0, index=futures_pnl.index, columns=[commodity])
    out[commodity] = pos.clip(-float(max_lot), float(max_lot)).fillna(0.0)
    return out


def _metrics(bt):
    table = split_performance(bt, TEST_START)
    oos = bt.loc[bt.index >= TEST_START]
    active = oos.loc[oos["held_gross_exposure"] > 1.0e-12, "net_pnl"].dropna()
    win_days = int((active > 0.0).sum()) if len(active) else 0
    return {
        "is_sharpe": table.loc["sharpe", "in_sample"],
        "oos_sharpe": table.loc["sharpe", "out_of_sample"],
        "oos_pnl": table.loc["total_pnl", "out_of_sample"],
        "oos_dd": table.loc["max_drawdown", "out_of_sample"],
        "full_sharpe": table.loc["sharpe", "full_period"],
        "full_pnl": table.loc["total_pnl", "full_period"],
        "full_dd": table.loc["max_drawdown", "full_period"],
        "hit_rate": table.loc["hit_rate", "out_of_sample"],
        "win_days": win_days,
        "turnover": table.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": table.loc["avg_gross_exposure", "full_period"],
    }


def _evaluate(strategy, commodity, signal, futures_pnl, mode="long_short"):
    positions = _positions_from_signal(signal, futures_pnl, commodity, mode=mode)
    rows = []
    for cost_adjusted in [False, True]:
        if cost_adjusted:
            bt, _ = backtest_positions_with_costs(positions, futures_pnl[[commodity]], 8.75, 0.05)
        else:
            bt, _ = backtest_positions(positions, futures_pnl[[commodity]], 0.0)
        row = {"commodity": commodity, "strategy": strategy, "mode": mode, "cost_adjusted": cost_adjusted}
        row.update(_metrics(bt))
        rows.append(row)
    return rows


def _standardized_prediction_to_signal(pred):
    mean = pred.rolling(252, min_periods=60).mean().shift(1)
    std = pred.rolling(252, min_periods=60).std().shift(1).replace(0.0, np.nan)
    return ((pred - mean) / std).clip(-5.0, 5.0).fillna(0.0)


def _fit_ridge_beta(x_train, y_train, alpha):
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    xtx = x.T.dot(x)
    penalty = np.eye(xtx.shape[0]) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(xtx + penalty, x.T.dot(y))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + penalty).dot(x.T).dot(y)


def _expanding_ridge_signal(signals, futures_pnl, commodity):
    x = pd.DataFrame(signals).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = futures_pnl[commodity].shift(-1)
    pred = pd.Series(np.nan, index=x.index)
    beta = None
    last_fit = None
    for i, date in enumerate(x.index):
        train_mask = (x.index < date) & y.notna()
        if int(train_mask.sum()) < MIN_TRAIN_OBS:
            continue
        x_train_raw = x.loc[train_mask]
        mean = x_train_raw.mean()
        std = x_train_raw.std().replace(0.0, np.nan).fillna(1.0)
        if beta is None or last_fit is None or (i - last_fit) >= REFIT_EVERY:
            beta = _fit_ridge_beta((x_train_raw - mean) / std, y.loc[train_mask], RIDGE_ALPHA)
            last_fit = i
        x_row = (x.loc[date] - mean) / std
        pred.loc[date] = np.r_[1.0, np.asarray(x_row, dtype=float)].dot(beta)
    return _standardized_prediction_to_signal(pred)


def _kalman_ols_signal(signals, futures_pnl, commodity):
    x = pd.DataFrame(signals).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = futures_pnl[commodity].shift(-1)
    beta = np.zeros(x.shape[1] + 1)
    p = np.eye(len(beta)) * 10.0
    pred = pd.Series(np.nan, index=x.index)
    mean = pd.Series(0.0, index=x.columns)
    var = pd.Series(1.0, index=x.columns)
    target_var = 1.0
    n = 0
    for date in x.index:
        row = x.loc[date]
        if n > MIN_TRAIN_OBS:
            std = np.sqrt(var.clip(lower=1.0e-8))
            z = ((row - mean) / std).clip(-5.0, 5.0)
            phi = np.r_[1.0, np.asarray(z, dtype=float)]
            pred.loc[date] = phi.dot(beta)
        y_value = y.loc[date]
        if pd.notnull(y_value):
            n += 1
            old_mean = mean.copy()
            mean = mean + (row - mean) / float(n)
            var = ((n - 2.0) / max(n - 1.0, 1.0)) * var + ((row - old_mean) * (row - mean)) / max(n - 1.0, 1.0)
            target_var = target_var + (float(y_value) ** 2 - target_var) / float(n)
            if n > MIN_TRAIN_OBS:
                std = np.sqrt(var.clip(lower=1.0e-8))
                z = ((row - mean) / std).clip(-5.0, 5.0)
                phi = np.r_[1.0, np.asarray(z, dtype=float)]
                p = p + np.eye(len(beta)) * KALMAN_Q
                r = max(target_var, 1.0)
                innovation_var = float(phi.dot(p).dot(phi) + r)
                gain = p.dot(phi) / innovation_var
                err = float(y_value - phi.dot(beta))
                beta = beta + gain * err
                p = p - np.outer(gain, phi).dot(p)
    return _standardized_prediction_to_signal(pred)


def _strategy_signals(feature_panels, futures_pnl, commodity):
    signals, families, family_members = _build_signals(feature_panels, commodity)

    all_avg = sum(signals.values()) / float(len(signals))
    family_avg = sum(families.values()) / float(len(families))

    family_ic = _family_ic_table(families, futures_pnl, commodity)
    passed = family_ic.loc[family_ic["passes_train_ic"]].copy()
    if passed.empty:
        ic_selected_name = family_ic.sort_values("validation_ic", ascending=False).iloc[0]["family"]
        ic_selected_signal = families[ic_selected_name]
    else:
        passed["oriented_validation_ic"] = passed["validation_ic"].abs()
        best = passed.sort_values(["oriented_validation_ic", "train_ic"], ascending=[False, False]).iloc[0]
        ic_selected_name = best["family"]
        ic_selected_signal = _orient(families[ic_selected_name], best["train_ic"])

    regimes = _regime_masks(feature_panels, futures_pnl, commodity)
    regime_selected_pieces = []
    regime_avg_pieces = []
    regime_selection_rows = []
    for regime_name, regime_mask in regimes.items():
        table = _family_ic_table(families, futures_pnl, commodity, regime_mask=regime_mask)
        passed_regime = table.loc[table["passes_train_ic"]].copy()
        if passed_regime.empty:
            best = table.sort_values("validation_ic", ascending=False).iloc[0]
            selected_family = best["family"]
            selected_signal = families[selected_family]
            avg_signal = family_avg
        else:
            passed_regime["abs_validation_ic"] = passed_regime["validation_ic"].abs()
            best = passed_regime.sort_values(["abs_validation_ic", "train_ic"], ascending=[False, False]).iloc[0]
            selected_family = best["family"]
            selected_signal = _orient(families[selected_family], best["train_ic"])
            oriented = [_orient(families[row["family"]], row["train_ic"]) for _, row in passed_regime.iterrows()]
            avg_signal = sum(oriented) / float(len(oriented))
        mask = pd.Series(regime_mask, index=futures_pnl.index).astype(float)
        regime_selected_pieces.append(selected_signal * mask)
        regime_avg_pieces.append(avg_signal * mask)
        item = best.to_dict()
        item["regime"] = regime_name
        item["selected_family"] = selected_family
        regime_selection_rows.append(item)

    regime_selected = sum(regime_selected_pieces).fillna(0.0)
    regime_avg = sum(regime_avg_pieces).fillna(0.0)

    ridge_signal = _expanding_ridge_signal(signals, futures_pnl, commodity)
    kalman_signal = _kalman_ols_signal(signals, futures_pnl, commodity)

    return {
        "signals": {
            "avg_all_signals": all_avg,
            "equal_family_weight": family_avg,
            "ic_family_selected_" + ic_selected_name: ic_selected_signal,
            "regime_best_family_trend_mr": regime_selected,
            "regime_avg_families_trend_mr": regime_avg,
            "ols_kalman_filter": kalman_signal,
            "ridge_expanding_alpha100": ridge_signal,
        },
        "families": families,
        "family_members": family_members,
        "family_ic": family_ic,
        "regime_selection": pd.DataFrame(regime_selection_rows),
    }


STRATEGY_META = {
    "avg_all_signals": {
        "explain": "Simple average of every provided economic signal.",
        "formula": "mean(price, curve, COT, public physical, Cargill inventory, Cargill crush signals)",
        "rationale": "Broad diversification; assumes weak signals are safer in a simple basket.",
        "overfit": "Low model overfit, but can be economically diluted.",
    },
    "equal_family_weight": {
        "explain": "Average within each family, then equal weight each family.",
        "formula": "mean(price_family, curve_family, cot_family, public_physical_family, cargill_physical_family)",
        "rationale": "Prevents a family with many columns from dominating the signal.",
        "overfit": "Low model overfit; family choices are researcher-defined.",
    },
    "ic_family_selected": {
        "explain": "Keep families with train IC above threshold, select highest validation IC family.",
        "formula": "selected_family_signal, oriented by train IC sign",
        "rationale": "Use only families with evidence of predictive rank correlation.",
        "overfit": "Moderate/high selection risk; validation IC can be noisy.",
    },
    "regime_best_family_trend_mr": {
        "explain": "Trend/MR observable regime; pick best validation-IC family separately per regime.",
        "formula": "1(trend)*best_trend_family + 1(MR/chop)*best_mr_family",
        "rationale": "Signals may work in different states; wheat/corn can trend while soy can revert.",
        "overfit": "Moderate selection risk; weights are fixed masks, not optimized.",
    },
    "regime_avg_families_trend_mr": {
        "explain": "Trend/MR observable regime; average all IC-passing families in each regime.",
        "formula": "1(trend)*avg(IC-passing trend families) + 1(MR/chop)*avg(IC-passing MR families)",
        "rationale": "Keeps regime conditioning but avoids picking a single family winner.",
        "overfit": "Lower than best-family regime selection, but still regime-research risk.",
    },
    "ols_kalman_filter": {
        "explain": "Dynamic linear OLS coefficients updated with a Kalman filter.",
        "formula": "y_t = x_t beta_t + error; beta_t follows a random walk",
        "rationale": "Allows relationships to drift through time without a full static fit.",
        "overfit": "Coefficient-estimation risk; parameters are fixed, not OOS tuned.",
    },
    "ridge_expanding_alpha100": {
        "explain": "Expanding-window Ridge with fixed alpha 100.",
        "formula": "argmin ||y-Xb||^2 + 100||b||^2, refit every 21 trading days",
        "rationale": "Regularized linear combination of all signals.",
        "overfit": "Coefficient-estimation risk; fixed alpha reduces but does not remove it.",
    },
}


def _base_strategy_name(name):
    if name.startswith("ic_family_selected_"):
        return "ic_family_selected"
    return name


def _write_log(results, diagnostics, path="notes/family_regime_model_comparison.txt"):
    lines = []
    lines.append("Family/regime/model strategy comparison")
    lines.append("Date: 2026-05-02")
    lines.append("")
    lines.append("Universe")
    lines.append("--------")
    lines.append("Products: CORN, SOYABEAN, WHEAT_SRW, WHEAT_HRW.")
    lines.append("Signals: price momentum/reversion, curve, COT, public physical, Cargill inventory, Cargill crush.")
    lines.append("No external data was fetched in this experiment.")
    lines.append("")
    lines.append("Strategy Definitions")
    lines.append("--------------------")
    for key, meta in STRATEGY_META.items():
        lines.append("- {}: {}".format(key, meta["explain"]))
        lines.append("  Formula: {}".format(meta["formula"]))
        lines.append("  Economic rationale: {}".format(meta["rationale"]))
        lines.append("  Overfit note: {}".format(meta["overfit"]))
    lines.append("")
    lines.append("Results Summary")
    lines.append("---------------")
    cols = [
        "commodity",
        "strategy",
        "mode",
        "cost_adjusted",
        "is_sharpe",
        "oos_sharpe",
        "oos_pnl",
        "oos_dd",
        "full_sharpe",
        "full_dd",
        "hit_rate",
        "win_days",
        "turnover",
        "avg_gross_exposure",
        "overfit_comment",
    ]
    lines.append(results[cols].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    lines.append("")
    for commodity in COMMODITIES:
        lines.append("{} diagnostics".format(commodity))
        lines.append("-" * (len(commodity) + 12))
        lines.append("Family IC table:")
        lines.append(diagnostics[commodity]["family_ic"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
        lines.append("")
        lines.append("Regime family selections:")
        lines.append(diagnostics[commodity]["regime_selection"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def run_family_regime_model_comparison(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    rows = []
    diagnostics = {}
    for commodity in COMMODITIES:
        bundle = _strategy_signals(feature_panels, futures_pnl, commodity)
        diagnostics[commodity] = {
            "family_ic": bundle["family_ic"],
            "regime_selection": bundle["regime_selection"],
        }
        for strategy, signal in bundle["signals"].items():
            mode = "long_short"
            rows.extend(_evaluate(strategy, commodity, signal, futures_pnl, mode=mode))
    results = pd.DataFrame(rows)
    comments = []
    for _, row in results.iterrows():
        base = _base_strategy_name(row["strategy"])
        comment = STRATEGY_META[base]["overfit"]
        if row["cost_adjusted"] and row["oos_sharpe"] < 0.0:
            comment = comment + " Reject: negative cost-adjusted OOS."
        elif row["cost_adjusted"] and row["oos_sharpe"] > 1.0 and base in ["avg_all_signals", "equal_family_weight", "regime_avg_families_trend_mr"]:
            comment = comment + " Strong and comparatively simple."
        elif row["cost_adjusted"] and base in ["ols_kalman_filter", "ridge_expanding_alpha100"] and row["oos_sharpe"] > 0.0:
            comment = comment + " Positive OOS but keep as diagnostic unless validation/full-period behavior is clean."
        comments.append(comment)
    results["overfit_comment"] = comments
    results = results.sort_values(["commodity", "cost_adjusted", "oos_sharpe"], ascending=[True, True, False]).reset_index(drop=True)
    _write_log(results, diagnostics)
    return {
        "results": results,
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    out = run_family_regime_model_comparison()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    cols = [
        "commodity",
        "strategy",
        "cost_adjusted",
        "oos_sharpe",
        "oos_pnl",
        "oos_dd",
        "full_sharpe",
        "full_dd",
        "hit_rate",
        "win_days",
        "overfit_comment",
    ]
    print(out["results"][cols].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
