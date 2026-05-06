"""Corn backtest routines used by the corn notebook."""

import numpy as np
import pandas as pd

from grain_backtest_core import split_performance
from shared_backtest import (
    active_metrics,
    clean_signal,
    equal_family,
    expanding_ols_prediction,
    family_feature_frame,
    family_average,
    kalman_prediction,
    prediction_to_signal,
    rank_ic,
    run_family_tests,
    signal_average,
    split_masks,
)


def _signals(members):
    return list(members.values()) if isinstance(members, dict) else list(members)


def summarize_backtest(bt, train_end, oos_start):
    full = split_performance(bt, oos_start)
    train_val = split_performance(bt.loc[bt.index < pd.Timestamp(oos_start)], train_end)
    return {
        "train_sharpe": train_val.loc["sharpe", "in_sample"],
        "validation_sharpe": train_val.loc["sharpe", "out_of_sample"],
        "validation_dd": train_val.loc["max_drawdown", "out_of_sample"],
        "oos_sharpe": full.loc["sharpe", "out_of_sample"],
        "oos_pnl": full.loc["total_pnl", "out_of_sample"],
        "oos_dd": full.loc["max_drawdown", "out_of_sample"],
        "full_sharpe": full.loc["sharpe", "full_period"],
        "full_pnl": full.loc["total_pnl", "full_period"],
        "full_dd": full.loc["max_drawdown", "full_period"],
        "turnover": full.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": full.loc["avg_gross_exposure", "full_period"],
    }


def _candidate_metric(source_table, base_strategy, variant, guard, bt, oos_start, guard_oos_pct=0.0):
    oos = active_metrics(bt, bt.index >= pd.Timestamp(oos_start))
    full = active_metrics(bt)
    return {
        "source_table": source_table,
        "base_strategy": base_strategy,
        "variant": variant,
        "guard": guard,
        "oos_sharpe": oos.get("sharpe", np.nan),
        "oos_pnl": oos.get("total_pnl", np.nan),
        "oos_dd": oos.get("max_drawdown", np.nan),
        "oos_active_days": oos.get("days", np.nan),
        "full_sharpe": full.get("sharpe", np.nan),
        "full_pnl": full.get("total_pnl", np.nan),
        "full_dd": full.get("max_drawdown", np.nan),
        "turnover": full.get("avg_daily_turnover", np.nan),
        "avg_gross_exposure": full.get("avg_gross_exposure", np.nan),
        "guard_oos_pct": guard_oos_pct,
    }


def oos_metric_row(name, bt, oos_start):
    oos = active_metrics(bt, bt.index >= pd.Timestamp(oos_start))
    full = active_metrics(bt)
    return {
        "strategy": name,
        "oos_sharpe": oos["sharpe"],
        "oos_pnl": oos["total_pnl"],
        "oos_dd": oos["max_drawdown"],
        "oos_active_days": oos["days"],
        "full_sharpe": full["sharpe"],
        "full_pnl": full["total_pnl"],
        "full_dd": full["max_drawdown"],
    }


def _select_by_train_ic(families, target, target_index, train_end, oos_start, min_abs_ic):
    masks = split_masks(target_index, train_end, oos_start)
    rows, selected_signals = [], []
    for family, members in families.items():
        iterator = members.items() if isinstance(members, dict) else enumerate(members)
        for signal_name, signal in iterator:
            signal_name = f"{family}_{signal_name}" if isinstance(signal_name, int) else signal_name
            raw_signal = signal.reindex(target_index).fillna(0.0)
            train_ic = rank_ic(raw_signal, target, masks["train"])
            orientation = 1.0 if pd.isnull(train_ic) or train_ic >= 0.0 else -1.0
            oriented_signal = clean_signal(orientation * raw_signal, target_index)
            selected = bool(pd.notnull(train_ic) and abs(train_ic) >= float(min_abs_ic))
            if selected:
                selected_signals.append(oriented_signal)
            rows.append({
                "family": family,
                "signal": signal_name,
                "train_ic": train_ic,
                "orientation": orientation,
                "selected": selected,
                "validation_ic": rank_ic(oriented_signal, target, masks["validation"]),
                "test_ic": rank_ic(oriented_signal, target, masks["test"]),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["abs_train_ic"] = table["train_ic"].abs()
        table = table.sort_values(["selected", "abs_train_ic"], ascending=[False, False]).reset_index(drop=True)
    return signal_average(selected_signals, target_index), table


def _best_family_by_regime_ic(families, target, trend_strength, target_index, train_end, oos_start):
    masks = split_masks(target_index, train_end, oos_start)
    threshold = trend_strength.expanding(min_periods=252).median().shift(1)
    regimes = {
        "trend": (trend_strength > threshold).fillna(False),
        "mr_or_chop": (trend_strength <= threshold).fillna(True),
    }
    family_signals = {name: signal_average(_signals(members), target_index) for name, members in families.items()}
    rows, pieces = [], []
    for regime_name, regime_mask in regimes.items():
        candidates = []
        for family, signal in family_signals.items():
            train_mask = masks["train"] & regime_mask
            validation_mask = masks["validation"] & regime_mask
            train_ic = rank_ic(signal, target, train_mask)
            orientation = 1.0 if pd.isnull(train_ic) or train_ic >= 0.0 else -1.0
            candidates.append({
                "regime": regime_name,
                "family": family,
                "train_ic": train_ic,
                "orientation": orientation,
                "validation_ic": rank_ic(orientation * signal, target, validation_mask),
                "train_obs": int(train_mask.sum()),
                "validation_obs": int(validation_mask.sum()),
                "signal": orientation * signal,
            })
        table = pd.DataFrame([{k: v for k, v in row.items() if k != "signal"} for row in candidates])
        selected = table.sort_values("validation_ic", ascending=False).iloc[0]
        pieces.append(next(row["signal"] for row in candidates if row["family"] == selected["family"]) * regime_mask.astype(float))
        rows.append(selected.to_dict())
    return clean_signal(sum(pieces), target_index), pd.DataFrame(rows)


def _family_model_signals(families, target, trend_strength, target_index, train_end, oos_start, min_abs_ic):
    trend_signal, _ = _best_family_by_regime_ic(families, target, trend_strength, target_index, train_end, oos_start)
    ic_signal, _ = _select_by_train_ic(families, target, target_index, train_end, oos_start, min_abs_ic)
    features = family_feature_frame(families, target_index)
    return {
        "avg_all_signals": family_average(families, target_index),
        "equal_family": equal_family(families, target_index),
        "best_family_by_trend_mr": trend_signal,
        "select_by_ic": ic_signal,
        "expanding_ols_family_model": prediction_to_signal(expanding_ols_prediction(features, target), target_index),
        "kalman_family_model": prediction_to_signal(kalman_prediction(features, target), target_index),
    }


def evaluate_regime_family_signal_sets(
    families_by_set,
    target,
    trend_strength,
    target_index,
    backtest_for_signal,
    train_end,
    oos_start,
    min_abs_ic,
    signal_sets=("A", "B"),
    test="generic",
    mode="long_short",
):
    def make_tests(families, target, target_index, _signal_set):
        return _family_model_signals(
            families,
            target,
            trend_strength,
            target_index,
            train_end,
            oos_start,
            min_abs_ic,
        )

    def row_for_backtest(signal_set, strategy, bt):
        row = {"test": test, "signal_set": signal_set, "strategy": strategy, "mode": mode, "note": ""}
        row.update(summarize_backtest(bt, train_end, oos_start))
        return row

    selected_family_sets = {name: families_by_set[name] for name in signal_sets}
    return run_family_tests(
        selected_family_sets,
        target,
        target_index,
        backtest_for_signal,
        row_for_backtest,
        make_tests=make_tests,
        key_mode=mode,
    )


def momentum_reversal_signals(panel, target_index):
    trend_strength = panel["mom_60"].abs()
    trend_cut = trend_strength.expanding(min_periods=252).median().shift(1)
    trend_mom_else_mr = pd.Series(
        np.where((trend_strength > trend_cut).fillna(False), panel["mom_60"], panel["rev_5"]),
        index=target_index,
    )
    return {
        "mom_20": clean_signal(panel["mom_20"], target_index),
        "mom_60": clean_signal(panel["mom_60"], target_index),
        "rev_5": clean_signal(panel["rev_5"], target_index),
        "mom_60_rev_5_equal": clean_signal(signal_average([panel["mom_60"], panel["rev_5"]], target_index), target_index),
        "trend_mom_else_mr": clean_signal(trend_mom_else_mr, target_index),
    }


def evaluate_signal_candidates(
    signals,
    backtest_for_signal,
    train_end,
    oos_start,
    test,
    signal_set,
    source_table,
    variant="raw",
    guard="no_guard",
    mode="long_short",
):
    rows, candidate_rows = [], []
    for name, signal in signals.items():
        bt = backtest_for_signal(signal)
        row = {"test": test, "signal_set": signal_set, "strategy": name, "mode": mode, "note": ""}
        row.update(summarize_backtest(bt, train_end, oos_start))
        rows.append(row)
        candidate_rows.append(_candidate_metric(source_table, name, variant, guard, bt, oos_start, 0.0))
    return {
        "results": pd.DataFrame(rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False]),
        "guard_results": pd.DataFrame(candidate_rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False]),
    }


def abundant_supply_masks(price, momentum, target_index):
    price = pd.Series(price, index=target_index).ffill()
    below_ma = price < price.rolling(252, min_periods=120).mean().shift(1)
    negative_momentum = pd.Series(momentum, index=target_index).fillna(0.0) < 0.0
    return {"below_ma_or_negative_mom": (below_ma | negative_momentum).fillna(False)}


def _guarded_positions(positions, guard_name, masks, target_column, oos_start):
    if guard_name == "no_guard":
        return positions.copy(), 0.0
    mask_name, action = guard_name.rsplit("_", 1)
    mask = pd.Series(masks[mask_name], index=positions.index).fillna(False).astype(bool)
    guarded = positions.copy()
    guarded.loc[mask, target_column] *= 0.50 if action == "half" else 0.0
    return guarded.fillna(0.0), float(mask.loc[mask.index >= pd.Timestamp(oos_start)].mean())


def _guard_menu(source_table, base_strategy, variant, signal, masks, positions_for_signal, backtest_positions, target_column, oos_start):
    base_positions = positions_for_signal(signal)
    guard_names = ["no_guard"] + [f"{mask_name}_{action}" for mask_name in masks for action in ["half", "flat"]]
    rows, backtests = [], {}
    for guard_name in guard_names:
        positions, guard_oos_pct = _guarded_positions(base_positions, guard_name, masks, target_column, oos_start)
        bt = backtest_positions(positions)
        key = (base_strategy, variant, guard_name)
        rows.append(_candidate_metric(source_table, base_strategy, variant, guard_name, bt, oos_start, guard_oos_pct))
        backtests[key] = bt
    return rows, backtests


def physical_disagreement_candidate_signals(
    base_signals,
    physical_signal,
    target_index,
    overlay_name="cargill",
    overlay_weights=(0.90, 0.85),
    base_threshold=0.05,
    physical_threshold=0.25,
):
    physical_signal = clean_signal(physical_signal, target_index)
    candidate_signals = {}
    for base_name, base_signal in base_signals.items():
        aligned_base = clean_signal(base_signal, target_index)
        disagreement = (
            (aligned_base * physical_signal < 0.0)
            & (aligned_base.abs() > float(base_threshold))
            & (physical_signal.abs() > float(physical_threshold))
        )
        half_filter = aligned_base.copy()
        half_filter.loc[disagreement] = 0.50 * half_filter.loc[disagreement]
        flat_filter = aligned_base.copy()
        flat_filter.loc[disagreement] = 0.0

        candidate_signals[(base_name, f"base_no_{overlay_name}")] = aligned_base
        for weight in overlay_weights:
            overlay_pct = int(round(weight * 100.0))
            physical_pct = int(round((1.0 - weight) * 100.0))
            candidate_signals[(base_name, f"{overlay_name}_overlay_{overlay_pct}_{physical_pct}")] = clean_signal(
                weight * aligned_base + (1.0 - weight) * physical_signal,
                target_index,
            )
        candidate_signals[(base_name, f"{overlay_name}_disagree_half")] = clean_signal(half_filter, target_index)
        candidate_signals[(base_name, f"{overlay_name}_disagree_flat")] = clean_signal(flat_filter, target_index)
    return candidate_signals


def candidate_metric_table(candidate_signals, backtest_for_signal, source_table, oos_start):
    rows = [
        _candidate_metric(source_table, base_strategy, variant, "no_guard", backtest_for_signal(signal), oos_start, 0.0)
        for (base_strategy, variant), signal in candidate_signals.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["base_strategy", "oos_sharpe", "full_sharpe"],
        ascending=[True, False, False],
    )


def evaluate_guarded_candidate_signals(
    candidate_signals,
    masks,
    positions_for_signal,
    backtest_positions,
    target_column,
    oos_start,
    source_table,
):
    guard_rows, guard_backtests = [], {}
    for (base_strategy, variant), signal in candidate_signals.items():
        rows, backtests = _guard_menu(
            source_table,
            base_strategy,
            variant,
            signal,
            masks,
            positions_for_signal,
            backtest_positions,
            target_column,
            oos_start,
        )
        guard_rows.extend(rows)
        guard_backtests.update(backtests)
    return (
        pd.DataFrame(guard_rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False]),
        guard_backtests,
    )


def candidate_reference_row(results, base_strategy, variant=None, guard=None, sort_columns=("oos_sharpe", "full_sharpe")):
    selected = results.loc[results["base_strategy"] == base_strategy]
    if variant is not None:
        selected = selected.loc[selected["variant"] == variant]
    if guard is not None:
        selected = selected.loc[selected["guard"] == guard]
    return selected.sort_values(list(sort_columns), ascending=[False] * len(sort_columns)).iloc[0]


def candidate_comparison_table(entries):
    return pd.DataFrame([
        {
            "strategy": label,
            "variant": row["variant"],
            "guard": row["guard"],
            "oos_sharpe": row["oos_sharpe"],
            "oos_pnl": row["oos_pnl"],
            "oos_dd": row["oos_dd"],
            "full_sharpe": row["full_sharpe"],
            "full_dd": row["full_dd"],
        }
        for label, row in entries
    ])


def walk_forward_signal_selection(signals, target_index, backtest_for_signal, start):
    rebalance_dates = list(pd.date_range(pd.Timestamp(start), target_index.max(), freq="YS"))
    selected_signal = pd.Series(0.0, index=target_index)
    rows = []
    for i, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else target_index.max() + pd.Timedelta(days=1)
        train_end = rebalance_date - pd.DateOffset(years=2)
        train_mask = target_index < train_end
        validation_mask = (target_index >= train_end) & (target_index < rebalance_date)
        trade_mask = (target_index >= rebalance_date) & (target_index < next_rebalance)
        candidates = []
        for name, signal in signals.items():
            bt = backtest_for_signal(signal)
            train = active_metrics(bt, train_mask)
            validation = active_metrics(bt, validation_mask)
            trade = active_metrics(bt, trade_mask)
            eligible = bool(
                pd.notnull(train["sharpe"])
                and pd.notnull(validation["sharpe"])
                and train["sharpe"] > 0.0
                and validation["sharpe"] > 0.0
            )
            candidates.append({
                "rebalance": rebalance_date.date(),
                "candidate": name,
                "eligible": eligible,
                "score": validation["sharpe"] + 0.25 * train["sharpe"] + 0.001 * validation["max_drawdown"] if eligible else -np.inf,
                "train_sharpe": train["sharpe"],
                "validation_sharpe": validation["sharpe"],
                "validation_dd": validation["max_drawdown"],
                "trade_sharpe": trade["sharpe"],
                "trade_pnl": trade["total_pnl"],
                "trade_dd": trade["max_drawdown"],
            })
        candidate_table = pd.DataFrame(candidates)
        eligible = candidate_table.loc[candidate_table["eligible"]].copy()
        if eligible.empty:
            selected = candidate_table.sort_values(["validation_sharpe", "train_sharpe"], ascending=[False, False]).iloc[0]
            selection_read = "Fallback: no candidate passed positive train/validation Sharpe gate."
        else:
            selected = eligible.sort_values(["score", "validation_sharpe"], ascending=[False, False]).iloc[0]
            selection_read = "Selected using only data before this rebalance date."
        selected_signal.loc[trade_mask] = signals[selected["candidate"]].loc[trade_mask]
        selected = selected.copy()
        selected["selected"] = True
        selected["selection_read"] = selection_read
        rows.append(selected)
    return clean_signal(selected_signal, target_index), pd.DataFrame(rows)
