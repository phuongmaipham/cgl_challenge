"""Lower-overfit wheat improvement experiments.

The previous standalone WHEAT_SRW and WHEAT_HRW outright sleeves were weak.
This script tests wheat as a SRW/HRW relative-value sleeve instead:

- no fitted coefficients;
- no external data;
- fixed economic signal families;
- fixed trend/MR regime masks;
- leg-level costs and margin funding.

The goal is not to discover a complex wheat model. It is to check whether wheat
has a more fund-usable role as a risk-balanced pair trade.
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
    period_performance,
    rolling_zscore,
    split_performance,
)


SPLIT_DATE = "2018-01-01"
TRAIN_END = pd.Timestamp("2016-01-01")
WHEAT = ["WHEAT_SRW", "WHEAT_HRW"]
TARGET_DAILY_PAIR_VOL = 45.0
MAX_LEG_LOT = 0.45
SIGNAL_THRESHOLD = 0.05


def _clean(series, index):
    return series.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)


def _tanh(series, divisor=2.0):
    return pd.Series(np.tanh(series.astype(float) / float(divisor)), index=series.index)


def _components(panel):
    index = panel.index
    price_mr = panel["rev_5"]
    price_trend = (panel["mom_20"] + panel["mom_60"]) / 2.0
    curve = (panel["curve_spread"] + panel["curve_ratio"] + panel["curve_change_20"]) / 3.0
    cot = (
        panel["cot_mm_level"]
        + panel["cot_mm_change"]
        + panel["cot_pm_oi_level"]
        + panel["cot_pm_oi_change"]
    ) / 4.0
    physical_public = (-panel["public_inventory_change"] - panel["receipts_change"]) / 2.0
    physical_cargill = (
        -panel["cgl_inventory_change"]
        + panel["crush_surprise"]
        + panel["crush_utilization"]
    ) / 3.0
    physical = (physical_public + physical_cargill) / 2.0
    all_family = (price_mr + price_trend + curve + cot + physical_public + physical_cargill) / 6.0
    return {
        "price_mr": _clean(price_mr, index),
        "price_trend": _clean(price_trend, index),
        "curve": _clean(curve, index),
        "cot": _clean(cot, index),
        "physical_public": _clean(physical_public, index),
        "physical_cargill": _clean(physical_cargill, index),
        "physical": _clean(physical, index),
        "all_family_equal": _clean(all_family, index),
    }


def _pair_difference(feature_panels, component_name):
    srw = _components(feature_panels["WHEAT_SRW"])[component_name]
    hrw = _components(feature_panels["WHEAT_HRW"])[component_name]
    return (srw - hrw).fillna(0.0)


def build_pair_signals(feature_panels, futures_pnl):
    index = futures_pnl.index
    pair = {
        "pair_price_mr": _pair_difference(feature_panels, "price_mr"),
        "pair_price_trend": _pair_difference(feature_panels, "price_trend"),
        "pair_curve": _pair_difference(feature_panels, "curve"),
        "pair_cot": _pair_difference(feature_panels, "cot"),
        "pair_physical_public": _pair_difference(feature_panels, "physical_public"),
        "pair_physical_cargill": _pair_difference(feature_panels, "physical_cargill"),
        "pair_physical": _pair_difference(feature_panels, "physical"),
        "pair_all_family_equal": _pair_difference(feature_panels, "all_family_equal"),
    }

    pair["pair_mr_curve_physical_equal"] = (
        pair["pair_price_mr"] + pair["pair_curve"] + pair["pair_physical"]
    ) / 3.0
    pair["pair_given_physical_equal"] = (
        pair["pair_physical_public"] + pair["pair_physical_cargill"]
    ) / 2.0
    pair["pair_balanced_low_overfit"] = (
        pair["pair_price_mr"]
        + pair["pair_price_trend"]
        + pair["pair_curve"]
        + pair["pair_cot"]
        + pair["pair_physical_public"]
        + pair["pair_physical_cargill"]
    ) / 6.0

    srw_trend = feature_panels["WHEAT_SRW"]["mom_60"].reindex(index).abs()
    hrw_trend = feature_panels["WHEAT_HRW"]["mom_60"].reindex(index).abs()
    trend_strength = ((srw_trend + hrw_trend) / 2.0).fillna(0.0)
    trend_threshold = trend_strength.expanding(min_periods=252).median().shift(1)
    trend_regime = (trend_strength > trend_threshold).fillna(False)

    mr_source = (pair["pair_price_mr"] + pair["pair_curve"] + pair["pair_physical"]) / 3.0
    trend_source = (pair["pair_price_trend"] + pair["pair_curve"] + pair["pair_cot"]) / 3.0
    pair["pair_fixed_trend_mr_regime"] = (
        mr_source.where(~trend_regime, 0.0) + trend_source.where(trend_regime, 0.0)
    )
    pair["pair_soft_hybrid_70_30"] = (0.70 * mr_source + 0.30 * trend_source).fillna(0.0)

    return {name: _clean(signal, index) for name, signal in pair.items()}, trend_regime


def build_pair_price_trend_from_prices(data, futures_pnl):
    index = futures_pnl.index
    srw = data["adj1"]["WHEAT_SRW"].reindex(index).ffill()
    hrw = data["adj1"]["WHEAT_HRW"].reindex(index).ffill()
    ratio = (srw / hrw).replace([np.inf, -np.inf], np.nan).ffill()
    trend_20 = rolling_zscore(ratio.pct_change(20), 252, 60).reindex(index).fillna(0.0)
    trend_60 = rolling_zscore(ratio.pct_change(60), 252, 80).reindex(index).fillna(0.0)
    return ((trend_20 + trend_60) / 2.0).clip(-5.0, 5.0).fillna(0.0)


def positions_from_pair_signal(
    signal,
    futures_pnl,
    target_daily_pair_vol=TARGET_DAILY_PAIR_VOL,
    max_leg_lot=MAX_LEG_LOT,
    signal_threshold=SIGNAL_THRESHOLD,
    halflife=2.0,
    rebalance_every=1,
):
    signal = _tanh(signal.reindex(futures_pnl.index).fillna(0.0))
    signal = signal.ewm(halflife=float(halflife), adjust=False, min_periods=1).mean()
    signal[signal.abs() < float(signal_threshold)] = 0.0

    vol = futures_pnl[WHEAT].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    srw_pos = signal * (float(target_daily_pair_vol) / vol["WHEAT_SRW"])
    hrw_pos = -signal * (float(target_daily_pair_vol) / vol["WHEAT_HRW"])

    positions = pd.DataFrame(0.0, index=futures_pnl.index, columns=WHEAT)
    positions["WHEAT_SRW"] = srw_pos.clip(-float(max_leg_lot), float(max_leg_lot))
    positions["WHEAT_HRW"] = hrw_pos.clip(-float(max_leg_lot), float(max_leg_lot))
    positions = positions.fillna(0.0)
    if int(rebalance_every) > 1:
        rebalance_mask = pd.Series(False, index=positions.index)
        rebalance_mask.iloc[:: int(rebalance_every)] = True
        positions = positions.where(rebalance_mask).ffill().fillna(0.0)
    return positions


def _active_win_days(bt):
    active = bt.loc[bt["held_gross_exposure"] > 1.0e-12, "net_pnl"].dropna()
    return int((active > 0.0).sum()) if len(active) else 0


def _metrics(bt):
    table = split_performance(bt, SPLIT_DATE)
    oos = bt.loc[bt.index >= SPLIT_DATE]
    full_active_margin = bt.loc[bt["held_gross_exposure"] > 1.0e-12]
    avg_margin = (
        full_active_margin["margin_used"].mean()
        if "margin_used" in bt.columns and not full_active_margin.empty
        else np.nan
    )
    oos_dd = table.loc["max_drawdown", "out_of_sample"]
    full_dd = table.loc["max_drawdown", "full_period"]
    return {
        "is_sharpe": table.loc["sharpe", "in_sample"],
        "oos_sharpe": table.loc["sharpe", "out_of_sample"],
        "oos_pnl": table.loc["total_pnl", "out_of_sample"],
        "oos_dd": oos_dd,
        "oos_dd_pct_avg_margin": (abs(oos_dd) / avg_margin * 100.0) if pd.notnull(avg_margin) and avg_margin else np.nan,
        "full_sharpe": table.loc["sharpe", "full_period"],
        "full_pnl": table.loc["total_pnl", "full_period"],
        "full_dd": full_dd,
        "full_dd_pct_avg_margin": (abs(full_dd) / avg_margin * 100.0) if pd.notnull(avg_margin) and avg_margin else np.nan,
        "hit_rate": table.loc["hit_rate", "out_of_sample"],
        "win_days": _active_win_days(oos),
        "turnover": table.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": table.loc["avg_gross_exposure", "full_period"],
        "avg_margin_used": avg_margin,
    }


def _evaluate_pair(strategy, signal, futures_pnl, position_kwargs=None):
    if position_kwargs is None:
        position_kwargs = {}
    positions = positions_from_pair_signal(signal, futures_pnl, **position_kwargs)
    rows = []
    backtests = {}
    for cost_adjusted in [False, True]:
        if cost_adjusted:
            bt, _ = backtest_positions_with_costs(positions, futures_pnl[WHEAT], 8.75, 0.05)
        else:
            bt, _ = backtest_positions(positions, futures_pnl[WHEAT], 0.0)
        row = {
            "book": "SRW_HRW_PAIR",
            "strategy": strategy,
            "cost_adjusted": cost_adjusted,
            "formula": STRATEGY_FORMULAS[strategy],
            "overfit": STRATEGY_OVERFIT[strategy],
        }
        row.update(_metrics(bt))
        rows.append(row)
        backtests[strategy + ("_cost" if cost_adjusted else "_zero")] = bt
    return rows, backtests, positions


STRATEGY_FORMULAS = {
    "pair_price_mr": "signal = SRW.rev_5 - HRW.rev_5",
    "pair_price_trend": "signal = mean(SRW.mom_20,mom_60) - mean(HRW.mom_20,mom_60)",
    "pair_curve": "signal = SRW curve family - HRW curve family",
    "pair_cot": "signal = SRW COT family - HRW COT family",
    "pair_physical_public": "signal = SRW public inventory/receipts pressure - HRW public pressure",
    "pair_physical_cargill": "signal = SRW Cargill inventory/crush family - HRW Cargill family",
    "pair_physical": "signal = SRW total physical pressure - HRW total physical pressure",
    "pair_all_family_equal": "signal = equal-weight family score(SRW) - equal-weight family score(HRW)",
    "pair_mr_curve_physical_equal": "signal = equal_weight(pair MR, pair curve, pair physical)",
    "pair_given_physical_equal": "signal = equal_weight(pair public physical, pair Cargill physical)",
    "pair_balanced_low_overfit": "signal = equal_weight(pair MR, trend, curve, COT, public physical, Cargill physical)",
    "pair_fixed_trend_mr_regime": "if observable trend regime: trend/curve/COT pair; otherwise MR/curve/physical pair",
    "pair_soft_hybrid_70_30": "signal = 70% MR/curve/physical pair + 30% trend/curve/COT pair",
    "pair_price_mr_cost_control": "same signal as pair_price_mr, but halflife 5, threshold 0.12, weekly rebalance, 40 target daily pair vol",
    "pair_mr_curve_physical_cost_control": "same signal as pair_mr_curve_physical_equal, but halflife 5, threshold 0.12, weekly rebalance, 40 target daily pair vol",
    "pair_price_mr_cargill_90_10_cost_control": "90% SRW/HRW price MR + 10% SRW/HRW Cargill physical family, with fixed cost-control execution",
    "pair_price_mr_cargill_95_05_cost_control": "95% SRW/HRW price MR + 5% SRW/HRW Cargill physical family, with fixed cost-control execution",
    "pair_price_mr_physical_90_10_cost_control": "90% SRW/HRW price MR + 10% SRW/HRW total physical family, with fixed cost-control execution",
    "pair_price_mr_cargill_80_20_pair_trend_cost_control": "80% price-MR/Cargill pair signal + 20% SRW/HRW price-ratio trend, with fixed cost-control execution",
    "pair_price_mr_cargill_trend_conflict_flat_cost_control": "price-MR/Cargill pair signal, but flat when observable SRW/HRW trend strongly conflicts with MR",
}


STRATEGY_OVERFIT = {
    "pair_price_mr": "Very low: one fixed price-reversion hypothesis.",
    "pair_price_trend": "Very low: one fixed price-trend hypothesis.",
    "pair_curve": "Very low: one fixed curve relative-tightness hypothesis.",
    "pair_cot": "Very low: one fixed positioning hypothesis.",
    "pair_physical_public": "Low: fixed public physical relative-tightness hypothesis.",
    "pair_physical_cargill": "Low: fixed Cargill physical relative-tightness hypothesis; uses both cgl_inv and cgl_crush.",
    "pair_physical": "Low: fixed public+Cargill physical hypothesis.",
    "pair_all_family_equal": "Low: equal family weights; no selected coefficients.",
    "pair_mr_curve_physical_equal": "Low/moderate: curated three-family economic basket.",
    "pair_given_physical_equal": "Low: physical-only equal family basket.",
    "pair_balanced_low_overfit": "Low: broad equal family basket; no selected coefficients.",
    "pair_fixed_trend_mr_regime": "Moderate research risk: fixed observable regime switch, no optimized weights.",
    "pair_soft_hybrid_70_30": "Moderate research risk: fixed hybrid weight, chosen as conservative MR-dominant wheat prior.",
    "pair_price_mr_cost_control": "Low/moderate: no alpha fitting, but adds a fixed execution filter after seeing costs matter.",
    "pair_mr_curve_physical_cost_control": "Low/moderate: no alpha fitting, but adds a fixed execution filter after seeing costs matter.",
    "pair_price_mr_cargill_90_10_cost_control": "Low/moderate: round-number 10% Cargill physical overlay; no fitted coefficients, but still a researched blend.",
    "pair_price_mr_cargill_95_05_cost_control": "Moderate: strongest OOS Sharpe, but choosing 5% instead of 10% is more selection-sensitive.",
    "pair_price_mr_physical_90_10_cost_control": "Low/moderate: round-number 10% total physical overlay; no fitted coefficients, but still a researched blend.",
    "pair_price_mr_cargill_80_20_pair_trend_cost_control": "Low/moderate: fixed 80/20 blend chosen as a trend-risk diversifier, not fitted coefficients.",
    "pair_price_mr_cargill_trend_conflict_flat_cost_control": "Low/moderate: observable trend-risk filter; avoids fighting strong pair trends but can undertrade.",
}


def _write_log(results, period_tables=None, path="notes/wheat_improvement.txt"):
    if period_tables is None:
        period_tables = {}
    lines = []
    lines.append("Wheat lower-overfit improvement experiments")
    lines.append("Date: 2026-05-02")
    lines.append("")
    lines.append("Goal")
    lines.append("----")
    lines.append("Improve WHEAT_SRW and WHEAT_HRW without fitting coefficients by testing wheat as a SRW/HRW relative-value sleeve.")
    lines.append("")
    lines.append("Controls")
    lines.append("--------")
    lines.append("- No OLS, Ridge, Kalman, or fitted model weights.")
    lines.append("- No external data.")
    lines.append("- Fixed signal formulas, fixed trend/MR regime rule, fixed risk target.")
    lines.append("- Leg-level SRW and HRW positions are risk-balanced with 60-day realized PnL volatility.")
    lines.append("- Cost-adjusted rows include 8.75 USD per one-way lot plus 5% annual margin funding.")
    lines.append("- Cargill requirement: pair_physical_cargill and all combined physical/family rows use both cgl_inventory_change from cgl_inv and crush_surprise/crush_utilization from cgl_crush.")
    lines.append("")
    lines.append("Position rule")
    lines.append("-------------")
    lines.append("pair_score > 0: long WHEAT_SRW, short WHEAT_HRW.")
    lines.append("pair_score < 0: short WHEAT_SRW, long WHEAT_HRW.")
    lines.append("Default leg_position = tanh(pair_score / 2) * 45 / 60d_leg_pnl_vol, clipped to +/-0.45 lots.")
    lines.append("Cost-control variants use halflife 5, threshold 0.12, weekly rebalance, and target daily pair vol 40.")
    lines.append("")
    lines.append("Results")
    lines.append("-------")
    cols = [
        "book",
        "strategy",
        "cost_adjusted",
        "is_sharpe",
        "oos_sharpe",
        "oos_pnl",
        "oos_dd",
        "oos_dd_pct_avg_margin",
        "full_sharpe",
        "full_pnl",
        "full_dd",
        "full_dd_pct_avg_margin",
        "hit_rate",
        "win_days",
        "turnover",
        "avg_gross_exposure",
        "avg_margin_used",
    ]
    lines.append(results[cols].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    lines.append("")
    lines.append("Formulas and overfit notes")
    lines.append("--------------------------")
    for strategy in results["strategy"].drop_duplicates():
        lines.append("- {}: {}".format(strategy, STRATEGY_FORMULAS[strategy]))
        lines.append("  Overfit: {}".format(STRATEGY_OVERFIT[strategy]))
    lines.append("")
    if period_tables:
        lines.append("Key period performance")
        lines.append("----------------------")
        for name, table in period_tables.items():
            lines.append("")
            lines.append(name)
            lines.append(table.to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
        lines.append("")
    cost = results.loc[results["cost_adjusted"]].sort_values(["oos_sharpe", "oos_dd"], ascending=[False, False])
    best = cost.iloc[0]
    recommended = cost.loc[cost["strategy"] == "pair_price_mr_cargill_90_10_cost_control"].iloc[0]
    lines.append("Recommendation")
    lines.append("--------------")
    lines.append(
        "Highest cost-adjusted SRW/HRW pair row: {}, OOS Sharpe {:.3f}, OOS PnL {:.3f}, OOS DD {:.3f}, OOS DD {:.2f}% of average margin, full Sharpe {:.3f}.".format(
            best["strategy"],
            best["oos_sharpe"],
            best["oos_pnl"],
            best["oos_dd"],
            best["oos_dd_pct_avg_margin"],
            best["full_sharpe"],
        )
    )
    lines.append(
        "Lower-overfit recommended row: {}, OOS Sharpe {:.3f}, OOS PnL {:.3f}, OOS DD {:.3f}, OOS DD {:.2f}% of average margin, full Sharpe {:.3f}.".format(
            recommended["strategy"],
            recommended["oos_sharpe"],
            recommended["oos_pnl"],
            recommended["oos_dd"],
            recommended["oos_dd_pct_avg_margin"],
            recommended["full_sharpe"],
        )
    )
    lines.append("Use as a wheat relative-value sleeve only if the fund mandate allows SRW/HRW pair trading. Do not present it as two independent outright wheat alphas.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def run_wheat_improvement_experiment(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    pair_signals, trend_regime = build_pair_signals(feature_panels, futures_pnl)
    pair_price_trend = build_pair_price_trend_from_prices(data, futures_pnl)
    rows = []
    positions = {}
    backtests = {}
    for strategy, signal in pair_signals.items():
        new_rows, new_backtests, new_positions = _evaluate_pair(strategy, signal, futures_pnl)
        rows.extend(new_rows)
        backtests.update(new_backtests)
        positions[strategy] = new_positions
    cost_control = {
        "target_daily_pair_vol": 40.0,
        "max_leg_lot": 0.40,
        "signal_threshold": 0.12,
        "halflife": 5.0,
        "rebalance_every": 5,
    }
    cost_control_map = {
        "pair_price_mr_cost_control": pair_signals["pair_price_mr"],
        "pair_mr_curve_physical_cost_control": pair_signals["pair_mr_curve_physical_equal"],
        "pair_price_mr_cargill_90_10_cost_control": (
            0.90 * pair_signals["pair_price_mr"] + 0.10 * pair_signals["pair_physical_cargill"]
        ),
        "pair_price_mr_cargill_95_05_cost_control": (
            0.95 * pair_signals["pair_price_mr"] + 0.05 * pair_signals["pair_physical_cargill"]
        ),
        "pair_price_mr_physical_90_10_cost_control": (
            0.90 * pair_signals["pair_price_mr"] + 0.10 * pair_signals["pair_physical"]
        ),
    }
    mr_cargill_90_10 = cost_control_map["pair_price_mr_cargill_90_10_cost_control"]
    cost_control_map["pair_price_mr_cargill_80_20_pair_trend_cost_control"] = (
        0.80 * mr_cargill_90_10 + 0.20 * pair_price_trend
    )
    trend_strength = pair_price_trend.abs()
    trend_state = (
        trend_strength > trend_strength.expanding(min_periods=252).median().shift(1)
    ).fillna(False)
    conflict = trend_state & ((mr_cargill_90_10 * pair_price_trend) < 0.0)
    cost_control_map["pair_price_mr_cargill_trend_conflict_flat_cost_control"] = mr_cargill_90_10.where(
        ~conflict,
        0.0,
    )
    for strategy, signal in cost_control_map.items():
        new_rows, new_backtests, new_positions = _evaluate_pair(
            strategy,
            signal,
            futures_pnl,
            position_kwargs=cost_control,
        )
        rows.extend(new_rows)
        backtests.update(new_backtests)
        positions[strategy] = new_positions
    results = pd.DataFrame(rows).sort_values(
        ["cost_adjusted", "oos_sharpe", "oos_dd"], ascending=[True, False, False]
    ).reset_index(drop=True)
    period_tables = {}
    for key in [
        "pair_price_mr_cargill_90_10_cost_control_cost",
        "pair_price_mr_cargill_80_20_pair_trend_cost_control_cost",
        "pair_price_mr_cargill_trend_conflict_flat_cost_control_cost",
    ]:
        if key in backtests:
            period_tables[key] = period_performance(backtests[key])[
                ["period", "total_pnl", "sharpe", "max_drawdown", "hit_rate", "days"]
            ]
    _write_log(results, period_tables=period_tables)
    return {
        "results": results,
        "positions": positions,
        "backtests": backtests,
        "trend_regime": trend_regime,
    }


if __name__ == "__main__":
    out = run_wheat_improvement_experiment()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    cols = [
        "book",
        "strategy",
        "cost_adjusted",
        "oos_sharpe",
        "oos_pnl",
        "oos_dd",
        "oos_dd_pct_avg_margin",
        "full_sharpe",
        "full_dd",
        "hit_rate",
        "win_days",
        "turnover",
    ]
    print(out["results"][cols].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
