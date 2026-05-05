"""IC-threshold family selection experiments for corn and soybeans.

Research rule:
1. Build a transparent signal universe for each commodity.
2. Keep only signals whose train IC clears a fixed threshold.
3. Build fixed equal-weight family candidates from the surviving signals.
4. Select the best family candidate using validation IC only.
5. Report the untouched 2018-2020 OOS backtest.
"""

from __future__ import print_function

import os

import numpy as np
import pandas as pd

from corn_external_signal_experiment import (
    _download_yfinance as _download_corn_yfinance,
    _ethanol_family,
    _external_yfinance_families as _corn_external_yfinance_families,
    _weather_family as _corn_weather_family,
)
from eia_ethanol_experiment import build_ethanol_feature_panel, fetch_eia_ethanol
from grain_futures_strategy import (
    backtest_positions,
    backtest_positions_with_costs,
    build_feature_panels,
    load_train_set,
    split_performance,
)
from meteostat_experiment import fetch_meteostat_weather
from soybean_external_signal_experiment import (
    _download_yfinance as _download_soy_yfinance,
    _external_weather_family as _soy_weather_family,
    _external_yfinance_families as _soy_external_yfinance_families,
)


TRAIN_END = pd.Timestamp("2016-01-01")
TEST_START = pd.Timestamp("2018-01-01")
IC_THRESHOLD = 0.015
VALIDATION_IC_FLOOR = 0.0
MAX_LOT = 0.50
TARGET_DAILY_PNL_VOL = 75.0


def _tanh(series, divisor=2.0):
    return pd.Series(np.tanh(series.astype(float) / float(divisor)), index=series.index)


def _clean_signal(series, index):
    signal = series.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return signal.clip(lower=-5.0, upper=5.0)


def _smooth_threshold(series, mode):
    signal = _tanh(series).ewm(halflife=2.0, adjust=False, min_periods=1).mean()
    signal[signal.abs() < 0.05] = 0.0
    if mode == "long_only":
        signal = signal.clip(lower=0.0)
    elif mode == "short_only":
        signal = signal.clip(upper=0.0)
    elif mode != "long_short":
        raise ValueError("Unknown mode: {}".format(mode))
    return signal


def _positions_from_signal(signal, futures_pnl, commodity, mode="long_only"):
    cleaned = _smooth_threshold(signal.reindex(futures_pnl.index).fillna(0.0), mode=mode)
    pnl = futures_pnl[[commodity]]
    asset_vol = pnl[commodity].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    pos = cleaned * (TARGET_DAILY_PNL_VOL / asset_vol)
    out = pd.DataFrame(0.0, index=futures_pnl.index, columns=[commodity])
    out[commodity] = pos.clip(lower=-MAX_LOT, upper=MAX_LOT).fillna(0.0)
    return out


def _split_masks(index):
    return {
        "train": index < TRAIN_END,
        "validation": (index >= TRAIN_END) & (index < TEST_START),
        "test": index >= TEST_START,
    }


def _ic(signal, futures_pnl, commodity, mask):
    aligned = pd.concat(
        [
            signal.reindex(futures_pnl.index),
            futures_pnl[commodity].shift(-1),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return np.nan
    aligned = aligned.loc[mask.reindex(aligned.index).fillna(False)]
    if len(aligned) < 40 or aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")


def _signal_ic_table(signals, futures_pnl, commodity):
    masks = _split_masks(futures_pnl.index)
    rows = []
    for name, signal in signals.items():
        row = {"signal": name}
        for split_name, mask in masks.items():
            row[split_name + "_ic"] = _ic(signal, futures_pnl, commodity, pd.Series(mask, index=futures_pnl.index))
        row["passes_ic_threshold"] = bool(
            pd.notnull(row["train_ic"]) and abs(row["train_ic"]) >= IC_THRESHOLD
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["passes_ic_threshold", "train_ic"], ascending=[False, False])


def _orient_and_filter_signals(signals, ic_table, futures_pnl):
    oriented = {}
    selected_rows = ic_table.loc[ic_table["passes_ic_threshold"]].copy()
    for _, row in selected_rows.iterrows():
        name = row["signal"]
        sign = 1.0 if row["train_ic"] >= 0 else -1.0
        oriented[name] = sign * signals[name].reindex(futures_pnl.index).fillna(0.0)
    return oriented


def _mean_available(signals, names):
    available = [signals[name] for name in names if name in signals]
    if not available:
        return None
    return sum(available) / float(len(available))


def _candidate_families(commodity, selected_signals):
    if commodity == "SOYABEAN":
        family_definitions = {
            "price": ["given_mom_20", "given_mom_60", "given_rev_5", "given_price_family"],
            "physical": [
                "given_inventory_pressure",
                "given_cgl_inventory_pressure",
                "given_crush_pressure",
                "given_curve_tightness",
                "given_physical_family",
            ],
            "external_crush": ["external_crush_family"],
            "fx_export": ["external_fx_export_family"],
            "weather": ["external_weather_hdd_cdd_family"],
            "macro": ["external_macro_risk_family", "external_relative_grain_family"],
        }
    else:
        family_definitions = {
            "price": ["given_mom_20", "given_mom_60", "given_rev_5", "given_price_family"],
            "physical": [
                "given_inventory_pressure",
                "given_cgl_inventory_pressure",
                "given_cgl_crush_activity",
                "given_curve_tightness",
                "given_physical_family",
            ],
            "ethanol": ["external_ethanol_family"],
            "fx_export": ["external_fx_export_family"],
            "weather": ["external_weather_hdd_cdd_family"],
            "macro": ["external_macro_risk_family", "external_relative_grain_family"],
        }

    families = {}
    family_members = {}
    for family, names in family_definitions.items():
        signal = _mean_available(selected_signals, names)
        if signal is not None:
            families[family] = signal
            family_members[family] = [name for name in names if name in selected_signals]
    return families, family_members


def _candidate_composites(commodity, families):
    definitions = {
        "selected_all_equal": list(families.keys()),
        "physical_only": ["physical"],
        "price_physical_equal": ["price", "physical"],
        "physical_fx_equal": ["physical", "fx_export"],
        "physical_weather_equal": ["physical", "weather"],
        "physical_macro_equal": ["physical", "macro"],
    }
    if commodity == "SOYABEAN":
        definitions.update(
            {
                "physical_extcrush_equal": ["physical", "external_crush"],
                "physical_extcrush_fx_equal": ["physical", "external_crush", "fx_export"],
                "physical_extcrush_weather_equal": ["physical", "external_crush", "weather"],
                "physical_extcrush_fx_weather_equal": ["physical", "external_crush", "fx_export", "weather"],
            }
        )
    else:
        definitions.update(
            {
                "physical_ethanol_equal": ["physical", "ethanol"],
                "physical_ethanol_fx_equal": ["physical", "ethanol", "fx_export"],
                "physical_ethanol_weather_equal": ["physical", "ethanol", "weather"],
                "physical_ethanol_fx_weather_equal": ["physical", "ethanol", "fx_export", "weather"],
            }
        )

    candidates = {}
    candidate_members = {}
    for candidate, family_names in definitions.items():
        used = [families[name] for name in family_names if name in families]
        if not used:
            continue
        if candidate != "selected_all_equal" and len(used) != len(family_names):
            continue
        candidates[candidate] = sum(used) / float(len(used))
        candidate_members[candidate] = [name for name in family_names if name in families]
    return candidates, candidate_members


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


def _evaluate_candidate(name, signal, futures_pnl, commodity, mode):
    positions = _positions_from_signal(signal, futures_pnl, commodity, mode)
    rows = []
    backtests = {}
    for cost_adjusted in [False, True]:
        if cost_adjusted:
            bt, _ = backtest_positions_with_costs(positions, futures_pnl[[commodity]], 8.75, 0.05)
        else:
            bt, _ = backtest_positions(positions, futures_pnl[[commodity]], 0.0)
        row = {"candidate": name, "mode": mode, "cost_adjusted": cost_adjusted}
        row.update(_metrics(bt))
        rows.append(row)
        backtests[name + "_" + mode + ("_cost" if cost_adjusted else "_zero")] = bt
    return rows, backtests


def _candidate_selection_table(candidates, candidate_members, futures_pnl, commodity):
    masks = _split_masks(futures_pnl.index)
    rows = []
    backtests = {}
    modes = ["long_only", "long_short"]
    for name, signal in candidates.items():
        for mode in modes:
            candidate_signal = signal.reindex(futures_pnl.index).fillna(0.0)
            if mode == "long_only":
                ic_signal = candidate_signal.clip(lower=0.0)
            else:
                ic_signal = candidate_signal
            row = {
                "candidate": name,
                "mode": mode,
                "families": ",".join(candidate_members[name]),
            }
            for split_name, mask in masks.items():
                row[split_name + "_ic"] = _ic(
                    ic_signal,
                    futures_pnl,
                    commodity,
                    pd.Series(mask, index=futures_pnl.index),
                )
            eligible = (
                pd.notnull(row["train_ic"])
                and pd.notnull(row["validation_ic"])
                and row["train_ic"] >= IC_THRESHOLD
                and row["validation_ic"] >= VALIDATION_IC_FLOOR
            )
            row["eligible"] = bool(eligible)
            row["selection_score"] = (
                row["validation_ic"] + 0.25 * row["train_ic"] if eligible else -np.inf
            )
            rows.append(row)
            metric_rows, new_bt = _evaluate_candidate(name, signal, futures_pnl, commodity, mode)
            for metric_row in metric_rows:
                metric_row["families"] = row["families"]
                metric_row["train_ic"] = row["train_ic"]
                metric_row["validation_ic"] = row["validation_ic"]
                metric_row["test_ic"] = row["test_ic"]
                metric_row["eligible"] = row["eligible"]
                metric_row["selection_score"] = row["selection_score"]
            backtests.update(new_bt)
            rows.extend(metric_rows)
    table = pd.DataFrame([row for row in rows if "cost_adjusted" not in row])
    results = pd.DataFrame([row for row in rows if "cost_adjusted" in row])
    eligible = table.loc[table["eligible"]].copy()
    if eligible.empty:
        selected = table.sort_values(["validation_ic", "train_ic"], ascending=[False, False]).iloc[0]
    else:
        selected = eligible.sort_values(["selection_score", "validation_ic"], ascending=[False, False]).iloc[0]
    return selected, table, results, backtests


def _given_signal_universe(feature_panels, commodity):
    panel = feature_panels[commodity]
    inventory_pressure = (
        -panel["public_inventory_change"] - panel["receipts_change"] - panel["cgl_inventory_change"]
    ) / 3.0
    curve_tightness = (panel["curve_spread"] + panel["curve_ratio"]) / 2.0
    price_family = (panel["mom_20"] + panel["mom_60"] + panel["rev_5"]) / 3.0
    trend = (panel["mom_20"] + panel["mom_60"] + panel["curve_spread"] + panel["cot_pm_oi_level"]) / 4.0
    signals = {
        "given_mom_20": panel["mom_20"],
        "given_mom_60": panel["mom_60"],
        "given_rev_5": panel["rev_5"],
        "given_curve_spread": panel["curve_spread"],
        "given_curve_ratio": panel["curve_ratio"],
        "given_cot_pm_oi_level": panel["cot_pm_oi_level"],
        "given_inventory_pressure": inventory_pressure,
        "given_cgl_inventory_pressure": -panel["cgl_inventory_change"],
        "given_curve_tightness": curve_tightness,
        "given_price_family": price_family,
        "given_trend": trend,
    }
    if commodity == "SOYABEAN":
        crush_pressure = (panel["crush_surprise"] + panel["crush_utilization"]) / 2.0
        signals["given_crush_pressure"] = crush_pressure
        signals["given_physical_family"] = (inventory_pressure + crush_pressure + curve_tightness) / 3.0
    else:
        cgl_crush_activity = (panel["crush_surprise"] + panel["crush_utilization"]) / 2.0
        signals["given_cgl_crush_activity"] = cgl_crush_activity
        signals["given_physical_family"] = (inventory_pressure + curve_tightness + 0.25 * cgl_crush_activity) / 2.25
    return {name: signal.fillna(0.0) for name, signal in signals.items()}


def _fetch_external_signals(commodity, futures_pnl):
    errors = []
    external = {}
    prices = pd.DataFrame()
    weather = pd.DataFrame()
    ethanol = pd.DataFrame()
    try:
        if commodity == "SOYABEAN":
            prices = _download_soy_yfinance(futures_pnl.index.min(), futures_pnl.index.max())
            external.update(_soy_external_yfinance_families(prices, futures_pnl.index))
        else:
            prices = _download_corn_yfinance(futures_pnl.index.min(), futures_pnl.index.max())
            external.update(_corn_external_yfinance_families(prices, futures_pnl.index))
    except Exception as exc:
        errors.append("yfinance: {}".format(exc))

    try:
        weather = fetch_meteostat_weather(futures_pnl.index.min(), futures_pnl.index.max())
        if commodity == "SOYABEAN":
            external.update(_soy_weather_family(weather, futures_pnl.index))
        else:
            external.update(_corn_weather_family(weather, futures_pnl.index))
    except Exception as exc:
        errors.append("meteostat: {}".format(exc))

    if commodity == "CORN":
        try:
            ethanol = fetch_eia_ethanol()
            ethanol_features = build_ethanol_feature_panel(ethanol, futures_pnl.index)
            external.update(_ethanol_family(ethanol_features))
        except Exception as exc:
            errors.append("eia_ethanol: {}".format(exc))

    external = {name: _clean_signal(signal, futures_pnl.index) for name, signal in external.items()}
    return external, errors, {"prices": prices, "weather": weather, "ethanol": ethanol}


def _run_one_commodity(commodity, feature_panels, futures_pnl):
    given = _given_signal_universe(feature_panels, commodity)
    external, errors, raw_external = _fetch_external_signals(commodity, futures_pnl)
    signals = dict(given)
    signals.update(external)
    signals = {name: _clean_signal(signal, futures_pnl.index) for name, signal in signals.items()}
    ic_table = _signal_ic_table(signals, futures_pnl, commodity)
    selected_signals = _orient_and_filter_signals(signals, ic_table, futures_pnl)
    families, family_members = _candidate_families(commodity, selected_signals)
    candidates, candidate_members = _candidate_composites(commodity, families)
    if not candidates:
        raise RuntimeError("No IC-selected family candidates for {}".format(commodity))
    selected, selection_table, results, backtests = _candidate_selection_table(
        candidates,
        candidate_members,
        futures_pnl,
        commodity,
    )
    selected_results = results.loc[
        (results["candidate"] == selected["candidate"]) & (results["mode"] == selected["mode"])
    ].copy()
    zero = results.loc[~results["cost_adjusted"]].copy()
    robust_pool = zero.loc[
        zero["eligible"]
        & (zero["train_sharpe"] > 0.0)
        & (zero["validation_sharpe"] > 0.0)
        & (zero["validation_max_drawdown"] > -250.0)
    ].copy()
    if robust_pool.empty:
        robust_selected = selected
    else:
        robust_pool["robust_score"] = (
            robust_pool["validation_sharpe"]
            + 0.25 * robust_pool["train_sharpe"]
            + 0.001 * robust_pool["validation_max_drawdown"]
        )
        robust_selected = robust_pool.sort_values(
            ["robust_score", "validation_sharpe", "validation_ic"],
            ascending=[False, False, False],
        ).iloc[0]
    robust_results = results.loc[
        (results["candidate"] == robust_selected["candidate"]) & (results["mode"] == robust_selected["mode"])
    ].copy()
    return {
        "commodity": commodity,
        "errors": errors,
        "ic_table": ic_table.reset_index(drop=True),
        "selected_signal_names": list(selected_signals.keys()),
        "families": families,
        "family_members": family_members,
        "candidate_members": candidate_members,
        "selection_table": selection_table.reset_index(drop=True),
        "results": results.sort_values(["cost_adjusted", "test_sharpe"], ascending=[True, False]).reset_index(drop=True),
        "selected": selected,
        "selected_results": selected_results.sort_values("cost_adjusted").reset_index(drop=True),
        "robust_selected": robust_selected,
        "robust_results": robust_results.sort_values("cost_adjusted").reset_index(drop=True),
        "backtests": backtests,
        "raw_external": raw_external,
    }


def _format_table(df, float_format="{:.3f}".format, max_rows=None):
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_string(index=False, float_format=float_format)


def write_ic_threshold_log(outputs, path="notes/ic_threshold_corn_soybean.txt"):
    lines = []
    lines.append("IC-threshold sleeve selection experiments")
    lines.append("Date: 2026-05-02")
    lines.append("")
    lines.append("Method")
    lines.append("------")
    lines.append("- Signal IC is Spearman correlation of signal_t versus next-day futures PnL.")
    lines.append("- Signals pass if abs(train IC) >= {:.3f} before 2016.".format(IC_THRESHOLD))
    lines.append("- Passed signals are oriented by train IC sign, then grouped into economic families.")
    lines.append("- Family candidates are fixed equal-weight composites.")
    lines.append("- Candidate selection uses validation IC from 2016-2017 only.")
    lines.append("- 2018-2020 is reported as the untouched OOS period.")
    lines.append("- Backtests are also cost-adjusted with trade and holding costs.")
    lines.append("")
    for name, out in outputs.items():
        lines.append("")
        lines.append("{} results".format(name))
        lines.append("=" * (len(name) + 8))
        if out["errors"]:
            lines.append("External data warnings: {}".format("; ".join(out["errors"])))
        else:
            lines.append("External data warnings: none")
        lines.append("")
        lines.append("Signals passing IC threshold")
        lines.append("----------------------------")
        passed_cols = ["signal", "train_ic", "validation_ic", "test_ic", "passes_ic_threshold"]
        passed = out["ic_table"].loc[out["ic_table"]["passes_ic_threshold"], passed_cols]
        lines.append(_format_table(passed.sort_values("train_ic", ascending=False)))
        lines.append("")
        lines.append("Family members after IC filtering")
        lines.append("---------------------------------")
        for family, members in out["family_members"].items():
            lines.append("- {}: {}".format(family, ", ".join(members)))
        lines.append("")
        lines.append("Candidate selection by validation IC")
        lines.append("------------------------------------")
        sel_cols = [
            "candidate",
            "mode",
            "families",
            "eligible",
            "selection_score",
            "train_ic",
            "validation_ic",
            "test_ic",
        ]
        lines.append(_format_table(out["selection_table"][sel_cols].sort_values("selection_score", ascending=False)))
        lines.append("")
        lines.append("Cost-adjusted OOS performance")
        lines.append("-----------------------------")
        perf_cols = [
            "candidate",
            "mode",
            "cost_adjusted",
            "train_ic",
            "validation_ic",
            "test_ic",
            "test_sharpe",
            "test_pnl",
            "test_max_drawdown",
            "full_sharpe",
            "max_drawdown",
        ]
        cost = out["results"].loc[out["results"]["cost_adjusted"], perf_cols]
        lines.append(_format_table(cost.sort_values("test_sharpe", ascending=False)))
        lines.append("")
        selected = out["selected"]
        lines.append("Selected by validation IC")
        lines.append("-------------------------")
        lines.append(
            "- {} {} families=[{}] train_IC={:.3f} validation_IC={:.3f} test_IC={:.3f}".format(
                selected["candidate"],
                selected["mode"],
                selected["families"],
                selected["train_ic"],
                selected["validation_ic"],
                selected["test_ic"],
            )
        )
        lines.append(_format_table(out["selected_results"][perf_cols]))
        lines.append("")
        robust = out["robust_selected"]
        lines.append("IC + validation Sharpe/DD sanity selection")
        lines.append("------------------------------------------")
        lines.append(
            "- {} {} families=[{}] train_IC={:.3f} validation_IC={:.3f} test_IC={:.3f}".format(
                robust["candidate"],
                robust["mode"],
                robust["families"],
                robust["train_ic"],
                robust["validation_ic"],
                robust["test_ic"],
            )
        )
        lines.append(
            "- Sanity gates: candidate passed train IC threshold, train Sharpe > 0, validation Sharpe > 0, validation DD > -250."
        )
        lines.append(_format_table(out["robust_results"][perf_cols]))
        lines.append("")
        lines.append("Overfit read")
        lines.append("------------")
        best_cost = cost.sort_values("test_sharpe", ascending=False).iloc[0]
        selected_cost = out["selected_results"].loc[out["selected_results"]["cost_adjusted"]].iloc[0]
        robust_cost = out["robust_results"].loc[out["robust_results"]["cost_adjusted"]].iloc[0]
        lines.append(
            "- Best OOS cost-adjusted row was {} {} with Sharpe {:.3f} and DD {:.3f}.".format(
                best_cost["candidate"],
                best_cost["mode"],
                best_cost["test_sharpe"],
                best_cost["test_max_drawdown"],
            )
        )
        lines.append(
            "- Validation-selected row had OOS Sharpe {:.3f} and DD {:.3f}.".format(
                selected_cost["test_sharpe"],
                selected_cost["test_max_drawdown"],
            )
        )
        lines.append(
            "- IC+Sharpe/DD sanity row had OOS Sharpe {:.3f} and DD {:.3f}.".format(
                robust_cost["test_sharpe"],
                robust_cost["test_max_drawdown"],
            )
        )
        if selected_cost["test_sharpe"] < best_cost["test_sharpe"] - 0.25:
            lines.append("- Gap is large, so candidate selection remains a research-overfit risk.")
        else:
            lines.append("- Gap is moderate/small, so the IC selection is comparatively stable.")
        lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def run_ic_threshold_sleeve_experiment(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    outputs = {
        "Corn": _run_one_commodity("CORN", feature_panels, futures_pnl),
        "Soybeans": _run_one_commodity("SOYABEAN", feature_panels, futures_pnl),
    }
    write_ic_threshold_log(outputs)
    return outputs


if __name__ == "__main__":
    out = run_ic_threshold_sleeve_experiment()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    for label, result in out.items():
        print("\n", label)
        print("errors:", result["errors"])
        print("selected:", result["selected"]["candidate"], result["selected"]["mode"])
        print(result["selected_results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
