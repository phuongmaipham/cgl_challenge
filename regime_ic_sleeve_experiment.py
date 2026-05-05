"""Regime-conditional IC experiments for corn and soybeans.

This tests whether IC screening improves when done inside observable sleeves:
- low / normal / high volatility;
- trend / mean-reversion-or-chop.

The experiment keeps the same fixed-family philosophy as
ic_threshold_sleeve_experiment.py, but selects signals and family candidates
inside each regime bucket using train/validation IC only.
"""

from __future__ import print_function

import os

import numpy as np
import pandas as pd

from grain_futures_strategy import build_feature_panels, load_train_set
from ic_threshold_sleeve_experiment import (
    IC_THRESHOLD,
    TEST_START,
    TRAIN_END,
    _candidate_composites,
    _candidate_families,
    _clean_signal,
    _evaluate_candidate,
    _fetch_external_signals,
    _format_table,
    _given_signal_universe,
    _ic,
    _positions_from_signal,
    _signal_ic_table,
    _split_masks,
)
from grain_futures_strategy import backtest_positions, backtest_positions_with_costs


MIN_REGIME_TRAIN_OBS = 120
MIN_REGIME_VALIDATION_OBS = 40


def _regime_masks(feature_panels, futures_pnl, commodity):
    index = futures_pnl.index
    pnl = futures_pnl[commodity].fillna(0.0)
    vol = pnl.rolling(60, min_periods=20).std().shift(1)
    lt_vol = vol.expanding(min_periods=252).median().shift(1)
    high_q = vol.expanding(min_periods=252).quantile(0.75).shift(1)

    high_vol = ((vol > 1.20 * lt_vol) | (vol > high_q)).reindex(index).fillna(False)
    low_vol = (vol < 0.80 * lt_vol).reindex(index).fillna(False)
    normal_vol = (~high_vol & ~low_vol).reindex(index).fillna(True)

    panel = feature_panels[commodity].reindex(index).fillna(0.0)
    trend_strength = panel["mom_60"].abs()
    trend_threshold = trend_strength.expanding(min_periods=252).median().shift(1)
    trend = (trend_strength > trend_threshold).reindex(index).fillna(False)
    mr_or_chop = (~trend).reindex(index).fillna(True)

    return {
        "vol": {
            "low_vol": low_vol.astype(bool),
            "normal_vol": normal_vol.astype(bool),
            "high_vol": high_vol.astype(bool),
        },
        "trend": {
            "trend": trend.astype(bool),
            "mr_or_chop": mr_or_chop.astype(bool),
        },
    }


def _regime_signal_ic_table(signals, futures_pnl, commodity, regime_mask):
    split_masks = _split_masks(futures_pnl.index)
    rows = []
    regime = pd.Series(regime_mask, index=futures_pnl.index).fillna(False).astype(bool)
    for name, signal in signals.items():
        row = {"signal": name}
        for split_name, split_mask in split_masks.items():
            mask = pd.Series(split_mask, index=futures_pnl.index).astype(bool) & regime
            row[split_name + "_obs"] = int(mask.sum())
            row[split_name + "_ic"] = _ic(signal, futures_pnl, commodity, mask)
        row["passes_ic_threshold"] = bool(
            row["train_obs"] >= MIN_REGIME_TRAIN_OBS
            and row["validation_obs"] >= MIN_REGIME_VALIDATION_OBS
            and pd.notnull(row["train_ic"])
            and abs(row["train_ic"]) >= IC_THRESHOLD
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["passes_ic_threshold", "train_ic"], ascending=[False, False])


def _orient_regime_signals(signals, ic_table, futures_pnl):
    out = {}
    for _, row in ic_table.loc[ic_table["passes_ic_threshold"]].iterrows():
        sign = 1.0 if row["train_ic"] >= 0 else -1.0
        out[row["signal"]] = sign * signals[row["signal"]].reindex(futures_pnl.index).fillna(0.0)
    return out


def _select_candidate_for_regime(commodity, signals, futures_pnl, regime_mask):
    signal_ic = _regime_signal_ic_table(signals, futures_pnl, commodity, regime_mask)
    selected_signals = _orient_regime_signals(signals, signal_ic, futures_pnl)
    if not selected_signals:
        return None, signal_ic, pd.DataFrame(), {}

    families, family_members = _candidate_families(commodity, selected_signals)
    candidates, candidate_members = _candidate_composites(commodity, families)
    if not candidates:
        return None, signal_ic, pd.DataFrame(), {}

    rows = []
    regime = pd.Series(regime_mask, index=futures_pnl.index).fillna(False).astype(bool)
    split_masks = _split_masks(futures_pnl.index)
    for candidate, signal in candidates.items():
        for mode in ["long_only", "long_short"]:
            candidate_signal = signal.reindex(futures_pnl.index).fillna(0.0)
            ic_signal = candidate_signal.clip(lower=0.0) if mode == "long_only" else candidate_signal
            row = {
                "candidate": candidate,
                "mode": mode,
                "families": ",".join(candidate_members[candidate]),
            }
            for split_name, split_mask in split_masks.items():
                mask = pd.Series(split_mask, index=futures_pnl.index).astype(bool) & regime
                row[split_name + "_obs"] = int(mask.sum())
                row[split_name + "_ic"] = _ic(ic_signal, futures_pnl, commodity, mask)
            eligible = (
                row["train_obs"] >= MIN_REGIME_TRAIN_OBS
                and row["validation_obs"] >= MIN_REGIME_VALIDATION_OBS
                and pd.notnull(row["train_ic"])
                and pd.notnull(row["validation_ic"])
                and row["train_ic"] >= IC_THRESHOLD
                and row["validation_ic"] >= 0.0
            )
            row["eligible"] = bool(eligible)
            row["score"] = row["validation_ic"] + 0.25 * row["train_ic"] if eligible else -np.inf
            rows.append(row)

    table = pd.DataFrame(rows)
    eligible = table.loc[table["eligible"]].copy()
    if eligible.empty:
        selected = table.sort_values(["validation_ic", "train_ic"], ascending=[False, False]).iloc[0]
    else:
        selected = eligible.sort_values(["score", "validation_ic"], ascending=[False, False]).iloc[0]

    selected_signal = candidates[selected["candidate"]]
    if selected["mode"] == "long_only":
        selected_signal = selected_signal.clip(lower=0.0)
    return selected, signal_ic, table, {
        "signal": selected_signal.reindex(futures_pnl.index).fillna(0.0),
        "families": families,
        "family_members": family_members,
        "candidates": candidates,
    }


def _combined_regime_signal(commodity, signals, futures_pnl, regime_group):
    selected_rows = []
    signal_ic_tables = {}
    candidate_tables = {}
    pieces = []
    for regime_name, regime_mask in regime_group.items():
        selected, signal_ic, candidate_table, details = _select_candidate_for_regime(
            commodity, signals, futures_pnl, regime_mask
        )
        signal_ic_tables[regime_name] = signal_ic
        candidate_tables[regime_name] = candidate_table
        if selected is None:
            continue
        selected = selected.copy()
        selected["regime"] = regime_name
        selected_rows.append(selected)
        mask = pd.Series(regime_mask, index=futures_pnl.index).fillna(False).astype(float)
        pieces.append(details["signal"] * mask)

    if not pieces:
        return None, pd.DataFrame(), signal_ic_tables, candidate_tables
    combined = sum(pieces).reindex(futures_pnl.index).fillna(0.0)
    return combined, pd.DataFrame(selected_rows), signal_ic_tables, candidate_tables


def _evaluate_strategy(name, signal, futures_pnl, commodity, mode="long_short"):
    rows, _ = _evaluate_candidate(name, signal, futures_pnl, commodity, mode)
    return pd.DataFrame(rows)


def _run_one(commodity, feature_panels, futures_pnl):
    given = _given_signal_universe(feature_panels, commodity)
    external, errors, _ = _fetch_external_signals(commodity, futures_pnl)
    signals = dict(given)
    signals.update(external)
    signals = {name: _clean_signal(signal, futures_pnl.index) for name, signal in signals.items()}

    flat_ic = _signal_ic_table(signals, futures_pnl, commodity)
    regimes = _regime_masks(feature_panels, futures_pnl, commodity)
    outputs = {
        "errors": errors,
        "flat_ic": flat_ic,
        "regime_results": {},
    }

    result_rows = []
    for group_name, regime_group in regimes.items():
        combined, selected_table, signal_ics, candidate_tables = _combined_regime_signal(
            commodity, signals, futures_pnl, regime_group
        )
        if combined is None:
            continue
        result = _evaluate_strategy("regime_ic_" + group_name, combined, futures_pnl, commodity, mode="long_short")
        result_rows.append(result)
        outputs["regime_results"][group_name] = {
            "selected_table": selected_table.reset_index(drop=True),
            "signal_ics": signal_ics,
            "candidate_tables": candidate_tables,
            "performance": result,
        }

    outputs["performance"] = pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()
    return outputs


def _append_log_section(lines, label, out):
    lines.append("")
    lines.append("{} regime IC results".format(label))
    lines.append("=" * (len(label) + 18))
    lines.append("External data warnings: {}".format("; ".join(out["errors"]) if out["errors"] else "none"))
    lines.append("")
    for group_name, result in out["regime_results"].items():
        lines.append("{} regime group".format(group_name))
        lines.append("-" * (len(group_name) + 13))
        selected = result["selected_table"]
        if selected.empty:
            lines.append("No selected regimes.")
            continue
        cols = [
            "regime",
            "candidate",
            "mode",
            "families",
            "eligible",
            "score",
            "train_ic",
            "validation_ic",
            "test_ic",
            "train_obs",
            "validation_obs",
            "test_obs",
        ]
        lines.append(_format_table(selected[cols]))
        lines.append("")
        perf_cols = [
            "candidate",
            "mode",
            "cost_adjusted",
            "test_sharpe",
            "test_pnl",
            "test_max_drawdown",
            "full_sharpe",
            "max_drawdown",
        ]
        lines.append(_format_table(result["performance"][perf_cols]))
        lines.append("")

    lines.append("Comparison summary")
    lines.append("------------------")
    if not out["performance"].empty:
        cost = out["performance"].loc[out["performance"]["cost_adjusted"]].copy()
        lines.append(_format_table(cost.sort_values("test_sharpe", ascending=False)))
    lines.append("")


def write_regime_ic_log(outputs, path="notes/regime_ic_corn_soybean.txt"):
    lines = []
    lines.append("Regime-conditional IC sleeve experiments")
    lines.append("Date: 2026-05-02")
    lines.append("")
    lines.append("Method")
    lines.append("------")
    lines.append("- Test IC separately inside low/normal/high volatility buckets.")
    lines.append("- Test IC separately inside trend vs mean-reversion-or-chop buckets.")
    lines.append("- Inside each bucket, select individual signals by train IC threshold.")
    lines.append("- Build fixed equal-weight family candidates from surviving signals.")
    lines.append("- Select candidate by validation IC inside the same bucket.")
    lines.append("- Combine the selected bucket signals with observable regime masks.")
    lines.append("- Report 2018-2020 OOS backtest and compare with the flat IC experiment.")
    for label, out in outputs.items():
        _append_log_section(lines, label, out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def run_regime_ic_sleeve_experiment(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    outputs = {
        "Corn": _run_one("CORN", feature_panels, futures_pnl),
        "Soybeans": _run_one("SOYABEAN", feature_panels, futures_pnl),
    }
    write_regime_ic_log(outputs)
    return outputs


if __name__ == "__main__":
    out = run_regime_ic_sleeve_experiment()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for label, result in out.items():
        print("\n", label)
        print("errors:", result["errors"])
        print(result["performance"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
