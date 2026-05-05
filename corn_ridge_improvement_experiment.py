"""Experiment 18 — CORN-specific Ridge model improvements.

Two pre-registered, economically motivated additions tested in walk-forward
only (no static OOS cherry-picking).

Improvement A — cot_mm_change added to CORN core block
-------------------------------------------------------
Economic rationale: The baseline core uses cot_mm_level (speculator
positioning state). Adding cot_mm_change (5-day rate of change) captures
speculator *flow* — when trend-following funds are rapidly building or
exiting CORN, the price impact often persists for several days. This is
CORN-specific because CORN is the most speculator-driven of the four grains
(highest money-manager participation relative to open interest). Zero new
data: cot_mm_change is already computed in build_feature_panels().

Improvement B — ethanol_prod_to_stocks added to CORN physical block
--------------------------------------------------------------------
Economic rationale: Ethanol demand ≈ 40% of US corn consumption. The
production-to-stocks ratio captures supply/demand pressure in the ethanol
sub-market, which transmits to corn demand within days (refiners draw down
corn stocks when ethanol margins are tight). This feature survived the IS
sub-period stability filter in Experiment 16 (the only EIA feature that did
for CORN). CORN-only: wheat/soybean have no ethanol connection. Placed in
the physical block (alpha=1000) for maximum regularisation.

Method
------
Both improvements are tested using the same walk-forward framework as the
baseline (annual retraining, expanding window, 20-day forward PnL target).
Per-commodity results are reported so the CORN impact is isolated from any
spillover across the 4-commodity cross-section.
"""

from __future__ import print_function

import numpy as np
import pandas as pd

from grain_futures_strategy import (
    COMMODITIES,
    OUTRIGHT_CORE_FEATURES,
    OUTRIGHT_PHYSICAL_FEATURES,
    backtest_positions,
    build_feature_panels,
    build_walk_forward_model_signals,
    edge_filtered_positions,
    fit_ridge_predict,
    load_train_set,
    model_predictions_to_positions,
    rolling_zscore,
    split_performance,
)

SPLIT_DATE = "2018-01-01"
DATA_DIR   = "train_set"

# CORN-augmented feature lists (pre-registered, not searched)
CORN_CORE_FEATURES_A  = OUTRIGHT_CORE_FEATURES + ["cot_mm_change"]
CORN_PHYS_FEATURES_B  = OUTRIGHT_PHYSICAL_FEATURES + ["ethanol_prod_to_stocks"]
CORN_CORE_FEATURES_AB = OUTRIGHT_CORE_FEATURES + ["cot_mm_change"]
CORN_PHYS_FEATURES_AB = OUTRIGHT_PHYSICAL_FEATURES + ["ethanol_prod_to_stocks"]


# ── EIA ethanol feature construction ──────────────────────────────────────

def _add_ethanol_to_corn_panel(feature_panels, futures_pnl):
    """Fetch EIA ethanol and add ethanol_prod_to_stocks to the CORN panel."""
    from eia_ethanol_experiment import fetch_eia_ethanol, build_ethanol_feature_panel

    ethanol = fetch_eia_ethanol()
    eth_features = build_ethanol_feature_panel(ethanol, futures_pnl.index)

    if "ethanol_prod_to_stocks" not in eth_features.columns:
        raise RuntimeError("ethanol_prod_to_stocks not found in EIA feature panel.")

    panels = {c: feature_panels[c].copy() for c in COMMODITIES}
    panels["CORN"]["ethanol_prod_to_stocks"] = eth_features["ethanol_prod_to_stocks"]
    return panels


# ── Core model builders ────────────────────────────────────────────────────

def _build_static_signals(feature_panels, futures_pnl, core_feats, phys_feats,
                          split_date=SPLIT_DATE, horizon=5,
                          core_alpha=25.0, phys_alpha=1000.0):
    """Fit static (single IS window) two-block Ridge, one model per commodity.

    For CORN: uses corn_feats. For all others: falls back to OUTRIGHT_* lists.
    """
    split_date = pd.Timestamp(split_date)
    train_mask = futures_pnl.index < split_date
    target_horizon = int(horizon)

    core_preds = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float)
    phys_preds = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float)

    for commodity in COMMODITIES:
        if target_horizon <= 1:
            target = futures_pnl[commodity].shift(-1)
        else:
            target = (futures_pnl[commodity]
                      .shift(-1)
                      .rolling(target_horizon, min_periods=target_horizon)
                      .sum()
                      .shift(-(target_horizon - 1)))

        cf = core_feats if commodity == "CORN" else OUTRIGHT_CORE_FEATURES
        pf = phys_feats if commodity == "CORN" else OUTRIGHT_PHYSICAL_FEATURES

        # Keep only features that exist in the panel
        cf = [f for f in cf if f in feature_panels[commodity].columns]
        pf = [f for f in pf if f in feature_panels[commodity].columns]

        cp, _ = fit_ridge_predict(feature_panels[commodity][cf], target, train_mask, alpha=core_alpha)
        pp, _ = fit_ridge_predict(feature_panels[commodity][pf], target, train_mask, alpha=phys_alpha)
        core_preds[commodity] = cp
        phys_preds[commodity] = pp

    return core_preds.fillna(0.0) + phys_preds.fillna(0.0)


def _build_wf_signals(feature_panels, futures_pnl, corn_core_feats, corn_phys_feats,
                      start_date="2014-01-01", horizon=20,
                      core_alpha=25.0, phys_alpha=1000.0):
    """Walk-forward expanding-window signals, CORN-specific feature lists."""
    trading_index = futures_pnl.index
    all_preds = pd.DataFrame(np.nan, index=trading_index, columns=COMMODITIES, dtype=float)

    retrain_dates = pd.date_range(start=start_date, end=trading_index[-1], freq="YE")

    for retrain_end in retrain_dates:
        train_mask = trading_index < retrain_end
        if int(train_mask.sum()) < 200:
            continue

        next_retrain = retrain_dates[retrain_dates > retrain_end]
        predict_end  = next_retrain[0] if len(next_retrain) else trading_index[-1] + pd.Timedelta(days=1)
        predict_mask = (trading_index >= retrain_end) & (trading_index < predict_end)
        if not predict_mask.any():
            continue

        for commodity in COMMODITIES:
            if horizon <= 1:
                target = futures_pnl[commodity].shift(-1)
            else:
                target = (futures_pnl[commodity]
                          .shift(-1)
                          .rolling(horizon, min_periods=horizon)
                          .sum()
                          .shift(-(horizon - 1)))

            cf = corn_core_feats if commodity == "CORN" else OUTRIGHT_CORE_FEATURES
            pf = corn_phys_feats if commodity == "CORN" else OUTRIGHT_PHYSICAL_FEATURES
            cf = [f for f in cf if f in feature_panels[commodity].columns]
            pf = [f for f in pf if f in feature_panels[commodity].columns]

            cp, _ = fit_ridge_predict(feature_panels[commodity][cf], target, train_mask, alpha=core_alpha)
            pp, _ = fit_ridge_predict(feature_panels[commodity][pf], target, train_mask, alpha=phys_alpha)
            combined = cp.fillna(0.0) + pp.fillna(0.0)
            all_preds.loc[predict_mask, commodity] = combined.loc[predict_mask]

    return all_preds


# ── Metrics helpers ────────────────────────────────────────────────────────

def _row(label, m, variant=None):
    r = {
        "variant":     variant or label,
        "strategy":    label,
        "is_sharpe":   round(m.loc["sharpe",             "in_sample"],     3),
        "oos_sharpe":  round(m.loc["sharpe",             "out_of_sample"], 3),
        "oos_pnl":     round(m.loc["total_pnl",          "out_of_sample"], 0),
        "full_sharpe": round(m.loc["sharpe",             "full_period"],   3),
        "max_dd":      round(m.loc["max_drawdown",        "full_period"],   0),
        "turnover":    round(m.loc["avg_daily_turnover",  "full_period"],   3),
    }
    return r


def _bt_metrics(preds, futures_pnl, split_date, edge=True):
    if edge:
        pos, _, _ = edge_filtered_positions(preds, futures_pnl, quantile=0.50)
    else:
        pos = model_predictions_to_positions(preds, futures_pnl)
    bt, _ = backtest_positions(pos, futures_pnl, 0.0)
    return split_performance(bt, split_date)


def _corn_only_metrics(preds, futures_pnl, split_date):
    """Metrics for CORN column only — edge filter is cross-sectional so we use
    full 4-commodity predictions for position sizing, then isolate CORN PnL."""
    pos, _, _ = edge_filtered_positions(preds, futures_pnl, quantile=0.50)
    corn_pos = pos[["CORN"]]
    corn_pnl = futures_pnl[["CORN"]]
    bt, _ = backtest_positions(corn_pos, corn_pnl, 0.0)
    return split_performance(bt, split_date)


# ── Main experiment ────────────────────────────────────────────────────────

def run_corn_ridge_improvement_experiment(data_dir=DATA_DIR):
    print("Loading data...")
    data          = load_train_set(data_dir)
    panels_base, futures_pnl = build_feature_panels(data)

    print("Adding EIA ethanol feature to CORN panel...")
    panels_eth = _add_ethanol_to_corn_panel(panels_base, futures_pnl)

    # Baseline walk-forward (same as main strategy)
    print("Running baseline walk-forward...")
    wf_base = build_walk_forward_model_signals(panels_base, futures_pnl)
    pred_base_wf = wf_base[0]

    rows = []

    # ── Walk-forward variants ──────────────────────────────────────────────
    print("Running Variant A (cot_mm_change in CORN core) walk-forward...")
    pred_A_wf = _build_wf_signals(
        panels_base, futures_pnl,
        corn_core_feats=CORN_CORE_FEATURES_A,
        corn_phys_feats=OUTRIGHT_PHYSICAL_FEATURES,
    )

    print("Running Variant B (ethanol_prod_to_stocks in CORN physical) walk-forward...")
    pred_B_wf = _build_wf_signals(
        panels_eth, futures_pnl,
        corn_core_feats=OUTRIGHT_CORE_FEATURES,
        corn_phys_feats=CORN_PHYS_FEATURES_B,
    )

    print("Running Variant AB (both improvements) walk-forward...")
    pred_AB_wf = _build_wf_signals(
        panels_eth, futures_pnl,
        corn_core_feats=CORN_CORE_FEATURES_AB,
        corn_phys_feats=CORN_PHYS_FEATURES_AB,
    )

    # ── All-commodity results (walk-forward, edge-filtered) ────────────────
    for label, pred in [
        ("Baseline WF",  pred_base_wf),
        ("A: +cot_mm_change WF",         pred_A_wf),
        ("B: +ethanol_prod_to_stocks WF", pred_B_wf),
        ("AB: both WF",                  pred_AB_wf),
    ]:
        m_ef = _bt_metrics(pred, futures_pnl, SPLIT_DATE, edge=True)
        rows.append(_row(label + " | all-commodity edge-filt", m_ef, label))

    # ── CORN-only results (walk-forward, edge-filtered) ────────────────────
    for label, pred in [
        ("Baseline WF",  pred_base_wf),
        ("A: +cot_mm_change WF",         pred_A_wf),
        ("B: +ethanol_prod_to_stocks WF", pred_B_wf),
        ("AB: both WF",                  pred_AB_wf),
    ]:
        m_corn = _corn_only_metrics(pred, futures_pnl, SPLIT_DATE)
        rows.append(_row(label + " | CORN-only edge-filt", m_corn, label + " [CORN only]"))

    results = pd.DataFrame(rows)
    return {
        "results":        results,
        "futures_pnl":    futures_pnl,
        "pred_base_wf":   pred_base_wf,
        "pred_A_wf":      pred_A_wf,
        "pred_B_wf":      pred_B_wf,
        "pred_AB_wf":     pred_AB_wf,
        "panels_base":    panels_base,
        "panels_eth":     panels_eth,
    }


if __name__ == "__main__":
    out = run_corn_ridge_improvement_experiment()
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    print()
    print(out["results"].to_string(index=False, float_format=lambda v: "{:.3f}".format(v)))
