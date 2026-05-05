#!/usr/bin/env python3
"""Streamlit backtest dashboard for the Cargill grain strategy."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from grain_futures_strategy import (
    COMMODITIES,
    COST_CASES,
    FINAL_OPPORTUNITY_QUANTILES,
    backtest_positions_with_costs,
    grain_volatility_state,
    multi_condition_filter_positions,
    observable_three_sleeve_positions,
    performance_metrics,
    period_performance,
    run_final_strategy_selection,
    run_observable_regime_weight_experiment,
    run_research_pipeline,
    skip_rebalance_positions,
    split_performance,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "train_set"
SPLIT_DATE = "2018-01-01"

MARKET_COST_CASE = COST_CASES[1]
SIGNAL_CATALOG = pd.DataFrame(
    [
        {
            "family": "Core price",
            "signal": "mom_60",
            "role": "Core",
            "used_in_regime": "Normal, low-vol/churn, shock",
            "interpretation": "60-day trend baseline; ranks grains by persistent repricing.",
        },
        {
            "family": "Core price",
            "signal": "rev_5",
            "role": "Core",
            "used_in_regime": "Normal, low-vol/churn",
            "interpretation": "5-day reversal; dampens short-term overshoot.",
        },
        {
            "family": "Core curve",
            "signal": "curve_spread",
            "role": "Core",
            "used_in_regime": "Normal, shock/high-opportunity",
            "interpretation": "Front-vs-second contract tightness and carry pressure.",
        },
        {
            "family": "Core curve",
            "signal": "curve_ratio",
            "role": "Core",
            "used_in_regime": "Normal, shock/high-opportunity",
            "interpretation": "Scale-normalized curve tightness signal.",
        },
        {
            "family": "Positioning",
            "signal": "cot_mm_level",
            "role": "Core",
            "used_in_regime": "Normal",
            "interpretation": "Managed-money crowding and speculative pressure.",
        },
        {
            "family": "Positioning",
            "signal": "cot_pm_oi_level",
            "role": "Core",
            "used_in_regime": "Normal",
            "interpretation": "Producer/merchant positioning relative to open interest.",
        },
        {
            "family": "Physical",
            "signal": "public_inventory_change",
            "role": "Overlay",
            "used_in_regime": "Normal, shock/high-opportunity",
            "interpretation": "Inventory draw/build pressure after reporting lag.",
        },
        {
            "family": "Physical",
            "signal": "receipts_change",
            "role": "Overlay",
            "used_in_regime": "Normal",
            "interpretation": "Flow/arrival proxy for nearby supply pressure.",
        },
        {
            "family": "Cargill",
            "signal": "cgl_inventory_change",
            "role": "Overlay",
            "used_in_regime": "Normal, shock/high-opportunity",
            "interpretation": "Private inventory change, shifted to avoid lookahead.",
        },
        {
            "family": "Cargill",
            "signal": "crush_surprise",
            "role": "Overlay",
            "used_in_regime": "Soybean physical overlay",
            "interpretation": "Processed minus planned crush activity.",
        },
        {
            "family": "Cargill",
            "signal": "crush_utilization",
            "role": "Overlay",
            "used_in_regime": "Soybean physical overlay",
            "interpretation": "Processed/planned utilization pressure.",
        },
        {
            "family": "Regime/opportunity",
            "signal": "prediction dispersion",
            "role": "Regime gate",
            "used_in_regime": "Shock/high-opportunity",
            "interpretation": "Cross-sectional conviction in model predictions.",
        },
        {
            "family": "Regime/opportunity",
            "signal": "curve dispersion",
            "role": "Regime gate",
            "used_in_regime": "Shock/high-opportunity",
            "interpretation": "Dispersion in curve tightness across grains.",
        },
        {
            "family": "Regime/opportunity",
            "signal": "momentum dispersion",
            "role": "Regime gate",
            "used_in_regime": "Shock/high-opportunity",
            "interpretation": "Dispersion in 60-day momentum across grains.",
        },
        {
            "family": "Risk state",
            "signal": "ewm_vol / expanding_long_vol",
            "role": "Regime weight",
            "used_in_regime": "Low-vol/churn, stress",
            "interpretation": "Live-observable volatility state used for sleeve weights.",
        },
    ]
)


st.set_page_config(
    page_title="Grain Strategy Backtest",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --card-bg: #525866;
    --card-border: rgba(255, 255, 255, 0.12);
    --card-text: rgba(248, 250, 252, 0.96);
    --muted-text: rgba(226, 232, 240, 0.78);
    --pnl-pos: #2e7d32;
    --pnl-neg: #c62828;
    --sig-long-bg: rgba(76, 175, 80, 0.18);
    --sig-long-fg: #1b5e20;
    --sig-short-bg: rgba(229, 57, 53, 0.18);
    --sig-short-fg: #b71c1c;
    --sig-flat-bg: rgba(107, 114, 128, 0.14);
    --sig-flat-fg: rgba(17, 24, 39, 0.82);
}
html[data-theme="dark"], body[data-theme="dark"], .stApp[data-theme="dark"], [data-theme="dark"] {
    --card-bg: #525866;
    --card-border: rgba(255, 255, 255, 0.12);
    --card-text: rgba(248, 250, 252, 0.96);
    --muted-text: rgba(226, 232, 240, 0.78);
    --pnl-pos: #66bb6a;
    --pnl-neg: #ef5350;
    --sig-long-bg: #243229;
    --sig-long-fg: #a5d6a7;
    --sig-short-bg: #372628;
    --sig-short-fg: #ef9a9a;
    --sig-flat-bg: #30333a;
    --sig-flat-fg: rgba(232, 234, 240, 0.84);
}
.block-container {padding-top: 1rem; padding-bottom: 0rem;}
[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    box-shadow: inset 0 0 0 1px var(--card-border);
    border-radius: 8px;
    padding: 8px 12px;
    border-left: 4px solid var(--card-border);
}
[data-testid="stMetric"] div {background: transparent !important;}
[data-testid="stMetric"] label {font-size: 0.72rem;}
div[data-testid="stMetricValue"] {font-size: 1.25rem; color: var(--card-text) !important;}
div[data-testid="stMetricLabel"] p {color: var(--muted-text) !important;}
div[data-testid="stMetricDelta"] {color: var(--card-text) !important;}
.sig-badge {display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem;}
.sig-long {background: var(--sig-long-bg); color: var(--sig-long-fg);}
.sig-short {background: var(--sig-short-bg); color: var(--sig-short-fg);}
.sig-flat {background: var(--sig-flat-bg); color: var(--sig-flat-fg);}
.note-box {
    border-left: 4px solid rgba(148, 163, 184, 0.80);
    background: rgba(148, 163, 184, 0.10);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 5px 0 12px 0;
}
.section-caption {color: rgba(148, 163, 184, 0.95); font-size: 0.90rem;}
</style>
""",
    unsafe_allow_html=True,
)


def _fmt_num(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def _fmt_money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${float(value):,.0f}"


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.1%}"


def _drawdown(cum_pnl: pd.Series) -> pd.Series:
    running_high = cum_pnl.cummax()
    return cum_pnl - running_high


def _slice_bt(bt: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = bt.copy()
    if start:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def _window_metric_row(bt: pd.DataFrame) -> pd.Series:
    metrics = performance_metrics(bt)
    active = bt.loc[bt["held_gross_exposure"] > 1.0e-12]
    gross_pnl = float(active["gross_pnl"].sum()) if len(active) else 0.0
    total_cost = float(active.get("costs", pd.Series(dtype=float)).sum()) if len(active) else 0.0
    metrics["gross_pnl"] = gross_pnl
    metrics["total_cost"] = total_cost
    metrics["cost_drag_fraction"] = np.nan if abs(gross_pnl) < 1.0e-12 else total_cost / abs(gross_pnl)
    metrics["winning_days"] = int((active["net_pnl"] > 0.0).sum()) if len(active) else 0
    metrics["losing_days"] = int((active["net_pnl"] < 0.0).sum()) if len(active) else 0
    return metrics


@st.cache_data(show_spinner="Running grain backtests...")
def load_backtest_bundle() -> dict:
    results = run_research_pipeline(data_dir=str(DATA_DIR), split_date=SPLIT_DATE)
    pnl = results["futures_pnl"]
    base_positions = results["model_positions"]
    skip_positions = skip_rebalance_positions(base_positions, 2)
    multi_positions, multi_active = multi_condition_filter_positions(
        results["predictions"],
        pnl,
        results["feature_panels"],
        FINAL_OPPORTUNITY_QUANTILES["prediction"],
        FINAL_OPPORTUNITY_QUANTILES["curve"],
        FINAL_OPPORTUNITY_QUANTILES["momentum"],
    )
    final_positions = 0.50 * skip_positions + 0.50 * multi_positions
    observable_positions, sleeve_weights, diagnostics = observable_three_sleeve_positions(
        results["predictions"],
        pnl,
        results["feature_panels"],
        base_positions,
        apply_vol_scale=False,
    )
    observable_scaled_positions, sleeve_weights_scaled, diagnostics_scaled = observable_three_sleeve_positions(
        results["predictions"],
        pnl,
        results["feature_panels"],
        base_positions,
        apply_vol_scale=True,
    )
    positions = {
        "Static edge-filtered": base_positions,
        "2-day skip-rebalance": skip_positions,
        "Multi-condition filter": multi_positions,
        "Final 50/50 blend": final_positions,
        "Observable three-sleeve blend": observable_positions,
        "Observable three-sleeve + vol scale": observable_scaled_positions,
        "Annual walk-forward Ridge": results["walk_forward_positions"],
    }

    backtests = {}
    pnl_by_asset = {}
    for name, pos in positions.items():
        bt, asset_pnl = backtest_positions_with_costs(
            pos,
            pnl,
            trade_cost_per_lot=MARKET_COST_CASE["trade_cost_per_lot"],
            holding_cost_rate=MARKET_COST_CASE["holding_cost_rate"],
            margin_budget=MARKET_COST_CASE["margin_budget"],
        )
        backtests[name] = bt
        pnl_by_asset[name] = asset_pnl

    regime_experiment = run_observable_regime_weight_experiment(str(DATA_DIR), split_date=SPLIT_DATE)
    final_selection = run_final_strategy_selection(str(DATA_DIR), split_date=SPLIT_DATE)
    vol_state = grain_volatility_state(pnl)

    return {
        "pipeline": results,
        "positions": positions,
        "backtests": backtests,
        "pnl_by_asset": pnl_by_asset,
        "multi_active": multi_active,
        "sleeve_weights": sleeve_weights,
        "sleeve_weights_scaled": sleeve_weights_scaled,
        "diagnostics": diagnostics,
        "diagnostics_scaled": diagnostics_scaled,
        "regime_experiment": regime_experiment,
        "final_selection": final_selection,
        "vol_state": vol_state,
    }


def render_kpi_row(metrics: pd.Series, bt_window: pd.DataFrame) -> None:
    dd = _drawdown(bt_window["cum_pnl"]).min() if len(bt_window) else np.nan
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sharpe", _fmt_num(metrics.get("sharpe"), 2))
    c2.metric("Total P&L", _fmt_money(metrics.get("total_pnl")))
    c3.metric("Max DD", _fmt_money(dd))
    c4.metric("Hit Rate", _fmt_pct(metrics.get("hit_rate")))
    c5.metric("Turnover", _fmt_num(metrics.get("avg_daily_turnover"), 3))
    c6.metric("Cost Drag", _fmt_pct(metrics.get("cost_drag_fraction")))


def render_equity_and_drawdown(bt: pd.DataFrame, initial_capital: float) -> None:
    chart = pd.DataFrame(index=bt.index)
    chart["Equity ($)"] = initial_capital + bt["cum_pnl"]
    chart["Drawdown ($)"] = _drawdown(bt["cum_pnl"])
    eq_tab, dd_tab, daily_tab = st.tabs(["Equity Curve", "Drawdown", "Daily P&L"])
    with eq_tab:
        st.line_chart(chart[["Equity ($)"]], use_container_width=True)
    with dd_tab:
        st.line_chart(chart[["Drawdown ($)"]], use_container_width=True)
    with daily_tab:
        st.bar_chart(bt["net_pnl"].rename("Daily net P&L"), use_container_width=True)


def render_signal_breakdown(bundle: dict, strategy_name: str, bt: pd.DataFrame) -> None:
    positions = bundle["positions"][strategy_name].reindex(bt.index).fillna(0.0)
    if len(positions) == 0:
        st.info("No positions in selected window.")
        return
    last_pos = positions.iloc[-1]
    rows = []
    for commodity in COMMODITIES:
        pos = float(last_pos.get(commodity, 0.0))
        if pos > 0.05:
            direction, css = "LONG", "sig-long"
        elif pos < -0.05:
            direction, css = "SHORT", "sig-short"
        else:
            direction, css = "FLAT", "sig-flat"
        rows.append(
            {
                "Commodity": commodity,
                "Direction": direction,
                "Position": pos,
                "Current weight": abs(pos) / max(float(last_pos.abs().sum()), 1.0e-12),
                "badge": css,
            }
        )
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        col.markdown(f"#### {row['Commodity']}")
        col.markdown(
            f"<span class='sig-badge {row['badge']}'>{row['Direction']}</span>",
            unsafe_allow_html=True,
        )
        col.metric("Position", f"{row['Position']:+.2f}")
    st.dataframe(
        pd.DataFrame(rows).drop(columns=["badge"]).round({"Position": 3, "Current weight": 3}),
        use_container_width=True,
        hide_index=True,
    )


def render_cost_audit(bundle: dict) -> None:
    table = bundle["regime_experiment"]["regime_weight_table"].copy()
    cols = [
        "strategy",
        "case",
        "trade_cost_per_lot",
        "holding_cost_rate",
        "oos_sharpe",
        "oos_pnl",
        "full_sharpe",
        "turnover",
        "total_turnover_lots",
        "margin_dollar_days",
        "trade_cost",
        "expected_trade_cost",
        "holding_cost",
        "expected_holding_cost",
        "total_cost",
        "CDF",
    ]
    st.dataframe(table[cols].round(3), use_container_width=True, hide_index=True)
    st.caption(
        "Every row uses the same unit assumptions. Dollar costs differ only because "
        "turnover and margin exposure differ by strategy."
    )


def render_regime_panel(bundle: dict) -> None:
    regime_table = bundle["regime_experiment"]["regime_weight_table"].copy()
    weight_summary = bundle["regime_experiment"]["weight_summary"].copy()
    assumptions = bundle["regime_experiment"]["assumptions"].copy()

    st.markdown(
        """
<div class="note-box">
The named drought, flood, COVID, and trade-war periods are diagnostics only.
The live-testable regime weighting uses lagged realized volatility and lagged
opportunity dispersion, not the historical label.
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        st.markdown("#### Observable regime assumptions")
        st.dataframe(
            assumptions.assign(value=assumptions["value"].astype(str)),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown("#### Average sleeve weights")
        st.dataframe(weight_summary.round(3), use_container_width=True, hide_index=True)

    st.markdown("#### Regime-weight test results")
    show_cols = [
        "strategy",
        "oos_sharpe",
        "oos_pnl",
        "full_sharpe",
        "full_pnl",
        "max_dd",
        "turnover",
        "total_cost",
        "CDF",
    ]
    st.dataframe(regime_table[show_cols].round(3), use_container_width=True, hide_index=True)


def render_period_table(bundle: dict, strategy_name: str) -> None:
    bt = bundle["backtests"][strategy_name]
    periods = period_performance(bt)
    show_cols = ["period", "start", "end", "total_pnl", "sharpe", "max_drawdown", "hit_rate"]
    st.dataframe(periods[show_cols].round(3), use_container_width=True, hide_index=True)


def render_signal_catalog(role_filter: list[str], regime_filter: str) -> None:
    catalog = SIGNAL_CATALOG.copy()
    if role_filter:
        catalog = catalog.loc[catalog["role"].isin(role_filter)]
    if regime_filter != "All":
        catalog = catalog.loc[catalog["used_in_regime"].str.contains(regime_filter, case=False, regex=False)]
    st.dataframe(catalog, use_container_width=True, hide_index=True)


st.sidebar.title("Dashboard")
st.sidebar.radio("Page", ["Backtest"], label_visibility="collapsed")
st.sidebar.markdown("**Backtest Controls**")

bundle = load_backtest_bundle()
strategy_names = list(bundle["backtests"].keys())
default_strategy = "Final 50/50 blend"
strategy_name = st.sidebar.selectbox(
    "Strategy",
    strategy_names,
    index=strategy_names.index(default_strategy),
    help="Backtest sleeve shown in the main charts and signal panel.",
)

period_options = {
    "Full History": (None, None),
    "In Sample: 2010-2017": (None, "2017-12-31"),
    "Out of Sample: 2018-2020": ("2018-01-01", None),
    "Russian drought/export ban": ("2010-07-01", "2011-06-30"),
    "US drought rally/retrace": ("2012-06-01", "2013-05-31"),
    "Low-price abundant supply": ("2014-06-01", "2017-12-31"),
    "US-China trade war": ("2018-07-06", "2020-01-15"),
    "2019 prevented planting floods": ("2019-05-01", "2019-07-31"),
    "COVID demand shock": ("2020-02-24", "2020-06-30"),
    "COVID recovery/China buying": ("2020-07-01", "2020-12-31"),
}
period_choice = st.sidebar.selectbox("Period", list(period_options.keys()), index=0)
display_start, display_end = period_options[period_choice]
initial_capital = st.sidebar.number_input("Initial capital ($)", 10_000, 10_000_000, 50_000, 5_000)

st.sidebar.markdown("**Signal Catalog**")
role_filter = st.sidebar.multiselect(
    "Signal role",
    ["Core", "Overlay", "Regime gate", "Regime weight"],
    default=["Core", "Overlay", "Regime gate", "Regime weight"],
)
regime_filter = st.sidebar.selectbox(
    "Regime filter",
    ["All", "Normal", "Low-vol/churn", "Shock/high-opportunity", "Soybean physical overlay"],
)

st.title("Backtest")
st.caption(
    "Cargill grain futures strategy dashboard: cost-adjusted backtests, sleeve comparisons, "
    "signal catalog, and regime diagnostics."
)

bt_full = bundle["backtests"][strategy_name]
bt_window = _slice_bt(bt_full, display_start, display_end)
metrics = _window_metric_row(bt_window)

render_kpi_row(metrics, bt_window)

st.markdown("### Strategy Summary")
left, right = st.columns([1.2, 1.0])
with left:
    st.markdown(
        """
<div class="note-box">
Final selected strategy is the fixed 50/50 blend: 50% 2-day skip-rebalance and
50% multi-condition opportunity filter. The observable three-sleeve regime
blend is documented, but not promoted because it does not beat the fixed blend
on cost-adjusted Sharpe/drawdown.
</div>
""",
        unsafe_allow_html=True,
    )
with right:
    st.dataframe(
        bundle["final_selection"]["assumptions"].assign(
            value=bundle["final_selection"]["assumptions"]["value"].astype(str)
        ),
        use_container_width=True,
        hide_index=True,
    )

render_equity_and_drawdown(bt_window, float(initial_capital))

tabs = st.tabs(
    [
        "Overview",
        "Signals",
        "Regime",
        "Costs",
        "Periods",
        "Daily Log",
    ]
)

with tabs[0]:
    st.markdown("### Strategy Comparison")
    rows = []
    for name, bt in bundle["backtests"].items():
        m = _window_metric_row(_slice_bt(bt, display_start, display_end))
        rows.append(
            {
                "Strategy": name,
                "Sharpe": m.get("sharpe", np.nan),
                "Total P&L": m.get("total_pnl", np.nan),
                "Hit Rate": m.get("hit_rate", np.nan),
                "Winning Days": m.get("winning_days", 0),
                "Max DD": _drawdown(_slice_bt(bt, display_start, display_end)["cum_pnl"]).min(),
                "Turnover": m.get("avg_daily_turnover", np.nan),
                "Cost Drag": m.get("cost_drag_fraction", np.nan),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)

    st.markdown("### Current Position Snapshot")
    render_signal_breakdown(bundle, strategy_name, bt_window)

with tabs[1]:
    st.markdown("### Signals Used")
    render_signal_catalog(role_filter, regime_filter)
    st.markdown("### Latest model predictions")
    latest_pred = bundle["pipeline"]["predictions"].dropna(how="all").iloc[-1].rename("latest_prediction")
    latest_df = latest_pred.reset_index()
    latest_df.columns = ["Commodity", "Latest prediction"]
    st.dataframe(latest_df.round(3), use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown("### Regime Weighting")
    render_regime_panel(bundle)
    st.markdown("### Volatility State")
    vol_state = bundle["vol_state"].dropna().tail(252)
    if len(vol_state):
        st.line_chart(vol_state[["vol_ratio"]], use_container_width=True)
    st.caption("Low-vol threshold is 0.70; high-vol threshold is 1.30; crisis threshold is 2.00.")

with tabs[3]:
    st.markdown("### Cost Audit")
    render_cost_audit(bundle)

with tabs[4]:
    st.markdown(f"### Named Period Performance: {strategy_name}")
    render_period_table(bundle, strategy_name)

with tabs[5]:
    st.markdown("### Daily P&L Log")
    n = min(60, len(bt_window))
    log = bt_window.tail(n).copy()
    log = log[["gross_pnl", "trade_cost", "holding_cost", "costs", "net_pnl", "turnover", "gross_exposure"]]
    log.index = log.index.strftime("%Y-%m-%d")
    st.dataframe(log.iloc[::-1].round(3), use_container_width=True)

st.caption(
    f"As of {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
    f"{strategy_name} | {period_choice} | "
    f"Cost case: {MARKET_COST_CASE['case']} "
    f"(${MARKET_COST_CASE['trade_cost_per_lot']:.2f}/lot, "
    f"{MARKET_COST_CASE['holding_cost_rate']:.1%} annual holding cost)"
)
