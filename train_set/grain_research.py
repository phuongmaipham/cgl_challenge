"""Core utilities for the grain-strategy research notebook.

Everything is plain pandas/numpy so the notebook stays portable. External
data (yfinance, weather, EIA) is read from CSVs in the `train_set` folder
that I downloaded once at the start of the project.
"""

import os
import numpy as np
import pandas as pd

from research_config import (
    COMMODITIES,
    CONTRACT_MULTIPLIER,
    DEFAULT_MARGIN_PER_LOT,
    SPLIT_DATE,
    OUTRIGHT_CORE_FEATURES,
    OUTRIGHT_PHYSICAL_FEATURES,
    COST_CASES,
    REGIME_PERIODS,
    METEOSTAT_LOCATIONS,
    COMMODITY_LOCATION_WEIGHTS,
    families_for_variant,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_train_set(data_dir="train_set"):
    """Load every CSV in train_set into a dict of DataFrames keyed by name."""
    names = {
        "adj1":        "train_adjPrices1.csv",
        "adj2":        "train_adjPrices2.csv",
        "unadj1":      "train_unadjPrices1.csv",
        "unadj2":      "train_unadjPrices2.csv",
        "cot_mm":      "train_cot_mm.csv",
        "cot_pm_oi":   "train_cot_pm_oi.csv",
        "inventories": "train_inventories.csv",
        "receipts":    "train_receipts.csv",
        "cgl_inv":     "train_cgl_inv.csv",
        "cgl_crush":   "train_cgl_crush.csv",
    }
    data = {}
    for key, filename in names.items():
        df = pd.read_csv(os.path.join(data_dir, filename), index_col=0, parse_dates=True)
        data[key] = df.sort_index().apply(pd.to_numeric, errors="coerce")
    return data


def load_external_yfinance(data_dir="train_set"):
    return pd.read_csv(os.path.join(data_dir, "external_yfinance.csv"),
                       index_col=0, parse_dates=True).sort_index()


def load_external_weather(data_dir="train_set"):
    df = pd.read_csv(os.path.join(data_dir, "external_weather.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_external_eia_ethanol(data_dir="train_set"):
    return pd.read_csv(os.path.join(data_dir, "external_eia_ethanol.csv"),
                       index_col=0, parse_dates=True).sort_index()


# ═══════════════════════════════════════════════════════════════════════════
# Lag-aware alignment, feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def to_available_calendar(df, trading_index, lag_days):
    """Treat each row's date as observation date, shift by `lag_days`, ffill."""
    if df is None or df.empty:
        return pd.DataFrame(index=trading_index)
    available = df.copy()
    available.index = available.index + pd.Timedelta(days=lag_days)
    return available.reindex(trading_index, method="ffill")


def rolling_zscore(series, lookback, min_periods=None):
    if min_periods is None:
        min_periods = max(20, lookback // 4)
    mean = series.rolling(lookback, min_periods=min_periods).mean()
    std = series.rolling(lookback, min_periods=min_periods).std()
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-5.0, 5.0).fillna(0.0)


def build_feature_panels(data):
    """Return a dict {commodity -> feature DataFrame} plus the futures pnl.

    Features are computed from adjusted prices (returns, momentum, vol),
    unadjusted prices (curve spread, curve change), COT, inventories,
    receipts, Cargill inventories and crush.
    """
    adj = data["adj1"][COMMODITIES].copy()
    unadj1 = data["unadj1"][COMMODITIES].copy()
    unadj2 = data["unadj2"][COMMODITIES].copy()
    trading_index = adj.index

    futures_pnl = adj.diff() * CONTRACT_MULTIPLIER

    feature_panels = {c: pd.DataFrame(index=trading_index) for c in COMMODITIES}

    # ── Price-derived features (per commodity)
    # All features are z-scored over rolling windows so they are comparable
    # in magnitude across commodities and across families.
    for c in COMMODITIES:
        price = adj[c]
        feature_panels[c]["mom_20"] = rolling_zscore(price.pct_change(20), 252)
        feature_panels[c]["mom_60"] = rolling_zscore(price.pct_change(60), 252)
        feature_panels[c]["rev_5"]  = -rolling_zscore(price.pct_change(5), 126)
        feature_panels[c]["vol_20"] = rolling_zscore(price.pct_change().rolling(20).std(), 252)

        spread = unadj2[c] - unadj1[c]
        ratio  = unadj2[c] / unadj1[c].replace(0.0, np.nan) - 1.0
        feature_panels[c]["curve_spread"]    = rolling_zscore(spread, 252)
        feature_panels[c]["curve_ratio"]     = rolling_zscore(ratio, 252)
        feature_panels[c]["curve_change_20"] = rolling_zscore(spread.diff(20), 252)

    # ── COT features (1-day lag for safety)
    for tbl, label in [("cot_mm", "cot_mm_level"), ("cot_pm_oi", "cot_pm_oi_level")]:
        if tbl in data and not data[tbl].empty:
            aligned = to_available_calendar(data[tbl], trading_index, lag_days=1)
            for c in COMMODITIES:
                if c in aligned.columns:
                    feature_panels[c][label] = rolling_zscore(aligned[c], 156)

    # ── Public inventories / receipts (1-day lag)
    if "inventories" in data and not data["inventories"].empty:
        aligned = to_available_calendar(data["inventories"], trading_index, 1)
        for c in COMMODITIES:
            if c in aligned.columns:
                feature_panels[c]["public_inventory_level"]  = rolling_zscore(aligned[c], 252)
                feature_panels[c]["public_inventory_change"] = rolling_zscore(aligned[c].diff(20), 252)

    if "receipts" in data and not data["receipts"].empty:
        aligned = to_available_calendar(data["receipts"], trading_index, 1)
        for c in COMMODITIES:
            if c in aligned.columns:
                feature_panels[c]["receipts_change"] = rolling_zscore(aligned[c].diff(20), 252)

    # ── Cargill inventory / crush (1-day lag)
    if "cgl_inv" in data and not data["cgl_inv"].empty:
        aligned = to_available_calendar(data["cgl_inv"], trading_index, 1)
        for c in COMMODITIES:
            if c in aligned.columns:
                feature_panels[c]["cgl_inventory_change"] = rolling_zscore(aligned[c].diff(20), 252)

    if "cgl_crush" in data and not data["cgl_crush"].empty:
        crush = to_available_calendar(data["cgl_crush"], trading_index, 1)
        if "SOYABEAN" in crush.columns:
            feature_panels["SOYABEAN"]["crush_surprise"]    = rolling_zscore(crush["SOYABEAN"].diff(5), 252)
            feature_panels["SOYABEAN"]["crush_utilization"] = rolling_zscore(crush["SOYABEAN"], 252)

    # Trim warm-up rows.
    for c in COMMODITIES:
        feature_panels[c] = feature_panels[c].iloc[252:].copy()
    futures_pnl = futures_pnl.loc[feature_panels[COMMODITIES[0]].index]

    return feature_panels, futures_pnl


# ═══════════════════════════════════════════════════════════════════════════
# External signal builders (read from CSV — no live downloads)
# ═══════════════════════════════════════════════════════════════════════════

def build_yfinance_features(trading_index, data_dir="train_set"):
    closes = load_external_yfinance(data_dir)
    closes = closes.reindex(trading_index, method="ffill")
    rets = closes.pct_change()

    feats = pd.DataFrame(index=trading_index)
    for col in closes.columns:
        feats[f"{col}_mom_20"]  = rolling_zscore(rets[col].rolling(20).mean(), 252)
        feats[f"{col}_mom_60"]  = rolling_zscore(rets[col].rolling(60).mean(), 252)
        feats[f"{col}_level_z"] = rolling_zscore(closes[col], 252)
    return feats


def build_full_panel(commodity, feature_panels, weather_features, ethanol_features, yfin_features):
    """Join all feature sources into one DataFrame for the requested commodity."""
    panel = feature_panels[commodity].copy()
    if commodity in weather_features:
        panel = panel.join(weather_features[commodity], rsuffix="_wx")
    panel = panel.join(ethanol_features, rsuffix="_eia")
    panel = panel.join(yfin_features, rsuffix="_yf")
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# Family-based signal aggregation
# ═══════════════════════════════════════════════════════════════════════════

def family_signal(panel, family_features, smooth=10):
    """Equal-weight z-scored features inside one family.

    `family_features` is a {feature_name: sign} dict.  Missing features are
    silently dropped — that way the same recipe can be used with the PROVIDED
    or FULL feature panel.
    """
    cols = [(f, s) for f, s in family_features.items() if f in panel.columns]
    if not cols:
        return pd.Series(0.0, index=panel.index)
    df = pd.concat([s * panel[f] for f, s in cols], axis=1)
    sig = df.mean(axis=1)
    return sig.rolling(smooth, min_periods=1).mean()


def equal_family_weight(panel, families, smooth=10):
    """Equal-weight every family, then average families together.

    Equal-weighting at the family level (instead of feature level) keeps any
    one large family from dominating just because it has more features.
    """
    sigs = [family_signal(panel, feats, smooth=smooth) for feats in families.values()]
    if not sigs:
        return pd.Series(0.0, index=panel.index)
    return pd.concat(sigs, axis=1).mean(axis=1)


def ic_family_selected(panel, families, target, train_mask, top_n=2, smooth=10):
    """Pick the top-N families by absolute IS information coefficient, equal-weight them.

    The IS IC fixes which families to use — selection is locked before OOS.
    A negative IC family is included with its sign flipped (so it still
    contributes positively).
    """
    ics = {}
    for fname, feats in families.items():
        sig = family_signal(panel, feats, smooth=smooth)
        valid = train_mask & sig.notna() & target.notna()
        if valid.sum() < 100:
            continue
        ics[fname] = sig[valid].corr(target[valid])

    if not ics:
        return pd.Series(0.0, index=panel.index), {}

    ranked = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    sigs = []
    for fname, ic in ranked:
        sig = family_signal(panel, families[fname], smooth=smooth)
        sigs.append(np.sign(ic) * sig)
    out = pd.concat(sigs, axis=1).mean(axis=1)
    return out, dict(ranked)


# ═══════════════════════════════════════════════════════════════════════════
# Wheat: SRW/HRW pair MR with Cargill physical-pressure overlay
# ═══════════════════════════════════════════════════════════════════════════

def _wheat_pair_components(panel):
    """Per-leg wheat features grouped into the components used in pair signals."""
    return {
        "price_mr":         panel["rev_5"],
        "curve":            (panel["curve_spread"] + panel["curve_ratio"] + panel["curve_change_20"]) / 3.0,
        "physical_public":  (-panel["public_inventory_change"] - panel["receipts_change"]) / 2.0,
        "physical_cargill": (-panel["cgl_inventory_change"] + panel.get("crush_surprise", 0.0)
                              + panel.get("crush_utilization", 0.0)) / 3.0,
    }


def wheat_pair_mr_with_cargill(feature_panels, futures_pnl, mr_weight=0.9, cargill_weight=0.1,
                                target_daily_pair_vol=40.0, max_leg_lot=0.45,
                                halflife=5.0, signal_threshold=0.12, rebalance_every=5):
    """Wheat SRW/HRW pair: 5-day reversal MR + Cargill physical-pressure overlay.

    Faithful to the original `pair_price_mr_cargill_90_10_cost_control`:
      • Per-leg `rev_5` and Cargill physical components are differenced (SRW - HRW).
      • A weighted blend (default 90% MR + 10% Cargill) gives the pair score.
      • The pair score is squashed by tanh, EWM-smoothed (halflife 5), and
        zeroed below |score| < threshold.
      • Positions are *fractional* lots, vol-scaled, clipped to ±0.45 per leg,
        and rebalanced weekly to keep turnover low.

    The wheat pair is a small, low-volume sleeve — fractional lot sizing
    matters a lot here. Integer-lot rounding would kill the strategy.
    """
    srw_panel = feature_panels["WHEAT_SRW"].reindex(futures_pnl.index)
    hrw_panel = feature_panels["WHEAT_HRW"].reindex(futures_pnl.index)
    srw_components = _wheat_pair_components(srw_panel)
    hrw_components = _wheat_pair_components(hrw_panel)

    pair_price_mr        = (srw_components["price_mr"]         - hrw_components["price_mr"]).fillna(0.0)
    pair_physical_cargill = (srw_components["physical_cargill"] - hrw_components["physical_cargill"]).fillna(0.0)

    pair_score = mr_weight * pair_price_mr + cargill_weight * pair_physical_cargill
    pair_score = np.tanh(pair_score)
    pair_score = pair_score.ewm(halflife=float(halflife), adjust=False, min_periods=1).mean()
    pair_score = pair_score.where(pair_score.abs() >= float(signal_threshold), 0.0)

    # Fractional vol-scaled positions, clipped to ±max_leg_lot.
    leg_vol = futures_pnl[["WHEAT_SRW", "WHEAT_HRW"]].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    srw_pos = pair_score * (target_daily_pair_vol / leg_vol["WHEAT_SRW"])
    hrw_pos = -pair_score * (target_daily_pair_vol / leg_vol["WHEAT_HRW"])

    positions = pd.DataFrame(0.0, index=futures_pnl.index, columns=COMMODITIES)
    positions["WHEAT_SRW"] = srw_pos.clip(-max_leg_lot, max_leg_lot).fillna(0.0)
    positions["WHEAT_HRW"] = hrw_pos.clip(-max_leg_lot, max_leg_lot).fillna(0.0)

    # Weekly rebalance: only update positions every `rebalance_every` days.
    if int(rebalance_every) > 1:
        rebalance_mask = pd.Series(False, index=positions.index)
        rebalance_mask.iloc[::int(rebalance_every)] = True
        positions = positions.where(rebalance_mask, axis=0).ffill().fillna(0.0)

    return positions, {"pair_price_mr": pair_price_mr,
                        "pair_physical_cargill": pair_physical_cargill,
                        "pair_score": pair_score}


def build_weather_features(trading_index, data_dir="train_set"):
    weather = load_external_weather(data_dir)
    feats_by_commodity = {}
    for commodity, weights in COMMODITY_LOCATION_WEIGHTS.items():
        frames = []
        for location, w in weights.items():
            sub = weather.loc[weather["location"] == location].copy()
            sub = sub.set_index("date")[["tavg", "prcp"]] * float(w)
            frames.append(sub)
        combined = sum(frames).sort_index() if frames else pd.DataFrame()
        if combined.empty:
            feats_by_commodity[commodity] = pd.DataFrame(index=trading_index)
            continue
        aligned = combined.reindex(trading_index, method="ffill")
        out = pd.DataFrame(index=trading_index)
        # Crude "heating/cooling degree day" proxies relative to 18C base.
        out["hdd"] = (18.0 - aligned["tavg"]).clip(lower=0).rolling(20).sum()
        out["cdd"] = (aligned["tavg"] - 18.0).clip(lower=0).rolling(20).sum()
        out["prcp_20d"] = aligned["prcp"].rolling(20).sum()
        out = out.apply(lambda s: rolling_zscore(s, 252))
        feats_by_commodity[commodity] = out
    return feats_by_commodity


def build_ethanol_features(trading_index, data_dir="train_set"):
    eth = load_external_eia_ethanol(data_dir)
    # Push observation date forward by 7 days to respect EIA's weekly release lag.
    eth.index = eth.index + pd.Timedelta(days=7)
    aligned = eth.reindex(trading_index, method="ffill").shift(1)
    feats = pd.DataFrame(index=trading_index)
    if "ethanol_production" in aligned:
        feats["ethanol_production_z"]  = rolling_zscore(aligned["ethanol_production"], 156)
        feats["ethanol_production_d4"] = rolling_zscore(aligned["ethanol_production"].diff(20), 156)
    if "ethanol_stocks" in aligned:
        feats["ethanol_stocks_z"]  = rolling_zscore(aligned["ethanol_stocks"], 156)
        feats["ethanol_stocks_d4"] = rolling_zscore(aligned["ethanol_stocks"].diff(20), 156)
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# Signal → position → backtest
# ═══════════════════════════════════════════════════════════════════════════

def signal_to_positions(predictions, futures_pnl, vol_target_daily=0.01):
    """Convert per-commodity z-score-like predictions into integer lot positions.

    Sign of the position follows the sign of the prediction; size is scaled
    by recent realised vol so a high-vol contract gets a smaller lot count.
    """
    pos = pd.DataFrame(0.0, index=futures_pnl.index, columns=futures_pnl.columns)
    for c in pos.columns:
        if c not in predictions.columns:
            continue
        sig = predictions[c].fillna(0.0)
        vol = futures_pnl[c].rolling(60, min_periods=20).std().bfill().fillna(1.0)
        target_pnl = vol_target_daily * 1_000_000.0
        scale = (target_pnl / vol).clip(upper=20.0)
        pos[c] = (sig.clip(-3, 3) / 3.0 * scale).round().astype(float)
    return pos


def backtest_positions(positions, futures_pnl, trade_cost_per_lot=0.0,
                       holding_cost_rate=0.0, margin_budget=np.inf):
    """Daily PnL accounting with simple cost and margin-budget controls."""
    pos = positions.shift(1).fillna(0.0)

    # Margin cap: scale all positions down on days we exceed the budget.
    if np.isfinite(margin_budget):
        notional_margin = pd.Series(0.0, index=pos.index)
        for c in pos.columns:
            notional_margin += pos[c].abs() * DEFAULT_MARGIN_PER_LOT.get(c, 2500.0)
        scale = (margin_budget / notional_margin.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        pos = pos.mul(scale, axis=0)

    gross = (pos * futures_pnl).sum(axis=1)
    turnover = pos.diff().abs().sum(axis=1).fillna(pos.abs().sum(axis=1))
    trade_cost = turnover * trade_cost_per_lot
    holding_cost = pos.abs().sum(axis=1) * (holding_cost_rate / 252.0) * 1000.0
    net = gross - trade_cost - holding_cost
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover, "positions": pos.abs().sum(axis=1)})


def perf_summary(bt, split_date=SPLIT_DATE):
    """Sharpe/PnL/drawdown over in-sample, OOS, and full-period."""
    out = []
    full = bt["net"]
    is_mask  = full.index <  pd.Timestamp(split_date)
    oos_mask = full.index >= pd.Timestamp(split_date)

    for label, mask in [("in_sample", is_mask), ("out_of_sample", oos_mask), ("full_period", slice(None))]:
        seg = full.loc[mask] if mask is not slice(None) else full
        if len(seg) == 0 or seg.std() == 0:
            sharpe, pnl, dd = 0.0, 0.0, 0.0
        else:
            sharpe = seg.mean() / seg.std() * np.sqrt(252)
            pnl = seg.sum()
            cum = seg.cumsum()
            dd = (cum - cum.cummax()).min()
        out.append({"segment": label, "sharpe": sharpe, "pnl": pnl, "max_drawdown": dd})
    return pd.DataFrame(out).set_index("segment")


def evaluate_under_cost_cases(positions, futures_pnl):
    """Backtest across all four cost cases and stack into one table."""
    rows = []
    for case in COST_CASES:
        bt = backtest_positions(
            positions, futures_pnl,
            trade_cost_per_lot=case["trade_cost_per_lot"],
            holding_cost_rate=case["holding_cost_rate"],
            margin_budget=case["margin_budget"],
        )
        perf = perf_summary(bt)
        rows.append({
            "cost_case": case["case"],
            "is_sharpe":  perf.loc["in_sample", "sharpe"],
            "oos_sharpe": perf.loc["out_of_sample", "sharpe"],
            "oos_pnl":    perf.loc["out_of_sample", "pnl"],
            "full_sharpe": perf.loc["full_period", "sharpe"],
            "full_pnl":    perf.loc["full_period", "pnl"],
            "max_drawdown": perf.loc["full_period", "max_drawdown"],
            "avg_turnover": bt["turnover"].mean(),
        })
    return pd.DataFrame(rows)


def regime_performance(bt):
    """Return PnL/Sharpe for each named historical regime."""
    rows = []
    full = bt["net"]
    for r in REGIME_PERIODS:
        mask = (full.index >= pd.Timestamp(r["start"])) & (full.index <= pd.Timestamp(r["end"]))
        seg = full.loc[mask]
        if len(seg) == 0:
            rows.append({"period": r["period"], "pnl": 0.0, "sharpe": 0.0, "days": 0})
            continue
        pnl = seg.sum()
        sharpe = (seg.mean() / seg.std() * np.sqrt(252)) if seg.std() > 0 else 0.0
        rows.append({"period": r["period"], "pnl": pnl, "sharpe": sharpe, "days": int(mask.sum())})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Signal aggregation patterns
# ═══════════════════════════════════════════════════════════════════════════

def features_to_per_commodity_signal(feature_panels, feature_names, futures_pnl):
    """Equal-weight a list of features into a per-commodity prediction.

    All features are z-scored already, so a simple mean works as a baseline
    aggregator. Sign reflects "expected positive return" by convention.
    """
    preds = pd.DataFrame(0.0, index=futures_pnl.index, columns=COMMODITIES)
    for c in COMMODITIES:
        panel = feature_panels.get(c)
        if panel is None or panel.empty:
            continue
        cols = [f for f in feature_names if f in panel.columns]
        if not cols:
            continue
        preds[c] = panel[cols].mean(axis=1)
    return preds


def online_ols_predictions(feature_panel, target, train_mask, min_train_days=252):
    """Walk-forward OLS predictions, expanding window after the warm-up.

    Used only as a single fitted-model benchmark — I want to see whether
    fitting coefficients beats fixed economic recipes. (No Ridge or RLS
    variants — those mostly add hyperparameter risk for this dataset.)
    """
    feats = feature_panel.fillna(0.0).values
    y = target.fillna(0.0).values
    preds = np.zeros(len(y))

    # Fit once on the full in-sample, then apply forward.
    train_idx = np.where(train_mask & (np.arange(len(y)) >= min_train_days))[0]
    if len(train_idx) < 50:
        return pd.Series(preds, index=target.index)

    x_train, y_train = feats[train_idx], y[train_idx]
    # Standardise X.
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-9
    x_std = (x_train - mu) / sd
    beta = np.linalg.lstsq(x_std, y_train, rcond=None)[0]

    x_all_std = (feats - mu) / sd
    preds = x_all_std @ beta
    return pd.Series(preds, index=target.index)


# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════

def header(title):
    """Print a Markdown-style separator (works in notebooks and plain stdout)."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"### {title}"))
    except Exception:
        print(f"\n— {title} —\n")
