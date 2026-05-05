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
    CORN_TRAIN_END,
    CORN_TARGET_DAILY_PNL_VOL,
    CORN_MAX_ABS_LOT,
    CORN_TRADE_COST_PER_LOT,
    CORN_HOLDING_COST_RATE,
    CORN_IC_THRESHOLD,
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
# Corn product-flow-aligned helpers
# ═══════════════════════════════════════════════════════════════════════════

def rolling_zscore_product_flow(obj, window=252, min_periods=40):
    """Rolling z-score matching the product-flow notebook convention."""
    mean = obj.rolling(window=window, min_periods=min_periods).mean()
    std = obj.rolling(window=window, min_periods=min_periods).std()
    return (obj - mean) / std.replace(0.0, np.nan)


def to_available_calendar_product_flow(df, trading_index, lag_days):
    """Release-lag alignment used by the product-flow feature builder."""
    out = df.copy()
    out.index = pd.to_datetime(out.index) + pd.DateOffset(days=int(lag_days))
    out = out.sort_index()
    out = out.groupby(out.index).last()
    return out.reindex(trading_index).ffill()


def build_product_flow_feature_panels(data, commodities=("CORN",)):
    """Build feature panels with the same timing/features as product-flow corn.

    This exists because the lighter `build_feature_panels` intentionally keeps
    a simpler portable feature set. The product-flow corn sleeve used:
      - COT lag 3 calendar days;
      - public inventories/receipts lag 2 calendar days;
      - Cargill inventory/crush lag 1 calendar day;
      - processed/planned Cargill crush activity for corn as a shared physical
        processing proxy.
    """
    commodities = list(commodities)
    trading_index = data["adj1"].index
    adj1 = data["adj1"].reindex(trading_index).ffill()
    unadj1 = data["unadj1"].reindex(trading_index).ffill()
    unadj2 = data["unadj2"].reindex(trading_index).ffill()

    futures_pnl = adj1[commodities].diff() * CONTRACT_MULTIPLIER
    pct_change = adj1.pct_change()

    cot_mm = to_available_calendar_product_flow(data["cot_mm"], trading_index, 3)
    cot_pm_oi = to_available_calendar_product_flow(data["cot_pm_oi"], trading_index, 3)
    inventories = to_available_calendar_product_flow(data["inventories"], trading_index, 2)
    receipts = to_available_calendar_product_flow(data["receipts"], trading_index, 2)
    cgl_inv = to_available_calendar_product_flow(data["cgl_inv"], trading_index, 1)
    cgl_crush = to_available_calendar_product_flow(data["cgl_crush"], trading_index, 1)

    curve_spread = unadj1 - unadj2
    curve_ratio = unadj1 / unadj2.replace(0.0, np.nan) - 1.0

    blocks = {
        "mom_20": rolling_zscore_product_flow(adj1.pct_change(20), 252, 60),
        "mom_60": rolling_zscore_product_flow(adj1.pct_change(60), 252, 80),
        "rev_5": -rolling_zscore_product_flow(adj1.pct_change(5), 126, 30),
        "vol_20": rolling_zscore_product_flow(pct_change.rolling(20, min_periods=10).std(), 252, 60),
        "curve_spread": rolling_zscore_product_flow(curve_spread, 252, 60),
        "curve_ratio": rolling_zscore_product_flow(curve_ratio, 252, 60),
        "curve_change_20": rolling_zscore_product_flow(curve_spread.diff(20), 252, 60),
        "cot_mm_level": rolling_zscore_product_flow(cot_mm, 156, 40),
        "cot_mm_change": rolling_zscore_product_flow(cot_mm.diff(5), 156, 40),
        "cot_pm_oi_level": rolling_zscore_product_flow(cot_pm_oi, 156, 40),
        "cot_pm_oi_change": rolling_zscore_product_flow(cot_pm_oi.diff(5), 156, 40),
        "public_inventory_level": rolling_zscore_product_flow(inventories, 156, 40),
        "public_inventory_change": rolling_zscore_product_flow(inventories.diff(5), 156, 40),
        "receipts_level": rolling_zscore_product_flow(receipts, 126, 30),
        "receipts_change": rolling_zscore_product_flow(receipts.diff(5), 126, 30),
        "cgl_inventory_level": rolling_zscore_product_flow(cgl_inv, 252, 60),
        "cgl_inventory_change": rolling_zscore_product_flow(cgl_inv.diff(5), 252, 60),
    }

    crush = pd.DataFrame(index=trading_index)
    crush["crush_processed"] = cgl_crush["processed"]
    crush["crush_planned"] = cgl_crush["planned"]
    crush["crush_surprise"] = cgl_crush["processed"] - cgl_crush["planned"]
    crush["crush_utilization"] = cgl_crush["processed"] / cgl_crush["planned"].replace(0.0, np.nan) - 1.0
    crush_features = rolling_zscore_product_flow(crush, 252, 60)

    panels = {}
    for commodity in commodities:
        frame = pd.DataFrame(index=trading_index)
        for feature_name, block in blocks.items():
            frame[feature_name] = block[commodity]
        for feature_name in crush_features.columns:
            frame[feature_name] = crush_features[feature_name]
        panels[commodity] = frame.clip(-5.0, 5.0).fillna(0.0)
    return panels, futures_pnl


def corn_given_signal_universe(feature_panels):
    """Corn provided-data signals used by the product-flow research path."""
    panel = feature_panels["CORN"]
    inventory_pressure = (
        -panel["public_inventory_change"]
        - panel["receipts_change"]
        - panel["cgl_inventory_change"]
    ) / 3.0
    curve_tightness = (panel["curve_spread"] + panel["curve_ratio"]) / 2.0
    price_family = (panel["mom_20"] + panel["mom_60"] + panel["rev_5"]) / 3.0
    trend = (panel["mom_20"] + panel["mom_60"] + panel["curve_spread"] + panel["cot_pm_oi_level"]) / 4.0
    cgl_crush_activity = (panel["crush_surprise"] + panel["crush_utilization"]) / 2.0
    physical_family = (inventory_pressure + curve_tightness + 0.25 * cgl_crush_activity) / 2.25
    conservative = 0.40 * physical_family + 0.30 * trend + 0.30 * price_family
    signals = {
        "given_mom_20": panel["mom_20"],
        "given_mom_60": panel["mom_60"],
        "given_rev_5": panel["rev_5"],
        "given_curve_spread": panel["curve_spread"],
        "given_curve_ratio": panel["curve_ratio"],
        "given_inventory_pressure": inventory_pressure,
        "given_cgl_inventory_pressure": -panel["cgl_inventory_change"],
        "given_cgl_crush_activity": cgl_crush_activity,
        "given_curve_tightness": curve_tightness,
        "given_price_family": price_family,
        "given_physical_family": physical_family,
        "given_trend": trend,
        "given_conservative_blend": conservative,
    }
    return {name: clean_product_flow_signal(signal, panel.index) for name, signal in signals.items()}


def clean_product_flow_signal(series, index=None):
    if index is None:
        index = series.index
    return (
        pd.Series(series, index=index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-5.0, 5.0)
    )


def mean_product_flow_signals(items, index):
    values = [item.reindex(index) for item in items if item is not None]
    if not values:
        return pd.Series(0.0, index=index)
    return clean_product_flow_signal(sum(values) / float(len(values)), index)


def weighted_product_flow_signals(signals, weights, index):
    total = pd.Series(0.0, index=index)
    used = 0.0
    for name, weight in weights.items():
        if name in signals and float(weight) != 0.0:
            total = total + float(weight) * signals[name].reindex(index)
            used += float(weight)
    if used == 0.0:
        return pd.Series(0.0, index=index)
    return clean_product_flow_signal(total / used, index)


def build_corn_product_flow_yfinance_families(trading_index, data_dir="train_set"):
    """Build corn external price/FX/macro families from saved yfinance CSV."""
    px = load_external_yfinance(data_dir).reindex(trading_index).ffill().shift(1)
    families = {}
    if {"corn", "soybean", "wheat"}.issubset(px.columns):
        corn_soy = rolling_zscore_product_flow((px["corn"] / px["soybean"]).pct_change(20, fill_method=None), 252, 60)
        corn_wheat = rolling_zscore_product_flow((px["corn"] / px["wheat"]).pct_change(20, fill_method=None), 252, 60)
        soy_corn_mr = -rolling_zscore_product_flow((px["soybean"] / px["corn"]).pct_change(20, fill_method=None), 252, 60)
        families["external_relative_grain_family"] = ((corn_soy + corn_wheat + soy_corn_mr) / 3.0).fillna(0.0)

    fx_parts = []
    if "usd_index" in px:
        fx_parts.append(-rolling_zscore_product_flow(px["usd_index"].pct_change(20, fill_method=None), 252, 60))
    if "brl" in px:
        fx_parts.append(-rolling_zscore_product_flow(px["brl"].pct_change(20, fill_method=None), 252, 60))
    if "cny" in px:
        fx_parts.append(-rolling_zscore_product_flow(px["cny"].pct_change(20, fill_method=None), 252, 60))
    if fx_parts:
        families["external_fx_export_family"] = (sum(fx_parts) / float(len(fx_parts))).fillna(0.0)

    macro_parts = []
    if "crude" in px:
        macro_parts.append(rolling_zscore_product_flow(px["crude"].pct_change(20, fill_method=None), 252, 60))
    if "equity" in px:
        macro_parts.append(rolling_zscore_product_flow(px["equity"].pct_change(20, fill_method=None), 252, 60))
    if "usd_index" in px:
        macro_parts.append(-rolling_zscore_product_flow(px["usd_index"].pct_change(60, fill_method=None), 252, 80))
    if macro_parts:
        families["external_macro_risk_family"] = (sum(macro_parts) / float(len(macro_parts))).fillna(0.0)
    return {name: clean_product_flow_signal(signal, trading_index) for name, signal in families.items()}


def build_corn_product_flow_ethanol_family(trading_index, data_dir="train_set"):
    """Build the EIA ethanol family used in the corn product-flow path."""
    ethanol = load_external_eia_ethanol(data_dir)
    available = ethanol.copy()
    available.index = available.index + pd.DateOffset(days=7)
    aligned = available.reindex(trading_index).ffill().shift(1)

    features = pd.DataFrame(index=trading_index)
    features["ethanol_production_change_4w"] = rolling_zscore_product_flow(aligned["ethanol_production"].diff(20), 156, 40)
    features["ethanol_stocks_change_4w"] = rolling_zscore_product_flow(aligned["ethanol_stocks"].diff(20), 156, 40)
    ratio = aligned["ethanol_production"] / aligned["ethanol_stocks"].replace(0.0, np.nan)
    features["ethanol_prod_to_stocks"] = rolling_zscore_product_flow(ratio, 156, 40)
    pressure = aligned["ethanol_production"].diff(20) - aligned["ethanol_stocks"].diff(20)
    features["ethanol_demand_pressure"] = rolling_zscore_product_flow(pressure, 156, 40)

    family = (
        features["ethanol_production_change_4w"]
        + features["ethanol_prod_to_stocks"]
        + features["ethanol_demand_pressure"]
        - features["ethanol_stocks_change_4w"]
    ) / 4.0
    return {
        "external_ethanol_family": clean_product_flow_signal(family, trading_index),
        "ethanol_features": features.clip(-5.0, 5.0).fillna(0.0),
    }


def _season_mask(index, months):
    return pd.Series(index.month.isin(months), index=index).astype(float)


def _add_product_flow_weather_features(aligned, seasonal=True):
    features = pd.DataFrame(index=aligned.index)
    if "tavg" in aligned:
        cdd = (aligned["tavg"] - 18.0).clip(lower=0.0)
        hdd = (18.0 - aligned["tavg"]).clip(lower=0.0)
        features["meteo_cdd_20d"] = rolling_zscore_product_flow(cdd.rolling(20, min_periods=5).sum(), 252, 60)
        features["meteo_hdd_20d"] = rolling_zscore_product_flow(hdd.rolling(20, min_periods=5).sum(), 252, 60)
    if {"tmin", "tmax"}.issubset(aligned.columns):
        temp_avg = (aligned["tmin"] + aligned["tmax"]) / 2.0
        gdd = (temp_avg.clip(upper=30.0) - 10.0).clip(lower=0.0)
        features["meteo_gdd_60d"] = rolling_zscore_product_flow(gdd.rolling(60, min_periods=15).sum(), 252, 60)
    if "tmax" in aligned:
        heat_stress = (aligned["tmax"] - 32.0).clip(lower=0.0)
        features["meteo_heat_stress_20d"] = rolling_zscore_product_flow(heat_stress.rolling(20, min_periods=5).sum(), 252, 60)
    if "prcp" in aligned:
        precip = aligned["prcp"].fillna(0.0)
        precip_20 = precip.rolling(20, min_periods=5).sum()
        features["meteo_precip_20d"] = rolling_zscore_product_flow(precip_20, 252, 60)
        features["meteo_dryness_20d"] = -features["meteo_precip_20d"]
        if "meteo_cdd_20d" in features:
            features["meteo_dry_cdd_20d"] = (features["meteo_dryness_20d"] * features["meteo_cdd_20d"]).clip(-5.0, 5.0)
    if "tmin" in aligned:
        freeze_stress = (0.0 - aligned["tmin"]).clip(lower=0.0)
        features["meteo_freeze_stress_5d"] = rolling_zscore_product_flow(freeze_stress.rolling(5, min_periods=3).sum(), 252, 60)

    if seasonal:
        planting = _season_mask(features.index, [3, 4, 5])
        growing = _season_mask(features.index, [6, 7, 8])
        harvest = _season_mask(features.index, [9, 10, 11])
        seasonal_features = {}
        for column in features.columns:
            seasonal_features[column + "_planting"] = features[column] * planting
            seasonal_features[column + "_growing"] = features[column] * growing
            seasonal_features[column + "_harvest"] = features[column] * harvest
        features = pd.concat([features, pd.DataFrame(seasonal_features, index=features.index)], axis=1)
    return features.clip(-5.0, 5.0).fillna(0.0)


def build_corn_product_flow_weather_family(trading_index, data_dir="train_set"):
    """Build the crop-belt weather family used in the corn product-flow path."""
    weather = load_external_weather(data_dir)
    weights = COMMODITY_LOCATION_WEIGHTS["CORN"]
    value_cols = [c for c in ["tavg", "tmin", "tmax", "prcp"] if c in weather.columns]
    frames = []
    for location, weight in weights.items():
        sub = weather.loc[weather["location"] == location, ["date"] + value_cols].copy()
        if sub.empty:
            continue
        sub[value_cols] = sub[value_cols] * float(weight)
        frames.append(sub)
    combined = pd.concat(frames, ignore_index=True).groupby("date")[value_cols].sum().sort_index()
    aligned = combined.reindex(trading_index).ffill().shift(1)
    features = _add_product_flow_weather_features(aligned, seasonal=True)
    candidates = [
        "meteo_cdd_20d_growing",
        "meteo_hdd_20d_growing",
        "meteo_gdd_60d_growing",
        "meteo_heat_stress_20d_growing",
        "meteo_dryness_20d_growing",
        "meteo_dry_cdd_20d_growing",
        "meteo_precip_20d_planting",
        "meteo_dryness_20d_planting",
        "meteo_freeze_stress_5d_harvest",
    ]
    existing = [c for c in candidates if c in features.columns]
    family = features[existing].mean(axis=1) if existing else pd.Series(0.0, index=trading_index)
    return {
        "external_weather_hdd_cdd_family": clean_product_flow_signal(family, trading_index),
        "weather_features": features,
    }


def build_corn_product_flow_signal_universe(feature_panels, futures_pnl, data_dir="train_set"):
    """Return all corn signals used by the product-flow-aligned tests."""
    index = futures_pnl.index
    signals = corn_given_signal_universe(feature_panels)
    signals.update(build_corn_product_flow_yfinance_families(index, data_dir))
    ethanol = build_corn_product_flow_ethanol_family(index, data_dir)
    weather = build_corn_product_flow_weather_family(index, data_dir)
    signals["external_ethanol_family"] = ethanol["external_ethanol_family"]
    signals["external_weather_hdd_cdd_family"] = weather["external_weather_hdd_cdd_family"]
    return {name: clean_product_flow_signal(signal, index) for name, signal in signals.items()}


def corn_signal_set_families(signals):
    """Families used for the requested Signal A / Signal B corn tests."""
    prices = {
        "given_mom_20": signals["given_mom_20"],
        "given_mom_60": signals["given_mom_60"],
        "given_rev_5": signals["given_rev_5"],
        "given_curve_spread": signals["given_curve_spread"],
        "given_curve_ratio": signals["given_curve_ratio"],
        "given_price_family": signals["given_price_family"],
    }
    if "external_relative_grain_family" in signals:
        prices["external_relative_grain_family"] = signals["external_relative_grain_family"]
    fundamentals_core = {
        "given_inventory_pressure": signals["given_inventory_pressure"],
        "given_cgl_inventory_pressure": signals["given_cgl_inventory_pressure"],
        "given_cgl_crush_activity": signals["given_cgl_crush_activity"],
        "given_curve_tightness": signals["given_curve_tightness"],
        "given_physical_family": signals["given_physical_family"],
    }
    fundamentals_a = dict(fundamentals_core)
    fundamentals_a["external_ethanol_family"] = signals["external_ethanol_family"]
    fundamentals_a["external_weather_hdd_cdd_family"] = signals["external_weather_hdd_cdd_family"]
    macro = {
        "external_fx_export_family": signals["external_fx_export_family"],
        "external_macro_risk_family": signals["external_macro_risk_family"],
    }
    return {
        "A": {"prices": prices, "fundamentals": fundamentals_a, "macro": macro},
        "B": {"prices": prices, "fundamentals": fundamentals_core},
        "alpha": {
            "eia": {"external_ethanol_family": signals["external_ethanol_family"]},
            "macro": macro,
            "weather": {"external_weather_hdd_cdd_family": signals["external_weather_hdd_cdd_family"]},
        },
    }


def corn_family_signal(signal_dict, index):
    return mean_product_flow_signals(list(signal_dict.values()), index)


def corn_average_all_signals(families, index):
    values = []
    for signals in families.values():
        values.extend(signals.values())
    return mean_product_flow_signals(values, index)


def corn_equal_family_signal(families, index):
    return mean_product_flow_signals([corn_family_signal(v, index) for v in families.values()], index)


def corn_select_by_ic_signal(families, futures_pnl, min_abs_ic=CORN_IC_THRESHOLD):
    """Select and orient individual Signal A/B members by train-period IC."""
    index = futures_pnl.index
    target = futures_pnl["CORN"].shift(-1)
    train_mask = pd.Series(index < pd.Timestamp(CORN_TRAIN_END), index=index)
    validation_mask = pd.Series((index >= pd.Timestamp(CORN_TRAIN_END)) & (index < pd.Timestamp(SPLIT_DATE)), index=index)
    test_mask = pd.Series(index >= pd.Timestamp(SPLIT_DATE), index=index)

    rows, selected_signals = [], []
    for family, members in families.items():
        for signal_name, signal in members.items():
            raw_signal = signal.reindex(index).fillna(0.0)
            train_ic = rank_ic_product_flow(raw_signal, target, train_mask)
            orientation = 1.0 if pd.isnull(train_ic) or train_ic >= 0.0 else -1.0
            oriented_signal = clean_product_flow_signal(orientation * raw_signal, index)
            selected = bool(pd.notnull(train_ic) and abs(train_ic) >= float(min_abs_ic))
            if selected:
                selected_signals.append(oriented_signal)
            rows.append({
                "family": family,
                "signal": signal_name,
                "train_ic": train_ic,
                "orientation": orientation,
                "selected": selected,
                "validation_ic": rank_ic_product_flow(oriented_signal, target, validation_mask),
                "test_ic": rank_ic_product_flow(oriented_signal, target, test_mask),
            })

    table = pd.DataFrame(rows)
    if not table.empty:
        table["abs_train_ic"] = table["train_ic"].abs()
        table = table.sort_values(["selected", "abs_train_ic"], ascending=[False, False]).reset_index(drop=True)
    selected_signal = mean_product_flow_signals(selected_signals, index)
    return selected_signal, table


def rank_ic_product_flow(signal, target, mask):
    aligned = pd.concat([signal, target], axis=1).dropna()
    if aligned.empty:
        return np.nan
    mask = pd.Series(mask, index=signal.index).reindex(aligned.index).fillna(False).astype(bool)
    aligned = aligned.loc[mask]
    if len(aligned) < 40 or aligned.iloc[:, 0].std() == 0.0 or aligned.iloc[:, 1].std() == 0.0:
        return np.nan
    return aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank())


def corn_trend_mr_family_signal(families, futures_pnl, feature_panels):
    index = futures_pnl.index
    target = futures_pnl["CORN"].shift(-1)
    trend_strength = feature_panels["CORN"]["mom_60"].abs().reindex(index).fillna(0.0)
    threshold = trend_strength.expanding(min_periods=252).median().shift(1)
    regimes = {
        "trend": (trend_strength > threshold).fillna(False),
        "mr_or_chop": (trend_strength <= threshold).fillna(True),
    }
    family_signals = {name: corn_family_signal(signals, index) for name, signals in families.items()}
    rows, pieces = [], []
    for regime_name, regime_mask in regimes.items():
        candidates = []
        for family, signal in family_signals.items():
            train_mask = (index < pd.Timestamp(CORN_TRAIN_END)) & regime_mask
            validation_mask = (index >= pd.Timestamp(CORN_TRAIN_END)) & (index < pd.Timestamp(SPLIT_DATE)) & regime_mask
            train_ic = rank_ic_product_flow(signal, target, train_mask)
            orientation = 1.0 if pd.isnull(train_ic) or train_ic >= 0.0 else -1.0
            validation_ic = rank_ic_product_flow(orientation * signal, target, validation_mask)
            candidates.append({
                "regime": regime_name,
                "family": family,
                "train_ic": train_ic,
                "orientation": orientation,
                "validation_ic": validation_ic,
                "train_obs": int(train_mask.sum()),
                "validation_obs": int(validation_mask.sum()),
                "signal": orientation * signal,
            })
        table = pd.DataFrame([{k: v for k, v in row.items() if k != "signal"} for row in candidates])
        selected = table.sort_values("validation_ic", ascending=False).iloc[0]
        selected_signal = next(row["signal"] for row in candidates if row["family"] == selected["family"])
        pieces.append(selected_signal * regime_mask.astype(float))
        rows.append(selected.to_dict())
    return clean_product_flow_signal(sum(pieces), index), pd.DataFrame(rows)


def corn_dynamic_linear_family_signal(families, futures_pnl, min_train_days=504, refit_every=21, alpha=100.0):
    index = futures_pnl.index
    x = pd.DataFrame({name: corn_family_signal(signals, index) for name, signals in families.items()}, index=index).fillna(0.0)
    y = futures_pnl["CORN"].shift(-1)
    pred = pd.Series(np.nan, index=index)
    beta, last_fit = None, None
    rows = []
    for i, date in enumerate(index):
        train_mask = (index < date) & y.notna()
        if int(train_mask.sum()) < min_train_days:
            continue
        x_train_raw = x.loc[train_mask]
        mean = x_train_raw.mean()
        std = x_train_raw.std().replace(0.0, np.nan).fillna(1.0)
        if beta is None or last_fit is None or (i - last_fit) >= refit_every:
            x_train = ((x_train_raw - mean) / std).values.astype(float)
            y_train = y.loc[train_mask].values.astype(float)
            x_aug = np.column_stack([np.ones(len(x_train)), x_train])
            xtx = x_aug.T @ x_aug
            penalty = np.eye(xtx.shape[0]) * float(alpha)
            penalty[0, 0] = 0.0
            try:
                beta = np.linalg.solve(xtx + penalty, x_aug.T @ y_train)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(xtx + penalty) @ x_aug.T @ y_train
            last_fit = i
            rows.append({"date": date, "intercept": beta[0], **{f"beta_{c}": beta[j + 1] for j, c in enumerate(x.columns)}})
        x_row = ((x.loc[date] - mean) / std).replace([np.inf, -np.inf], 0.0)
        pred.loc[date] = np.r_[1.0, np.asarray(x_row, dtype=float)] @ beta
    mean = pred.rolling(252, min_periods=60).mean().shift(1)
    std = pred.rolling(252, min_periods=60).std().shift(1).replace(0.0, np.nan)
    return clean_product_flow_signal(((pred - mean) / std).clip(-5.0, 5.0), index), pd.DataFrame(rows)


def smooth_corn_signal(signal, mode="long_short"):
    index = signal.index
    out = pd.Series(np.tanh(signal.astype(float) / 2.0), index=index)
    out = out.ewm(halflife=2.0, adjust=False, min_periods=1).mean()
    out[out.abs() < 0.05] = 0.0
    if mode == "long_only":
        out = out.clip(lower=0.0)
    elif mode == "short_only":
        out = out.clip(upper=0.0)
    elif mode != "long_short":
        raise ValueError(f"Unknown mode: {mode}")
    return out.fillna(0.0)


def corn_positions_from_signal(signal, futures_pnl, mode="long_short",
                               target_daily_pnl_vol=CORN_TARGET_DAILY_PNL_VOL,
                               max_abs_lot=CORN_MAX_ABS_LOT):
    index = futures_pnl.index
    cleaned = smooth_corn_signal(signal.reindex(index).fillna(0.0), mode=mode)
    asset_vol = futures_pnl["CORN"].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    lots = cleaned * (float(target_daily_pnl_vol) / asset_vol)
    positions = pd.DataFrame(0.0, index=index, columns=["CORN"])
    if mode == "long_only":
        positions["CORN"] = lots.clip(0.0, float(max_abs_lot)).fillna(0.0)
    else:
        positions["CORN"] = lots.clip(-float(max_abs_lot), float(max_abs_lot)).fillna(0.0)
    return positions


def backtest_positions_product_flow(positions, futures_pnl, trade_cost_per_lot=CORN_TRADE_COST_PER_LOT,
                                    holding_cost_rate=CORN_HOLDING_COST_RATE):
    """Product-flow-style cost accounting using margin funding per lot."""
    adjusted_positions = positions.reindex(futures_pnl.index).fillna(0.0)
    held_positions = adjusted_positions.shift(1).fillna(0.0)
    pnl = futures_pnl.reindex(adjusted_positions.index).fillna(0.0)
    gross = held_positions * pnl
    turnover_by_asset = adjusted_positions.diff().abs().fillna(0.0)
    trade_cost = turnover_by_asset * float(trade_cost_per_lot)
    margin = pd.Series({c: DEFAULT_MARGIN_PER_LOT.get(c, 2500.0) for c in adjusted_positions.columns})
    margin_use = held_positions.abs().mul(margin, axis=1)
    holding_cost = margin_use * (float(holding_cost_rate) / 252.0)
    net = gross - trade_cost - holding_cost
    result = pd.DataFrame(index=adjusted_positions.index)
    result["gross_pnl"] = gross.sum(axis=1)
    result["trade_cost"] = trade_cost.sum(axis=1)
    result["holding_cost"] = holding_cost.sum(axis=1)
    result["costs"] = result["trade_cost"] + result["holding_cost"]
    result["net_pnl"] = net.sum(axis=1)
    result["turnover"] = turnover_by_asset.sum(axis=1)
    result["gross_exposure"] = adjusted_positions.abs().sum(axis=1)
    result["held_gross_exposure"] = held_positions.abs().sum(axis=1)
    result["margin_used"] = margin_use.sum(axis=1)
    result["cum_pnl"] = result["net_pnl"].cumsum()
    return result, net


def product_flow_performance_metrics(bt):
    active = bt["held_gross_exposure"] > 1.0e-12
    pnl = bt.loc[active, "net_pnl"].dropna()
    if len(pnl) == 0:
        return pd.Series(dtype=float)
    vol = pnl.std()
    sharpe = np.nan if vol == 0.0 else pnl.mean() / vol * np.sqrt(252.0)
    cum = pnl.cumsum()
    drawdown = cum - cum.cummax()
    return pd.Series({
        "days": float(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "sharpe": float(sharpe) if pd.notnull(sharpe) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((pnl > 0.0).mean()),
        "avg_daily_turnover": float(bt["turnover"].reindex(pnl.index).mean()),
        "avg_gross_exposure": float(bt["gross_exposure"].reindex(pnl.index).mean()),
    })


def product_flow_split_performance(bt, split_date=SPLIT_DATE):
    split_date = pd.Timestamp(split_date)
    return pd.DataFrame({
        "in_sample": product_flow_performance_metrics(bt.loc[bt.index < split_date]),
        "out_of_sample": product_flow_performance_metrics(bt.loc[bt.index >= split_date]),
        "full_period": product_flow_performance_metrics(bt),
    })


def summarize_corn_backtest(bt, train_end=CORN_TRAIN_END, split_date=SPLIT_DATE):
    full = product_flow_split_performance(bt, split_date)
    train_val = product_flow_split_performance(bt.loc[bt.index < pd.Timestamp(split_date)], train_end)
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


def product_flow_period_performance(bt, periods=None):
    if periods is None:
        periods = REGIME_PERIODS
    rows = []
    for item in periods:
        start, end = pd.Timestamp(item["start"]), pd.Timestamp(item["end"])
        metrics = product_flow_performance_metrics(bt.loc[(bt.index >= start) & (bt.index <= end)])
        row = {"period": item["period"], "start": start, "end": end}
        for key, value in metrics.items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def corn_vol_regime_masks(feature_panels, futures_pnl):
    index = futures_pnl.index
    pnl = futures_pnl["CORN"].fillna(0.0)
    vol = pnl.rolling(60, min_periods=20).std().shift(1)
    lt_vol = vol.expanding(min_periods=252).median().shift(1)
    high_q = vol.expanding(min_periods=252).quantile(0.75).shift(1)
    high_vol = ((vol > 1.20 * lt_vol) | (vol > high_q)).reindex(index).fillna(False)
    low_vol = (vol < 0.80 * lt_vol).reindex(index).fillna(False)
    normal_vol = (~high_vol & ~low_vol).reindex(index).fillna(True)
    return {"low_vol": low_vol.astype(bool), "normal_vol": normal_vol.astype(bool), "high_vol": high_vol.astype(bool)}


def corn_regime_signal_ic_table(signals, futures_pnl, regime_mask):
    index = futures_pnl.index
    target = futures_pnl["CORN"].shift(-1)
    splits = {
        "train": pd.Series(index < pd.Timestamp(CORN_TRAIN_END), index=index),
        "validation": pd.Series((index >= pd.Timestamp(CORN_TRAIN_END)) & (index < pd.Timestamp(SPLIT_DATE)), index=index),
        "test": pd.Series(index >= pd.Timestamp(SPLIT_DATE), index=index),
    }
    rows = []
    regime = pd.Series(regime_mask, index=index).fillna(False).astype(bool)
    for name, signal in signals.items():
        row = {"signal": name}
        for split_name, split_mask in splits.items():
            mask = split_mask & regime
            row[f"{split_name}_obs"] = int(mask.sum())
            row[f"{split_name}_ic"] = rank_ic_product_flow(signal.reindex(index), target, mask)
        row["passes_ic_threshold"] = bool(
            row["train_obs"] >= 120
            and row["validation_obs"] >= 40
            and pd.notnull(row["train_ic"])
            and abs(row["train_ic"]) >= CORN_IC_THRESHOLD
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["passes_ic_threshold", "train_ic"], ascending=[False, False])


def corn_candidate_families(selected_signals):
    definitions = {
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
    families, members = {}, {}
    index = next(iter(selected_signals.values())).index
    for family, names in definitions.items():
        used = [selected_signals[name] for name in names if name in selected_signals]
        if used:
            families[family] = mean_product_flow_signals(used, index)
            members[family] = [name for name in names if name in selected_signals]
    return families, members


def corn_candidate_composites(families):
    definitions = {
        "selected_all_equal": list(families.keys()),
        "physical_only": ["physical"],
        "price_physical_equal": ["price", "physical"],
        "physical_fx_equal": ["physical", "fx_export"],
        "physical_weather_equal": ["physical", "weather"],
        "physical_macro_equal": ["physical", "macro"],
        "physical_ethanol_equal": ["physical", "ethanol"],
        "physical_ethanol_fx_equal": ["physical", "ethanol", "fx_export"],
        "physical_ethanol_weather_equal": ["physical", "ethanol", "weather"],
        "physical_ethanol_fx_weather_equal": ["physical", "ethanol", "fx_export", "weather"],
    }
    candidates, members = {}, {}
    index = next(iter(families.values())).index
    for candidate, family_names in definitions.items():
        used = [families[name] for name in family_names if name in families]
        if not used:
            continue
        if candidate != "selected_all_equal" and len(used) != len(family_names):
            continue
        candidates[candidate] = mean_product_flow_signals(used, index)
        members[candidate] = [name for name in family_names if name in families]
    return candidates, members


def select_corn_candidate_for_regime(signals, futures_pnl, regime_mask):
    index = futures_pnl.index
    signal_ic = corn_regime_signal_ic_table(signals, futures_pnl, regime_mask)
    selected_signals = {}
    for _, row in signal_ic.loc[signal_ic["passes_ic_threshold"]].iterrows():
        sign = 1.0 if row["train_ic"] >= 0.0 else -1.0
        selected_signals[row["signal"]] = clean_product_flow_signal(sign * signals[row["signal"]], index)
    if not selected_signals:
        return None, signal_ic, pd.DataFrame(), None

    families, _ = corn_candidate_families(selected_signals)
    candidates, candidate_members = corn_candidate_composites(families)
    if not candidates:
        return None, signal_ic, pd.DataFrame(), None

    target = futures_pnl["CORN"].shift(-1)
    regime = pd.Series(regime_mask, index=index).fillna(False).astype(bool)
    splits = {
        "train": pd.Series(index < pd.Timestamp(CORN_TRAIN_END), index=index),
        "validation": pd.Series((index >= pd.Timestamp(CORN_TRAIN_END)) & (index < pd.Timestamp(SPLIT_DATE)), index=index),
        "test": pd.Series(index >= pd.Timestamp(SPLIT_DATE), index=index),
    }
    rows = []
    for candidate, signal in candidates.items():
        candidate_signal = signal.clip(lower=0.0)
        row = {"candidate": candidate, "mode": "long_only", "families": ",".join(candidate_members[candidate])}
        for split_name, split_mask in splits.items():
            mask = split_mask & regime
            row[f"{split_name}_obs"] = int(mask.sum())
            row[f"{split_name}_ic"] = rank_ic_product_flow(candidate_signal, target, mask)
        eligible = (
            row["train_obs"] >= 120
            and row["validation_obs"] >= 40
            and pd.notnull(row["train_ic"])
            and pd.notnull(row["validation_ic"])
            and row["train_ic"] >= CORN_IC_THRESHOLD
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
    selected_signal = candidates[selected["candidate"]].clip(lower=0.0)
    return selected, signal_ic, table, selected_signal


def build_corn_vol_regime_signal(signals, feature_panels, futures_pnl):
    pieces, rows = [], []
    signal_ics, candidate_tables = {}, {}
    for regime_name, regime_mask in corn_vol_regime_masks(feature_panels, futures_pnl).items():
        selected, signal_ic, candidate_table, selected_signal = select_corn_candidate_for_regime(signals, futures_pnl, regime_mask)
        signal_ics[regime_name] = signal_ic
        candidate_tables[regime_name] = candidate_table
        if selected_signal is None:
            continue
        selected = selected.copy()
        selected["regime"] = regime_name
        rows.append(selected)
        pieces.append(selected_signal * pd.Series(regime_mask, index=futures_pnl.index).astype(float))
    if not pieces:
        return pd.Series(0.0, index=futures_pnl.index), pd.DataFrame(), signal_ics, candidate_tables
    return clean_product_flow_signal(sum(pieces), futures_pnl.index), pd.DataFrame(rows), signal_ics, candidate_tables


def corn_abundant_supply_masks(data, feature_panels, futures_pnl):
    index = futures_pnl.index
    price = data["adj1"]["CORN"].reindex(index).ffill()
    below_ma = price < price.rolling(252, min_periods=120).mean().shift(1)
    mom60_negative = feature_panels["CORN"]["mom_60"].reindex(index).fillna(0.0) < 0.0
    pnl = futures_pnl["CORN"].fillna(0.0)
    vol = pnl.rolling(60, min_periods=20).std().shift(1)
    lt_vol = vol.expanding(min_periods=252).median().shift(1)
    low_or_normal_vol = (vol <= 1.05 * lt_vol).fillna(False)
    low_vol = (vol < 0.80 * lt_vol).fillna(False)
    curve_weak = feature_panels["CORN"]["curve_spread"].reindex(index).fillna(0.0) <= 0.0
    return {
        "below_ma_and_negative_mom": (below_ma & mom60_negative).fillna(False),
        "below_ma_or_negative_mom": (below_ma | mom60_negative).fillna(False),
        "abundant_low_or_normal": (below_ma & mom60_negative & low_or_normal_vol).fillna(False),
        "abundant_low_vol": (below_ma & mom60_negative & low_vol).fillna(False),
        "abundant_curve_confirmed": (below_ma & mom60_negative & curve_weak).fillna(False),
    }


def scale_corn_positions_when(positions, condition, scale):
    out = positions.copy()
    mask = pd.Series(condition, index=positions.index).fillna(False).astype(bool)
    out.loc[mask, "CORN"] = float(scale) * out.loc[mask, "CORN"]
    return out.fillna(0.0)


def corn_candidate_key(candidate):
    return (
        f'{candidate["source_table"]}|{candidate["signal_set"]}|'
        f'{candidate["strategy"]}|{candidate["mode"]}|{candidate["note"]}'
    )


def build_corn_carry_forward_candidates(specs, combo_results, combo_positions, combo_signals,
                                        alpha_results, alpha_positions, alpha_context_signals):
    """Build the small set of hand-carried candidates used in the final guard test."""
    candidates = []
    for spec in specs:
        signal_set = spec["signal_set"]
        strategy = spec["strategy"]
        mode = spec["mode"]
        note = spec["note"]

        if spec["source_table"] == "alpha_combinations":
            row = combo_results.loc[
                (combo_results["signal_set"] == signal_set)
                & (combo_results["strategy"] == strategy)
                & (combo_results["alpha_combo"] == note)
                & (combo_results["mode"] == mode)
            ].iloc[0]
            signal = combo_signals[(signal_set, strategy, note)]
            positions = combo_positions[(signal_set, strategy, note, mode)]
        elif spec["source_table"] == "standalone_alpha_sleeves":
            row = alpha_results.loc[
                (alpha_results["signal_set"] == signal_set)
                & (alpha_results["strategy"] == strategy)
                & (alpha_results["alpha"] == note)
                & (alpha_results["mode"] == mode)
            ].iloc[0]
            signal = alpha_context_signals[(signal_set, note)]
            positions = alpha_positions[(signal_set, note, mode)]
        else:
            raise ValueError(f'Unknown source table: {spec["source_table"]}')

        candidates.append({
            "source_table": spec["source_table"],
            "selection_rule": spec["selection_rule"],
            "signal_set": signal_set,
            "strategy": strategy,
            "mode": mode,
            "note": note,
            "signal": signal,
            "positions": positions,
            "validation_sharpe_at_selection": row["validation_sharpe"],
        })
    return candidates


def make_corn_candidate(source_table, selection_rule, signal_set, strategy, mode, note, signal, positions):
    return {
        "source_table": source_table,
        "selection_rule": selection_rule,
        "signal_set": signal_set,
        "strategy": strategy,
        "mode": mode,
        "note": note,
        "signal": signal,
        "positions": positions,
    }


def summarize_corn_candidates(candidates, futures_pnl):
    rows = []
    for candidate in candidates:
        bt, _ = backtest_positions_product_flow(candidate["positions"], futures_pnl)
        row = {k: v for k, v in candidate.items() if k not in ["positions", "signal"]}
        row["candidate_key"] = corn_candidate_key(candidate)
        row.update(summarize_corn_backtest(bt))
        rows.append(row)
    return pd.DataFrame(rows)


def run_corn_supply_guard_tests(candidates, supply_masks, futures_pnl, trading_index, oos_start=SPLIT_DATE):
    rows, backtests, positions_by_key = [], {}, {}
    candidate_by_key = {corn_candidate_key(candidate): candidate for candidate in candidates}
    for candidate in candidates:
        base_key = corn_candidate_key(candidate)
        guard_tests = {"no_guard": candidate["positions"]}
        for mask_name, mask in supply_masks.items():
            guard_tests[f"{mask_name}_half"] = scale_corn_positions_when(candidate["positions"], mask, 0.50)
            guard_tests[f"{mask_name}_flat"] = scale_corn_positions_when(candidate["positions"], mask, 0.0)

        for guard_name, positions in guard_tests.items():
            bt, _ = backtest_positions_product_flow(positions, futures_pnl)
            row = {
                "candidate_key": base_key,
                "source_table": candidate["source_table"],
                "selection_rule": candidate["selection_rule"],
                "signal_set": candidate["signal_set"],
                "base_strategy": candidate["strategy"],
                "base_mode": candidate["mode"],
                "note": candidate["note"],
                "guard": guard_name,
                "strategy": f'{candidate["strategy"]}__{guard_name}',
            }
            row.update(summarize_corn_backtest(bt))
            if guard_name == "no_guard":
                row["guard_oos_pct"] = 0.0
            else:
                mask_name = guard_name.rsplit("_", 1)[0]
                mask = pd.Series(supply_masks[mask_name], index=trading_index)
                row["guard_oos_pct"] = float(mask.loc[trading_index >= pd.Timestamp(oos_start)].mean())
            rows.append(row)
            backtests[(base_key, guard_name)] = bt
            positions_by_key[(base_key, guard_name)] = positions

    results = pd.DataFrame(rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False])
    return results, backtests, positions_by_key, candidate_by_key


def compare_selected_corn_long_only(best_row, candidate_by_key, supply_masks, futures_pnl,
                                    trading_index, selected_positions_by_key,
                                    oos_start=SPLIT_DATE):
    base_candidate = candidate_by_key[best_row["candidate_key"]]
    long_only_candidate = make_corn_candidate(
        base_candidate["source_table"],
        "same_signal_as_selected_long_short",
        base_candidate["signal_set"],
        base_candidate["strategy"],
        "long_only",
        base_candidate["note"],
        base_candidate["signal"],
        corn_positions_from_signal(base_candidate["signal"], futures_pnl, mode="long_only"),
    )
    results, backtests, positions_by_key, _ = run_corn_supply_guard_tests(
        [long_only_candidate],
        supply_masks,
        futures_pnl,
        trading_index,
        oos_start=oos_start,
    )
    best_long_only = results.iloc[0]
    same_guard = results.loc[results["guard"] == best_row["guard"]].iloc[0]

    long_only_key = corn_candidate_key(long_only_candidate)
    selected_positions = selected_positions_by_key[(best_row["candidate_key"], best_row["guard"])]
    same_guard_positions = positions_by_key[(long_only_key, best_row["guard"])]
    diff = selected_positions["CORN"] - same_guard_positions["CORN"]
    exposure_check = pd.DataFrame([{
        "position_check": "selected_long_short_vs_same_signal_long_only",
        "pct_short_days": float((selected_positions["CORN"] < 0.0).mean()),
        "min_lot": float(selected_positions["CORN"].min()),
        "max_lot": float(selected_positions["CORN"].max()),
        "max_abs_diff_vs_long_only": float(diff.abs().max()),
    }])

    cols = [
        "source_table", "signal_set", "base_strategy", "base_mode", "note", "guard",
        "validation_sharpe", "oos_sharpe", "oos_pnl", "oos_dd",
        "full_sharpe", "full_dd", "turnover", "guard_oos_pct",
    ]
    comparison_rows = []
    for label, row in [
        ("selected_long_short", best_row),
        ("same_signal_long_only_same_guard", same_guard),
        ("same_signal_long_only_best_fixed_guard", best_long_only),
    ]:
        item = row[cols].to_dict()
        item["check"] = label
        comparison_rows.append(item)
    return results, pd.DataFrame(comparison_rows)[["check"] + cols], exposure_check, backtests, positions_by_key


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
