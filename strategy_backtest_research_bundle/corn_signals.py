"""Corn signal construction used by the corn notebook."""

from pathlib import Path

import numpy as np
import pandas as pd

from grain_backtest_core import load_train_set as _load_train_set
from research_config import COMMODITY_LOCATION_WEIGHTS
from shared_backtest import clean_signal, rolling_zscore


BUNDLE_DIR = Path(__file__).resolve().parent

PRICE_SIGNAL_NAMES = (
    "given_mom_20",
    "given_mom_60",
    "given_rev_5",
    "given_curve_spread",
    "given_curve_ratio",
    "given_price_family",
)
FUNDAMENTAL_CORE_SIGNAL_NAMES = (
    "given_inventory_pressure",
    "given_cgl_inventory_pressure",
    "given_cgl_crush_activity",
    "given_curve_tightness",
    "given_physical_family",
)
MACRO_SIGNAL_NAMES = (
    "external_fx_export_family",
    "external_macro_risk_family",
)
WEATHER_FAMILY_FEATURES = (
    "meteo_cdd_20d_growing",
    "meteo_hdd_20d_growing",
    "meteo_gdd_60d_growing",
    "meteo_heat_stress_20d_growing",
    "meteo_dryness_20d_growing",
    "meteo_dry_cdd_20d_growing",
    "meteo_precip_20d_planting",
    "meteo_dryness_20d_planting",
    "meteo_freeze_stress_5d_harvest",
)
DD_COLUMNS = ["validation_dd", "trade_dd", "oos_dd", "full_dd", "max_drawdown"]


def _bundle_path(path_like):
    path = Path(path_like)
    candidates = [path if path.is_absolute() else Path.cwd() / path]
    if not path.is_absolute():
        candidates.append(BUNDLE_DIR / path)
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(BUNDLE_DIR)
        except ValueError:
            continue
        return resolved
    raise ValueError("corn_signals only supports files inside this bundle")


def load_train_set(data_dir="train_set"):
    return _load_train_set(_bundle_path(data_dir))


def _read_indexed_numeric_csv(data_dir, filename):
    df = pd.read_csv(_bundle_path(Path(data_dir) / filename), index_col=0, parse_dates=True)
    return df.sort_index().apply(pd.to_numeric, errors="coerce")


def _load_external_yfinance(data_dir="train_set"):
    return _read_indexed_numeric_csv(data_dir, "external_yfinance.csv")


def _load_external_weather(data_dir="train_set"):
    df = pd.read_csv(_bundle_path(Path(data_dir) / "external_weather.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_external_eia_ethanol(data_dir="train_set"):
    return _read_indexed_numeric_csv(data_dir, "external_eia_ethanol.csv")


def _corn_given_signal_universe(feature_panels):
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
    return {name: clean_signal(signal, panel.index) for name, signal in signals.items()}


def _build_yfinance_families(trading_index, data_dir="train_set"):
    px = _load_external_yfinance(data_dir).reindex(trading_index).ffill().shift(1)
    families = {}
    if {"corn", "soybean", "wheat"}.issubset(px.columns):
        corn_soy = rolling_zscore((px["corn"] / px["soybean"]).pct_change(20, fill_method=None), 252, 60)
        corn_wheat = rolling_zscore((px["corn"] / px["wheat"]).pct_change(20, fill_method=None), 252, 60)
        soy_corn_mr = -rolling_zscore((px["soybean"] / px["corn"]).pct_change(20, fill_method=None), 252, 60)
        families["external_relative_grain_family"] = ((corn_soy + corn_wheat + soy_corn_mr) / 3.0).fillna(0.0)

    fx_parts = []
    if "usd_index" in px:
        fx_parts.append(-rolling_zscore(px["usd_index"].pct_change(20, fill_method=None), 252, 60))
    if "brl" in px:
        fx_parts.append(-rolling_zscore(px["brl"].pct_change(20, fill_method=None), 252, 60))
    if "cny" in px:
        fx_parts.append(-rolling_zscore(px["cny"].pct_change(20, fill_method=None), 252, 60))
    if fx_parts:
        families["external_fx_export_family"] = (sum(fx_parts) / float(len(fx_parts))).fillna(0.0)

    macro_parts = []
    if "crude" in px:
        macro_parts.append(rolling_zscore(px["crude"].pct_change(20, fill_method=None), 252, 60))
    if "equity" in px:
        macro_parts.append(rolling_zscore(px["equity"].pct_change(20, fill_method=None), 252, 60))
    if "usd_index" in px:
        macro_parts.append(-rolling_zscore(px["usd_index"].pct_change(60, fill_method=None), 252, 80))
    if macro_parts:
        families["external_macro_risk_family"] = (sum(macro_parts) / float(len(macro_parts))).fillna(0.0)
    return {name: clean_signal(signal, trading_index) for name, signal in families.items()}


def _build_ethanol_family(trading_index, data_dir="train_set"):
    ethanol = _load_external_eia_ethanol(data_dir)
    available = ethanol.copy()
    available.index = available.index + pd.DateOffset(days=7)
    aligned = available.reindex(trading_index).ffill().shift(1)

    features = pd.DataFrame(index=trading_index)
    features["ethanol_production_change_4w"] = rolling_zscore(aligned["ethanol_production"].diff(20), 156, 40)
    features["ethanol_stocks_change_4w"] = rolling_zscore(aligned["ethanol_stocks"].diff(20), 156, 40)
    ratio = aligned["ethanol_production"] / aligned["ethanol_stocks"].replace(0.0, np.nan)
    features["ethanol_prod_to_stocks"] = rolling_zscore(ratio, 156, 40)
    pressure = aligned["ethanol_production"].diff(20) - aligned["ethanol_stocks"].diff(20)
    features["ethanol_demand_pressure"] = rolling_zscore(pressure, 156, 40)

    family = (
        features["ethanol_production_change_4w"]
        + features["ethanol_prod_to_stocks"]
        + features["ethanol_demand_pressure"]
        - features["ethanol_stocks_change_4w"]
    ) / 4.0
    return clean_signal(family, trading_index)


def _weather_features(aligned):
    features = pd.DataFrame(index=aligned.index)
    if "tavg" in aligned:
        cdd = (aligned["tavg"] - 18.0).clip(lower=0.0)
        hdd = (18.0 - aligned["tavg"]).clip(lower=0.0)
        features["meteo_cdd_20d"] = rolling_zscore(cdd.rolling(20, min_periods=5).sum(), 252, 60)
        features["meteo_hdd_20d"] = rolling_zscore(hdd.rolling(20, min_periods=5).sum(), 252, 60)
    if {"tmin", "tmax"}.issubset(aligned.columns):
        temp_avg = (aligned["tmin"] + aligned["tmax"]) / 2.0
        gdd = (temp_avg.clip(upper=30.0) - 10.0).clip(lower=0.0)
        features["meteo_gdd_60d"] = rolling_zscore(gdd.rolling(60, min_periods=15).sum(), 252, 60)
    if "tmax" in aligned:
        heat_stress = (aligned["tmax"] - 32.0).clip(lower=0.0)
        features["meteo_heat_stress_20d"] = rolling_zscore(heat_stress.rolling(20, min_periods=5).sum(), 252, 60)
    if "prcp" in aligned:
        precip = aligned["prcp"].fillna(0.0)
        precip_20 = precip.rolling(20, min_periods=5).sum()
        features["meteo_precip_20d"] = rolling_zscore(precip_20, 252, 60)
        features["meteo_dryness_20d"] = -features["meteo_precip_20d"]
        if "meteo_cdd_20d" in features:
            features["meteo_dry_cdd_20d"] = (features["meteo_dryness_20d"] * features["meteo_cdd_20d"]).clip(-5.0, 5.0)
    if "tmin" in aligned:
        freeze_stress = (0.0 - aligned["tmin"]).clip(lower=0.0)
        features["meteo_freeze_stress_5d"] = rolling_zscore(freeze_stress.rolling(5, min_periods=3).sum(), 252, 60)

    month = features.index.month
    planting = pd.Series(month.isin([3, 4, 5]), index=features.index).astype(float)
    growing = pd.Series(month.isin([6, 7, 8]), index=features.index).astype(float)
    harvest = pd.Series(month.isin([9, 10, 11]), index=features.index).astype(float)
    seasonal_features = {}
    for column in features.columns:
        seasonal_features[column + "_planting"] = features[column] * planting
        seasonal_features[column + "_growing"] = features[column] * growing
        seasonal_features[column + "_harvest"] = features[column] * harvest
    return pd.concat([features, pd.DataFrame(seasonal_features, index=features.index)], axis=1).clip(-5.0, 5.0).fillna(0.0)


def _build_weather_family(trading_index, data_dir="train_set"):
    weather = _load_external_weather(data_dir)
    value_cols = [c for c in ["tavg", "tmin", "tmax", "prcp"] if c in weather.columns]
    frames = []
    for location, weight in COMMODITY_LOCATION_WEIGHTS["CORN"].items():
        sub = weather.loc[weather["location"] == location, ["date"] + value_cols].copy()
        if sub.empty:
            continue
        sub[value_cols] = sub[value_cols] * float(weight)
        frames.append(sub)
    if not frames:
        return pd.Series(0.0, index=trading_index)
    combined = pd.concat(frames, ignore_index=True).groupby("date")[value_cols].sum().sort_index()
    features = _weather_features(combined.reindex(trading_index).ffill().shift(1))
    existing = [c for c in WEATHER_FAMILY_FEATURES if c in features.columns]
    return clean_signal(features[existing].mean(axis=1), trading_index) if existing else pd.Series(0.0, index=trading_index)


def build_corn_product_flow_signal_universe(feature_panels, futures_pnl, data_dir="train_set"):
    """Return all corn signals used by the product-flow-aligned tests."""
    index = futures_pnl.index
    signals = _corn_given_signal_universe(feature_panels)
    signals.update(_build_yfinance_families(index, data_dir))
    signals["external_ethanol_family"] = _build_ethanol_family(index, data_dir)
    signals["external_weather_hdd_cdd_family"] = _build_weather_family(index, data_dir)
    return {name: clean_signal(signal, index) for name, signal in signals.items()}


def corn_signal_set_families(signals):
    """Families used for the requested Signal A / Signal B corn tests."""
    prices = {name: signals[name] for name in PRICE_SIGNAL_NAMES}
    if "external_relative_grain_family" in signals:
        prices["external_relative_grain_family"] = signals["external_relative_grain_family"]
    fundamentals_core = {name: signals[name] for name in FUNDAMENTAL_CORE_SIGNAL_NAMES}
    fundamentals_a = dict(fundamentals_core)
    fundamentals_a["external_ethanol_family"] = signals["external_ethanol_family"]
    fundamentals_a["external_weather_hdd_cdd_family"] = signals["external_weather_hdd_cdd_family"]
    macro = {name: signals[name] for name in MACRO_SIGNAL_NAMES}
    return {
        "A": {"prices": prices, "fundamentals": fundamentals_a, "macro": macro},
        "B": {"prices": prices, "fundamentals": fundamentals_core},
        "alpha": {
            "eia": {"external_ethanol_family": signals["external_ethanol_family"]},
            "macro": macro,
            "weather": {"external_weather_hdd_cdd_family": signals["external_weather_hdd_cdd_family"]},
        },
    }


def corn_dd_pct_table(table, columns=None, dd_capital_usd=10000.0):
    out = table.copy()
    if columns is not None:
        out = out[columns].copy()
    rename = {}
    for column in DD_COLUMNS:
        if column in out.columns:
            out[column] = 100.0 * out[column] / float(dd_capital_usd)
            rename[column] = "max_dd_pct" if column == "max_drawdown" else f"{column}_pct"
    return out.rename(columns=rename)
