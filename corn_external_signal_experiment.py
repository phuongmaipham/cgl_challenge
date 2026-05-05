"""Corn lower-overfit strategy experiments with EIA ethanol.

This mirrors the soybean research style but keeps corn-specific economics:
provided physical/Cargill signals, EIA ethanol production/stocks, crop-belt
weather, export FX pressure, grain relative value, and macro/risk.
"""

from __future__ import print_function

import numpy as np
import pandas as pd

from eia_ethanol_experiment import build_ethanol_feature_panel, fetch_eia_ethanol
from grain_futures_strategy import (
    backtest_positions,
    backtest_positions_with_costs,
    build_feature_panels,
    load_train_set,
    rolling_zscore,
    split_performance,
)
from meteostat_experiment import fetch_meteostat_weather, build_meteostat_feature_panels


COMMODITY = "CORN"
TRAIN_END = "2016-01-01"
TEST_START = "2018-01-01"
YF_TICKERS = {
    "corn": "ZC=F",
    "soybean": "ZS=F",
    "wheat": "ZW=F",
    "usd_index": "DX-Y.NYB",
    "brl": "BRL=X",
    "cny": "CNY=X",
    "crude": "CL=F",
    "equity": "SPY",
}


def _tanh(series, divisor=2.0):
    return pd.Series(np.tanh(series.astype(float) / float(divisor)), index=series.index)


def _smooth_threshold(series, mode="long_only"):
    signal = _tanh(series).ewm(halflife=2.0, adjust=False, min_periods=1).mean()
    signal[signal.abs() < 0.05] = 0.0
    if mode == "long_only":
        signal = signal.clip(lower=0.0)
    elif mode == "short_only":
        signal = signal.clip(upper=0.0)
    elif mode != "long_short":
        raise ValueError("Unknown mode: {}".format(mode))
    return signal


def _download_yfinance(start, end):
    import yfinance as yf

    tickers = list(YF_TICKERS.values())
    raw = yf.download(
        tickers,
        start=str(pd.Timestamp(start).date()),
        end=str((pd.Timestamp(end) + pd.DateOffset(days=7)).date()),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"]
        else:
            close = raw.xs(raw.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        close = raw.to_frame(tickers[0]) if isinstance(raw, pd.Series) else raw
    close = close.rename(columns={ticker: name for name, ticker in YF_TICKERS.items()})
    close = close[[name for name in YF_TICKERS if name in close.columns]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def _given_components(feature_panels):
    corn = feature_panels[COMMODITY]
    inventory_pressure = (
        -corn["public_inventory_change"] - corn["receipts_change"] - corn["cgl_inventory_change"]
    ) / 3.0
    cgl_crush_activity = (corn["crush_surprise"] + corn["crush_utilization"]) / 2.0
    trend = (corn["mom_20"] + corn["mom_60"] + corn["curve_spread"] + corn["cot_pm_oi_level"]) / 4.0
    curve_tightness = (corn["curve_spread"] + corn["curve_ratio"]) / 2.0
    price_family = (corn["mom_20"] + corn["mom_60"] + corn["rev_5"]) / 3.0
    physical_family = (inventory_pressure + curve_tightness + 0.25 * cgl_crush_activity) / 2.25
    conservative = 0.40 * physical_family + 0.30 * trend + 0.30 * price_family
    return {
        "given_price_family": price_family.fillna(0.0),
        "given_physical_family": physical_family.fillna(0.0),
        "given_trend": trend.fillna(0.0),
        "given_inventory_pressure": inventory_pressure.fillna(0.0),
        "given_cgl_crush_activity": cgl_crush_activity.fillna(0.0),
        "given_curve_tightness": curve_tightness.fillna(0.0),
        "given_conservative_blend": conservative.fillna(0.0),
    }


def _ethanol_family(ethanol_features):
    parts = []
    if "ethanol_production_change_4w" in ethanol_features:
        parts.append(ethanol_features["ethanol_production_change_4w"])
    if "ethanol_prod_to_stocks" in ethanol_features:
        parts.append(ethanol_features["ethanol_prod_to_stocks"])
    if "ethanol_demand_pressure" in ethanol_features:
        parts.append(ethanol_features["ethanol_demand_pressure"])
    if "ethanol_stocks_change_4w" in ethanol_features:
        parts.append(-ethanol_features["ethanol_stocks_change_4w"])
    if not parts:
        return {}
    return {"external_ethanol_family": (sum(parts) / float(len(parts))).clip(lower=-5.0, upper=5.0).fillna(0.0)}


def _external_yfinance_families(prices, trading_index):
    px = prices.reindex(trading_index).ffill().shift(1)
    families = {}
    if {"corn", "soybean", "wheat"}.issubset(px.columns):
        corn_soy = rolling_zscore((px["corn"] / px["soybean"]).pct_change(20, fill_method=None), 252, 60)
        corn_wheat = rolling_zscore((px["corn"] / px["wheat"]).pct_change(20, fill_method=None), 252, 60)
        soy_corn_mr = -rolling_zscore((px["soybean"] / px["corn"]).pct_change(20, fill_method=None), 252, 60)
        families["external_relative_grain_family"] = ((corn_soy + corn_wheat + soy_corn_mr) / 3.0).fillna(0.0)

    fx_parts = []
    if "usd_index" in px.columns:
        fx_parts.append(-rolling_zscore(px["usd_index"].pct_change(20, fill_method=None), 252, 60))
    if "brl" in px.columns:
        fx_parts.append(-rolling_zscore(px["brl"].pct_change(20, fill_method=None), 252, 60))
    if "cny" in px.columns:
        fx_parts.append(-rolling_zscore(px["cny"].pct_change(20, fill_method=None), 252, 60))
    if fx_parts:
        families["external_fx_export_family"] = (sum(fx_parts) / float(len(fx_parts))).fillna(0.0)

    macro_parts = []
    if "crude" in px.columns:
        macro_parts.append(rolling_zscore(px["crude"].pct_change(20, fill_method=None), 252, 60))
    if "equity" in px.columns:
        macro_parts.append(rolling_zscore(px["equity"].pct_change(20, fill_method=None), 252, 60))
    if "usd_index" in px.columns:
        macro_parts.append(-rolling_zscore(px["usd_index"].pct_change(60, fill_method=None), 252, 80))
    if macro_parts:
        families["external_macro_risk_family"] = (sum(macro_parts) / float(len(macro_parts))).fillna(0.0)

    out = {}
    for name, values in families.items():
        cleaned = values.clip(lower=-5.0, upper=5.0).replace([np.inf, -np.inf], np.nan)
        if cleaned.notna().sum() > 20 and cleaned.fillna(0.0).abs().sum() > 0.0:
            out[name] = cleaned.fillna(0.0)
    return out


def _weather_family(weather, trading_index):
    panels = build_meteostat_feature_panels(weather, trading_index, mode="commodity_seasonal")
    corn = panels[COMMODITY].reindex(trading_index).fillna(0.0)
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
    existing = [col for col in candidates if col in corn.columns]
    if not existing:
        return {}
    return {"external_weather_hdd_cdd_family": corn[existing].mean(axis=1).clip(lower=-5.0, upper=5.0).fillna(0.0)}


def _positions_from_signal(signal, futures_pnl, mode="long_only"):
    cleaned = _smooth_threshold(signal.reindex(futures_pnl.index).fillna(0.0), mode=mode)
    pnl = futures_pnl[[COMMODITY]]
    asset_vol = pnl[COMMODITY].rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    pos = cleaned.reindex(pnl.index).fillna(0.0) * (75.0 / asset_vol)
    out = pd.DataFrame(0.0, index=pnl.index, columns=[COMMODITY])
    out[COMMODITY] = pos.clip(lower=-0.50, upper=0.50).fillna(0.0)
    return out


def _metrics(bt):
    table = split_performance(bt, TEST_START)
    train_val = split_performance(bt.loc[bt.index < pd.Timestamp(TEST_START)], TRAIN_END)
    if table.empty or "sharpe" not in table.index:
        return {
            "train_sharpe": np.nan,
            "validation_sharpe": np.nan,
            "validation_max_drawdown": np.nan,
            "test_sharpe": np.nan,
            "test_pnl": 0.0,
            "test_max_drawdown": np.nan,
            "full_sharpe": np.nan,
            "full_pnl": 0.0,
            "max_drawdown": np.nan,
            "turnover": 0.0,
            "avg_gross_exposure": 0.0,
        }
    return {
        "train_sharpe": train_val.loc["sharpe", "in_sample"] if "sharpe" in train_val.index else np.nan,
        "validation_sharpe": train_val.loc["sharpe", "out_of_sample"] if "sharpe" in train_val.index else np.nan,
        "validation_max_drawdown": train_val.loc["max_drawdown", "out_of_sample"] if "max_drawdown" in train_val.index else np.nan,
        "test_sharpe": table.loc["sharpe", "out_of_sample"],
        "test_pnl": table.loc["total_pnl", "out_of_sample"],
        "test_max_drawdown": table.loc["max_drawdown", "out_of_sample"],
        "full_sharpe": table.loc["sharpe", "full_period"],
        "full_pnl": table.loc["total_pnl", "full_period"],
        "max_drawdown": table.loc["max_drawdown", "full_period"],
        "turnover": table.loc["avg_daily_turnover", "full_period"],
        "avg_gross_exposure": table.loc["avg_gross_exposure", "full_period"],
    }


def _evaluate_signal(name, family, signal, futures_pnl, mode="long_only"):
    positions = _positions_from_signal(signal, futures_pnl, mode=mode)
    rows = []
    backtests = {}
    for cost_adjusted in [False, True]:
        if cost_adjusted:
            bt, _ = backtest_positions_with_costs(positions, futures_pnl[[COMMODITY]], 8.75, 0.05)
        else:
            bt, _ = backtest_positions(positions, futures_pnl[[COMMODITY]], 0.0)
        row = {"experiment": name, "family": family, "mode": mode, "cost_adjusted": cost_adjusted}
        row.update(_metrics(bt))
        rows.append(row)
        backtests[name + ("_cost_adjusted" if cost_adjusted else "_zero_cost")] = bt
    return rows, backtests, positions


def _zero_cost_metrics(signal, futures_pnl, mode="long_only"):
    positions = _positions_from_signal(signal, futures_pnl, mode=mode)
    bt, _ = backtest_positions(positions, futures_pnl[[COMMODITY]], 0.0)
    return _metrics(bt)


def _mean_available(items):
    values = [series for series in items if series is not None]
    if not values:
        raise ValueError("No series available")
    return sum(values) / float(len(values))


def _weighted_sum(families, weights):
    total = 0.0
    used = 0.0
    for name, weight in weights.items():
        if name in families and float(weight) != 0.0:
            total = total + float(weight) * families[name]
            used += float(weight)
    if used == 0.0:
        raise ValueError("No available families")
    return total / used


def _build_regime_frames(given, futures_pnl):
    corn_pnl = futures_pnl[COMMODITY].fillna(0.0)
    vol = corn_pnl.rolling(60, min_periods=20).std().shift(1)
    lt_vol = vol.expanding(min_periods=252).median().shift(1)
    high_threshold = vol.expanding(min_periods=252).quantile(0.75).shift(1)
    low_vol = (vol < 0.80 * lt_vol).fillna(False)
    high_vol = ((vol > 1.20 * lt_vol) | (vol > high_threshold)).fillna(False)
    trend_positive = (given["given_trend"] > 0.0).reindex(futures_pnl.index).fillna(False)
    return {"low_vol": low_vol.astype(float), "high_vol": high_vol.astype(float), "trend_positive": trend_positive.astype(float)}


def _build_candidate_signals(given, external, futures_pnl):
    families = dict(given)
    families.update(external)
    regimes = _build_regime_frames(given, futures_pnl)
    source = _mean_available([
        given["given_physical_family"],
        external.get("external_ethanol_family"),
        external.get("external_fx_export_family"),
        external.get("external_weather_hdd_cdd_family"),
    ])
    trend_signal = _mean_available([given["given_trend"], external.get("external_macro_risk_family")])
    high_vol = regimes["high_vol"]
    trend_w = (0.20 + 0.40 * high_vol).clip(0.20, 0.60)
    signals = {
        "given_physical_family": given["given_physical_family"],
        "given_conservative_blend": given["given_conservative_blend"],
        "external_ethanol_family": external.get("external_ethanol_family", pd.Series(0.0, index=futures_pnl.index)),
        "requirement_given_90_ethanol_10": _weighted_sum(
            families,
            {
                "given_conservative_blend": 0.90,
                "external_ethanol_family": 0.10,
            },
        ),
        "requirement_given_80_ethanol_20": _weighted_sum(
            families,
            {
                "given_conservative_blend": 0.80,
                "external_ethanol_family": 0.20,
            },
        ),
        "requirement_given_80_ethanol_10_fx_10": _weighted_sum(
            families,
            {
                "given_conservative_blend": 0.80,
                "external_ethanol_family": 0.10,
                "external_fx_export_family": 0.10,
            },
        ),
        "requirement_physical_60_trend_25_ethanol_15": _weighted_sum(
            families,
            {
                "given_physical_family": 0.60,
                "given_trend": 0.25,
                "external_ethanol_family": 0.15,
            },
        ),
        "fundamental_equal_physical_ethanol_fx_weather": _mean_available([
            given["given_physical_family"],
            external.get("external_ethanol_family"),
            external.get("external_fx_export_family"),
            external.get("external_weather_hdd_cdd_family"),
        ]),
        "fundamental_40_physical_30_ethanol_15_fx_15_weather": _weighted_sum(
            families,
            {
                "given_physical_family": 0.40,
                "external_ethanol_family": 0.30,
                "external_fx_export_family": 0.15,
                "external_weather_hdd_cdd_family": 0.15,
            },
        ),
        "fundamental_50_physical_30_ethanol_20_fx": _weighted_sum(
            families,
            {
                "given_physical_family": 0.50,
                "external_ethanol_family": 0.30,
                "external_fx_export_family": 0.20,
            },
        ),
        "regime_hybrid_vol_trend_weight": ((1.0 - trend_w) * source + trend_w * trend_signal),
        "low_vol_physical_weather_else_ethanol_fx": (
            regimes["low_vol"] * _mean_available([given["given_physical_family"], external.get("external_weather_hdd_cdd_family")])
            + (1.0 - regimes["low_vol"]) * _mean_available([external.get("external_ethanol_family"), external.get("external_fx_export_family"), given["given_trend"]])
        ),
    }
    if "external_ethanol_family" in external:
        high_vol_delever = 1.0 - 0.35 * regimes["high_vol"]
        signals["requirement_drawdown_guard_given_ethanol"] = (
            _weighted_sum(
                families,
                {
                    "given_conservative_blend": 0.80,
                    "external_ethanol_family": 0.10,
                    "external_weather_hdd_cdd_family": 0.10,
                },
            )
            * high_vol_delever
        )
    clean = {name: signal.clip(lower=-5.0, upper=5.0).fillna(0.0) for name, signal in signals.items()}
    return clean, regimes


def _select_pre2018(candidates, futures_pnl, require_ethanol=False, modes=("long_only", "long_short")):
    rows = []
    for name, signal in candidates.items():
        for mode in modes:
            metrics = _zero_cost_metrics(signal, futures_pnl, mode=mode)
            requirement_ok = (not require_ethanol) or ("ethanol" in name)
            eligible = (
                requirement_ok
                and pd.notnull(metrics.get("train_sharpe"))
                and pd.notnull(metrics.get("validation_sharpe"))
                and metrics["train_sharpe"] > 0.0
                and metrics["validation_sharpe"] >= 0.50
            )
            score = (
                metrics["validation_sharpe"] + 0.25 * metrics["train_sharpe"] + 0.002 * metrics["validation_max_drawdown"]
                if eligible
                else -np.inf
            )
            row = {"candidate": name, "mode": mode, "eligible": bool(eligible), "score": score}
            row.update(metrics)
            rows.append(row)
    table = pd.DataFrame(rows)
    if require_ethanol:
        selection_pool = table.loc[table["candidate"].str.contains("ethanol")].copy()
    else:
        selection_pool = table.copy()
    eligible = selection_pool.loc[selection_pool["eligible"]].copy()
    if eligible.empty:
        selected = selection_pool.sort_values(
            ["validation_sharpe", "validation_max_drawdown", "test_max_drawdown"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        selected = eligible.sort_values(["score", "validation_max_drawdown"], ascending=[False, False]).iloc[0]
    return selected["candidate"], selected["mode"], table


def run_corn_signal_experiment(data_dir="train_set", include_weather=True, include_eia=True):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    given = _given_components(feature_panels)
    external = {}
    errors = []
    prices = pd.DataFrame()
    weather = pd.DataFrame()
    ethanol = pd.DataFrame()

    try:
        prices = _download_yfinance(futures_pnl.index.min(), futures_pnl.index.max())
        external.update(_external_yfinance_families(prices, futures_pnl.index))
    except Exception as exc:
        errors.append("yfinance: {}".format(exc))

    if include_weather:
        try:
            weather = fetch_meteostat_weather(futures_pnl.index.min(), futures_pnl.index.max())
            external.update(_weather_family(weather, futures_pnl.index))
        except Exception as exc:
            errors.append("meteostat: {}".format(exc))

    if include_eia:
        try:
            ethanol = fetch_eia_ethanol()
            ethanol_features = build_ethanol_feature_panel(ethanol, futures_pnl.index)
            external.update(_ethanol_family(ethanol_features))
        except Exception as exc:
            ethanol_features = pd.DataFrame()
            errors.append("eia_ethanol: {}".format(exc))
    else:
        ethanol_features = pd.DataFrame()

    candidates, regime_frames = _build_candidate_signals(given, external, futures_pnl)
    selected, selected_mode, selection = _select_pre2018(candidates, futures_pnl, require_ethanol=True)
    rows = []
    backtests = {}
    positions = {}
    for name, signal in candidates.items():
        for mode in ["long_only", "long_short"]:
            new_rows, new_bt, pos = _evaluate_signal("corn_candidate_" + name, name, signal, futures_pnl, mode=mode)
            rows.extend(new_rows)
            backtests.update(new_bt)
            positions[name + "_" + mode] = pos
    new_rows, new_bt, pos = _evaluate_signal(
        "corn_pre2018_selection_" + selected,
        "pre2018_selected_corn_strategy",
        candidates[selected],
        futures_pnl,
        mode=selected_mode,
    )
    rows.extend(new_rows)
    backtests.update(new_bt)
    positions["selected"] = pos
    results = pd.DataFrame(rows).sort_values(["cost_adjusted", "test_sharpe"], ascending=[True, False]).reset_index(drop=True)
    return {
        "results": results,
        "selection": selection,
        "selected_candidate": selected,
        "selected_mode": selected_mode,
        "given_signals": given,
        "external_signals": external,
        "regime_frames": regime_frames,
        "positions": positions,
        "backtests": backtests,
        "yfinance_prices": prices,
        "weather": weather,
        "ethanol": ethanol,
        "ethanol_features": ethanol_features,
        "errors": errors,
    }


if __name__ == "__main__":
    out = run_corn_signal_experiment()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("Errors:", out["errors"])
    print("Selected:", out["selected_candidate"], out["selected_mode"])
    print(out["selection"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
