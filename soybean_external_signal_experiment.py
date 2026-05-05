"""Soybean given-data vs external-signal experiments.

The module keeps two research tracks separate:
1. given-data-only soybean signals from the provided training files;
2. optional external soybean signals from yfinance and Meteostat.

External data is grouped into economic families and tested both individually
and as equal-family-weight composites. The reported test period is 2018-2020.
"""

from __future__ import print_function

import numpy as np
import pandas as pd

from grain_futures_strategy import (
    backtest_positions,
    backtest_positions_with_costs,
    build_feature_panels,
    load_train_set,
    rolling_zscore,
    split_performance,
)
from meteostat_experiment import fetch_meteostat_weather, build_meteostat_feature_panels
from soybean_no_fit_experiment import (
    COMMODITY,
    TEST_START,
    build_soybean_signal,
    signal_to_soybean_positions,
)


YF_TICKERS = {
    "soybean": "ZS=F",
    "soymeal": "ZM=F",
    "soyoil": "ZL=F",
    "corn": "ZC=F",
    "wheat": "ZW=F",
    "usd_index": "DX-Y.NYB",
    "brl": "BRL=X",
    "cny": "CNY=X",
    "crude": "CL=F",
    "equity": "SPY",
}
TRAIN_END = "2016-01-01"
VALIDATION_END = "2018-01-01"


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
    soy = feature_panels[COMMODITY]
    inventory_pressure = (
        -soy["public_inventory_change"] - soy["receipts_change"] - soy["cgl_inventory_change"]
    ) / 3.0
    crush_pressure = (soy["crush_surprise"] + soy["crush_utilization"]) / 2.0
    trend = (soy["mom_20"] + soy["mom_60"] + soy["curve_spread"] + soy["cot_pm_oi_level"]) / 4.0
    curve_tightness = (soy["curve_spread"] + soy["curve_ratio"]) / 2.0
    price_family = (soy["mom_20"] + soy["mom_60"] + soy["rev_5"]) / 3.0
    physical_family = (inventory_pressure + crush_pressure + curve_tightness) / 3.0
    conservative = build_soybean_signal(feature_panels, "soy_conservative_long_blend")
    return {
        "given_price_family": price_family.fillna(0.0),
        "given_physical_family": physical_family.fillna(0.0),
        "given_trend": trend.fillna(0.0),
        "given_inventory_pressure": inventory_pressure.fillna(0.0),
        "given_crush_pressure": crush_pressure.fillna(0.0),
        "given_conservative_long_blend": conservative.fillna(0.0),
        "given_equal_price_physical": ((price_family + physical_family) / 2.0).fillna(0.0),
    }


def _external_yfinance_families(prices, trading_index):
    px = prices.reindex(trading_index).ffill().shift(1)
    families = {}

    if {"soybean", "soymeal", "soyoil"}.issubset(px.columns):
        soy_dollars = px["soybean"] / 100.0
        crush = 0.022 * px["soymeal"] + 0.11 * px["soyoil"] - soy_dollars
        crush_mom = rolling_zscore(crush.diff(20), 252, 60)
        meal_lead = rolling_zscore(
            px["soymeal"].pct_change(20, fill_method=None) - px["soybean"].pct_change(20, fill_method=None),
            252,
            60,
        )
        oil_lead = rolling_zscore(
            px["soyoil"].pct_change(20, fill_method=None) - px["soybean"].pct_change(20, fill_method=None),
            252,
            60,
        )
        families["external_crush_family"] = ((crush_mom + meal_lead + oil_lead) / 3.0).fillna(0.0)

    if {"soybean", "corn", "wheat"}.issubset(px.columns):
        soy_corn = rolling_zscore((px["soybean"] / px["corn"]).pct_change(20, fill_method=None), 252, 60)
        soy_wheat = rolling_zscore((px["soybean"] / px["wheat"]).pct_change(20, fill_method=None), 252, 60)
        corn_soy_mr = -rolling_zscore((px["corn"] / px["soybean"]).pct_change(20, fill_method=None), 252, 60)
        families["external_relative_grain_family"] = ((soy_corn + soy_wheat + corn_soy_mr) / 3.0).fillna(0.0)

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


def _external_weather_family(weather, trading_index):
    panels = build_meteostat_feature_panels(weather, trading_index, mode="commodity_seasonal")
    soy = panels[COMMODITY].reindex(trading_index).fillna(0.0)
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
    existing = [col for col in candidates if col in soy.columns]
    if not existing:
        return {}
    family = soy[existing].mean(axis=1).fillna(0.0)
    return {"external_weather_hdd_cdd_family": family.clip(lower=-5.0, upper=5.0)}


def _positions_from_signal(signal, futures_pnl, mode="long_only"):
    cleaned = _smooth_threshold(signal.reindex(futures_pnl.index).fillna(0.0), mode=mode)
    return signal_to_soybean_positions(cleaned, futures_pnl, target_daily_pnl_vol=75.0, max_lot=0.50, mode="long_short")


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
        "validation_max_drawdown": train_val.loc["max_drawdown", "out_of_sample"]
        if "max_drawdown" in train_val.index
        else np.nan,
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
        row = {
            "experiment": name,
            "family": family,
            "mode": mode,
            "cost_adjusted": cost_adjusted,
        }
        row.update(_metrics(bt))
        rows.append(row)
        backtests[name + ("_cost_adjusted" if cost_adjusted else "_zero_cost")] = bt
    return rows, backtests, positions


def _zero_cost_metric_for_signal(signal, futures_pnl, mode="long_only"):
    positions = _positions_from_signal(signal, futures_pnl, mode=mode)
    bt, _ = backtest_positions(positions, futures_pnl[[COMMODITY]], 0.0)
    return _metrics(bt)


def _passes_pre2018(metrics):
    return (
        pd.notnull(metrics.get("train_sharpe"))
        and pd.notnull(metrics.get("validation_sharpe"))
        and metrics["train_sharpe"] > 0.0
        and metrics["validation_sharpe"] > 0.0
        and pd.notnull(metrics.get("validation_max_drawdown"))
        and metrics["validation_max_drawdown"] > -250.0
    )


def _build_overfit_controlled_signal(given, external, futures_pnl):
    """Select family membership using only train/validation performance."""
    mandatory = {
        "given_physical_family": given["given_physical_family"],
    }
    optional = {
        name: external[name]
        for name in [
            "external_crush_family",
            "external_fx_export_family",
            "external_weather_hdd_cdd_family",
        ]
        if name in external
    }
    selection_rows = []
    selected = dict(mandatory)

    for name, signal in optional.items():
        metrics = _zero_cost_metric_for_signal(signal, futures_pnl, mode="long_only")
        keep = _passes_pre2018(metrics)
        row = {"family": name, "selected": bool(keep)}
        row.update(metrics)
        selection_rows.append(row)
        if keep:
            selected[name] = signal

    combined = sum(selected.values()) / float(len(selected))
    return combined, selected, pd.DataFrame(selection_rows)


def _weighted_sum(families, weights):
    total = 0.0
    used_weight = 0.0
    for name, weight in weights.items():
        if name in families and float(weight) != 0.0:
            total = total + float(weight) * families[name]
            used_weight += float(weight)
    if used_weight == 0.0:
        raise ValueError("No available families in weights: {}".format(weights))
    return total / used_weight


def _build_weight_relaxed_pre2018_signal(given, external, futures_pnl):
    """Choose fixed family weights using only pre-2018 validation data."""
    available = {
        "given_physical_family": given["given_physical_family"],
    }
    for name in [
        "external_crush_family",
        "external_fx_export_family",
        "external_weather_hdd_cdd_family",
    ]:
        if name in external:
            available[name] = external[name]

    candidates = [
        (
            "physical_only",
            {
                "given_physical_family": 1.00,
            },
        ),
        (
            "physical_70_fx_30",
            {
                "given_physical_family": 0.70,
                "external_fx_export_family": 0.30,
            },
        ),
        (
            "physical_50_fx_50",
            {
                "given_physical_family": 0.50,
                "external_fx_export_family": 0.50,
            },
        ),
        (
            "physical_30_fx_70",
            {
                "given_physical_family": 0.30,
                "external_fx_export_family": 0.70,
            },
        ),
        (
            "physical_70_weather_30",
            {
                "given_physical_family": 0.70,
                "external_weather_hdd_cdd_family": 0.30,
            },
        ),
        (
            "physical_70_extcrush_30",
            {
                "given_physical_family": 0.70,
                "external_crush_family": 0.30,
            },
        ),
        (
            "physical_50_fx_25_weather_25",
            {
                "given_physical_family": 0.50,
                "external_fx_export_family": 0.25,
                "external_weather_hdd_cdd_family": 0.25,
            },
        ),
        (
            "physical_50_fx_25_extcrush_25",
            {
                "given_physical_family": 0.50,
                "external_fx_export_family": 0.25,
                "external_crush_family": 0.25,
            },
        ),
        (
            "physical_40_fx_20_extcrush_20_weather_20",
            {
                "given_physical_family": 0.40,
                "external_fx_export_family": 0.20,
                "external_crush_family": 0.20,
                "external_weather_hdd_cdd_family": 0.20,
            },
        ),
    ]

    rows = []
    candidate_signals = {}
    for name, weights in candidates:
        usable = {family: weight for family, weight in weights.items() if family in available}
        if "given_physical_family" not in usable:
            continue
        signal = _weighted_sum(available, usable)
        metrics = _zero_cost_metric_for_signal(signal, futures_pnl, mode="long_only")
        score = -np.inf
        if (
            pd.notnull(metrics.get("train_sharpe"))
            and pd.notnull(metrics.get("validation_sharpe"))
            and metrics["train_sharpe"] > 0.0
            and metrics["validation_sharpe"] > 0.0
        ):
            score = (
                metrics["validation_sharpe"]
                + 0.25 * metrics["train_sharpe"]
                + 0.001 * metrics["validation_max_drawdown"]
            )
        row = {
            "candidate": name,
            "eligible": bool(np.isfinite(score)),
            "score": score,
            "weights": str(usable),
        }
        row.update(metrics)
        rows.append(row)
        candidate_signals[name] = signal

    table = pd.DataFrame(rows)
    eligible = table.loc[table["eligible"]].copy()
    drawdown_eligible = table.loc[
        table["eligible"] & (table["validation_sharpe"] >= 0.50) & (table["train_sharpe"] > 0.0)
    ].copy()

    if eligible.empty:
        selected_name = "physical_only"
    else:
        selected_name = eligible.sort_values(["score", "validation_max_drawdown"], ascending=[False, False]).iloc[0][
            "candidate"
        ]

    if drawdown_eligible.empty:
        drawdown_selected_name = selected_name
    else:
        drawdown_selected_name = drawdown_eligible.sort_values(
            ["validation_max_drawdown", "validation_sharpe"],
            ascending=[False, False],
        ).iloc[0]["candidate"]

    return (
        candidate_signals[selected_name],
        selected_name,
        candidate_signals[drawdown_selected_name],
        drawdown_selected_name,
        table,
        candidate_signals,
    )


def _soybean_regime_frames(given, external, futures_pnl):
    soy_pnl = futures_pnl[COMMODITY].fillna(0.0)
    vol = soy_pnl.rolling(60, min_periods=20).std().shift(1)
    lt_vol = vol.expanding(min_periods=252).median().shift(1)
    high_threshold = vol.expanding(min_periods=252).quantile(0.75).shift(1)
    low_vol = (vol < 0.80 * lt_vol).fillna(False)
    high_vol = ((vol > 1.20 * lt_vol) | (vol > high_threshold)).fillna(False)
    trend_positive = (given["given_trend"] > 0.0).reindex(futures_pnl.index).fillna(False)
    return {
        "vol": vol,
        "lt_vol": lt_vol,
        "low_vol": low_vol.astype(float),
        "high_vol": high_vol.astype(float),
        "trend_positive": trend_positive.astype(float),
    }


def _mean_available(items):
    values = [series for series in items if series is not None]
    if not values:
        raise ValueError("No series available to average")
    return sum(values) / float(len(values))


def _build_regime_shift_signals(given, external, futures_pnl):
    """Build observable regime-shift candidates with fixed equal-weight sleeves."""
    regimes = _soybean_regime_frames(given, external, futures_pnl)
    source = _mean_available(
        [
            given["given_physical_family"],
            external.get("external_fx_export_family"),
            external.get("external_crush_family"),
            external.get("external_weather_hdd_cdd_family"),
        ]
    )
    physical_weather = _mean_available(
        [
            given["given_physical_family"],
            external.get("external_weather_hdd_cdd_family"),
        ]
    )
    trend_signal = _mean_available(
        [
            given["given_trend"],
            external.get("external_macro_risk_family"),
        ]
    )
    high_vol_signal = _mean_available(
        [
            given["given_trend"],
            external.get("external_fx_export_family"),
            external.get("external_crush_family"),
        ]
    )
    low_vol_signal = _mean_available(
        [
            given["given_physical_family"],
            external.get("external_weather_hdd_cdd_family"),
        ]
    )

    high_vol = regimes["high_vol"]
    low_vol = regimes["low_vol"]
    trend_positive = regimes["trend_positive"]
    trend_w_vol = (0.20 + 0.40 * high_vol).clip(0.20, 0.60)
    trend_w_confirmed = (0.20 + 0.40 * ((high_vol > 0) | (trend_positive > 0)).astype(float)).clip(0.20, 0.60)

    candidates = {
        "regime_hybrid_vol_trend_weight": (1.0 - trend_w_vol) * source + trend_w_vol * trend_signal,
        "regime_hybrid_trend_confirmed_weight": (1.0 - trend_w_confirmed) * source
        + trend_w_confirmed * trend_signal,
        "regime_low_physical_high_trend_switch": high_vol * high_vol_signal + (1.0 - high_vol) * low_vol_signal,
        "regime_low_vol_physical_else_balanced": low_vol * low_vol_signal + (1.0 - low_vol) * source,
        "regime_high_vol_fx_trend_else_physical_weather": high_vol * high_vol_signal
        + (1.0 - high_vol) * physical_weather,
    }
    return {name: signal.clip(lower=-5.0, upper=5.0).fillna(0.0) for name, signal in candidates.items()}, regimes


def run_soybean_given_only_experiment(data_dir="train_set"):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    signals = _given_components(feature_panels)
    rows = []
    backtests = {}
    positions = {}
    for name, signal in signals.items():
        mode = "long_only" if name != "given_equal_price_physical" else "long_short"
        new_rows, new_bt, pos = _evaluate_signal(name, "given", signal, futures_pnl, mode=mode)
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions[name] = pos

    results = pd.DataFrame(rows).sort_values(["cost_adjusted", "test_sharpe"], ascending=[True, False])
    return {
        "results": results.reset_index(drop=True),
        "signals": signals,
        "positions": positions,
        "backtests": backtests,
    }


def run_soybean_external_signal_experiment(data_dir="train_set", include_weather=True):
    data = load_train_set(data_dir)
    feature_panels, futures_pnl = build_feature_panels(data)
    given = _given_components(feature_panels)

    external = {}
    external_errors = []
    try:
        prices = _download_yfinance(futures_pnl.index.min(), futures_pnl.index.max())
        external.update(_external_yfinance_families(prices, futures_pnl.index))
    except Exception as exc:
        prices = pd.DataFrame()
        external_errors.append("yfinance: {}".format(exc))

    weather = pd.DataFrame()
    if include_weather:
        try:
            weather = fetch_meteostat_weather(futures_pnl.index.min(), futures_pnl.index.max())
            external.update(_external_weather_family(weather, futures_pnl.index))
        except Exception as exc:
            external_errors.append("meteostat: {}".format(exc))

    rows = []
    backtests = {}
    positions = {}

    for name, signal in external.items():
        new_rows, new_bt, pos = _evaluate_signal(name, name, signal, futures_pnl, mode="long_only")
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions[name] = pos

    if external:
        external_equal = sum(external.values()) / float(len(external))
        new_rows, new_bt, pos = _evaluate_signal(
            "external_equal_family_weight",
            "external_equal_families",
            external_equal,
            futures_pnl,
            mode="long_only",
        )
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions["external_equal_family_weight"] = pos

        given_base = given["given_conservative_long_blend"]
        combined_50_50 = 0.50 * given_base + 0.50 * external_equal
        new_rows, new_bt, pos = _evaluate_signal(
            "given_plus_external_equal_family_50_50",
            "given_external_equal_families",
            combined_50_50,
            futures_pnl,
            mode="long_only",
        )
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions["given_plus_external_equal_family_50_50"] = pos

        all_families = dict(external)
        all_families["given_price_family"] = given["given_price_family"]
        all_families["given_physical_family"] = given["given_physical_family"]
        all_equal = sum(all_families.values()) / float(len(all_families))
        new_rows, new_bt, pos = _evaluate_signal(
            "given_and_external_all_families_equal_weight",
            "all_equal_families",
            all_equal,
            futures_pnl,
            mode="long_only",
        )
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions["given_and_external_all_families_equal_weight"] = pos

        if "external_weather_hdd_cdd_family" in external:
            crush_weather = 0.50 * given["given_crush_pressure"] + 0.50 * external["external_weather_hdd_cdd_family"]
            new_rows, new_bt, pos = _evaluate_signal(
                "given_crush_plus_weather_hdd_cdd_equal_weight",
                "given_crush_weather_equal",
                crush_weather,
                futures_pnl,
                mode="long_only",
            )
            rows.extend(new_rows)
            backtests.update(new_bt)
            positions["given_crush_plus_weather_hdd_cdd_equal_weight"] = pos

            conservative_weather = (
                0.50 * given["given_conservative_long_blend"] + 0.50 * external["external_weather_hdd_cdd_family"]
            )
            new_rows, new_bt, pos = _evaluate_signal(
                "given_conservative_plus_weather_hdd_cdd_equal_weight",
                "given_conservative_weather_equal",
                conservative_weather,
                futures_pnl,
                mode="long_only",
            )
            rows.extend(new_rows)
            backtests.update(new_bt)
            positions["given_conservative_plus_weather_hdd_cdd_equal_weight"] = pos

        fundamental_family_names = [
            ("given_physical_family", given.get("given_physical_family")),
            ("external_crush_family", external.get("external_crush_family")),
            ("external_fx_export_family", external.get("external_fx_export_family")),
            ("external_weather_hdd_cdd_family", external.get("external_weather_hdd_cdd_family")),
        ]
        fundamental_families = [values for _, values in fundamental_family_names if values is not None]
        if len(fundamental_families) >= 2:
            fundamental_equal = sum(fundamental_families) / float(len(fundamental_families))
            new_rows, new_bt, pos = _evaluate_signal(
                "given_physical_external_fundamentals_equal_weight",
                "physical_crush_fx_weather_equal",
                fundamental_equal,
                futures_pnl,
                mode="long_only",
            )
            rows.extend(new_rows)
            backtests.update(new_bt)
            positions["given_physical_external_fundamentals_equal_weight"] = pos

        overfit_controlled, selected_families, selection_table = _build_overfit_controlled_signal(
            given,
            external,
            futures_pnl,
        )
        new_rows, new_bt, pos = _evaluate_signal(
            "overfit_controlled_pre2018_family_selection",
            "pre2018_selected_equal_families",
            overfit_controlled,
            futures_pnl,
            mode="long_only",
        )
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions["overfit_controlled_pre2018_family_selection"] = pos

        (
            weight_relaxed,
            weight_relaxed_name,
            drawdown_relaxed,
            drawdown_relaxed_name,
            weight_relaxed_table,
            weight_relaxed_candidates,
        ) = _build_weight_relaxed_pre2018_signal(
            given,
            external,
            futures_pnl,
        )
        for candidate_name, candidate_signal in weight_relaxed_candidates.items():
            new_rows, new_bt, pos = _evaluate_signal(
                "weight_relaxed_candidate_" + candidate_name,
                "predefined_fixed_family_weights",
                candidate_signal,
                futures_pnl,
                mode="long_only",
            )
            rows.extend(new_rows)
            backtests.update(new_bt)
            positions["weight_relaxed_candidate_" + candidate_name] = pos

        new_rows, new_bt, pos = _evaluate_signal(
            "weight_relaxed_pre2018_selection_" + weight_relaxed_name,
            "pre2018_selected_fixed_family_weights",
            weight_relaxed,
            futures_pnl,
            mode="long_only",
        )
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions["weight_relaxed_pre2018_selection_" + weight_relaxed_name] = pos

        new_rows, new_bt, pos = _evaluate_signal(
            "drawdown_priority_pre2018_selection_" + drawdown_relaxed_name,
            "pre2018_drawdown_priority_fixed_family_weights",
            drawdown_relaxed,
            futures_pnl,
            mode="long_only",
        )
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions["drawdown_priority_pre2018_selection_" + drawdown_relaxed_name] = pos

        regime_signals, regime_frames = _build_regime_shift_signals(given, external, futures_pnl)
        regime_selection_rows = []
        for regime_name, regime_signal in regime_signals.items():
            metrics = _zero_cost_metric_for_signal(regime_signal, futures_pnl, mode="long_only")
            eligible = (
                pd.notnull(metrics.get("train_sharpe"))
                and pd.notnull(metrics.get("validation_sharpe"))
                and metrics["train_sharpe"] > 0.0
                and metrics["validation_sharpe"] >= 0.50
            )
            score = (
                metrics["validation_sharpe"] + 0.25 * metrics["train_sharpe"] + 0.001 * metrics["validation_max_drawdown"]
                if eligible
                else -np.inf
            )
            row = {"candidate": regime_name, "eligible": bool(eligible), "score": score}
            row.update(metrics)
            regime_selection_rows.append(row)
            new_rows, new_bt, pos = _evaluate_signal(
                "regime_candidate_" + regime_name,
                "observable_regime_shift",
                regime_signal,
                futures_pnl,
                mode="long_only",
            )
            rows.extend(new_rows)
            backtests.update(new_bt)
            positions["regime_candidate_" + regime_name] = pos

        regime_selection_table = pd.DataFrame(regime_selection_rows)
        regime_eligible = regime_selection_table.loc[regime_selection_table["eligible"]].copy()
        if regime_eligible.empty:
            regime_selected_name = regime_selection_table.sort_values(
                ["validation_sharpe", "validation_max_drawdown"],
                ascending=[False, False],
            ).iloc[0]["candidate"]
        else:
            regime_selected_name = regime_eligible.sort_values(
                ["score", "validation_max_drawdown"],
                ascending=[False, False],
            ).iloc[0]["candidate"]
        selected_regime_signal = regime_signals[regime_selected_name]
        new_rows, new_bt, pos = _evaluate_signal(
            "regime_pre2018_selection_" + regime_selected_name,
            "pre2018_selected_observable_regime_shift",
            selected_regime_signal,
            futures_pnl,
            mode="long_only",
        )
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions["regime_pre2018_selection_" + regime_selected_name] = pos

        fund_candidates = {
            "fund_blend_50_regime_50_drawdown": 0.50 * selected_regime_signal + 0.50 * drawdown_relaxed,
            "fund_blend_35_regime_65_drawdown": 0.35 * selected_regime_signal + 0.65 * drawdown_relaxed,
            "fund_blend_65_regime_35_drawdown": 0.65 * selected_regime_signal + 0.35 * drawdown_relaxed,
            "fund_consensus_min_regime_drawdown": pd.concat(
                [selected_regime_signal, drawdown_relaxed],
                axis=1,
            ).min(axis=1),
        }
        fund_selection_rows = []
        for fund_name, fund_signal in fund_candidates.items():
            metrics = _zero_cost_metric_for_signal(fund_signal, futures_pnl, mode="long_only")
            eligible = (
                pd.notnull(metrics.get("train_sharpe"))
                and pd.notnull(metrics.get("validation_sharpe"))
                and metrics["train_sharpe"] > 0.0
                and metrics["validation_sharpe"] >= 0.50
            )
            score = (
                metrics["validation_sharpe"]
                + 0.25 * metrics["train_sharpe"]
                + 0.002 * metrics["validation_max_drawdown"]
                if eligible
                else -np.inf
            )
            row = {"candidate": fund_name, "eligible": bool(eligible), "score": score}
            row.update(metrics)
            fund_selection_rows.append(row)
            new_rows, new_bt, pos = _evaluate_signal(
                "fund_candidate_" + fund_name,
                "fund_usable_regime_drawdown_blend",
                fund_signal,
                futures_pnl,
                mode="long_only",
            )
            rows.extend(new_rows)
            backtests.update(new_bt)
            positions["fund_candidate_" + fund_name] = pos

        fund_selection_table = pd.DataFrame(fund_selection_rows)
        fund_eligible = fund_selection_table.loc[fund_selection_table["eligible"]].copy()
        if fund_eligible.empty:
            fund_selected_name = fund_selection_table.sort_values(
                ["validation_sharpe", "validation_max_drawdown"],
                ascending=[False, False],
            ).iloc[0]["candidate"]
        else:
            fund_selected_name = fund_eligible.sort_values(
                ["score", "validation_max_drawdown"],
                ascending=[False, False],
            ).iloc[0]["candidate"]
        selected_fund_signal = fund_candidates[fund_selected_name]
        new_rows, new_bt, pos = _evaluate_signal(
            "fund_pre2018_selection_" + fund_selected_name,
            "pre2018_selected_fund_usable_blend",
            selected_fund_signal,
            futures_pnl,
            mode="long_only",
        )
        rows.extend(new_rows)
        backtests.update(new_bt)
        positions["fund_pre2018_selection_" + fund_selected_name] = pos
    else:
        selection_table = pd.DataFrame()
        selected_families = {}
        weight_relaxed_table = pd.DataFrame()
        weight_relaxed_name = None
        drawdown_relaxed_name = None
        regime_selection_table = pd.DataFrame()
        regime_selected_name = None
        regime_frames = {}
        fund_selection_table = pd.DataFrame()
        fund_selected_name = None

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["cost_adjusted", "test_sharpe"], ascending=[True, False]).reset_index(drop=True)
    return {
        "results": results,
        "given_signals": given,
        "external_signals": external,
        "positions": positions,
        "backtests": backtests,
        "yfinance_prices": prices,
        "weather": weather,
        "errors": external_errors,
        "overfit_control_selection": selection_table,
        "overfit_control_selected_families": list(selected_families.keys()),
        "weight_relaxed_selection": weight_relaxed_table,
        "weight_relaxed_selected_candidate": weight_relaxed_name,
        "drawdown_priority_selected_candidate": drawdown_relaxed_name,
        "regime_selection": regime_selection_table,
        "regime_selected_candidate": regime_selected_name,
        "regime_frames": regime_frames,
        "fund_selection": fund_selection_table,
        "fund_selected_candidate": fund_selected_name,
    }


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("Given-data-only soybean experiments")
    given_out = run_soybean_given_only_experiment()
    print(given_out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    print("\nExternal soybean experiments")
    external_out = run_soybean_external_signal_experiment()
    if external_out["errors"]:
        print("External data errors:")
        for item in external_out["errors"]:
            print(" - " + item)
    if external_out["results"].empty:
        print("No external results.")
    else:
        print(external_out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
