"""Optional Meteostat crop-belt weather overlay experiment.

Meteostat is open data and does not require an API key. This module keeps the
weather test separate from the selected Cargill-aware strategy.
"""

from __future__ import print_function

from datetime import datetime

import pandas as pd

from grain_futures_strategy import (
    COMMODITIES,
    backtest_positions,
    build_feature_panels,
    build_improved_model_signals,
    edge_filtered_positions,
    fit_ridge_predict,
    load_train_set,
    model_predictions_to_positions,
    rolling_zscore,
    split_performance,
)


SPLIT_DATE = "2018-01-01"
OVERLAY_WEIGHTS = [0.10, 0.25, 0.50, 1.00]
METEOSTAT_LOCATIONS = {
    "iowa_corn_belt": (42.03, -93.63),
    "illinois_corn_belt": (40.63, -89.40),
    "nebraska_plains": (41.26, -96.02),
    "kansas_wheat": (38.35, -98.20),
}
COMMODITY_LOCATION_WEIGHTS = {
    "CORN": {
        "iowa_corn_belt": 0.45,
        "illinois_corn_belt": 0.35,
        "nebraska_plains": 0.20,
    },
    "SOYABEAN": {
        "iowa_corn_belt": 0.40,
        "illinois_corn_belt": 0.40,
        "nebraska_plains": 0.20,
    },
    "WHEAT_SRW": {
        "illinois_corn_belt": 0.70,
        "kansas_wheat": 0.30,
    },
    "WHEAT_HRW": {
        "kansas_wheat": 0.70,
        "nebraska_plains": 0.30,
    },
}


def fetch_meteostat_weather(start, end, locations=None):
    """Fetch daily Meteostat weather for a compact crop-belt location set.

    Compatible with both the legacy meteostat API (Daily class) and the new
    function-based API introduced in meteostat 2.x (daily function + stations).
    """
    import meteostat as _ms

    if locations is None:
        locations = METEOSTAT_LOCATIONS

    start_dt = pd.Timestamp(start).to_pydatetime()
    end_dt = pd.Timestamp(end).to_pydatetime()

    # Detect API version: new API exposes 'daily' as a function, not a class.
    _use_new_api = callable(getattr(_ms, "daily", None)) and not isinstance(
        getattr(_ms, "daily", None), type
    )

    rows = []
    for name, (lat, lon) in locations.items():
        point = _ms.Point(float(lat), float(lon))

        if _use_new_api:
            # New API: find nearest station, then fetch by station ID.
            nearby = _ms.stations.nearby(point)
            if nearby.empty:
                continue
            station_id = nearby.index[0]
            ts = _ms.daily(station_id, start_dt, end_dt)
            frame = ts.fetch()
            if frame is None or (hasattr(frame, "empty") and frame.empty):
                continue
            # New API column names: 'temp' instead of 'tavg'
            frame = frame.rename(columns={"temp": "tavg"})
            frame = frame.reset_index().rename(columns={"time": "date"})
        else:
            # Legacy API: Daily class accepts a Point directly.
            Daily = getattr(_ms, "Daily")
            frame = Daily(point, start_dt, end_dt).fetch()
            if frame.empty:
                continue
            frame = frame.reset_index().rename(columns={"time": "date"})

        frame["location"] = name
        rows.append(frame)

    if not rows:
        raise RuntimeError("Meteostat returned no weather rows.")

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out


def _weighted_weather_by_commodity(weather, value_cols, commodity):
    weights = COMMODITY_LOCATION_WEIGHTS.get(commodity, {})
    frames = []
    for location, weight in weights.items():
        sub = weather.loc[weather["location"] == location, ["date"] + value_cols].copy()
        if sub.empty:
            continue
        sub[value_cols] = sub[value_cols] * float(weight)
        frames.append(sub)
    if not frames:
        return weather.groupby("date")[value_cols].mean().sort_index()

    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("date")[value_cols].sum().sort_index()


def _season_mask(index, months):
    return pd.Series(index.month.isin(months), index=index).astype(float)


def _add_weather_features(aligned, prefix, seasonal=False):
    features = pd.DataFrame(index=aligned.index)
    if "temp_mean" in aligned.columns:
        features[prefix + "temp_mean_level"] = rolling_zscore(aligned["temp_mean"], 252, 60)
        features[prefix + "temp_mean_change_20"] = rolling_zscore(aligned["temp_mean"].diff(20), 252, 60)
        cdd = (aligned["temp_mean"] - 18.0).clip(lower=0.0)
        hdd = (18.0 - aligned["temp_mean"]).clip(lower=0.0)
        for window in [5, 20, 60]:
            min_periods = max(3, int(window / 4))
            features[prefix + "cdd_{}d".format(window)] = rolling_zscore(
                cdd.rolling(window, min_periods=min_periods).sum(),
                252,
                60,
            )
            features[prefix + "hdd_{}d".format(window)] = rolling_zscore(
                hdd.rolling(window, min_periods=min_periods).sum(),
                252,
                60,
            )
        features[prefix + "cdd_change_20"] = rolling_zscore(cdd.rolling(20, min_periods=5).sum().diff(20), 252, 60)
        features[prefix + "hdd_change_20"] = rolling_zscore(hdd.rolling(20, min_periods=5).sum().diff(20), 252, 60)
    if "temp_max" in aligned.columns:
        features[prefix + "heat_level"] = rolling_zscore(aligned["temp_max"], 252, 60)
        heat_stress = (aligned["temp_max"] - 32.0).clip(lower=0.0)
        features[prefix + "heat_stress_5d"] = rolling_zscore(heat_stress.rolling(5, min_periods=3).sum(), 252, 60)
        features[prefix + "heat_stress_20d"] = rolling_zscore(heat_stress.rolling(20, min_periods=5).sum(), 252, 60)
    if "temp_min" in aligned.columns:
        features[prefix + "cold_level"] = -rolling_zscore(aligned["temp_min"], 252, 60)
        freeze_stress = (0.0 - aligned["temp_min"]).clip(lower=0.0)
        features[prefix + "freeze_stress_5d"] = rolling_zscore(freeze_stress.rolling(5, min_periods=3).sum(), 252, 60)
        features[prefix + "freeze_stress_20d"] = rolling_zscore(freeze_stress.rolling(20, min_periods=5).sum(), 252, 60)
    if "temp_min" in aligned.columns and "temp_max" in aligned.columns:
        temp_avg = (aligned["temp_min"] + aligned["temp_max"]) / 2.0
        gdd = (temp_avg.clip(upper=30.0) - 10.0).clip(lower=0.0)
        features[prefix + "gdd_20d"] = rolling_zscore(gdd.rolling(20, min_periods=5).sum(), 252, 60)
        features[prefix + "gdd_60d"] = rolling_zscore(gdd.rolling(60, min_periods=15).sum(), 252, 60)
    if "precipitation" in aligned.columns:
        precip = aligned["precipitation"].fillna(0.0)
        for window in [5, 20, 60]:
            precip_sum = precip.rolling(window, min_periods=max(3, int(window / 4))).sum()
            features[prefix + "precip_{}d".format(window)] = rolling_zscore(precip_sum, 252, 60)
            features[prefix + "dryness_{}d".format(window)] = -features[prefix + "precip_{}d".format(window)]
        if prefix + "heat_stress_20d" in features.columns:
            features[prefix + "dry_heat_20d"] = (
                features[prefix + "dryness_20d"] * features[prefix + "heat_stress_20d"]
            ).clip(lower=-5.0, upper=5.0)
        if prefix + "cdd_20d" in features.columns:
            features[prefix + "dry_cdd_20d"] = (
                features[prefix + "dryness_20d"] * features[prefix + "cdd_20d"]
            ).clip(lower=-5.0, upper=5.0)
    if "wind_speed" in aligned.columns:
        features[prefix + "wind_level"] = rolling_zscore(aligned["wind_speed"], 252, 60)

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

    return features.clip(lower=-5.0, upper=5.0)


def build_meteostat_feature_panels(weather, trading_index, mode="shared_basic"):
    """Build weather features from crop-belt daily weather."""
    rename = {
        "tavg": "temp_mean",
        "tmin": "temp_min",
        "tmax": "temp_max",
        "prcp": "precipitation",
        "wspd": "wind_speed",
    }
    weather = weather.rename(columns={k: v for k, v in rename.items() if k in weather.columns})
    value_cols = [
        col
        for col in ["temp_mean", "temp_min", "temp_max", "precipitation", "wind_speed"]
        if col in weather.columns
    ]
    if not value_cols:
        raise RuntimeError("Meteostat data has no recognized weather columns.")
    for column in value_cols:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")

    if mode == "shared_basic":
        daily = weather.groupby("date")[value_cols].mean().sort_index()
        aligned = daily.reindex(trading_index).ffill().shift(1)
        features = _add_weather_features(aligned, "meteo_", seasonal=False)
        keep = [
            "meteo_temp_mean_level",
            "meteo_temp_mean_change_20",
            "meteo_heat_level",
            "meteo_cold_level",
            "meteo_precip_20d",
            "meteo_dryness_20d",
            "meteo_wind_level",
        ]
        features = features[[col for col in keep if col in features.columns]]
        return {commodity: features.copy() for commodity in COMMODITIES}

    panels = {}
    seasonal = mode == "commodity_seasonal"
    for commodity in COMMODITIES:
        daily = _weighted_weather_by_commodity(weather, value_cols, commodity)
        aligned = daily.reindex(trading_index).ffill().shift(1)
        panels[commodity] = _add_weather_features(aligned, "meteo_", seasonal=seasonal)
    return panels


def build_meteostat_predictions(weather_panels, futures_pnl, split_date=SPLIT_DATE, alpha=1000.0, horizon=5):
    train_mask = futures_pnl.index < pd.Timestamp(split_date)
    predictions = pd.DataFrame(index=futures_pnl.index, columns=COMMODITIES, dtype=float)
    coefficients = {}
    for commodity in COMMODITIES:
        horizon = int(horizon)
        target = futures_pnl[commodity].shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
        pred, coef = fit_ridge_predict(weather_panels[commodity], target, train_mask, alpha=alpha)
        predictions[commodity] = pred
        coefficients[commodity] = coef
    return predictions.fillna(0.0), pd.DataFrame(coefficients)


def _metrics_row(name, predictions, futures_pnl, edge_filter=False):
    if edge_filter:
        positions, _, _ = edge_filtered_positions(predictions, futures_pnl, quantile=0.50)
        name = name + " | edge-filtered"
    else:
        positions = model_predictions_to_positions(predictions, futures_pnl)
        name = name + " | unfiltered"
    bt, _ = backtest_positions(positions, futures_pnl, 0.0)
    metrics = split_performance(bt, SPLIT_DATE)
    return {
        "experiment": name,
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "full_pnl": metrics.loc["total_pnl", "full_period"],
        "max_drawdown": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
    }


def _prediction_correlation(a, b):
    common = a.reindex_like(b).fillna(0.0)
    base = b.fillna(0.0)
    values = []
    for column in common.columns:
        joined = pd.concat([common[column], base[column]], axis=1).dropna()
        if len(joined) > 20 and joined.iloc[:, 0].std() > 0.0 and joined.iloc[:, 1].std() > 0.0:
            values.append(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
    if not values:
        return float("nan")
    return float(pd.Series(values).mean())


def run_meteostat_experiment():
    data = load_train_set("train_set")
    feature_panels, futures_pnl = build_feature_panels(data)
    selected_predictions, _, _, _ = build_improved_model_signals(feature_panels, futures_pnl, SPLIT_DATE)

    weather = fetch_meteostat_weather(futures_pnl.index.min(), futures_pnl.index.max())
    variants = [
        ("shared_basic", 1000.0, 5),
        ("shared_basic", 1000.0, 20),
        ("commodity_basic", 1000.0, 5),
        ("commodity_basic", 1000.0, 20),
        ("commodity_seasonal", 1000.0, 5),
        ("commodity_seasonal", 1000.0, 20),
        ("commodity_seasonal", 5000.0, 20),
        ("commodity_seasonal", 100.0, 20),
    ]

    rows = []
    rows.append(_metrics_row("Current selected core + physical", selected_predictions, futures_pnl, False))
    rows.append(_metrics_row("Current selected core + physical", selected_predictions, futures_pnl, True))
    coefficient_blocks = {}
    panel_blocks = {}
    prediction_blocks = {}
    for mode, alpha, horizon in variants:
        weather_panels = build_meteostat_feature_panels(weather, futures_pnl.index, mode=mode)
        weather_predictions, weather_coefficients = build_meteostat_predictions(
            weather_panels,
            futures_pnl,
            alpha=alpha,
            horizon=horizon,
        )
        variant_name = "Meteostat {} alpha={} horizon={}".format(mode, int(alpha), int(horizon))
        panel_blocks[variant_name] = weather_panels
        prediction_blocks[variant_name] = weather_predictions
        coefficient_blocks[variant_name] = weather_coefficients

        weather_only = _metrics_row(variant_name + " only", weather_predictions, futures_pnl, False)
        weather_only["weather_variant"] = variant_name
        weather_only["prediction_corr_to_selected"] = _prediction_correlation(weather_predictions, selected_predictions)
        rows.append(weather_only)

        weather_only_edge = _metrics_row(variant_name + " only", weather_predictions, futures_pnl, True)
        weather_only_edge["weather_variant"] = variant_name
        weather_only_edge["prediction_corr_to_selected"] = weather_only["prediction_corr_to_selected"]
        rows.append(weather_only_edge)

        for weight in OVERLAY_WEIGHTS:
            combined = selected_predictions.fillna(0.0) + float(weight) * weather_predictions.fillna(0.0)
            for edge_filter in [False, True]:
                row = _metrics_row(
                    "Current selected + {} overlay w={}".format(variant_name, weight),
                    combined,
                    futures_pnl,
                    edge_filter,
                )
                row["weather_variant"] = variant_name
                row["weather_weight"] = weight
                row["prediction_corr_to_selected"] = weather_only["prediction_corr_to_selected"]
                rows.append(row)

    results = pd.DataFrame(rows).sort_values(["oos_sharpe", "full_sharpe"], ascending=[False, False])
    return {
        "weather": weather,
        "weather_feature_panels": panel_blocks,
        "weather_predictions": prediction_blocks,
        "weather_coefficients": coefficient_blocks,
        "results": results.reset_index(drop=True),
    }


if __name__ == "__main__":
    out = run_meteostat_experiment()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
