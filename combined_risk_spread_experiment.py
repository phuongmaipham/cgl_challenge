"""Risk-control and spread diagnostics for the combined external overlay.

This script keeps the Macro + EIA ethanol overlay optional, then adds separate
rows for drawdown controls and tradable spread sleeves.
"""

from __future__ import print_function

import numpy as np
import pandas as pd

from grain_futures_strategy import (
    _backtest_weighted_sleeves,
    backtest_positions,
    build_calendar_spread_feature_panels,
    build_calendar_spread_pnl,
    build_feature_panels,
    build_improved_model_signals,
    build_intercommodity_feature_panels,
    build_intercommodity_spread_pnl,
    build_walk_forward_spread_signals,
    edge_filtered_positions,
    load_train_set,
    model_predictions_to_positions,
    split_performance,
)
from macro_yfinance_experiment import (
    _download_yfinance_prices,
    build_macro_feature_block,
    build_macro_predictions,
)
from eia_ethanol_experiment import (
    build_ethanol_feature_panel,
    build_ethanol_predictions,
    fetch_eia_ethanol,
)


SPLIT_DATE = "2018-01-01"


def _row_from_bt(name, bt, sleeve):
    metrics = split_performance(bt, SPLIT_DATE)
    return {
        "experiment": name,
        "sleeve": sleeve,
        "is_sharpe": metrics.loc["sharpe", "in_sample"],
        "oos_sharpe": metrics.loc["sharpe", "out_of_sample"],
        "oos_pnl": metrics.loc["total_pnl", "out_of_sample"],
        "full_sharpe": metrics.loc["sharpe", "full_period"],
        "full_pnl": metrics.loc["total_pnl", "full_period"],
        "max_drawdown": metrics.loc["max_drawdown", "full_period"],
        "turnover": metrics.loc["avg_daily_turnover", "full_period"],
    }


def _row_from_positions(name, positions, pnl, sleeve):
    bt, _ = backtest_positions(positions, pnl, 0.0)
    return _row_from_bt(name, bt, sleeve), bt


def _drawdown_scaled_positions(positions, pnl, trigger=-3500.0, scale=0.50, lookback=126):
    """Scale exposure after a trailing drawdown breach using only prior PnL."""
    base_bt, _ = backtest_positions(positions, pnl, 0.0)
    lagged_cum = base_bt["net_pnl"].cumsum().shift(1)
    trailing_peak = lagged_cum.rolling(int(lookback), min_periods=20).max()
    trailing_drawdown = lagged_cum - trailing_peak
    exposure_scale = pd.Series(1.0, index=positions.index)
    exposure_scale.loc[trailing_drawdown < float(trigger)] = float(scale)
    return positions.mul(exposure_scale.fillna(1.0), axis=0), exposure_scale


def _vol_scaled_positions(positions, pnl, target_ann_vol=10000.0, lookback=60, min_scale=0.35, max_scale=1.15):
    """Scale exposure to a trailing portfolio volatility target."""
    base_bt, _ = backtest_positions(positions, pnl, 0.0)
    trailing_vol = base_bt["net_pnl"].rolling(int(lookback), min_periods=20).std().shift(1) * np.sqrt(252.0)
    exposure_scale = (float(target_ann_vol) / trailing_vol.replace(0.0, np.nan)).clip(
        lower=float(min_scale),
        upper=float(max_scale),
    )
    return positions.mul(exposure_scale.fillna(1.0), axis=0), exposure_scale


def run_combined_risk_spread_experiment():
    data = load_train_set("train_set")
    feature_panels, futures_pnl = build_feature_panels(data)
    selected_predictions, _, _, _ = build_improved_model_signals(feature_panels, futures_pnl, SPLIT_DATE)

    macro_prices = _download_yfinance_prices(futures_pnl.index.min(), futures_pnl.index.max())
    macro_features = build_macro_feature_block(macro_prices, futures_pnl.index)
    macro_predictions, macro_coefficients = build_macro_predictions(macro_features, futures_pnl)

    ethanol = fetch_eia_ethanol()
    ethanol_features = build_ethanol_feature_panel(ethanol, futures_pnl.index)
    ethanol_predictions, ethanol_coefficients = build_ethanol_predictions(ethanol_features, futures_pnl)

    selected_positions, _, _ = edge_filtered_positions(selected_predictions, futures_pnl, quantile=0.50)

    def make_external_positions(macro_weight, ethanol_weight):
        predictions = (
            selected_predictions.fillna(0.0)
            + float(macro_weight) * macro_predictions.fillna(0.0)
            + float(ethanol_weight) * ethanol_predictions.fillna(0.0)
        )
        positions, _, _ = edge_filtered_positions(predictions, futures_pnl, quantile=0.50)
        return predictions, positions

    combined_predictions, combined_positions = make_external_positions(1.00, 1.00)

    rows = []
    selected_row, selected_bt = _row_from_positions(
        "Current selected core + physical | edge-filtered",
        selected_positions,
        futures_pnl,
        "outright",
    )
    rows.append(selected_row)
    combined_row, combined_bt = _row_from_positions(
        "Selected + macro 1.00 + ethanol 1.00 | edge-filtered",
        combined_positions,
        futures_pnl,
        "outright",
    )
    rows.append(combined_row)

    external_candidates = [
        ("Selected + macro 0.10 + ethanol 0.10 | edge-filtered", 0.10, 0.10),
        ("Selected + macro 0.25 + ethanol 0.00 | edge-filtered", 0.25, 0.00),
        ("Selected + macro 0.25 + ethanol 0.10 | edge-filtered", 0.25, 0.10),
        ("Selected + macro 0.50 + ethanol 0.25 | edge-filtered", 0.50, 0.25),
    ]
    candidate_positions = {
        "full_macro_eia": combined_positions,
    }
    for name, macro_weight, ethanol_weight in external_candidates:
        _, positions = make_external_positions(macro_weight, ethanol_weight)
        row, _ = _row_from_positions(name, positions, futures_pnl, "outright")
        row["macro_weight"] = macro_weight
        row["ethanol_weight"] = ethanol_weight
        rows.append(row)
        candidate_positions[name] = positions

    risk_rows = []
    for trigger in [-2500.0, -3500.0, -4500.0, -5500.0]:
        for scale in [0.25, 0.50, 0.75]:
            adjusted, exposure_scale = _drawdown_scaled_positions(
                combined_positions,
                futures_pnl,
                trigger=trigger,
                scale=scale,
                lookback=126,
            )
            name = "Macro+EIA drawdown scale trigger {} scale {}".format(int(trigger), scale)
            row, _ = _row_from_positions(name, adjusted, futures_pnl, "outright risk-control")
            row["risk_rule"] = "trailing_126d_drawdown"
            row["risk_trigger"] = trigger
            row["risk_scale"] = scale
            row["avg_scale"] = exposure_scale.mean()
            risk_rows.append(row)

    for target in [7000.0, 8500.0, 10000.0, 11500.0]:
        adjusted, exposure_scale = _vol_scaled_positions(
            combined_positions,
            futures_pnl,
            target_ann_vol=target,
            lookback=60,
            min_scale=0.35,
            max_scale=1.15,
        )
        name = "Macro+EIA volatility scale target {}".format(int(target))
        row, _ = _row_from_positions(name, adjusted, futures_pnl, "outright risk-control")
        row["risk_rule"] = "trailing_60d_vol"
        row["risk_trigger"] = target
        row["risk_scale"] = np.nan
        row["avg_scale"] = exposure_scale.mean()
        risk_rows.append(row)

    risk_results = pd.DataFrame(risk_rows).sort_values(
        ["max_drawdown", "oos_sharpe"],
        ascending=[False, False],
    )
    rows.extend(risk_results.head(8).to_dict("records"))

    calendar_pnl = build_calendar_spread_pnl(data)
    calendar_panels = build_calendar_spread_feature_panels(feature_panels)
    calendar_predictions, calendar_coefficients = build_walk_forward_spread_signals(
        calendar_panels,
        calendar_pnl,
        alpha=100.0,
    )
    calendar_positions = model_predictions_to_positions(calendar_predictions, calendar_pnl)
    calendar_row, calendar_bt = _row_from_positions(
        "Calendar spread only | front minus second",
        calendar_positions,
        calendar_pnl,
        "calendar spread",
    )
    rows.append(calendar_row)

    inter_pnl, hedge_ratios = build_intercommodity_spread_pnl(futures_pnl)
    inter_panels = build_intercommodity_feature_panels(feature_panels)
    inter_predictions, inter_coefficients = build_walk_forward_spread_signals(
        inter_panels,
        inter_pnl,
        alpha=100.0,
    )
    inter_positions = model_predictions_to_positions(inter_predictions, inter_pnl)
    inter_row, inter_bt = _row_from_positions(
        "Intercommodity spread only | vol-hedged pairs",
        inter_positions,
        inter_pnl,
        "intercommodity spread",
    )
    rows.append(inter_row)

    for spread_name, spread_positions, spread_pnl in [
        ("calendar", calendar_positions, calendar_pnl),
        ("intercommodity", inter_positions, inter_pnl),
    ]:
        for outright_weight, spread_weight in [(0.90, 0.10), (0.80, 0.20), (0.70, 0.30)]:
            overlay_bt = _backtest_weighted_sleeves(
                [(combined_positions, futures_pnl), (spread_positions, spread_pnl)],
                [outright_weight, spread_weight],
                0.0,
            )
            rows.append(
                _row_from_bt(
                    "Macro+EIA {} + {} spread {}".format(outright_weight, spread_name, spread_weight),
                    overlay_bt,
                    "{} overlay".format(spread_name),
                )
            )

    light_both_positions = candidate_positions["Selected + macro 0.10 + ethanol 0.10 | edge-filtered"]
    for outright_weight, spread_weight in [(0.90, 0.10), (0.80, 0.20)]:
        overlay_bt = _backtest_weighted_sleeves(
            [(light_both_positions, futures_pnl), (inter_positions, inter_pnl)],
            [outright_weight, spread_weight],
            0.0,
        )
        rows.append(
            _row_from_bt(
                "Light Macro+EIA {} + intercommodity spread {}".format(outright_weight, spread_weight),
                overlay_bt,
                "intercommodity overlay",
            )
        )

    results = pd.DataFrame(rows)
    results = results.sort_values(["oos_sharpe", "max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    return {
        "results": results,
        "risk_results": risk_results.reset_index(drop=True),
        "macro_coefficients": macro_coefficients,
        "ethanol_coefficients": ethanol_coefficients,
        "calendar_coefficients": calendar_coefficients,
        "intercommodity_coefficients": inter_coefficients,
        "selected_bt": selected_bt,
        "combined_bt": combined_bt,
        "calendar_bt": calendar_bt,
        "intercommodity_bt": inter_bt,
        "intercommodity_hedge_ratios": hedge_ratios,
    }


if __name__ == "__main__":
    out = run_combined_risk_spread_experiment()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 24)
    print(out["results"].to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
