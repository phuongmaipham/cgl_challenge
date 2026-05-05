"""Clean standalone runner for the product-specific grain strategies.

This script is intentionally presentation-friendly:
- it runs the basic model audit used in the research story;
- it runs the final product-specific soybean, corn, and wheat strategies;
- it saves compact CSV tables under outputs/product_strategy_standalone/;
- it prints a concise summary table.

The final preferred strategies are not fitted coefficient models:
- Soybeans: economic family blend with low-volatility half-exposure control.
- Corn: volatility-regime corn sleeve with abundant-supply flat guard.
- Wheat: SRW/HRW relative-value pair using price mean reversion and Cargill
  physical pressure, plus optional trend-aware variants.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from corn_abundant_supply_improvement import run_corn_abundant_supply_improvement
from family_regime_model_comparison import run_family_regime_model_comparison
from linear_online_model_experiment import run_linear_online_model_experiment
from soybean_low_vol_switch_experiment import run_soybean_low_vol_switch_experiment
from wheat_improvement_experiment import run_wheat_improvement_experiment


OUTPUT_DIR = Path("outputs/product_strategy_standalone")


def _fmt_table(df: pd.DataFrame) -> str:
    """Pretty string formatting for terminal output."""
    return df.to_string(index=False, float_format=lambda value: f"{value:.3f}")


def _save(df: pd.DataFrame, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def _cost_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    if "cost_adjusted" not in df.columns:
        return df.copy()
    return df.loc[df["cost_adjusted"].astype(bool)].copy()


def run_basic_audit(data_dir: str = "train_set") -> dict[str, pd.DataFrame]:
    """Run the basic tests that justify product-specific strategy design.

    The audit covers:
    1. average all signals;
    2. equal family weight;
    3. IC-selected family;
    4. regime best family;
    5. regime average families;
    6. OLS/Kalman;
    7. Ridge.
    """
    family_out = run_family_regime_model_comparison(data_dir=data_dir)
    linear_out = run_linear_online_model_experiment(data_dir=data_dir)

    family_results = family_out["results"].copy()
    linear_results = linear_out["results"].copy()

    _save(family_results, "basic_family_regime_audit.csv")
    _save(linear_results, "basic_ols_kalman_ridge_audit.csv")

    return {
        "family_regime": family_results,
        "linear_models": linear_results,
    }


def run_soybean_final(data_dir: str = "train_set") -> dict[str, object]:
    """Run the final soybean strategy family.

    Preferred row:
        low_vol_half_base_else_base

    Logic:
        base_signal = 40% provided physical/Cargill
                    + 20% FX/export
                    + 20% external crush
                    + 20% weather HDD/CDD

        if 60d soybean PnL volatility < 0.80 * expanding median volatility:
            position *= 0.50
    """
    out = run_soybean_low_vol_switch_experiment(data_dir=data_dir)
    results = out["results"].copy()
    cost = _cost_adjusted(results)

    preferred = cost.loc[cost["strategy"] == "low_vol_half_base_else_base"].copy()
    diagnostics = cost.loc[
        cost["strategy"].isin(
            [
                "base_drawdown_priority",
                "low_vol_flat_else_base",
                "low_vol_half_base_else_base",
                "low_vol_abundant_proxy_flat_else_base",
                "low_vol_weak_trend_flat_else_base",
            ]
        )
    ].copy()

    _save(results, "soybean_all_low_vol_tests.csv")
    _save(diagnostics, "soybean_key_tests.csv")
    _save(preferred, "soybean_recommended.csv")

    return {
        "results": results,
        "diagnostics": diagnostics,
        "recommended": preferred,
        "backtests": out["backtests"],
        "period_tables": out["period_tables"],
    }


def run_corn_final(data_dir: str = "train_set") -> dict[str, object]:
    """Run the final corn strategy family.

    Preferred row:
        below_ma_or_negative_mom_flat

    Logic:
        base = volatility-regime IC corn sleeve

        if corn price < 252d moving average OR corn mom_60 < 0:
            position = 0
        else:
            position = base position
    """
    out = run_corn_abundant_supply_improvement(data_dir=data_dir)
    results = out["results"].copy()
    cost = _cost_adjusted(results)

    preferred = cost.loc[cost["strategy"] == "below_ma_or_negative_mom_flat"].copy()
    diagnostics = cost.loc[
        cost["strategy"].isin(
            [
                "base_regime_ic_vol",
                "below_ma_or_negative_mom_flat",
                "below_ma_and_negative_mom_flat",
                "abundant_curve_confirmed_flat",
                "abundant_low_or_normal_flat",
            ]
        )
    ].copy()

    _save(results, "corn_all_abundant_supply_tests.csv")
    _save(diagnostics, "corn_key_tests.csv")
    _save(preferred, "corn_recommended.csv")

    return {
        "results": results,
        "diagnostics": diagnostics,
        "recommended": preferred,
        "backtests": out["backtests"],
        "period_tables": out["period_tables"],
        "selected_table": out["selected_table"],
        "errors": out["errors"],
    }


def run_wheat_final(data_dir: str = "train_set") -> dict[str, object]:
    """Run the final wheat SRW/HRW relative-value strategy family.

    Preferred base row:
        pair_price_mr_cargill_90_10_cost_control

    Trend-aware variant:
        pair_price_mr_cargill_80_20_pair_trend_cost_control

    Conservative risk-control diagnostic:
        pair_price_mr_cargill_trend_conflict_flat_cost_control
    """
    out = run_wheat_improvement_experiment(data_dir=data_dir)
    results = out["results"].copy()
    cost = _cost_adjusted(results)

    keep = [
        "pair_price_mr_cargill_90_10_cost_control",
        "pair_price_mr_cargill_80_20_pair_trend_cost_control",
        "pair_price_mr_cargill_trend_conflict_flat_cost_control",
    ]
    diagnostics = cost.loc[cost["strategy"].isin(keep)].copy()
    preferred = cost.loc[cost["strategy"] == "pair_price_mr_cargill_90_10_cost_control"].copy()

    _save(results, "wheat_all_pair_tests.csv")
    _save(diagnostics, "wheat_key_tests.csv")
    _save(preferred, "wheat_recommended.csv")

    return {
        "results": results,
        "diagnostics": diagnostics,
        "recommended": preferred,
        "backtests": out["backtests"],
        "trend_regime": out["trend_regime"],
    }


def build_final_summary(
    soybean: dict[str, object],
    corn: dict[str, object],
    wheat: dict[str, object],
) -> pd.DataFrame:
    """Create one clean presentation table for the recommended strategies."""
    soy = soybean["recommended"].iloc[0]
    corn_row = corn["recommended"].iloc[0]
    wheat_row = wheat["recommended"].iloc[0]

    rows = [
        {
            "product": "Soybeans",
            "strategy": soy["strategy"],
            "economic_logic": "Physical/Cargill + FX/export + external crush + weather, with low-vol half exposure.",
            "oos_sharpe": soy["oos_sharpe"],
            "oos_pnl": soy["oos_pnl"],
            "oos_dd": soy["oos_dd"],
            "full_sharpe": soy["full_sharpe"],
            "full_dd": soy["full_dd"],
            "overfit_read": "Low/moderate: fixed family weights and observable risk control; no fitted coefficients.",
        },
        {
            "product": "Corn",
            "strategy": corn_row["strategy"],
            "economic_logic": "Vol-regime corn sleeve with observable abundant-supply flat guard.",
            "oos_sharpe": corn_row["oos_sharpe"],
            "oos_pnl": corn_row["oos_pnl"],
            "oos_dd": corn_row["oos_dd"],
            "full_sharpe": corn_row["full_sharpe"],
            "full_dd": corn_row["full_dd"],
            "overfit_read": "Moderate: improved risk control, but corn remains a smaller satellite sleeve.",
        },
        {
            "product": "Wheat SRW/HRW",
            "strategy": wheat_row["strategy"],
            "economic_logic": "Risk-balanced SRW/HRW relative value: price MR plus Cargill physical pressure.",
            "oos_sharpe": wheat_row["oos_sharpe"],
            "oos_pnl": wheat_row["oos_pnl"],
            "oos_dd": wheat_row["oos_dd"],
            "full_sharpe": wheat_row["full_sharpe"],
            "full_dd": wheat_row["full_dd"],
            "overfit_read": "Low/moderate: no fitted coefficients; relative-value structure is economically motivated.",
        },
    ]
    summary = pd.DataFrame(rows)
    _save(summary, "final_recommended_strategy_summary.csv")
    return summary


def run_all(data_dir: str = "train_set", run_audit: bool = True) -> dict[str, object]:
    """Run the full standalone product strategy workflow."""
    audit = run_basic_audit(data_dir=data_dir) if run_audit else {}
    soybean = run_soybean_final(data_dir=data_dir)
    corn = run_corn_final(data_dir=data_dir)
    wheat = run_wheat_final(data_dir=data_dir)
    summary = build_final_summary(soybean, corn, wheat)

    return {
        "audit": audit,
        "soybean": soybean,
        "corn": corn,
        "wheat": wheat,
        "summary": summary,
    }


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    result = run_all()
    print("\nFinal recommended strategy summary")
    print("----------------------------------")
    print(_fmt_table(result["summary"]))
    print(f"\nSaved CSV outputs to: {OUTPUT_DIR.resolve()}")
