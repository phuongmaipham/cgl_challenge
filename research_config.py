"""Shared constants for the grain-strategy research notebook.

Keeping these in one file makes the notebook itself easier to read and lets
me reuse the same configuration across helper scripts.
"""

import numpy as np


# ── Universe & contracts ────────────────────────────────────────────────────
COMMODITIES = ["CORN", "SOYABEAN", "WHEAT_SRW", "WHEAT_HRW"]
CONTRACT_MULTIPLIER = 5000.0

DEFAULT_MARGIN_PER_LOT = {
    "CORN": 1500.0,
    "SOYABEAN": 3500.0,
    "WHEAT_SRW": 2500.0,
    "WHEAT_HRW": 2500.0,
}

# Train / OOS split. Everything before SPLIT_DATE is in-sample.
SPLIT_DATE = "2018-01-01"

# Corn-only research split used in the product-flow corn sleeve:
# train < 2016, validation 2016-2017, OOS >= 2018.
CORN_TRAIN_END = "2016-01-01"
CORN_TARGET_DAILY_PNL_VOL = 75.0
CORN_MAX_ABS_LOT = 0.50
CORN_TRADE_COST_PER_LOT = 8.75
CORN_HOLDING_COST_RATE = 0.05
CORN_IC_THRESHOLD = 0.015


# ── Feature groups ─────────────────────────────────────────────────────────
OUTRIGHT_CORE_FEATURES = [
    "mom_60",
    "rev_5",
    "curve_spread",
    "curve_ratio",
    "cot_mm_level",
    "cot_pm_oi_level",
]

OUTRIGHT_PHYSICAL_FEATURES = [
    "public_inventory_change",
    "receipts_change",
    "cgl_inventory_change",
    "crush_surprise",
    "crush_utilization",
]


# ── Cost cases ─────────────────────────────────────────────────────────────
# I check four cost regimes so a result that survives only the "no cost"
# baseline is flagged as fragile.
COST_CASES = [
    {
        "case": "zero_cost_no_margin_cap",
        "trade_cost_per_lot": 0.0,
        "holding_cost_rate": 0.0,
        "margin_budget": np.inf,
        "description": "Research baseline with no transaction or funding costs.",
    },
    {
        "case": "market_assumption",
        "trade_cost_per_lot": 8.75,
        "holding_cost_rate": 0.05,
        "margin_budget": np.inf,
        "description": "Approx. 0.5 tick bid/ask plus commissions/fees, 5% annual margin funding.",
    },
    {
        "case": "market_assumption_margin_cap",
        "trade_cost_per_lot": 8.75,
        "holding_cost_rate": 0.05,
        "margin_budget": 2500.0,
        "description": "Market cost assumption plus a 2,500 USD margin budget per aggregate book.",
    },
    {
        "case": "stress_cost_margin_cap",
        "trade_cost_per_lot": 15.00,
        "holding_cost_rate": 0.08,
        "margin_budget": 2500.0,
        "description": "Stress case: wider execution cost, higher funding rate, same margin budget.",
    },
]


# ── Named historical regimes ───────────────────────────────────────────────
# Used purely for diagnostics — I look at how each candidate behaves inside
# these named periods, but I never select strategies based on them.
REGIME_PERIODS = [
    {
        "period": "Russian drought/export ban shock",
        "start": "2010-07-01", "end": "2011-06-30",
        "reason": "Russian heat wave, drought, and grain export ban lifted in mid-2011.",
    },
    {
        "period": "US drought rally/retrace",
        "start": "2012-06-01", "end": "2013-05-31",
        "reason": "Historic US drought drove corn/soybean/wheat price shock and later retrace.",
    },
    {
        "period": "Crimea/Black Sea shock",
        "start": "2014-02-15", "end": "2014-05-31",
        "reason": "Ukraine/Crimea crisis raised Black Sea wheat and corn export risk.",
    },
    {
        "period": "Low-price abundant supply",
        "start": "2014-06-01", "end": "2017-12-31",
        "reason": "Post-drought supply rebuild and generally lower grain price regime.",
    },
    {
        "period": "US-China trade war",
        "start": "2018-07-06", "end": "2020-01-15",
        "reason": "Tariff escalation hit US soybean demand until the Phase One agreement.",
    },
    {
        "period": "2019 prevented planting floods",
        "start": "2019-05-01", "end": "2019-07-31",
        "reason": "Wet spring and Midwest flooding delayed corn and soybean planting.",
    },
    {
        "period": "COVID demand shock",
        "start": "2020-02-24", "end": "2020-06-30",
        "reason": "COVID restrictions reduced gasoline/ethanol demand and changed food demand.",
    },
    {
        "period": "COVID recovery/China buying",
        "start": "2020-07-01", "end": "2020-12-31",
        "reason": "Recovery phase with stronger Chinese buying and post-shock grain repricing.",
    },
]


# ── External data: Meteostat weather ────────────────────────────────────────
METEOSTAT_LOCATIONS = {
    "iowa_corn_belt":     (42.03, -93.63),
    "illinois_corn_belt": (40.63, -89.40),
    "nebraska_plains":    (41.26, -96.02),
    "kansas_wheat":       (38.35, -98.20),
}

# Approximate harvest acreage weights — used to combine station data into
# a per-commodity weather signal.
COMMODITY_LOCATION_WEIGHTS = {
    "CORN": {
        "iowa_corn_belt":     0.45,
        "illinois_corn_belt": 0.35,
        "nebraska_plains":    0.20,
    },
    "SOYABEAN": {
        "iowa_corn_belt":     0.40,
        "illinois_corn_belt": 0.40,
        "nebraska_plains":    0.20,
    },
    "WHEAT_SRW": {
        "illinois_corn_belt": 0.70,
        "kansas_wheat":       0.30,
    },
    "WHEAT_HRW": {
        "kansas_wheat":       0.70,
        "nebraska_plains":    0.30,
    },
}


# ── External data: EIA weekly ethanol ───────────────────────────────────────
EIA_SERIES = {
    "ethanol_production": "PET.W_EPOOXE_YOP_NUS_MBBLD.W",
    "ethanol_stocks":     "PET.W_EPOOXE_SAE_NUS_MBBL.W",
}


# ── External data: yfinance daily closes ───────────────────────────────────
YF_TICKERS = {
    "soybean":   "ZS=F",
    "soymeal":   "ZM=F",
    "soyoil":    "ZL=F",
    "corn":      "ZC=F",
    "wheat":     "ZW=F",
    "usd_index": "DX-Y.NYB",
    "brl":       "BRL=X",
    "cny":       "CNY=X",
    "crude":     "CL=F",
    "equity":    "SPY",
}


# ── Family taxonomy ────────────────────────────────────────────────────────
# Every feature gets a family label and a fixed sign (+1 means "high z-score
# is bullish for the commodity", -1 means "high z-score is bearish").
# I run each family-based test twice:
#   - PROVIDED: only train_set CSVs (prices, COT, public physical, Cargill physical)
#   - FULL:     PROVIDED + weather + EIA ethanol + yfinance macro
#
# This keeps the "what does the provided data alone get us?" question
# answerable separately from the "what does adding external data do?" question.

# Prices family — derived from adj/unadj price CSVs.
FAMILY_PRICES = {
    "mom_20":         +1,
    "mom_60":         +1,
    "rev_5":          +1,
    "curve_spread":   -1,   # contango -> bearish carry
    "curve_ratio":    -1,
    "curve_change_20": -1,
}

# Fundamentals — provided portion (always on).
FAMILY_FUNDAMENTALS_PROVIDED = {
    "cot_mm_level":           +1,   # spec-long positioning -> trend continuation
    "cot_pm_oi_level":        -1,   # producer hedge pressure -> bearish
    "cgl_inventory_change":   -1,
    "crush_surprise":         +1,
    "crush_utilization":      +1,
    "public_inventory_level": -1,
    "public_inventory_change": -1,
    "receipts_change":        -1,
}

# Fundamentals — extras only present in the FULL variant.
FAMILY_FUNDAMENTALS_EXTRA = {
    "hdd":                    +1,   # heat/cold stress in growing season -> bullish
    "cdd":                    +1,
    "prcp_20d":               -1,   # heavy rain -> bearish (good growing conditions)
    "ethanol_production_z":   +1,
    "ethanol_production_d4":  +1,
    "ethanol_stocks_z":       -1,
    "ethanol_stocks_d4":      -1,
}

# Macro — yfinance derived. Only present in the FULL variant.
FAMILY_MACRO = {
    "usd_index_level_z":  -1,
    "usd_index_mom_60":   -1,
    "brl_level_z":        +1,   # weak USD/BRL = stronger BRL = bullish South-American grains in USD
    "cny_level_z":        +1,
    "crude_mom_60":       +1,
    "crude_level_z":      +1,
    "equity_mom_60":      +1,
    "soybean_mom_60":     +1,
    "soymeal_mom_60":     +1,
    "soyoil_mom_60":      +1,
    "corn_mom_60":        +1,
    "wheat_mom_60":       +1,
}


def families_for_variant(variant):
    """Return a {family_name: {feature: sign}} dict for the requested variant.

    variant must be 'provided' (provided CSVs only) or 'full' (everything).
    """
    if variant == "provided":
        return {
            "prices":       dict(FAMILY_PRICES),
            "fundamentals": dict(FAMILY_FUNDAMENTALS_PROVIDED),
        }
    if variant == "full":
        return {
            "prices":       dict(FAMILY_PRICES),
            "fundamentals": {**FAMILY_FUNDAMENTALS_PROVIDED, **FAMILY_FUNDAMENTALS_EXTRA},
            "macro":        dict(FAMILY_MACRO),
        }
    raise ValueError(f"variant must be 'provided' or 'full', got {variant!r}")
