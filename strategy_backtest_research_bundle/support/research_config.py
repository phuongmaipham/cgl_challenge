"""Small config file for the three grain backtest notebooks."""

COMMODITIES = ["CORN", "SOYABEAN", "WHEAT_SRW", "WHEAT_HRW"]
CONTRACT_MULTIPLIER = 5000.0

DEFAULT_MARGIN_PER_LOT = {
    "CORN": 1500.0,
    "SOYABEAN": 3500.0,
    "WHEAT_SRW": 2500.0,
    "WHEAT_HRW": 2500.0,
}

SPLIT_DATE = "2018-01-01"

# Corn uses an extra train/validation split before the shared 2018 OOS start.
CORN_TRAIN_END = "2016-01-01"
CORN_TARGET_DAILY_PNL_VOL = 75.0
CORN_MAX_ABS_LOT = 0.50
CORN_TRADE_COST_PER_LOT = 8.75
CORN_HOLDING_COST_RATE = 0.05
CORN_IC_THRESHOLD = 0.015

REGIME_PERIODS = [
    {"period": "Russian drought/export ban shock", "start": "2010-07-01", "end": "2011-06-30"},
    {"period": "US drought rally/retrace", "start": "2012-06-01", "end": "2013-05-31"},
    {"period": "Crimea/Black Sea shock", "start": "2014-02-15", "end": "2014-05-31"},
    {"period": "Low-price abundant supply", "start": "2014-06-01", "end": "2017-12-31"},
    {"period": "US-China trade war", "start": "2018-07-06", "end": "2020-01-15"},
    {"period": "2019 prevented planting floods", "start": "2019-05-01", "end": "2019-07-31"},
    {"period": "COVID demand shock", "start": "2020-02-24", "end": "2020-06-30"},
    {"period": "COVID recovery/China buying", "start": "2020-07-01", "end": "2020-12-31"},
]

# Simple acreage-style weights for the corn weather signal.
COMMODITY_LOCATION_WEIGHTS = {
    "CORN": {
        "iowa_corn_belt": 0.45,
        "illinois_corn_belt": 0.35,
        "nebraska_plains": 0.20,
    },
}
