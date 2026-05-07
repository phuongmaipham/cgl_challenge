# Grain Strategy Results Pack

This folder is prepared for a skim-first review by a systematic trading expert,
while still remaining executable.

The notebooks include executed outputs. In JupyterLab-compatible viewers, code inputs
are hidden by default, but the code is still present and can be expanded or rerun.

## Layout

```
strategy_backtest_research_bundle/
├── grain_futures_strategy_standalone.ipynb     # main cross-grain pipeline + final blend
├── grain_portfolio_backtest_research.ipynb     # outright + cross-grain spreads
├── soybean_strategy_backtest_research.ipynb    # soybean walk-forward + Cargill
├── wheat_strategy_backtest_research.ipynb      # SRW/HRW pair sleeve
└── support/                                    # everything else
    ├── grain_backtest_core.py                  # data loaders, features, costs, metrics
    ├── shared_backtest.py                      # helpers used by soybean/wheat/portfolio
    ├── grain_futures_strategy.py               # `gfs.*` for the standalone notebook
    ├── grain_futures_strategy_config.json      # constants (cost cases, regime periods, …)
    ├── research_config.py                      # split date, margins, regime periods
    ├── requirements.txt                        # Python dependency pins
    └── train_set/                              # all CSV inputs (internal + external)
```

The notebooks add `support/` to `sys.path` automatically and use
`DATA_DIR = "support/train_set"`, so they always run from the bundle root with
no environment setup beyond installing dependencies.

## Suggested review order

1. `grain_futures_strategy_standalone.ipynb` — main cross-grain strategy case: model
   comparison, holding-period checks, opportunity filters, final blend, cost/margin audit.
2. `grain_portfolio_backtest_research.ipynb` — portfolio-level check across outright
   grain sleeves and simple spread books.
3. `soybean_strategy_backtest_research.ipynb` — soybean sleeve research, final
   candidate, robustness diagnostics.
4. `wheat_strategy_backtest_research.ipynb` — SRW/HRW wheat pair sleeve research,
   final candidate, robustness diagnostics.

## Quick start — launch in JupyterLab

From the bundle root:

```bash
# 1. Install dependencies (first time only)
python3 -m pip install -r support/requirements.txt

# 2. Launch JupyterLab
jupyter lab
```

Open any notebook and **Run All** (▶▶) to reproduce the cells. Each notebook
finishes in 30–90 seconds on a typical laptop.

## Or rerun from the command line

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
  --output-dir _executed_notebooks \
  grain_futures_strategy_standalone.ipynb \
  grain_portfolio_backtest_research.ipynb \
  soybean_strategy_backtest_research.ipynb \
  wheat_strategy_backtest_research.ipynb
```

This writes rerun copies to `_executed_notebooks/` and leaves the review copies untouched.

## Data

`support/train_set/` contains 13 CSVs:

- 10 internal CGL files: adjusted/unadjusted prices, COT, public inventories, receipts,
  Cargill inventories, Cargill crush.
- 3 external files (already saved as static CSVs — no live fetch):
  `external_yfinance.csv`, `external_weather.csv`, `external_eia_ethanol.csv`.

The bundle makes **no network calls**. Everything runs from local files.
