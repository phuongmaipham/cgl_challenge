# Strategy Backtest Research Bundle

One-week research handoff for the grain strategy notebooks. The notebooks are
the main deliverable; the Python files only keep repeated loading, feature, and
backtest code out of the cells.

## Files

- `corn_strategy_backtest_research.ipynb`
- `soybean_strategy_backtest_research.ipynb`
- `wheat_strategy_backtest_research.ipynb`
- `grain_portfolio_backtest_research.ipynb`
- `grain_backtest_core.py` - train-set loading, feature panels, costs, metrics
- `shared_backtest.py` - shared notebook backtest helpers
- `corn_signals.py` - corn signal construction from bundled data files
- `corn_backtest.py` - corn candidate tables and guard checks
- `research_config.py` - constants used by the notebooks
- `train_set/` - local CSV inputs

## Run

From this folder:

```bash
python3 -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute corn_strategy_backtest_research.ipynb --inplace
jupyter nbconvert --to notebook --execute soybean_strategy_backtest_research.ipynb --inplace
jupyter nbconvert --to notebook --execute wheat_strategy_backtest_research.ipynb --inplace
jupyter nbconvert --to notebook --execute grain_portfolio_backtest_research.ipynb --inplace
```

The notebooks assume the working directory is this folder, so `DATA_DIR = "train_set"` resolves locally.

## Current promoted rows

- Corn: `wf_momentum_mr + cargill_disagree_flat + below_ma_or_negative_mom_flat`
- Soybean: `wf_momentum_mr + cargill_disagree_half + low_vol_flat`
- Wheat: `pair_price_mr_cargill_trend_conflict_flat_cost_control`
- Portfolio: simple shared-score outright/spread check across corn, soybean, and wheat
