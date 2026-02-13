# Task 1: Research Existing Code

## Status: COMPLETED

## Approach
Explored both the Marimo notebook (`baybe-resources/notebooks/Goldmining_Demo.py`) and the Streamlit demo (`BayBE_goldmining/streamlit_demo.py`) to understand the current implementations and identify what the individual run demonstration should look like.

## Findings

### Marimo Notebook (before changes)
- Sets up a `GoldMine` landscape, defines BayBE parameters/searchspace/objective
- Jumps directly to `simulate_scenarios` with 30 MC iterations
- Shows a `backtest_plot` with mean + confidence intervals
- No individual run visualization

### Streamlit Demo
- Staged approach with button-driven progression:
  1. Single random search run with animated point-by-point sampling on landscape
  2. "Run many times" to show mean/variance from MC runs
  3. Single BayBE run with animated sampling
  4. MC runs for BayBE to show full comparison
- Uses `plot_landscape_with_samples()` to show numbered sample points on the landscape
- Uses `plot_learning_curves()` to show cumulative best values over iterations

### Shared Utilities (`utils.py`)
- `GoldMine` class with `evaluate()` and `plot()` methods
- `GoldMine.plot()` already supports a `samples` DataFrame parameter with numbered point annotations
- `backtest_plot()` designed for MC simulation DataFrame format (not suitable for single runs)

## Key Insight
The `GoldMine.plot(samples=...)` method already handles everything needed for the landscape visualization. For the learning curve, a simple matplotlib plot is more appropriate than `backtest_plot` since we're showing individual trajectories, not MC statistics.
