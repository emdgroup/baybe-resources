# Goldmine Individual Run - Implementation Plan

## Status: COMPLETED

## Problem Statement
The Goldmining_Demo.py Marimo notebook only demonstrated BayBE vs Random Search through a full MC simulation with multiple runs. It needed to first demonstrate a single individual run (like the Streamlit demo in BayBE_goldmining) to build intuition before showing the statistical comparison.

## High-Level Approach
Inserted new cells between the "Defining the Optimization Problem" section and the "Running Simulated Campaigns" section:
1. Run a single random search campaign showing points on the landscape
2. Run a single BayBE campaign showing points on the landscape
3. Compare the two single runs via a learning curve plot

## Tasks
1. [x] Research existing code and understand both implementations
2. [x] Add individual run section with markdown intro, single random search, single BayBE run, landscape plots, and comparison learning curve
3. [x] Test the notebook (`marimo check` passes clean, interactive run verified by user)

## Key Technical Decisions
- Reuse the existing `mine` object, `parameters`, `searchspace`, and `objective` from earlier cells
- Create fresh Campaign objects for the individual runs (campaigns are stateful and can't be reused for the MC simulation)
- Use the existing `GoldMine.plot()` method for landscape visualization with sample points
- Add a simple learning curve comparison using matplotlib directly (the existing `backtest_plot` from utils is designed for the MC simulation DataFrame format)
- Extracted `N_DOE_ITERATIONS`, `N_MC_ITERATIONS`, and `lookup_function` into a shared cell so both the individual run and MC simulation sections use the same values
- Used `_` prefixed variables for all cell-local imports and intermediates to satisfy marimo's one-definition-per-variable rule

## Alternatives Considered
- **Using `simulate_scenarios` with `n_mc_iterations=1`**: Rejected because it doesn't give us access to the individual sample points for landscape plotting, and the DataFrame format doesn't match what we need for step-by-step visualization
- **Adding animation**: Rejected since Marimo notebooks aren't naturally suited for step-by-step animation like Streamlit; static plots with all points are more appropriate

## Files Modified
- `notebooks/Goldmining_Demo.py` — Added 7 new cells (4 code + 3 markdown), refactored 1 existing cell
