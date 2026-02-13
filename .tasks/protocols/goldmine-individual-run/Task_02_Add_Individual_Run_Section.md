# Task 2: Add Individual Run Section

## Status: COMPLETED

## Approach
Inserted new cells into `Goldmining_Demo.py` between the problem setup (objective definition) and the MC simulation section. Used manual campaign loops instead of `simulate_scenarios` to get per-iteration sample points for landscape visualization.

## Implementation Details

### Cell Restructuring
- **Extracted shared config cell**: Moved `N_DOE_ITERATIONS`, `N_MC_ITERATIONS`, and `lookup_function` out of the MC simulation cell into their own cell. This allows both the individual run cells and the MC simulation to reference the same iteration count.
- **Split scenarios cell**: The original combined cell (imports + config + lookup + scenarios) was split into: (1) config/lookup cell, (2) individual run cells, (3) scenarios cell.

### New Cells Added (7 total)

1. **Markdown intro** (`hide_code=True`): Introduces the "Single Run Demonstration" section, explains why we look at a single run first, uses `rf"""` string with `{N_DOE_ITERATIONS}` interpolation.

2. **Random search single run**: Creates a `Campaign` with `RandomRecommender`, loops `N_DOE_ITERATIONS` times calling `recommend()` + `add_measurements()`, tracks cumulative best, plots landscape with `mine.plot(samples=...)`. Returns `random_best_values`.

3. **Markdown for BayBE** (`hide_code=True`): Explains BayBE's surrogate model and exploration/exploitation balance.

4. **BayBE single run**: Same structure as random search but with default recommender. Returns `baybe_best_values`.

5. **Markdown for comparison** (`hide_code=True`): Introduces the learning curve comparison.

6. **Comparison learning curve**: Plots both `random_best_values` and `baybe_best_values` on the same axes with theoretical optimum line. Uses `mo.mpl.interactive()` for interactivity.

7. **Updated MC simulation markdown**: Added motivating text explaining why multiple runs are needed after seeing a single run.

### Marimo Variable Isolation
All cell-local variables use `_` prefix to avoid marimo's one-definition-per-variable constraint:
- Imports: `import pandas as _pd`, `from baybe import Campaign as _Campaign`, etc.
- Loop variables: `_rec`, `_x_val`, `_y_val`, `_gold_val`
- Intermediates: `_random_campaign`, `_random_samples`, `_current_best_random`, etc.

Only exported variables (`random_best_values`, `baybe_best_values`, `fig_random`, `fig_baybe`, `scenarios`) keep clean names.

## Blockers Encountered
- **`marimo check` flagged 9 duplicate definitions**: Initial implementation reused variable names across cells (`pd`, `np`, `Campaign`, `RandomRecommender`, `rec`, `x_val`, `y_val`, `gold_val`, `plt`). Fixed by applying `_` prefix convention consistently.

## Testing
- `marimo check` passes with 0 issues (exit code 0, no output)
- User verified interactive run works correctly
