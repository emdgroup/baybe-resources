# Task 1: Refactor evaluate method

## Approach

Removed the dual-mode (scalar + DataFrame) dispatch from `GoldMine.evaluate` and made it accept only a `pd.DataFrame` parameter.

## Changes Made

### `notebooks/utils.py` - `GoldMine.evaluate`
- Removed `y` parameter
- Renamed `x` parameter to `df` with `pd.DataFrame` type hint
- Removed `isinstance` check and scalar code path
- Kept the interpolation logic for DataFrame mode unchanged
- Updated docstring to reflect the new single-mode API

### `notebooks/Goldmining_Demo.py` - Random search loop
- Replaced `mine.evaluate(x_val, y_val)` scalar calls with `mine.evaluate(random_rec)` DataFrame call
- Extracted gold value from the returned DataFrame's `"gold_richness"` column
- Removed redundant duplicate `mine.evaluate` call (was called twice before)

### `notebooks/Goldmining_Demo.py` - BayBE optimization loop
- Same pattern: replaced scalar evaluate with DataFrame evaluate
- The workflow is now: `recommend -> evaluate -> add_measurements`

## Key Insight

The `simulate_scenarios` call on line 332 already passed DataFrames to `mine.evaluate` as a lookup function, so no changes were needed there.
