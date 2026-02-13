# Refactor mine.evaluate to Accept Only a DataFrame

## Overview

Simplify the `GoldMine.evaluate` method to accept only a pandas DataFrame (as returned by BayBE's `campaign.recommend()`), removing the scalar `(x, y)` calling mode. This makes the API more consistent and directly compatible with BayBE's recommendation workflow.

## Requirement Analysis

### Business Requirements

- `mine.evaluate(df)` should accept a DataFrame with `"x"` and `"y"` columns (as returned by `campaign.recommend()`)
- It should add a `"gold_richness"` column and return the DataFrame
- The scalar mode `mine.evaluate(x, y)` is removed

### Technical Requirements

- Update `GoldMine.evaluate` in `utils.py` to only accept a DataFrame
- Update all call sites in `Goldmining_Demo.py` that used scalar mode
- Ensure `simulate_scenarios` still works (already passes DataFrames)

## Current System Analysis

### High-level approach/architecture

The `evaluate` method currently has dual-mode dispatch:
1. Scalar: `mine.evaluate(x_val, y_val)` returns `float`
2. DataFrame: `mine.evaluate(df)` returns `DataFrame` with `"gold_richness"` column

The notebook's individual run loops extract x,y from the recommendation, call scalar evaluate, then manually assign the result back. This can be simplified to just pass the recommendation DataFrame directly.

### Existing Components to Leverage

- The DataFrame mode of `evaluate` already does exactly what's needed
- `simulate_scenarios` already uses the DataFrame mode as a lookup function

### Key technical decisions and rationale

- Remove `y` parameter entirely, rename `x` to `df` with `pd.DataFrame` type hint
- The individual run loops in the notebook will be simplified: call `mine.evaluate(rec)` directly on the recommendation, then extract the gold value for tracking from the returned DataFrame
- No changes needed for `simulate_scenarios` usage since it already passes DataFrames

### Alternatives considered and why they were rejected

- **Keep both modes**: Rejected because the user explicitly wants DataFrame-only, and it simplifies the API
- **Add a separate method**: Rejected; unnecessary complexity, better to have a single clean method

## Implementation Plan

### Task 1: Refactor evaluate method
**Status**: COMPLETED
**Dependencies**: None
**Protocol**: `.tasks/protocols/evaluate-dataframe-only/Task_01_Refactor_Evaluate.md`

**Description**: Modify `GoldMine.evaluate` in `utils.py` to only accept a DataFrame parameter. Remove scalar mode and `y` parameter.

### Task 2: Update notebook call sites
**Status**: COMPLETED
**Dependencies**: Task 1
**Protocol**: `.tasks/protocols/evaluate-dataframe-only/Task_02_Update_Notebook.md`

**Description**: Update the random search and BayBE optimization loops in `Goldmining_Demo.py` to use the new DataFrame-only API.

### Task 3: Test
**Status**: COMPLETED
**Dependencies**: Task 2
**Protocol**: `.tasks/protocols/evaluate-dataframe-only/Task_03_Test.md`

**Description**: Run `marimo check` and verify the notebook works correctly.
