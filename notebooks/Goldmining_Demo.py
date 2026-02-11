import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", app_title="Gold Mining Optimization")


@app.cell
def _():
    import marimo as mo
    import warnings

    warnings.filterwarnings("ignore")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gold Mining Optimization

    This notebook demonstrates Bayesian optimization using `BayBE` through a gold mining metaphor. We compare random search with `BayBE`'s intelligent optimization approach on a 2D landscape containing multiple gold-rich regions.

    ## Overview

    The demo simulates searching for gold in an unknown 2D landscape:

    1. **Random Search Baseline**: Random sampling explores the landscape without strategy
    2. **`BayBE` Optimization**: Intelligent exploration that learns from each sample
    3. **Performance Comparison**: Quantitative comparison across multiple independent runs

    This example translates directly to real-world optimization problems such as:
    - Experimental parameter optimization (temperature, pH, concentration)
    - Formulation design (ingredient ratios)
    - Process optimization (reaction conditions)

    /// caution
    This notebook was developed for `BayBE` version 0.14.2. Although we do our best in keeping our breaking changes minimal and support outdated versions for a long time, this notebook might not be immediately applicable for other `BayBE` versions. If you install `BayBE` via the instructions in this repository, version 0.14.2 will thus be installed.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up the Landscape

    We create a 2D landscape with multiple Gaussian peaks representing gold-rich regions. This simulates an unknown terrain where we want to find the highest gold concentration using as few samples as possible.
    """)
    return


@app.cell
def _():
    from utils import GoldMine

    mine = GoldMine(grid_size=100, n_peaks=5, noise_level=0.1)
    return (mine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Let's visualize the landscape. The brighter regions indicate higher gold concentrations.
    """)
    return


@app.cell(hide_code=True)
def _(mine):
    fig_landscape = mine.plot(title="Gold Mining Landscape")
    fig_landscape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining the Optimization Problem

    To compare random search with `BayBE`, we need to set up the optimization campaign components.

    ### Define the Parameters

    We define two continuous parameters representing the 2D coordinates of our mining landscape. `BayBE` uses [`NumericalContinuousParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.numerical.NumericalContinuousParameter.html) for parameters that can take any value within a specified range.
    """)
    return


@app.cell
def _():
    from baybe.parameters import NumericalContinuousParameter

    parameters = [
        NumericalContinuousParameter(name="x", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="y", bounds=(0.0, 1.0)),
    ]
    return (parameters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define the Search Space

    The [`SearchSpace`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.searchspace.html) defines all possible parameter combinations that can be explored. We use [`SearchSpace.from_product`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.searchspace.core.SearchSpace.html#baybe.searchspace.core.SearchSpace.from_product) to create the Cartesian product of our parameters. In this example, our search space is effectively just the unit square in two dimensions.
    """)
    return


@app.cell
def _(parameters):
    from baybe.searchspace import SearchSpace

    searchspace = SearchSpace.from_product(parameters=parameters)
    return (searchspace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define the Target and Objective

    The [`NumericalTarget`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.targets.numerical.NumericalTarget.html) represents the quantity we want to optimize (gold richness). We wrap it in a [`SingleTargetObjective`](https://emdgroup.github.io/baybe/0.14.2/userguide/objectives.html#singletargetobjective) since we're optimizing a single objective. By default, `BayBE` maximizes the target.
    """)
    return


@app.cell
def _():
    from baybe.targets import NumericalTarget
    from baybe.objectives import SingleTargetObjective

    target = NumericalTarget(name="gold_richness")
    objective = SingleTargetObjective(target=target)
    return (objective,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now use a `RandomRecommender` to get 20 random recommendations and investigate them.
    """)
    return


@app.cell
def _(mine, objective, searchspace):
    import pandas as pd
    import numpy as np
    from baybe import Campaign
    from baybe.recommenders import RandomRecommender

    random_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=RandomRecommender(),
    )

    random_samples = []
    random_best_values = []
    current_best_random = -np.inf

    for _ in range(20):
        random_rec = random_campaign.recommend(batch_size=1)
        random_rec = mine.evaluate(random_rec)
        random_campaign.add_measurements(random_rec)

        random_gold_val = random_rec["gold_richness"].iloc[0]
        random_samples.append({"x": random_rec["x"].iloc[0], "y": random_rec["y"].iloc[0], "value": random_gold_val})
        current_best_random = max(current_best_random, random_gold_val)
        random_best_values.append(current_best_random)

    random_samples_df = pd.DataFrame(random_samples)

    fig_random = mine.plot(
        samples=random_samples_df,
        title="Random Search Samples",
    )
    fig_random
    return Campaign, RandomRecommender, np, pd, random_best_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `BayBE` Optimization

    `BayBE` uses a surrogate model to predict the landscape and an acquisition function to decide where to sample next. It balances **exploration** (sampling uncertain regions) with **exploitation** (refining known promising areas).
    """)
    return


@app.cell
def _(Campaign, mine, np, objective, pd, searchspace):
    baybe_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )

    baybe_samples = []
    baybe_best_values = []
    current_best_baybe = -np.inf

    for _i in range(20):
        baybe_rec = baybe_campaign.recommend(batch_size=1)
        baybe_rec = mine.evaluate(baybe_rec)
        baybe_campaign.add_measurements(baybe_rec)

        baybe_gold_val = baybe_rec["gold_richness"].iloc[0]
        baybe_samples.append({"x": baybe_rec["x"].iloc[0], "y": baybe_rec["y"].iloc[0], "value": baybe_gold_val})
        current_best_baybe = max(current_best_baybe, baybe_gold_val)
        baybe_best_values.append(current_best_baybe)

    baybe_samples_df = pd.DataFrame(baybe_samples)

    fig_baybe = mine.plot(
        samples=baybe_samples_df,
        title="BayBE Optimization Samples",
    )
    fig_baybe
    return (baybe_best_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Single Run Comparison

    The learning curve below compares the best gold richness found so far at each iteration for both methods. The dashed line indicates the theoretical optimum (the maximum value in the landscape).

    Notice how `BayBE` typically finds high-richness regions faster and reaches a better final value compared to random search.
    """)
    return


@app.cell
def _(baybe_best_values, mine, mo, np, random_best_values):
    import matplotlib.pyplot as plt

    fig_comparison, ax_comparison = plt.subplots(figsize=(10, 6))
    iterations = np.arange(1, len(random_best_values) + 1)

    ax_comparison.plot(
        iterations,
        random_best_values,
        "o-",
        label="Random Search",
    )
    ax_comparison.plot(
        iterations,
        baybe_best_values,
        "s-",
        label="BayBE Optimization",
    )

    theoretical_max = np.max(mine.Z)
    ax_comparison.axhline(
        y=theoretical_max,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Theoretical Optimum",
    )

    ax_comparison.set_xlabel("Number of Mining Attempts", fontsize=12)
    ax_comparison.set_ylabel("Best Gold Richness Found", fontsize=12)
    ax_comparison.set_title("Single Run: Random Search vs BayBE", fontsize=14, fontweight="bold")
    ax_comparison.legend(loc="lower right", fontsize=10)
    ax_comparison.grid(True, alpha=0.3)

    mo.mpl.interactive(fig_comparison)
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running Simulated Campaigns

    A single run can be misleading due to randomness. To get a statistically robust comparison, we now run both methods multiple times using `BayBE`'s simulation utilities. This averages over the randomness in the optimization process and provides confidence intervals.

    We'll set up two scenarios:
    1. **Random Search**: Uses [`RandomRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.pure.nonpredictive.sampling.RandomRecommender.html) for pure random sampling
    2. **`BayBE` Optimization**: Uses the default recommender for intelligent Bayesian optimization
    """)
    return


@app.cell
def _(Campaign, RandomRecommender, objective, searchspace):
    scenarios = {
        "Random Search": Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=RandomRecommender()
        ),
        "BayBE Optimization": Campaign(
            searchspace=searchspace,
            objective=objective,
        ),
    }
    return (scenarios,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run the Simulations

    We use [`simulate_scenarios`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.simulation.scenarios.html#baybe.simulation.scenarios.simulate_scenarios) to execute the optimization campaigns. This function runs each scenario multiple times and collects performance statistics.

    /// note
    When running the simulations, you may see warnings from `RandomRecommender` about unused objectives and measurements. This is expected and harmless. `RandomRecommender` samples points uniformly at random without considering the objective function or learning from previous measurements, which is why these inputs are ignored. The warnings serve as a reminder that random search does not utilize the optimization machinery that makes `BayBE` effective.
    ///
    """)
    return


@app.cell
def _(mine, scenarios):
    from baybe.simulation import simulate_scenarios

    N_DOE_ITERATIONS = 30  # Number of optimization iterations per run
    N_MC_ITERATIONS = 30  # Number of Monte Carlo runs

    results = simulate_scenarios(
        scenarios,
        mine.evaluate,
        batch_size=1,
        n_doe_iterations=N_DOE_ITERATIONS,
        n_mc_iterations=N_MC_ITERATIONS,
    )

    results.rename(
        columns={
            "Scenario": "Method",
            "Num_Experiments": "Number of Mining Attempts",
            "gold_richness_CumBest": "Best Gold Richness Found",
        },
        inplace=True,
    )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performance Comparison

    Let's visualize how both methods performed across multiple independent runs. The plot shows the mean performance (solid line) with confidence intervals (shaded regions) representing the variability across different optimization runs.
    """)
    return


@app.cell
def _(mo, plt, results):
    from utils import backtest_plot

    backtest_plot(
        df=results,
        x="Number of Mining Attempts",
        y="Best Gold Richness Found",
        hue="Method",
    )
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot demonstrates that `BayBE` consistently outperforms random search by:
    - **Faster convergence**: `BayBE` reaches high-quality solutions with fewer mining attempts
    - **Better final performance**: `BayBE` typically finds regions with higher gold concentrations
    - **More reliable**: Smaller confidence intervals indicate more consistent performance

    This advantage comes from `BayBE`'s ability to build a surrogate model of the landscape and intelligently balance exploration (discovering new regions) with exploitation (refining known promising areas). In contrast, random search does not learn from previous observations and continues to sample uniformly across the entire search space.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Real-World Applications

    This gold mining demo translates directly to optimization problems across various domains:

    | Demo Concept | Real-World Application |
    |--------------|------------------------|
    | Gold mining location (x, y) | Experimental parameters (temperature, pH, concentration) |
    | Gold richness value | Assay result, yield, purity, activity |
    | Number of attempts | Lab experiments (time and cost expensive) |
    | Random search | Traditional trial-and-error |
    | `BayBE` optimization | Smart experiment design |
    | Multiple Monte Carlo runs | Accounting for experimental variability |


    For more information, see the [`BayBE` documentation](https://emdgroup.github.io/baybe/).
    """)
    return


if __name__ == "__main__":
    app.run()
