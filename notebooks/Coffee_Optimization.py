import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Espresso Optimization")


@app.cell
def _():
    import marimo as mo
    import warnings

    warnings.filterwarnings("ignore")
    return (mo,)


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using `BayBE` to Optimize Espresso Brewing

    This notebook demonstrates how to use BayBE for optimizing espresso brewing parameters. Making a great espresso is challenging: there are many parameters to tune, and they interact in complex ways. Small changes in grind size, temperature, or extraction time can dramatically affect the taste.

    We'll use BayBE to efficiently explore the parameter space and find optimal brewing conditions that maximize taste quality. This example showcases BayBE's ability to handle both **discrete** and **hybrid** (mixed discrete-continuous) search spaces.

    /// caution
    This notebook was developed for `BayBE` version 0.14.2. Although we do our best in keeping our breaking changes minimal and support outdated versions for a long time, this notebook might not be immediately applicable for other `BayBE` versions.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Espresso Optimization Challenge

    Espresso extraction is a complex process involving multiple parameters:

    1. **Bean Type**: Different beans (Arabica, Robusta, Blend) have different flavor profiles and extraction characteristics.
    2. **Grind Size**: Finer grinds increase extraction but risk over-extraction and channeling.
    3. **Water Pressure**: Typically 7.5-10 bar; affects extraction rate and crema formation.
    4. **Water Temperature**: Higher temperatures extract more compounds but can cause bitterness.
    5. **Brewing Time**: Longer extractions yield more flavor but risk over-extraction.

    With 3 bean types, 5 grind settings, 6 pressure levels, 5 temperatures, and 6 brewing times, we have **2,700 possible combinations**. Our goal is to find excellent espresso (taste score > 8.5) using only a small fraction of experiments.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: Discrete Search Space

    We'll start by modeling all parameters as discrete choices. This is a common scenario when equipment has fixed settings (e.g., grinder with numbered settings, machine with preset temperatures).

    ### Defining the Parameters

    BayBE offers different parameter types. For this example, we use:
    - [`CategoricalParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.categorical.CategoricalParameter.html) for bean type
    - [`NumericalDiscreteParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.numerical.NumericalDiscreteParameter.html) for grind size, pressure, temperature, and time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Understanding Parameter Types

    **CategoricalParameter**: Use for distinct categories without inherent numerical ordering.
    - **Examples**: Bean types, processing methods
    - **Key insight**: "Arabica" is not numerically "between" Robusta and Blend

    **NumericalDiscreteParameter**: Use for numerical values from a finite set where the numerical relationships matter.
    - **Examples**: Fixed temperature settings (88, 90, 92°C), pressure levels from equipment presets
    - **Key insight**: 92°C is closer to 90°C than to 88°C - BayBE uses this structure

    **NumericalContinuousParameter** (Part 2): Use when parameters can take any value within a range.
    - **Examples**: Precise temperature control (88.0-96.0°C), continuously adjustable flow rates
    - **Key insight**: Enables fine-grained optimization - can recommend 92.37°C

    **Note**: Chemical substances (solvents, catalysts, ligands) are best modeled using `SubstanceParameter` (see the Reaction Optimization example), which can leverage chemical descriptors for better predictions.
    """)
    return


@app.cell
def _():
    from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter

    # Categorical parameters
    bean_type = CategoricalParameter(
        name="bean_type",
        values=["Arabica", "Robusta", "Blend"],
        encoding="OHE"
    )

    # Numerical discrete parameters
    # Grind size in microns (typical espresso range: 200-400 microns)
    grind_size = NumericalDiscreteParameter(
        name="grind_size",
        values=[200, 250, 300, 350, 400],
        tolerance=5.0
    )

    water_pressure = NumericalDiscreteParameter(
        name="water_pressure",
        values=[7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        tolerance=0.1
    )

    water_temperature = NumericalDiscreteParameter(
        name="water_temperature",
        values=[88, 90, 92, 94, 96],
        tolerance=0.5
    )

    brewing_time = NumericalDiscreteParameter(
        name="brewing_time",
        values=[20, 23, 26, 29, 32, 35],
        tolerance=0.5
    )
    return (
        bean_type,
        brewing_time,
        grind_size,
        water_pressure,
        water_temperature,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Creating the Search Space

    We combine the parameters into a [`SearchSpace`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.searchspace.core.SearchSpace.html) using the Cartesian product of all parameter values.
    """)
    return


@app.cell
def _(
    bean_type,
    brewing_time,
    grind_size,
    water_pressure,
    water_temperature,
):
    from baybe.searchspace import SearchSpace

    discrete_parameters = [
        bean_type,
        grind_size,
        water_pressure,
        water_temperature,
        brewing_time,
    ]

    searchspace_discrete = SearchSpace.from_product(parameters=discrete_parameters)

    print(f"Search space size: {len(searchspace_discrete.discrete.exp_rep)} combinations")
    return SearchSpace, searchspace_discrete


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Defining the Objective

    We want to maximize the taste score, which ranges from 1 (awful) to 10 (awesome). We use a [`SingleTargetObjective`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.objectives.single.SingleTargetObjective.html) with a [`NumericalTarget`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.targets.numerical.NumericalTarget.html).
    """)
    return


@app.cell
def _():
    from baybe.targets import NumericalTarget
    from baybe.objectives import SingleTargetObjective

    target = NumericalTarget(name="taste")
    objective = SingleTargetObjective(target=target)
    return (objective,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Creating the Campaign

    The [`Campaign`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.campaign.Campaign.html) combines the search space and objective. We use the default [`TwoPhaseMetaRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.meta.sequential.TwoPhaseMetaRecommender.html), which:
    - Initially uses random sampling for diverse exploration
    - Switches to Bayesian optimization once data is available
    """)
    return


@app.cell
def _(objective, searchspace_discrete):
    from baybe.campaign import Campaign

    campaign_discrete = Campaign(
        searchspace=searchspace_discrete,
        objective=objective
    )
    return Campaign, campaign_discrete


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Running the Optimization Loop

    We'll run the optimization iteratively. In each iteration:
    1. Get a recommendation from `BayBE` (one espresso)
    2. Brew and taste the espressos (evaluate the hidden taste function)
    3. Add the measurements back to the campaign
    4. Repeat

    We start with 5 initial random experiments, then perform 20 additional iterations of 1 recommendation each (25 total experiments).
    """)
    return


@app.cell
def _(campaign_discrete):
    # Get initial recommendations (5 random experiments)
    initial_recommendations = campaign_discrete.recommend(batch_size=5)
    initial_recommendations
    return (initial_recommendations,)


@app.cell
def _(initial_recommendations):
    from utils import espresso_taste

    # Evaluate the taste for initial recommendations
    initial_recommendations["taste"] = initial_recommendations.apply(
        lambda row: espresso_taste(
            bean_type=row["bean_type"],
            grind_size=row["grind_size"],
            water_pressure=row["water_pressure"],
            water_temperature=row["water_temperature"],
            brewing_time=row["brewing_time"],
        ),
        axis=1
    )

    initial_recommendations
    return (espresso_taste,)


@app.cell
def _(campaign_discrete, initial_recommendations):
    # Add measurements to campaign
    campaign_discrete.add_measurements(initial_recommendations)
    return


@app.cell
def _(campaign_discrete, espresso_taste, mo):
    # Run optimization iterations (one at a time)
    for iteration in mo.status.progress_bar(
        range(20),
        title="Optimizing espresso parameters (discrete)",
    ):
        # Get recommendation (one espresso)
        recommendations = campaign_discrete.recommend(batch_size=1)

        # Evaluate taste
        recommendations["taste"] = recommendations.apply(
            lambda row: espresso_taste(
                bean_type=row["bean_type"],
                grind_size=row["grind_size"],
                water_pressure=row["water_pressure"],
                water_temperature=row["water_temperature"],
                brewing_time=row["brewing_time"],
            ),
            axis=1
        )

        # Add to campaign
        campaign_discrete.add_measurements(recommendations)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Results: Discrete Search Space

    Let's examine the results. We can see all measurements and identify the best espresso found.
    """)
    return


@app.cell
def _(campaign_discrete):
    measurements_discrete = campaign_discrete.measurements
    best_discrete = measurements_discrete.loc[measurements_discrete["taste"].idxmax()]
    print(f"Best discrete taste: {best_discrete['taste']:.1f}")
    measurements_discrete
    return best_discrete, measurements_discrete


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Taste Distribution Across the Search Space

    To understand the difficulty of the optimization, let's look at how taste scores
    are distributed when we evaluate every point in the discrete search space.
    """)
    return


@app.cell
def _(espresso_taste, mo, searchspace_discrete):
    import matplotlib.pyplot as _plt
    import numpy as _np

    # Evaluate taste for every point in the discrete search space
    all_params = searchspace_discrete.discrete.exp_rep
    all_tastes = all_params.apply(
        lambda row: espresso_taste(
            bean_type=row["bean_type"],
            grind_size=row["grind_size"],
            water_pressure=row["water_pressure"],
            water_temperature=row["water_temperature"],
            brewing_time=row["brewing_time"],
        ),
        axis=1,
    )

    _fig, _ax = _plt.subplots(figsize=(8, 4))
    _ax.hist(all_tastes, bins=_np.arange(1, 11, 0.5), edgecolor="white", alpha=0.8)
    _ax.set(xlabel="Taste Score", ylabel="Count", title="Taste Distribution (Full Discrete Space)")
    _ax.grid(True, alpha=0.3, axis="y")
    _plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: Hybrid Search Space

    Modern espresso machines allow precise temperature and time control. We'll now convert two parameters to continuous ranges while keeping grind size discrete (since grinders typically have fixed burr settings):
    - **Grind size**: Stays discrete (200-400 microns in 50 micron steps)
    - **Water temperature**: 88°C to 96°C (continuous)
    - **Brewing time**: 20 to 35 seconds (continuous)

    This creates a **hybrid search space** with both discrete (categorical, grind, pressure) and continuous (temperature, time) parameters.
    """)
    return


@app.cell
def _(SearchSpace, bean_type, grind_size, water_pressure):
    from baybe.parameters import NumericalContinuousParameter

    # Convert some parameters to continuous
    water_temperature_continuous = NumericalContinuousParameter(
        name="water_temperature",
        bounds=(88.0, 96.0)
    )

    brewing_time_continuous = NumericalContinuousParameter(
        name="brewing_time",
        bounds=(20.0, 35.0)
    )

    # Create hybrid search space (grind_size stays discrete!)
    hybrid_parameters = [
        bean_type,
        grind_size,  # Keep grind size discrete
        water_pressure,
        water_temperature_continuous,
        brewing_time_continuous,
    ]

    searchspace_hybrid = SearchSpace.from_product(parameters=hybrid_parameters)
    return (searchspace_hybrid,)


@app.cell
def _(Campaign, objective, searchspace_hybrid):
    # Create new campaign with hybrid search space
    campaign_hybrid = Campaign(
        searchspace=searchspace_hybrid,
        objective=objective
    )
    return (campaign_hybrid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Running Hybrid Optimization

    We'll run the same optimization process with the hybrid search space.
    """)
    return


@app.cell
def _(campaign_hybrid, espresso_taste):
    # Initial recommendations for hybrid space (5 random experiments)
    initial_recommendations_hybrid = campaign_hybrid.recommend(batch_size=5)

    initial_recommendations_hybrid["taste"] = initial_recommendations_hybrid.apply(
        lambda row: espresso_taste(
            bean_type=row["bean_type"],
            grind_size=row["grind_size"],
            water_pressure=row["water_pressure"],
            water_temperature=row["water_temperature"],
            brewing_time=row["brewing_time"],
        ),
        axis=1
    )

    campaign_hybrid.add_measurements(initial_recommendations_hybrid)

    initial_recommendations_hybrid
    return (initial_recommendations_hybrid,)


@app.cell
def _(campaign_hybrid, espresso_taste, mo):
    # Run optimization iterations for hybrid space (one espresso at a time)
    for iteration_hybrid in mo.status.progress_bar(
        range(15),
        title="Optimizing espresso parameters (hybrid)",
    ):
        recommendation_hybrid = campaign_hybrid.recommend(batch_size=1)

        recommendation_hybrid["taste"] = recommendation_hybrid.apply(
            lambda row: espresso_taste(
                bean_type=row["bean_type"],
                grind_size=row["grind_size"],
                water_pressure=row["water_pressure"],
                water_temperature=row["water_temperature"],
                brewing_time=row["brewing_time"],
            ),
            axis=1
        )

        campaign_hybrid.add_measurements(recommendation_hybrid)
    return (iteration_hybrid, recommendation_hybrid)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Results: Hybrid Search Space

    Let's examine the hybrid optimization results and compare with the discrete approach.
    """)
    return


@app.cell
def _(campaign_hybrid):
    measurements_hybrid = campaign_hybrid.measurements
    best_hybrid = measurements_hybrid.loc[measurements_hybrid["taste"].idxmax()]
    print(f"Best hybrid taste: {best_hybrid['taste']:.1f}")
    measurements_hybrid
    return best_hybrid, measurements_hybrid


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Comparison: Discrete vs. Hybrid

    Let's visualize how both approaches performed over the course of optimization.
    """)
    return


@app.cell
def _(measurements_discrete, measurements_hybrid, mo, pd):
    import matplotlib.pyplot as _plt
    import seaborn as _sns

    # Build cumulative-best data
    disc = measurements_discrete[["taste"]].copy()
    disc["approach"], disc["experiment"] = "Discrete", range(len(disc))
    disc["best"] = disc["taste"].cummax()

    hyb = measurements_hybrid[["taste"]].copy()
    hyb["approach"], hyb["experiment"] = "Hybrid", range(len(hyb))
    hyb["best"] = hyb["taste"].cummax()

    combined = pd.concat([disc, hyb])

    _fig, _ax = _plt.subplots(figsize=(8, 5))
    _sns.lineplot(data=combined, x="experiment", y="best", hue="approach", marker="o", markersize=5, ax=_ax)
    _ax.set(xlabel="Experiment", ylabel="Best Taste Score", title="Optimization Progress")
    _ax.grid(True, alpha=0.3)
    _plt.tight_layout()
    mo.mpl.interactive(_fig)
    return





if __name__ == "__main__":
    app.run()
