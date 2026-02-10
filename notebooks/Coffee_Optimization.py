import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Espresso Optimization")


@app.cell
def _():
    import marimo as mo
    import warnings

    warnings.filterwarnings("ignore")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using `BayBE` to Optimize Espresso Brewing

    This notebook demonstrates how to use `BayBE` for optimizing espresso brewing parameters. Making a great espresso is challenging: there are many parameters to tune, and they interact in complex ways. Small changes in grind size, temperature, or extraction time can dramatically affect the taste.

    We'll use `BayBE` to efficiently explore the parameter space and find optimal brewing conditions that maximize taste quality. This example also showcases that the same problem can be modeled in different ways. For this, it uses `BayBE`'s ability to handle both **discrete** and **hybrid** search spaces.

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

    In this setting, we will use 3 bean types, 5 grind settings, 6 pressure levels, 5 temperatures, and 6 brewing times. This means that we have **2,700 possible combinations** from which we want to find an excellent espresso with only a small fraction of experiments.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modelling the problem with discrete parameters

    We'll start by modeling all parameters as discrete choices. This is a common scenario when equipment has fixed settings (e.g., grinder with numbered settings, machine with preset temperatures).

    `BayBE` offers a wide range of different parameter types:

    - **[`CategoricalParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.categorical.CategoricalParameter.html)**: These are used for distinct categories without inherent numerical ordering. In our example, we use them for the bean type as "Arabica" is not numerically "between" Robusta and Blend.
    - **[`NumericalDiscreteParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.numerical.NumericalDiscreteParameter.html)**: These are used for numerical values from a finite set where the numerical relationships matter. In our example, these are all other parameters that we described earlier. The key dfference when comparing those parameters with `CategoricalParameter`s is that the numbers carry a meaning: 92°C is closer to 90°C than to 88°C, and `BayBE` uses this structure.
    - **[`SubstanceParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.categorical.SubstanceParameter.html)**: These are used for modeling chemical substances like solvents, catalysts, ligands, and enable `BayBE` to leverage chemical descriptors for better predictions. More details on this kind of parameters can be found in the `ReactionOptimization` example.

    For more details, we refer to the [user guide on parameters](https://emdgroup.github.io/baybe/0.14.2/userguide/parameters.html).
    """)
    return


@app.cell
def _():
    from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter

    bean_type = CategoricalParameter(
        name="bean_type",
        values=["Arabica", "Robusta", "Blend"],
        encoding="OHE"
    )

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
def _(bean_type, brewing_time, grind_size, water_pressure, water_temperature):
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

    The [`Campaign`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.campaign.Campaign.html) combines the search space and objective. We could also specify the [recommender](https://emdgroup.github.io/baybe/0.14.2/userguide/recommenders.html) to use here, but we use the default [`TwoPhaseMetaRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.meta.sequential.TwoPhaseMetaRecommender.html). This recommender initially uses random sampling and switches to a Bayesian optimizer once data is available
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
    1. Get a recommendation from `BayBE` for one espresso
    2. "Brew" and "taste" the espressos. This is done by evaluating the hidden `espresso_taste` function which simulates us drinking the coffee and giving it a rating on a scale from 1 to 10.
    3. Add the measurements to the campaign.
    4. Repeat a certain number of times.

    We start with 5 initial random experiments, then perform 20 additional iterations of 1 recommendation each (25 total experiments).
    """)
    return


@app.cell
def _(campaign_discrete):
    from utils import espresso_taste

    initial_recommendations = campaign_discrete.recommend(batch_size=5)
    initial_recommendations = espresso_taste(initial_recommendations)
    initial_recommendations
    return espresso_taste, initial_recommendations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add the recommendation together with the measured target value to the campaign before beginning the iterative optimization.
    """)
    return


@app.cell
def _(campaign_discrete, initial_recommendations):
    campaign_discrete.add_measurements(initial_recommendations)
    return


@app.cell
def _(campaign_discrete, espresso_taste, mo):
    for iteration in mo.status.progress_bar(
        range(20),
        title="Optimizing your espresso",
    ):
        recommendations = campaign_discrete.recommend(batch_size=1)
        recommendations = espresso_taste(recommendations)
        campaign_discrete.add_measurements(recommendations)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Optimization Progress: Discrete Search Space

    Let's visualize how the optimization progressed by plotting the cumulative best taste score found over the course of the experiments.
    """)
    return


@app.cell(hide_code=True)
def _(campaign_discrete, mo, plt):
    # Plot cumulative best taste score
    discrete_measurements = campaign_discrete.measurements.copy()
    discrete_measurements["cumulative_best"] = discrete_measurements["taste"].cummax()

    discrete_progress_fig, discrete_progress_ax = plt.subplots(figsize=(8, 4))
    discrete_progress_ax.plot(
        range(len(discrete_measurements)), 
        discrete_measurements["cumulative_best"], 
        marker="o", 
    )
    discrete_progress_ax.set(
        xlabel="Experiment Number", 
        ylabel="Best Taste Score Found", 
        title="Cumulative Best Result"
    )
    discrete_progress_ax.grid(True, alpha=0.3)
    discrete_progress_ax.axhline(y=8.5, color='red', linestyle='--', alpha=0.5, label='Excellent threshold (8.5)')
    discrete_progress_ax.legend()
    plt.tight_layout()
    mo.mpl.interactive(discrete_progress_fig)
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
    return (measurements_discrete,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Taste Distribution Across the Search Space

    To understand the difficulty of the optimization, let's look at how taste scores
    are distributed when we evaluate every point in the discrete search space.
    """)
    return


@app.cell(hide_code=True)
def _(espresso_taste, mo, searchspace_discrete):
    import matplotlib.pyplot as plt
    import numpy as np

    all_params = searchspace_discrete.discrete.exp_rep
    all_tastes = espresso_taste(all_params)["taste"]

    taste_fig, taste_ax = plt.subplots(figsize=(8, 4))
    taste_ax.hist(all_tastes, bins=np.arange(1, 11, 0.1), edgecolor="white", alpha=0.8)
    taste_ax.set(xlabel="Taste Score", ylabel="Count", title="Taste Distribution (Full Discrete Space)")
    taste_ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    mo.mpl.interactive(taste_fig)
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Alternative modelling: Using discrete and continuous parameters

    Modern espresso machines allow precise temperature and time control. This opens up the possibility to model our process differently by modelling the `NumericalDiscreteParameter`s continuously instead.

    This creates a **hybrid search space** with both discrete (categorical) and continuous (all others) parameters.
    """)
    return


@app.cell
def _(SearchSpace, bean_type):
    from baybe.parameters import NumericalContinuousParameter

    water_temperature_continuous = NumericalContinuousParameter(
        name="water_temperature",
        bounds=(88.0, 96.0)
    )

    brewing_time_continuous = NumericalContinuousParameter(
        name="brewing_time",
        bounds=(20.0, 35.0)
    )

    grind_size_continuous = NumericalContinuousParameter(
        name="grind_size",
        bounds=(200, 400)
    )

    water_pressure_continuous = NumericalContinuousParameter(
        name="water_pressure",
        bounds=(7.5, 10.0)
    )

    hybrid_parameters = [
        bean_type,
        grind_size_continuous,
        water_pressure_continuous,
        water_temperature_continuous,
        brewing_time_continuous,
    ]

    searchspace_hybrid = SearchSpace.from_product(parameters=hybrid_parameters)
    return (searchspace_hybrid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `BayBE` offers two different ways of doing hybrid optimization:
    1. The [`NaiveHybridSpaceRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.naive.NaiveHybridSpaceRecommender.html) optimizes the discrete and the continuous parts of the search space independently and then combines the best found results.
    2. The [`BotorchRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.pure.bayesian.botorch.BotorchRecommender.html) uses a brute-force optimization that can be computationally expensive for larger discrete subspaces.

    As our discrete space only consists of a single parameter with only three different values, we use the `BotorchRecommender` in the following.
    """)
    return


@app.cell
def _(Campaign, objective, searchspace_hybrid):
    from baybe.recommenders import BotorchRecommender, TwoPhaseMetaRecommender

    campaign_hybrid = Campaign(
        searchspace=searchspace_hybrid,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(recommender=BotorchRecommender())
    )
    return (campaign_hybrid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Running Hybrid Optimization

    We'll run the same optimization process with the hybrid search space. To be fair, we provide the hybrid campaign with the same set of initial recommendations before running the optimization.
    """)
    return


@app.cell
def _(campaign_hybrid, initial_recommendations):
    campaign_hybrid.add_measurements(initial_recommendations)
    return


@app.cell
def _(campaign_hybrid, espresso_taste, mo):

    for iteration_hybrid in mo.status.progress_bar(
        range(20),
        title="Optimizing espresso parameters (hybrid)",
    ):
        recommendation_hybrid = campaign_hybrid.recommend(batch_size=1)
        recommendation_hybrid = espresso_taste(recommendation_hybrid)
        campaign_hybrid.add_measurements(recommendation_hybrid)
    return


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
    return (measurements_hybrid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Comparison: Discrete vs. Hybrid

    Let's visualize how both approaches performed over the course of optimization.
    """)
    return


@app.cell(hide_code=True)
def _(measurements_discrete, measurements_hybrid, mo, plt):
    import seaborn as sns
    import pandas as pd

    disc = measurements_discrete[["taste"]].copy()
    disc["approach"], disc["experiment"] = "Discrete", range(len(disc))
    disc["best"] = disc["taste"].cummax()

    hyb = measurements_hybrid[["taste"]].copy()
    hyb["approach"], hyb["experiment"] = "Hybrid", range(len(hyb))
    hyb["best"] = hyb["taste"].cummax()

    combined = pd.concat([disc, hyb])

    hybrid_fig, hybrid_ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=combined, x="experiment", y="best", hue="approach", marker="o",ax=hybrid_ax)
    hybrid_ax.set(xlabel="Experiment", ylabel="Best Taste Score", title="Optimization Progress")
    hybrid_ax.axhline(y=8.5, color='red', linestyle='--', alpha=0.5, label='Excellent threshold (8.5)')
    hybrid_ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.mpl.interactive(hybrid_fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
