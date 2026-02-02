import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Reaction Optimization")


@app.cell
def _():
    import marimo as mo
    import warnings

    warnings.filterwarnings("ignore")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using BayBE to optimize Reaction Conditions

    This notebook contains an example on how to use BayBE for the optimization of reaction conditions. It is inspired by the corresponding notebook developed by Pat Walters as part of his [Practical Cheminformatics Tutorial](https://github.com/PatWalters/practical_cheminformatics_tutorials). This notebook assumes basic familiarity with the core concepts of Bayesian Optimization. The intention of this notebook is *not* to introduce and explain all aspects of Bayesian Optimization, but to focus on the usage of BayBE.

    In drug discovery, we frequently encounter situations where we need to modify a set of reaction conditions to optimize the yield. This notebook shows how to use BayBE to model and optimize such a campaign.

    # Chemical encodings

    This notebook demonstrates the power and usefulness of BayBE's chemical encodings. If parameters in a process to be optimized are chemicals, this feature enables BayBE to automatically use meaningful chemical descriptors, automatically leveraging chamical knowledge for the optimization process.

    This notebook assumes some basic familiarity with using BayBE, and that it does not explain all of the core concepts. If you are interested in those, we recommend to first check out the `Reation_Optimization` example.

    /// caution
    This notebook was developed for BayBE version 0.14.2. Although we do our best in keeping our breaking changes minimal and support outdated versions for a long time, this notebook might not be immediately applicable for other BayBE versions. If you install BayBE via the instructions in this repository, version 0.14.2 will thus be installed.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Introduction

    In this notebook, we consider a reaction described in the supporting material of a 2020 paper by [Shields et al.](https://www.nature.com/articles/s41586-021-03213-y), in which the following reaction should be optimized:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="images/reaction.svg", caption="Reaction being optimized in this tutorial."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We can vary 5 different parameters in this experiment:

    1. **Ligand**: We are given a list of 12 different ligands that we can choose from.
    2. **Base:** We have 4 different bases available for our experiment.
    3. **Solvent**: We can use one of 4 available solvents.
    4. **Concentration:** We can choose from one of 3 available concentrations.
    5. **Temperature:** We can chose from one of 3 available temperatures.

    Consequently, this means that we have **1728** different potential experiments that we could run. Fortunately, Shields and coworkers have investigated all 1728 combinations and provided a table with the conditions and corresponding yields. Note that only 18 out of the 1728 potential experiments have a yield within the top 10 percent!
    """)
    return


@app.cell(hide_code=True)
def _():
    import pandas as pd

    df = pd.read_csv("data/shields.csv")
    df
    return df, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our goal is to identify one of the top candidates, that is, one of the 18 experiments with a yield larger than 90 using only a few experiments.

    We will begin by identifying 10 initial reaction conditions. In practice, we would then run experiments to evaluate these conditions and record the corresponding reaction yields. However, in this case, we will look up the yields in a table. With the conditions and yields in hand, we can build a Bayesian model and use this model to select another 5 reaction conditions. We will then look up the yields for the 5 conditions and use this information to update the model. We will repeat this process through 5 rounds of optimization and examine the reaction yields for each optimization cycle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview

    Setting up an experimentation campaign with `BayBE` requires us to set up the main components individually. In this notebook, we will set up the following components one after another.

    1. [**Parameters**](https://emdgroup.github.io/baybe/0.14.2/userguide/parameters.html): In our setting, a _parameter_ is something that we can control directly. An example of this is which ligand to choose, or at which of the available temperatures to run the experiment. Each of the 5 parameters described earlier will correspond to exactly one of BayBE's `Parameter`s.
    2. [**Search space**](https://emdgroup.github.io/baybe/0.14.2/userguide/searchspace.html): The search space defines the combination of parameters to be searched. It thus contains all possible experiments that we could conduct. The search space is typically defined using the function `Searchspace.from_product`, which creates a search space as the Cartesian product of the parameters.
    3. [**Target**](https://emdgroup.github.io/baybe/0.14.2/userguide/targets.html): The target is the quantity we are optimizing. In the case of reaction optimization, this is typically the yield. `BayBE` can optimize a single parameter or multiple parameters at once. In this notebook, we'll focus on single parameter optimization, where we are only optimizing the yield, and we hence stick to single target optimization.
    4. [**Recommender**](https://emdgroup.github.io/baybe/0.14.2/userguide/recommenders.html): The recommender selects the next set of experiments to be performed. In this case, we use the default [`TwoPhaseMetaRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.meta.sequential.TwoPhaseMetaRecommender.html). This recommender behaves differently depending on whether it has experimental data. At the beginning of an optimization process, we typically don't have experimental data and want to find a diverse set of conditions to gather some initial data. If the `TwoPhaseMetaRecommender` has no data available, it uses random sampling to select a set of initial experiments. If the recommender has data, it uses the [`BotorchRecommender`].(https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.pure.bayesian.botorch.BotorchRecommender.html), a Bayesian optimizer that balances exploration and exploitation when selecting sets of reaction conditions.
    5. [**Campaign**](https://emdgroup.github.io/baybe/0.14.2/userguide/campaigns.html): In `BayBE`, the search space, objective, and recommender are combined into an `campaign` object. The Campaign has two important methods: `recommend`, which recommends the next set of experiments, and `add_measurements', which adds a set of experiments and updates the underlying Bayesian model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Defining the [`Parameters`](https://emdgroup.github.io/baybe/0.14.2/userguide/parameters.html)

    In this section, we introduce two different parameter types: The [`CategoricalParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.categorical.CategoricalParameter.html) and the [`NumericalDiscreteParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.numerical.NumericalDiscreteParameter.html).


    The `CategoricalParameter` has a `name` field as well as a `values` field. The `name` is used to describe the parameter, while the `values` are the collection of values that the parameter can take. In addition, one can choose a specific `encoding`. For the sake of this tutorial, we use the `One-Hot-Encoding`, `BayBE`'s default choice for `CategoricalParameter`s.

    In this tutorial, we model the three different chemical parameters, that is, the solvent, the ligand, and the base as `CategoricalParameters`. Since we have access to the data, we extract the values for the parameters from there, and create the corresponding `CategoricalParameters`.

    /// admonition | Note
    As ligand, solvent and base are chmical substances, they should preferably be modeled using the [`SubstanceParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.substance.SubstanceParameter.html). This is not done in this example for simplicity. We refer to the `Chemical_Encodings` example for a tutorial on using `SubstanceParameter`s  and a demonstration of its effect.
    ///
    """)
    return


@app.cell
def _(df):
    from baybe.parameters import CategoricalParameter

    ligand = CategoricalParameter(values=df["Ligand_Name"].unique(), name="Ligand_Name")
    solvent = CategoricalParameter(
        values=df["Solvent_Name"].unique(), name="Solvent_Name"
    )
    base = CategoricalParameter(values=df["Base_Name"].unique(), name="Base_Name")
    return base, ligand, solvent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `NumericalDiscreteParameter` is another `DiscreteParameter` and is intended to be used for parameters that have numerical values.
    """)
    return


@app.cell
def _(df):
    from baybe.parameters import NumericalDiscreteParameter

    concentration = NumericalDiscreteParameter(
        values=df["Concentration"].unique(), name="Concentration"
    )
    temperature = NumericalDiscreteParameter(
        values=df["Temp_C"].unique(), name="Temp_C"
    )
    return concentration, temperature


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the [`SearchSpace`](https://emdgroup.github.io/baybe/0.14.2/userguide/searchspace.html)

    The parameters are now combined into a [`SearchSpace`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.searchspace.html) object. Using the [`SearchSpace.from_product`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.searchspace.core.SearchSpace.html#baybe.searchspace.core.SearchSpace.from_product) constructor, we construct the cartesian product of the parameters that we defined previously.
    """)
    return


@app.cell
def _(base, concentration, ligand, solvent, temperature):
    from baybe.searchspace import SearchSpace

    parameters = [ligand, solvent, base, concentration, temperature]
    searchspace = SearchSpace.from_product(parameters=parameters)
    return (searchspace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the [`Target`](https://emdgroup.github.io/baybe/0.14.2/userguide/targets.html) & objective

    In this example, we want to maximize the yield of the reaction. Since we are only optimizing a single objective, we use the [`SingleTargetObjective`](https://emdgroup.github.io/baybe/0.14.2/userguide/objectives.html#singletargetobjective) which assumes a maximization of the target as default.
    """)
    return


@app.cell
def _():
    from baybe.targets import NumericalTarget
    from baybe.objectives import SingleTargetObjective

    target = NumericalTarget(name="yield")
    objective = SingleTargetObjective(target=target)
    return (objective,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the [`Recommender`](https://emdgroup.github.io/baybe/0.14.2/userguide/recommenders.html)

    The [`Recommender`](https://emdgroup.github.io/baybe/0.14.2/userguide/recommenders.html) selects the next set of experiments to try.
    There are many different recommenders offered by `BayBE`, and a lot of ways of combining them. For this example, we use the default initial recommender, the [`RandomRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.pure.nonpredictive.sampling.RandomRecommender.html). This recommender samples initial points from the search space randomly. Once it has data available, BayBE will automatically switch to the [`BotorchRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.pure.bayesian.botorch.BotorchRecommender.html).

    /// admonition | Task
    Instead of using the default recommender, use the [`FPSRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.pure.nonpredictive.sampling.FPSRecommender.html). Also, think about which of the two recommenders should be used in this example, and under which circumstances which recommender might be more favourable.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the [`Campaign`](https://emdgroup.github.io/baybe/0.14.2/userguide/campaigns.html)

    Now, we combine all of the individual pieces into one of the core concepts of `BayBE` - the `campaign` object. This object is responsible for organizing and managing an experimental campaign.
    """)
    return


@app.cell
def _(objective, searchspace):
    from baybe.campaign import Campaign

    campaign = Campaign(
        searchspace=searchspace, objective=objective
    )
    return (campaign,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Starting [the recommendation loop](https://emdgroup.github.io/baybe/0.14.2/userguide/getting_recommendations.html)

    Now that the `campaign` is defined, we can ask it for recommendations. So far, we haven't done any experiments. As such, the `campaign` will use random sampling to select a diverse set of initial experiments.
    """)
    return


@app.cell
def _(campaign):
    initial_rec = campaign.recommend(batch_size=10)
    initial_rec
    return (initial_rec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    At this point, we would typically perform a set of experiments using the 10 recommendations provided by the `campaign`. In this tutorial, we simply grab the yield from the data.
    """)
    return


@app.cell(hide_code=True)
def _(df, initial_rec, pd):
    merge_columns = [
        "Ligand_Name",
        "Solvent_Name",
        "Base_Name",
        "Concentration",
        "Temp_C",
    ]
    initial_rec_results = pd.merge(
        initial_rec, df[merge_columns + ["yield"]], on=merge_columns, how="left"
    )

    initial_rec_results
    return initial_rec_results, merge_columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we've performed experiments, we need to add the data from the experiments to the Campaign. We do this with the [`add_measurements`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.campaign.Campaign.html#baybe.campaign.Campaign.add_measurements) method.
    """)
    return


@app.cell
def _(campaign, initial_rec_results):
    campaign.add_measurements(initial_rec_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's simulate what we would do in practice.

    1. Ask the `campaign` for another set of 5 recommendations. Now that we've added measurements, the Campaign uses the underlying Bayesian model to select the next set of reaction conditions.
    2. Next we will look up the yield for that set of conditions and use the yield data to update the Bayesian model.

    We'll repeat this process 5 times and examine the distribution of yields at each iteration.
    """)
    return


@app.cell
def _(campaign, df, merge_columns, mo):
    for _ in mo.status.progress_bar(
        range(10),
        title="Optimizing reaction conditions",
    ):
        rec = campaign.recommend(5)
        rec_results = rec.merge(
            df[merge_columns + ["yield"]], on=merge_columns, how="left"
        )
        campaign.add_measurements(rec_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Now, let's have a look at the results which are stored in the `campaign` object and compare them to the optimal value. Note how much `marimo`helps here with the inspection!
    """)
    return


@app.cell
def _(campaign):
    measurements = campaign.measurements
    measurements
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    As we can see, we found a very good candidate, and only needed to evaluate a fraction of the search space! This insight concludes this basic BayBE tutorial.
    """)
    return


if __name__ == "__main__":
    app.run()
