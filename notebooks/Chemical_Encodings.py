import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", app_title="Chemical encodings")


@app.cell
def _():
    import marimo as mo
    import warnings

    warnings.filterwarnings("ignore")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Chemical encodings

    This notebook demonstrates the power and usefulness of BayBE's chemical encodings. If parameters in a process to be optimized are chemicals, this feature enables BayBE to automatically use meaningful chemical descriptors, automatically leveraging chemical knowledge for the optimization process.

    This notebook assumes some basic familiarity with using BayBE, and that it does not explain all of the core concepts. If you are interested in those, we recommend to first check out the `Reaction_Optimization` example.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup

    We begin this example by defining a suitable recommender. We use a [`TwoPhaseMetaRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.meta.sequential.TwoPhaseMetaRecommender.html) equipped with a [`BotorchRecommender`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.recommenders.pure.bayesian.botorch.BotorchRecommender.html). We also use use a specific kernel that is suited particularly well for chemical problems - the [`EDBOKernel`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.surrogates.gaussian_process.presets.edbo.EDBOKernelFactory.html#baybe.surrogates.gaussian_process.presets.edbo.EDBOKernelFactory).
    """)
    return


@app.cell
def _():
    from baybe.surrogates.gaussian_process.presets.edbo import EDBOKernelFactory
    from baybe.recommenders import TwoPhaseMetaRecommender, BotorchRecommender
    from baybe.surrogates import GaussianProcessSurrogate

    recommender = TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(
            surrogate_model=GaussianProcessSurrogate(
                kernel_or_factory=EDBOKernelFactory()
            )
        )
    )
    return (recommender,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This examples uses the same chemical reaction like the `Reaction_Optimization` notebook. However, it uses different encodings of the chemical parameters. This is done by using BayBE's [`SubstanceParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.substance.SubstanceParameter.html) which automatically translates chemicals into meaningful descriptors. For more details on the `SubstanceParameter`, we refer to the [corresponding section of the user guide](https://emdgroup.github.io/baybe/0.14.2/userguide/parameters.html#substanceparameter).

    The specific encoding that should be used can be described by the `encoding` field. We investigate three different encodings here and create one campaign per chemical encoding.
    """)
    return


@app.cell
def _():
    import pandas as pd

    from utils import create_dict_from_columns

    df = pd.read_csv("data/shields.csv")

    substances = {
        "bases": create_dict_from_columns(df, "Base_Name", "Base_SMILES"),
        "ligands": create_dict_from_columns(df, "Ligand_Name", "Ligand_SMILES"),
        "solvents": create_dict_from_columns(df, "Solvent_Name", "Solvent_SMILES"),
    }
    return df, substances


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Now that we have collected the data, we can create different campaigns that we want to compare against each other. To facilitate the usage of `BayBE`'s simulation capabilities, we collect the different campaigns in a `dict`.
    """)
    return


@app.cell
def _(df, recommender, substances):
    from baybe import Campaign
    from baybe.parameters import (
        CategoricalParameter,
        NumericalDiscreteParameter,
        SubstanceParameter,
    )
    from baybe.parameters import CategoricalParameter
    from baybe.searchspace import SearchSpace
    from baybe.targets import NumericalTarget

    objective = NumericalTarget(name="yield").to_objective()

    substance_encodings = ["MORDRED", "ECFP", "RDKIT2DDESCRIPTORS"]
    scenarios = {
        encoding: Campaign(
            searchspace=SearchSpace.from_product(
                parameters=[
                    SubstanceParameter(
                        name="Solvent_Name",
                        data=substances["solvents"],
                        encoding=encoding,
                    ),
                    SubstanceParameter(
                        name="Base_Name", data=substances["bases"], encoding=encoding
                    ),
                    SubstanceParameter(
                        name="Ligand_Name",
                        data=substances["ligands"],
                        encoding=encoding,
                    ),
                    NumericalDiscreteParameter(
                        values=df["Concentration"].unique(), name="Concentration"
                    ),
                    NumericalDiscreteParameter(
                        values=df["Temp_C"].unique(), name="Temp_C"
                    ),
                ]
            ),
            objective=objective,
            recommender=recommender,
        )
        for encoding in substance_encodings
    }
    return (
        Campaign,
        CategoricalParameter,
        NumericalDiscreteParameter,
        SearchSpace,
        objective,
        scenarios,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Of course, we also want to compare the campaigns using the chemical encodings to other campaigns not using the special encoding.
    """)
    return


@app.cell
def _(
    Campaign,
    CategoricalParameter,
    NumericalDiscreteParameter,
    SearchSpace,
    objective,
    scenarios,
    substances,
):
    ohe_parameters = [
        CategoricalParameter(
            name="Solvent_Name", values=substances["solvents"], encoding="OHE"
        ),
        CategoricalParameter(
            name="Base_Name", values=substances["bases"], encoding="OHE"
        ),
        CategoricalParameter(
            name="Ligand_Name", values=substances["ligands"], encoding="OHE"
        ),
        NumericalDiscreteParameter(name="Temp_C", values=[90, 105, 120]),
        NumericalDiscreteParameter(name="Concentration", values=[0.057, 0.1, 0.153]),
    ]
    campaign_ohe = Campaign(
        searchspace=SearchSpace.from_product(parameters=ohe_parameters),
        objective=objective,
    )
    scenarios["OHE"] = campaign_ohe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note
    Sometimes, none of the well-known encodings might be suitable for a specific use case. BayBE thus also allows to supply an arbitrary, user defined "translation" from chemicals to meaningful descriptors. More generally, the [`CustomDiscreteParameter`](https://emdgroup.github.io/baybe/0.14.2/_autosummary/baybe.parameters.custom.CustomDiscreteParameter.html#baybe.parameters.custom.CustomDiscreteParameter) allows fully user-specified representations of data.
    ///

    /// admonition | Task
    Create an additional campaign using a `CustomDiscreteParameter` with some "dummy data" as a representation for one or more of the chemical substances. More information on this parameter can be found in the [corresponding section of the user guide](https://emdgroup.github.io/baybe/0.14.2/userguide/parameters.html#customdiscreteparameter).
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using `BayBE`'s [simulation capabilities](https://emdgroup.github.io/baybe/0.14.2/userguide/simulation.html)

    `BayBE` offers multiple functionalities to “simulate” experimental campaigns with a given lookup mechanism. `BayBE`’s simulation package enables a wide range of use cases and can even be used for “oracle predictions”. This is made possible through the flexible use of lookup mechanisms, which act as the loop-closing element of an optimization loop.

    Lookups can be provided in a variety of ways, by using fixed data sets, analytical functions, or any other form of black-box callable. In all cases, their role is the same: to retrieve target values for parameter configurations suggested by the recommendation engine.

    In our case, we can directly use the data that we stored in the `df` dataframe and do the simulation.

    /// Note
    For performance reasons, the default values used in this example are very low. We recommend to increase `N_DOE_ITERATIONS` resp. `N_MC_ITERATIONS` to values of at least 15 resp. 25 to observe the effects of the different encodings. For best results, increase them to 25 resp. 40.
    ///
    """)
    return


@app.cell
def _(df, scenarios):
    from baybe.simulation import simulate_scenarios

    BATCH_SIZE = 2
    N_DOE_ITERATIONS = 5
    N_MC_ITERATIONS = 10

    results = simulate_scenarios(
        scenarios,
        df,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
        n_mc_iterations=N_MC_ITERATIONS,
    )

    results.rename(
        columns={
            "Scenario": "Substance encoding",
            "Num_Experiments": "Number of experiments",
            "yield_CumBest": "Running best yield",
        },
        inplace=True,
    )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We now visualize the results using the `backtest_plot` utility. This utility "averages" the individual Monte Carlo iterations and shows the mean and a confidence interval. It can also be used to give visual guidance on the performance of individual scenarios. We mark the (fictional) goal of a yield of at least 90% and compare the different encodings.
    """)
    return


@app.cell
def _(mo, results):
    from utils import backtest_plot
    import matplotlib.pyplot as plt

    backtest_plot(
        df=results,
        x="Number of experiments",
        y="Running best yield",
        hue="Substance encoding",
        indicator_y=90
    )
    mo.mpl.interactive(plt.gcf())
    return (plt,)


@app.cell
def _(plt):
    plt.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot demonstrates that all of the chemical encodings outperform the one-hot-encoding. Even though all of the encodings perform comparably during the first few iterations, the campaigns leveraging the chemical encodings reach the marked target of achieving at least 90% yield significantly faster.
    """)
    return


if __name__ == "__main__":
    app.run()
