import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", app_title="Transfer Learning")


@app.cell
def _():
    import marimo as mo
    import warnings

    warnings.filterwarnings("ignore")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Transfer learning

    This file contains some examples regarding to the topic of **transfer learning**. It demonstrates how to use BayBE's transfer learning capabilities to improve the performance of campaigns if data from similar campaigns is available.

    /// Note
    The term *transfer learning* is somewhat ambiguous, and different people might have different interpretations of what is meant by this term. We thus recommend to first read [the userguide on transfer learning](https://emdgroup.github.io/baybe/0.14.2/userguide/transfer_learning.html) to ensure that it is clear how to interpret this term in the context of BayBE.
    ///

    /// Caution
    To really see the effects of transfer learning, it is necessary to run longer tests than the ones presented in this notebook. Since the goal of this notebook is to demonstrate how to use and set up transfer learning in BayBE, the results obtained by just executing this notebook might not be representative. We provide a pre-computed version in the `images` subfolder.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Some basic settings and data loading

    We begin by introducing some settings that will be used later in this notebook. We also load the data, which is the same data used in the `Chemical_Encoding` notebook.
    """)
    return


@app.cell
def _():
    import pandas as pd

    data = pd.read_csv("data/shields.csv")
    data
    return data, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figuring out if transfer learning should be used at all

    The first step of a potential transfer learning workflow should always be a detailed analysis of the data. The reason is that transfer learning assume that there is some positive correlation in the data which can then be leveraged. If there is no such positive correlation, then transfer learning is not the correct tool to use, and attempting to use can even be harmful.

    In this example, we want to evaluate if we can use transfer learning for leveraging knowledge that we gained for reactions in two labs for the third lab. We thus begin by visualizing the data, using a small helper function that is defined in the next cell.
    """)
    return


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    from pathlib import Path
    from collections import defaultdict
    import csv
    from itertools import combinations
    import scipy.stats as stats

    labs = ["A", "B", "C"]
    concentrations = [0.057, 0.1, 0.153]
    units = {"lab": "", "concentration": "mol / l"}
    parameters_to_analyze = {"lab": labs, "concentration": concentrations}

    def analyze_data(file_path: Path, parameter_to_analyze: str):
        data = defaultdict(lambda: defaultdict(list))
        with open(file_path, "r") as file:
            csv_reader = csv.DictReader(file)
            config = {}
            for row in csv_reader:
                config["base"] = row["Base_Name"]
                config["ligand"] = row["Ligand_Name"]
                config["solvent"] = row["Solvent_Name"]
                config["concentration"] = float(row["Concentration"])
                config["lab"] = row["Lab"]
                yield_value = float(row["yield"])

                # config key consists of every key in the config dictionary, except for
                # the parameter to analyze
                config_key = tuple(
                    value
                    for key, value in config.items()
                    if key != parameter_to_analyze
                )
                data[config_key][config[parameter_to_analyze]].append(yield_value)

        # form all possible combinations of the parameter to analyze
        permutations = list(
            combinations(parameters_to_analyze[parameter_to_analyze], 2)
        )

        for v1, v2 in permutations:
            slopes = []

            yields_v1: list[float] = []
            yields_v2: list[float] = []

            for parameter_config, parameter_combination in data.items():
                if v1 in parameter_combination and v2 in parameter_combination:
                    yield_v1 = parameter_combination[v1][0]
                    yields_v1.append(yield_v1)
                    yield_v2 = parameter_combination[v2][0]
                    yields_v2.append(yield_v2)
                    if parameter_to_analyze != "lab":
                        slope = (yield_v2 - yield_v1) / (v2 - v1)
                        slopes.append(slope)

            ## Correlation between yields at v1 and v2
            corr, _ = stats.pearsonr(yields_v1, yields_v2)
            unit_str = f" {units[parameter_to_analyze]}" if units[parameter_to_analyze] else ""
            print(
                f"Pearson correlation coefficient (PCC) between yields at "
                f"{v1}{unit_str} and "
                f"{v2}{unit_str}: {corr:.2f}"
            )
            scc, _ = stats.spearmanr(yields_v1, yields_v2)
            print(
                f"Spearman rank correlation coefficient (SCC) between yields at "
                f"{v1}{unit_str} and "
                f"{v2}{unit_str}: {scc:.2f}"
            )
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                yields_v1, yields_v2
            )
            print(f"R2 value: {r_value**2:.2f}")

            # plot the correlation between the yields at v1 and v2
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(
                yields_v1,
                yields_v2,
                color="#0e69af",
                edgecolor="black",
                s=15,
                alpha=0.8,
                linewidth=0.5,
            )
            # include the regression line
            x = [min(yields_v1), max(yields_v1)]
            y = [slope * x_i + intercept for x_i in x]
            ax.plot(
                x,
                y,
                color="grey",
                label=f"R$^2$ = {r_value**2:.2f}, PCC = {corr:.2f}, SCC = {scc:.2f}",
            )
            ax.set_xlabel(f"yield at {v1}{unit_str}")
            ax.set_ylabel(f"yield at {v2}{unit_str}")
            ax.set_title(
                f"{parameter_to_analyze} combination: "
                f"{v1}{unit_str} to "
                f"{v2}{unit_str}"
            )
            # include 1:1 line
            ax.plot(x, x, color="black", linestyle="--", label="1:1 line")
            ax.grid(True)
            # show the legend
            ax.legend()
            plt.show()
    return Path, analyze_data, concentrations, labs, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Task
    The helper function is written in a generic way, allowing to also investigate other potential parameters. Instead of investigating whether the lab is a suitable candidate for transfer learning, investigate whether or not the concentration could also be used. Also think about why makes the concentration parameter inherently different from the lab parameter, and what the influence of this on the question of whether or not to use it as a `TaskParameter` is.
    ///
    """)
    return


@app.cell
def _(Path, analyze_data):
    analyze_data(Path("data/shields.csv"), "lab")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the data is positively correlated, this is indeed a use case in which transfer learning can be used. We thus continue with describing how we can now setup BayBE to leverage this.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up the BayBE campaign

    After we have analyzed the data and came to the conclusion that we want to approach this optimization with transfer learning, we now set up our `BayBE` campaign. We first begin by collecting all parts of the campaign that are not related to Transfer Learning.

    /// Note
    This example uses substance encodings. In case you are interested in more details on them, check out the `ChemicalEncodings` example!
    ///
    """)
    return


@app.cell
def _(concentrations, pd):
    from baybe.parameters import (
        NumericalDiscreteParameter,
        SubstanceParameter,
    )
    from baybe.targets import NumericalTarget
    from baybe.objectives import SingleTargetObjective
    from utils import create_dict_from_columns

    encoding = "RDKIT2DDESCRIPTORS"
    df = pd.read_csv("data/shields.csv")

    substances = {
        "bases": create_dict_from_columns(df, "Base_Name", "Base_SMILES"),
        "ligands": create_dict_from_columns(df, "Ligand_Name", "Ligand_SMILES"),
        "solvents": create_dict_from_columns(df, "Solvent_Name", "Solvent_SMILES"),
    }

    parameters = [
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
            name="Concentration", values=concentrations, tolerance=0.001
        ),
    ]

    objective = SingleTargetObjective(NumericalTarget(name="yield"))
    return objective, parameters


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Transfer Learning in `BayBE` is enabled by using a special parameter - the [`TaskParameter`](https://emdgroup.github.io/baybe/0.14.2/userguide/transfer_learning.html#the-role-of-the-taskparameter). This parameter is used to "mark" the context of individual experiments, and thus to "align" different campaigns along their context dimension. The set of all possible contexts is provided upon the initialization of the `TaskParameter` by providing them as `values`.

    In this example, each lab corresponds to a different `context`. The set of `values` is thus the set of all labs. The `active_values` describes for which tasks recommendations should be given. This ensures that `BayBE` does not recommend to conduct experiments for a context that might no longer be available.

    We can then combine the `TaskParameters` together with the components defined above to create one campaign for each lab.
    """)
    return


@app.cell
def _(labs, objective, parameters):
    from baybe.parameters import TaskParameter
    from baybe import Campaign
    from baybe.searchspace import SearchSpace

    tl_campaigns = {
        lab: Campaign(
            searchspace=SearchSpace.from_product(
                parameters=parameters
                + [
                    TaskParameter(
                        name="Lab",
                        values=labs,
                        active_values=[lab],
                    )
                ]
            ),
            objective=objective,
        )
        for lab in labs
    }
    return Campaign, tl_campaigns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up the simulation

    We now set up the simulation loop. This requires us to define the number of DoE iterations as well as the number of Monte Carlo iterations.

    Since we want to investigate the influence of Transfer Learning, we will provide the campaigns with batches of initial data from the labs that are *not* being active for the corresponding campaign. The percentage of points sampled for this is given in the `SAMPLED_FRACTIONS` list. For each Monte Carlo iteration, we sample a different batch of initial data that is then being used by the algorithm. In addition, we compare the results to a "baseline" that is not using any Transfer Learning.

    /// Caution
    Leveraging existing data requires a significant amount of data, in particular when we want to compare different amounts of sampled data. The execution can thus take quite some time if using settings that really show the effect.

    To really see the impact of Transfer Learning, you need to run the following code with more Monte Carlo Iterations, which might take quite some time. If you are interested in looking at some pre-computed results, have a look at the pre-computed image in the `images` subfolder.
    ///
    """)
    return


@app.cell
def _(Campaign, data, labs, pd, tl_campaigns):
    from baybe.simulation import simulate_scenarios

    from baybe.utils.random import set_random_seed

    N_DOE_ITERATIONS = 2
    BATCH_SIZE = 2
    N_MC_ITERATIONS = 3
    set_random_seed(1337)

    SAMPLE_FRACTIONS = [0.01, 0.05, 0.1, 0.15]

    def optimize_for_lab(
        lab: str,
        tl_campaigns: dict[str, Campaign] = tl_campaigns,
        data: pd.DataFrame = data,
        sample_fractions: list[float] = SAMPLE_FRACTIONS
    ):

        lookup = data.copy(deep=True)

        print(f"\n\nLab: {lab}")
        excluded_labs = [l for l in labs if l != lab]

        print(f"Taking additional data from {excluded_labs} into account.\n")
        campaign = tl_campaigns[lab]
        # Lookup table that contains all data except the data for the current lab.
        lookup_other_data = lookup[lookup["Lab"] != lab].copy(deep=True)

        results: list[pd.DataFrame] = []
        for p in sample_fractions:
            print("Percentage of data used: ", p)
            result_fraction = simulate_scenarios(
                {f"{int(100 * p)}": campaign},
                lookup,
                initial_data=[
                    lookup_other_data.sample(frac=p) for _ in range(N_MC_ITERATIONS)
                ],
                batch_size=BATCH_SIZE,
                n_doe_iterations=N_DOE_ITERATIONS,
            )
            results.append(result_fraction)

        print("Percentage of data used: 0.0")
        result_baseline = simulate_scenarios(
            {"0": campaign},
            lookup,
            batch_size=BATCH_SIZE,
            n_doe_iterations=N_DOE_ITERATIONS,
            n_mc_iterations=N_MC_ITERATIONS,
        )
        results = pd.concat([result_baseline, *results])

        results.rename(
            columns={
                "Scenario": "% of data used",
                "Num_Experiments": "Number of experiments",
                "yield_CumBest": "Running best yield",
            },
            inplace=True,
        )

        return results
    return (optimize_for_lab,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now finally run the simulation code and investigate the results. The variable `lab_to_investigate` can be changed to any of the available labs.
    """)
    return


@app.cell
def _(mo, optimize_for_lab, plt):
    from utils import backtest_plot

    LAB_TO_INVESTIGATE = "B"

    backtest_plot(
        df=optimize_for_lab(LAB_TO_INVESTIGATE),
        x="Number of experiments",
        y="Running best yield",
        hue="% of data used",
    )
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following image shows the pre-computed result that was obtained with the following settings:
    * `SAMPLED_FRACTIONS = [0.01, 0.05, 0.1, 0.15]`
    * `N_DOE_ITERATIONS = 30`
    * `BATCH_SIZE = 2`
    * `N_MC_ITERATIONS = 40`
    * `LAB_TO_INVESTIGATE = "B"`

    It demonstrates that already a little data can help in the beginning of an experimental campaign, and that although more data is more favourable, the effects are diminishing at some point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="images/transfer_learning_precomputed.png", caption="Reaction being optimized in this tutorial."
    )
    return


if __name__ == "__main__":
    app.run()
