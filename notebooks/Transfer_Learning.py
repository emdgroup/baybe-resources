import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium", app_title="Transfer Learning")


@app.cell
def _():
    import marimo as mo
    import warnings

    warnings.filterwarnings("ignore")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Transfer learning

        This file contains some examples regarding to the topic of **transfer learning**. It demonstrates how to use `BayBE's` transfer learning capabilities to improve the performance of campaigns if data from similar campaigns is available.

        /// admonition | Note
        The term *transfer learning* is somewhat ambiguous, and different people might have different interpretations of what is meant by this term. We thus recommend to first read [the userguide on transfer learning](https://emdgroup.github.io/baybe/stable/userguide/transfer_learning.html) to ensure that it is clear how to interpret this term in the context of `BayBE`.
        ///

        /// admonition | Note
        To really see the effects of transfer learning, it is necessary to run longer tests. Since the goal of this notebook is to demonstrate how to use and set up transfer learning in BayBE, the results obtained by just executing this notebook might not be representative. We thus refer to our [documentation](https://emdgroup.github.io/baybe/stable/) for more detailed plots.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Some basic settings and data loading

        We begin by introducing some settings that will be used later in this notebook. We also load the data, which is the same data used in the `Reaction_Optimization` and `Chemical_Encoding` notebooks.
        """
    )
    return


@app.cell
def _():
    import pandas as pd

    temperatures = [90, 105, 120]
    concentrations = [0.057, 0.1, 0.153]
    parameters_to_analyze = {
        "temperature": temperatures,
        "concentration": concentrations,
    }
    units = {"temperature": "°C", "concentration": "mol / l"}

    data = pd.read_csv("data/shields.csv")
    data
    return concentrations, data, parameters_to_analyze, pd, temperatures, units


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data visualization: Correlation of Data Between Different Temperatures

        We first visualize the data, using a small helper function that is defined in the next cell.
        """
    )
    return


@app.cell(hide_code=True)
def _(parameters_to_analyze, units):
    import matplotlib.pyplot as plt
    from pathlib import Path
    from collections import defaultdict
    import csv
    from itertools import combinations
    import scipy.stats as stats

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
                config["temperature"] = int(row["Temp_C"])
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
                    slope = (yield_v2 - yield_v1) / (v2 - v1)
                    slopes.append(slope)

            ## Correlation between yields at v1 and v2
            corr, _ = stats.pearsonr(yields_v1, yields_v2)
            print(
                f"Pearson correlation coefficient (PCC) between yields at "
                f"{v1} {units[parameter_to_analyze]} and "
                f"{v2} {units[parameter_to_analyze]}: {corr:.2f}"
            )
            scc, _ = stats.spearmanr(yields_v1, yields_v2)
            print(
                f"Spearman rank correlation coefficient (SCC) between yields at "
                f"{v1} {units[parameter_to_analyze]} and "
                f"{v2} {units[parameter_to_analyze]}: {scc:.2f}"
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
            ax.set_xlabel(f"yield at {v1} {units[parameter_to_analyze]}")
            ax.set_ylabel(f"yield at {v2} {units[parameter_to_analyze]}")
            ax.set_title(
                f"{parameter_to_analyze} combination: "
                f"{v1} {units[parameter_to_analyze]} to "
                f"{v2} {units[parameter_to_analyze]}"
            )
            # include 1:1 line
            ax.plot(x, x, color="black", linestyle="--", label="1:1 line")
            ax.grid(True)
            # show the legend
            ax.legend()
            plt.show()

    return Path, analyze_data, plt


@app.cell
def _(Path, analyze_data):
    analyze_data(Path("data/shields.csv"), "temperature")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setting up the `BayBE` campaign

        After we have analyzed the data and came to the conclusion that we want to approach this optimization with transfer learning, we now set up our `BayBE` campaign. We first begin by collecting all parts of the campaign that are not related to Transfer Learning.

        ///admonition | Note
        This example uses substance encodings. In case you are interested in more details on them, check out the `ChemicalEncodings` example!
        ///
        """
    )
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

    objective = SingleTargetObjective(NumericalTarget(name="yield", mode="MAX"))
    return objective, parameters


@app.cell
def _(mo):
    mo.md(
        """
        Transfer Learning in `BayBE` is enabled by using a special parameter - the [`TaskParameter`](https://emdgroup.github.io/baybe/stable/userguide/transfer_learning.html#the-role-of-the-taskparameter). This parameter is used to "mark" the context of individual experiments, and thus to "align" different campaigns along their context dimension. The set of all possible contexts is provided upon the initialization of the `TaskParameter` by providing them as `values`.

        In this example, each temperature corresponds to a different `context`. The set of `values` is thus the set of all temperatures. The `active_values` describes for which tasks recommendations should be given. This ensures that `BayBE` does not recommend to conduct experiments for a context that might no longer be available.

        We can then combine the `TaskParameters` together with the components defined above to create one campaign for each temperature.
        """
    )
    return


@app.cell
def _(objective, parameters, temperatures):
    from baybe.parameters import TaskParameter
    from baybe import Campaign
    from baybe.searchspace import SearchSpace

    tl_campaigns = {
        temp: Campaign(
            searchspace=SearchSpace.from_product(
                parameters=parameters
                + [
                    TaskParameter(
                        name="Temp_C",
                        values=[str(t) for t in temperatures],
                        active_values=[str(temp)],
                    )
                ]
            ),
            objective=objective,
        )
        for temp in temperatures
    }
    return Campaign, tl_campaigns


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Setting up the simulation

        We now set up the simulation loop. This requires us to define the number of DoE iterations as well as the number of Monte Carlo iterations.

        Since we want to investigate the influence of Transfer Learning, we will provide the campaigns with batches of initial data from the temperatures that are *not* being active for the corresponding campaign. For each Monte Carlo iteration, we sample a different batch of initial data that is then being used by the algorithm. In addition, we compare the results to a "baseline" that is not using any Transfer Learning.

        ///admonition | Note
        To really see the impact of Transfer Learning, you need to run the following code with more Monte Carlo Iterations, which might take quite some time. If you are interested in looking at some pre-computed results, have a look at the corresponding `.ipynb` version of this notebook.
        ///
        """
    )
    return


@app.cell
def _(Campaign, data, pd, temperatures, tl_campaigns):
    from baybe.simulation import simulate_scenarios

    from baybe.utils.random import set_random_seed

    N_DOE_ITERATIONS = 2
    BATCH_SIZE = 2
    N_MC_ITERATIONS = 3
    set_random_seed(1337)

    sample_fractions = [0.01, 0.1]

    def optimize_for_temperature(
        temp: str,
        tl_campaigns: dict[str, Campaign] = tl_campaigns,
        data: pd.DataFrame = data,
    ):

        lookup_T = data.copy(deep=True)
        lookup_T["Temp_C"] = lookup_T["Temp_C"].astype(str)

        print(f"\n\nTemperature: {temp}")
        excluded_temps = [str(t) for t in temperatures if str(t) != str(temp)]

        print(f"Taking additional data from {excluded_temps} into account.\n")
        campaign = tl_campaigns[temp]
        # Lookup table that contains all data except the data for the current temperature.
        lookup_other_data = lookup_T[lookup_T["Temp_C"] != str(temp)].copy(deep=True)

        results: list[pd.DataFrame] = []
        for p in sample_fractions:
            print("Percentage of data used: ", p)
            result_fraction = simulate_scenarios(
                {f"{int(100 * p)}": campaign},
                lookup_T,
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
            lookup_T,
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

    return (optimize_for_temperature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We now finally run the simulation code and investigate the results.""")
    return


@app.cell
def _(mo, optimize_for_temperature, plt):
    from utils import backtest_plot

    temp_to_investigate = 105

    backtest_plot(
        df=optimize_for_temperature(temp_to_investigate),
        x="Number of experiments",
        y="Running best yield",
        hue="% of data used",
    )
    mo.mpl.interactive(plt.gcf())
    return


if __name__ == "__main__":
    app.run()
