import seaborn as sns
import pandas as pd
from typing import Sequence
from matplotlib.ticker import MaxNLocator

sns.set_context("paper", font_scale=1.7)


def create_dict_from_columns(df: pd.DataFrame, col_a: str, col_b: str):
    unique_pairs = df[[col_a, col_b]].drop_duplicates()
    result_dict = dict(zip(unique_pairs[col_a], unique_pairs[col_b]))
    return result_dict


def backtest_plot(
    df: pd.DataFrame,
    y: str,
    hue: str,
    x: str = "Num_Experiments",
    indicator_y: float | None = None,
    indicator_labels: list[str] | None = None,
    xlim: Sequence | None = None,
    ylim: Sequence | None = None,
):
    """Plot utility.

    Shows a backtest plot based on seaborn's lineplot and stores the figure. Optionally
    adds guidelines at what x position a certain y level is hit.

    Args:
        df: Results of the backtest simulation.
        y: Name of the column that will be used as y.
        hue: One line will be plotted for each group in this column.
        x: Name of the column that will be used as x.
        indicator_y: The y level for which indicator lines should be shown.
        indicator_labels: Subset of entries in the 'hue' column. Indicators will only
            be shown for those groups.
    """
    # Creat plot
    ax = sns.lineplot(
        data=df,
        marker="o",
        markersize=7,
        x=x,
        y=y,
        hue=hue,
    )
    ax.figure.set_size_inches(10, 6)

    # Add inidactors if requested
    if indicator_y is not None:
        indicator_labels = indicator_labels or df[hue].unique().tolist()

        xmax = 0.0
        for label in indicator_labels:
            label_data = df[df[hue] == label]
            grouped = label_data.groupby(x)[y].mean().reset_index()

            closest_point = grouped.iloc[(grouped[y] - indicator_y).abs().argmin()]
            closest_x = closest_point[x]  # .values[0]
            xmax = max(xmax, closest_x)

            ax.axvline(
                closest_x,
                color="grey",
                linestyle="--",
                ymax=(indicator_y - ax.get_ylim()[0])
                / (ax.get_ylim()[1] - ax.get_ylim()[0]),
            )

        ax.axhline(
            indicator_y, color="grey", linestyle="--", xmax=xmax / ax.get_xlim()[1]
        )

    # Set axis limits if requested
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Set reasonable integer xtick size
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
