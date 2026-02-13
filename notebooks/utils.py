import seaborn as sns
import pandas as pd
import numpy as np
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


def espresso_taste(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate the taste quality of espresso shots based on brewing parameters.
    
    This function models a complex, realistic espresso extraction landscape with
    multiple local optima representing different espresso styles (ristretto, classic,
    lungo) and one global optimum. The function incorporates realistic interactions
    between parameters based on espresso physics.
    
    Vectorized implementation using NumPy for efficient batch evaluation.
    
    Args:
        df: DataFrame with columns:
            - bean_type: Type of coffee bean ("Arabica", "Robusta", or "Blend")
            - grind_size: Grind size in microns (typically 200-400 for espresso)
            - water_pressure: Brewing pressure in bar (typically 8.0-10.0)
            - water_temperature: Water temperature in Celsius (typically 88-96)
            - brewing_time: Extraction time in seconds (typically 20-35)
    
    Returns:
        DataFrame with added 'taste' column containing scores from 1.0 (awful) 
        to 10.0 (awesome), rounded to 1 decimal place
    """
    result = df.copy()
    
    # Normalize inputs to [0, 1] range for easier computation (vectorized)
    grind_norm = (df["grind_size"].values - 200) / 200  # 200-400 microns -> 0-1
    pressure_norm = (df["water_pressure"].values - 7.5) / 2.5  # 7.5-10 -> 0-1
    temp_norm = (df["water_temperature"].values - 88) / 8  # 88-96 -> 0-1
    time_norm = (df["brewing_time"].values - 20) / 15  # 20-35 -> 0-1
    
    # Bean type base quality and sensitivity (vectorized mapping)
    bean_quality_map = {"Arabica": 1.0, "Blend": 0.85, "Robusta": 0.7}
    bean_sensitivity_map = {"Arabica": 1.2, "Blend": 1.0, "Robusta": 0.8}
    
    base_quality = df["bean_type"].map(bean_quality_map).fillna(0.8).values
    sensitivity = df["bean_type"].map(bean_sensitivity_map).fillna(1.0).values
    
    # Temperature normalized to [0, 1]
    adjusted_temp = temp_norm
    
    # === Define three local optima (different espresso styles) ===
    
    # 1. RISTRETTO optimum: Fine grind, short time, high pressure, lower temp
    ristretto_grind = 0.2  # Fine
    ristretto_time = 0.2   # Short (23-24s)
    ristretto_pressure = 0.8  # High (9.5 bar)
    ristretto_temp = 0.4   # Medium-low temp
    
    ristretto_score = np.exp(-5 * (
        (grind_norm - ristretto_grind)**2 +
        (time_norm - ristretto_time)**2 +
        (pressure_norm - ristretto_pressure)**2 +
        (adjusted_temp - ristretto_temp)**2
    ))
    
    # 2. CLASSIC optimum: Medium-fine grind, medium time, medium pressure (GLOBAL MAX)
    classic_grind = 0.35   # Medium-fine
    classic_time = 0.5     # Medium (27-28s)
    classic_pressure = 0.6 # Medium-high (9 bar)
    classic_temp = 0.5     # Medium temp
    
    classic_score = np.exp(-4 * (
        (grind_norm - classic_grind)**2 +
        (time_norm - classic_time)**2 +
        (pressure_norm - classic_pressure)**2 +
        (adjusted_temp - classic_temp)**2
    ))
    
    # 3. LUNGO optimum: Coarser grind, longer time, lower pressure
    lungo_grind = 0.6      # Coarser
    lungo_time = 0.75      # Longer (31-32s)
    lungo_pressure = 0.3   # Lower (8.25 bar)
    lungo_temp = 0.6       # Medium-high temp
    
    lungo_score = np.exp(-5 * (
        (grind_norm - lungo_grind)**2 +
        (time_norm - lungo_time)**2 +
        (pressure_norm - lungo_pressure)**2 +
        (adjusted_temp - lungo_temp)**2
    ))
    
    # STYLE SCORE: Measures how well parameters match known espresso styles
    style_score = 1.3 * classic_score + 0.8 * ristretto_score + 0.7 * lungo_score
    
    # === Add realistic extraction physics penalties ===
    
    # Over-extraction penalty (too fine + too long = bitter)
    over_extraction = (1 - grind_norm) * time_norm
    over_extraction_penalty = -1.1 * over_extraction**2
    
    # Under-extraction penalty (too coarse + too short = sour/weak)
    under_extraction = grind_norm * (1 - time_norm)
    under_extraction_penalty = -0.8 * under_extraction**2
    
    # Temperature extremes penalty
    temp_extreme_penalty = -1.5 * (adjusted_temp - 0.5)**4
    
    # Pressure-grind interaction: fine grinds need precise pressure control
    pressure_grind_mismatch = np.abs(pressure_norm - 0.6) * (1 - grind_norm)**2
    pressure_penalty = -0.8 * pressure_grind_mismatch
    
    # Channeling risk (too fine grind + too high pressure = uneven extraction)
    channeling_risk = (1 - grind_norm)**2 * (pressure_norm - 0.5)**2
    channeling_penalty = -1 * channeling_risk
    
    # === Combine all components ===
    
    # BASE SCORE: Converts style_score into a taste score on 1-10 scale
    base_score = 4.5 + 3.0 * style_score
    
    # Apply quality modifiers
    quality_modifier = base_quality
    
    # Apply penalties with sensitivity
    penalties = sensitivity * (
        over_extraction_penalty +
        under_extraction_penalty +
        temp_extreme_penalty +
        pressure_penalty +
        channeling_penalty
    )
    
    # Final score
    final_score = base_score * quality_modifier + penalties
    
    # Add small random noise to simulate measurement uncertainty (vectorized)
    noise = np.random.normal(0, 0.25, size=len(df))
    final_score += noise
    
    # Clip to valid range [1, 10]
    final_score = np.clip(final_score, 1.0, 10.0)
    
    # Round to 1 decimal place (realistic measurement precision)
    final_score = np.round(final_score, 1)
    
    result["taste"] = final_score
    return result


class GoldMine:
    """A 2D landscape with multiple Gaussian peaks for optimization demonstration.

    This class represents a gold mining landscape where peaks correspond to gold-rich
    regions. It can be used to demonstrate optimization algorithms by providing a
    continuous 2D function that can be evaluated at any point.

    Attributes:
        grid_size: Resolution of the landscape grid.
        n_peaks: Number of gold-rich regions (peaks).
        noise_level: Amount of random noise added to the landscape.
        X: Meshgrid x-coordinates.
        Y: Meshgrid y-coordinates.
        Z: Gold richness values at each grid point.
    """

    def __init__(
        self,
        grid_size: int = 100,
        n_peaks: int = 5,
        noise_level: float = 0.1,
    ):
        """Initialize a gold mining landscape.

        Args:
            grid_size: Resolution of the grid.
            n_peaks: Number of gold-rich regions (peaks).
            noise_level: Amount of random noise.
        """
        import numpy as np

        self.grid_size = grid_size
        self.n_peaks = n_peaks
        self.noise_level = noise_level

        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.zeros_like(self.X)

        # Generate random peaks
        for _ in range(n_peaks):
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            amplitude = np.random.uniform(5, 10)
            sigma_x = np.random.uniform(0.05, 0.15)
            sigma_y = np.random.uniform(0.05, 0.15)

            self.Z += amplitude * np.exp(
                -(
                    (self.X - x_center) ** 2 / (2 * sigma_x**2)
                    + (self.Y - y_center) ** 2 / (2 * sigma_y**2)
                )
            )

        # Add noise
        self.Z += np.random.randn(*self.Z.shape) * noise_level
        self.Z = np.maximum(self.Z, 0)

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the gold richness for a DataFrame of locations.

        Takes a DataFrame with 'x' and 'y' columns (as returned by
        ``campaign.recommend()``), adds a 'gold_richness' column, and returns
        the DataFrame. This makes it directly usable as a lookup function for
        ``simulate_scenarios``.

        Args:
            df: DataFrame with 'x' and 'y' columns. Typically the output of
                ``campaign.recommend()``.

        Returns:
            The input DataFrame with an added 'gold_richness' column.
        """
        import numpy as np
        from scipy.interpolate import RegularGridInterpolator

        x_coords = self.X[0, :]
        y_coords = self.Y[:, 0]
        interpolator = RegularGridInterpolator((y_coords, x_coords), self.Z)

        points = np.column_stack([df["y"].values, df["x"].values])
        df["gold_richness"] = interpolator(points)
        return df

    def plot(self, samples: pd.DataFrame | None = None, title: str = "Gold Mining Landscape"):
        """Visualize the landscape with optional sample points.

        Args:
            samples: Optional DataFrame with 'x', 'y', and 'value' columns.
            title: Plot title.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create gold-themed colormap
        colors = ["#2C1810", "#8B4513", "#DAA520", "#FFD700", "#FFED4E"]
        cmap = LinearSegmentedColormap.from_list("gold", colors, N=100)

        # Plot landscape
        contour = ax.contourf(self.X, self.Y, self.Z, levels=20, cmap=cmap, alpha=0.8)
        plt.colorbar(contour, ax=ax, label="Gold Richness")

        # Plot sample points if provided
        if samples is not None and len(samples) > 0:
            ax.scatter(
                samples["x"],
                samples["y"],
                c=samples["value"],
                s=100,
                cmap=cmap,
                edgecolors="black",
                linewidths=2,
                marker="X",
                vmin=self.Z.min(),
                vmax=self.Z.max(),
                zorder=5,
            )

            # Add numbers to show sampling order
            for idx, row in samples.iterrows():
                ax.annotate(
                    str(idx + 1),
                    (row["x"], row["y"]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="circle,pad=0.3", facecolor="black", alpha=0.7),
                )

        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        return fig
