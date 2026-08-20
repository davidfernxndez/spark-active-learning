"""
This module implements the methods used to plot the performance and
scalability results obtained in Spark Active Learning experiments
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Config variables
from .config import RESULTS_DIR

def plot_performance_results(file_name, metric_name = "accuracy", custom_range = [0.6, 0.7]):
    """
    Displays Active Learning performance results as:
    - Comparison table shows the accuracy achieved by Random Selection
    and the Uncertainty + K-Means selection strategy for each labeled
    percentage
    - Accuracy plots:  One using the complete accuracy
    range from 0 to 1, and another focused on a detailed view to
    highlight small differences between the selection strategies.

    Parameters
    ----------
    file_name : str 
        Name of the CSV file containing the Active Learning performance
        results.

    custom_range : list
        Custom accuracy range of the detailed view plot. 
    Returns
    -------
    None
    """

    # Load the performance results from the CSV file.
    results_df = pd.read_csv(RESULTS_DIR / file_name)

    # Reshape the results so that each selection strategy has its own column.
    comparison_df = (
        results_df[
            ["method", "labeled_percentage", metric_name]
        ]
        .pivot(
            index="labeled_percentage",
            columns="method",
            values=metric_name
        )
        .reset_index()
    )

    # Rename columns to provide descriptive names in the comparison table.
    comparison_df = comparison_df.rename(
        columns={
            "random": "Random Selection",
            "smart": "Uncertainty + K-means"
        }
    )
    # Round accuracy values to three decimal places for display.
    accuracy_columns = [
        "Random Selection",
        "Uncertainty + K-means"
    ]

    comparison_df[accuracy_columns] = comparison_df[accuracy_columns].round(4)
    # Display the comparison table in the notebook.
    display(comparison_df)

    # Create two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot both selection strategies on each subplot.
    for method, label in [
        ("smart", "Uncertainty + K-means selection"),
        ("random", "Random Selection")
    ]:
        method_df = results_df[results_df["method"] == method]

        # Full accuracy range.
        axes[0].plot(
            method_df["labeled_percentage"],
            method_df[metric_name],
            marker="o",
            label=label
        )

        # Zoomed accuracy range.
        axes[1].plot(
            method_df["labeled_percentage"],
            method_df[metric_name],
            marker="o",
            label=label
        )

    # Configure the full-scale plot.
    axes[0].set_xlabel("Labeled Percentage (%)")
    axes[0].set_ylabel(metric_name)
    axes[0].set_title("Full Accuracy Range")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)
    axes[0].legend()

    # Configure the zoomed-in plot.
    axes[1].set_xlabel("Labeled Percentage (%)")
    axes[1].set_ylabel(metric_name)
    axes[1].set_title("Detailed View")
    axes[1].set_ylim(custom_range[0], custom_range[1])
    axes[1].grid(True)
    axes[1].legend()

    # Global title
    fig.suptitle("Active Learning Performance", fontsize=14)

    # Adjust the layout to prevent the global title from overlapping the plots.
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


def plot_scalability(speed_up_filename, size_up_filename, scale_up_filename):
    """
    Plot Speed-Up, Size-Up, and Scale-Up results from scalability experiments.

    The function reads the execution-time results stored in the specified files from
    ''RESULTS_DIR''. For each experiment, repeated measurements are averaged by
    selection method, number of cores, and dataset size before computing the corresponding
    scalability metric.

    Speed-Up is computed relative to the average execution time using one
    core and the same dataset size:

        Speed-Up(n) = T(1) / T(n)

    Size-Up is computed relative to the average execution time for 10% of
    the dataset using the same number of cores:

        Size-Up = T(percentage) / T(10%)

    Scale-Up is computed relative to the configuration using one core and
    10% of the dataset:

        Scale-Up = T(1 core, 10%) / T(n cores, percentage)

    Parameters
    -------
    speed_up_filename: str
        Filename os speed up experiment results.  

    size_up_filename: str
        Filename os size up experiment results.  

    scale_up_filename: str
        Filename os scale up experiment results.  

    Returns
    -------
    None
        Displays the scalability plots.
    """

    # Set file paths
    speed_up_path = RESULTS_DIR / speed_up_filename
    size_up_path = RESULTS_DIR / size_up_filename
    scale_up_path = RESULTS_DIR / scale_up_filename

    # Load the experimental results.
    speed_up_df = pd.read_csv(speed_up_path)
    size_up_df = pd.read_csv(size_up_path)
    scale_up_df = pd.read_csv(scale_up_path)

    # Compute the mean execution time for repeated experiments.
    speed_up_mean = (
        speed_up_df
        .groupby(["method", "cores", "percentage_set"], as_index=False)
        ["time"]
        .mean()
    )

    size_up_mean = (
        size_up_df
        .groupby(["method", "cores", "percentage_set"], as_index=False)
        ["time"]
        .mean()
    )

    scale_up_mean = (
        scale_up_df
        .groupby(["method", "cores", "percentage_set"], as_index=False)
        ["time"]
        .mean()
    )

    # ============================================================
    # SPEED-UP
    # ============================================================

    # Obtain the reference execution time for each method and dataset size.
    # The reference corresponds to the configuration using one core.
    speed_reference = (
        speed_up_mean[speed_up_mean["cores"] == 1]
        [["method", "percentage_set", "time"]]
        .rename(columns={"time": "reference_time"})
    )

    speed_up_mean = speed_up_mean.merge(
        speed_reference,
        on=["method", "percentage_set"],
        how="left"
    )

    speed_up_mean["speed_up"] = (
        speed_up_mean["reference_time"] /
        speed_up_mean["time"]
    )

    # ============================================================
    # SIZE-UP
    # ============================================================

    # Obtain the reference execution time for each method and number
    # of cores using 10% of the dataset.
    size_reference = (
        size_up_mean[size_up_mean["percentage_set"] == 10]
        [["method", "cores", "time"]]
        .rename(columns={"time": "reference_time"})
    )

    size_up_mean = size_up_mean.merge(
        size_reference,
        on=["method", "cores"],
        how="left"
    )

    size_up_mean["size_up"] = (
        size_up_mean["time"] /
        size_up_mean["reference_time"]
    )

    # ============================================================
    # SCALE-UP
    # ============================================================

    # Obtain the execution time of the reference configuration:
    # one core processing 10% of the dataset.
    scale_reference = (
        scale_up_mean[
            (scale_up_mean["cores"] == 1) &
            (scale_up_mean["percentage_set"] == 10)
        ]
        [["method", "time"]]
        .rename(columns={"time": "reference_time"})
    )

    scale_up_mean = scale_up_mean.merge(
        scale_reference,
        on="method",
        how="left"
    )

    scale_up_mean["scale_up"] = (
        scale_up_mean["reference_time"] /
        scale_up_mean["time"]
    )

    # Create a categorical label representing the core/dataset-size pair.
    scale_up_mean["configuration"] = (
        scale_up_mean["cores"].astype(str)
        + "C / "
        + scale_up_mean["percentage_set"].astype(str)
        + "%"
    )

    # Sort configurations by number of cores and dataset size.
    scale_up_mean = scale_up_mean.sort_values(
        ["cores", "percentage_set"]
    )

    # ============================================================
    # PLOT SCALABILITY GRAHPS
    # ============================================================

    # -----------
    # SPEED UP
    # -----------

    plt.figure(figsize=(8, 5))

    # Plot speed up for each dataset Size
    for percentage_set in sorted(speed_up_mean["percentage_set"].unique()):
        data = speed_up_mean[
            speed_up_mean["percentage_set"] == percentage_set
        ].sort_values("cores")

        plt.plot(
            data["cores"],
            data["speed_up"],
            marker="o",
            label=f"Dataset Size {percentage_set}%"
        )

    # Lineal Speed Up
    cores = sorted(speed_up_mean["cores"].unique())
    plt.plot(
        cores,
        cores,
        color="black",
        linestyle="--",
        marker="o",
        label="Linear speed Up"
    )

    # Configuration
    plt.xlabel("Number of Cores")
    plt.ylabel("Speed-Up")
    plt.title("Speed-Up scalability analysis")
    plt.xticks(cores)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------
    # SIZE UP
    # -----------
    plt.figure(figsize=(8, 5))

    # Plot Size Up for each number of cores
    for cores in sorted(size_up_mean["cores"].unique()):
        data = size_up_mean[
            size_up_mean["cores"] == cores
        ].sort_values("percentage_set")

        plt.plot(
            data["percentage_set"],
            data["size_up"],
            marker="o",
            label=f"{cores} cores Size-Up"
        )

    # Linear Size Up
    percentage_set = sorted(size_up_mean["percentage_set"].unique())

    plt.plot(
        percentage_set,
        percentage_set / percentage_set[0],
        color="black",
        linestyle="--",
        marker="o",
        label="Linear Size Up"
    )

    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Size-Up")
    plt.title("Size-Up scalability Analysis")
    plt.xticks(percentage_set)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # -----------
    # SCALE UP
    # -----------

    plt.figure(figsize=(8, 5))

    plt.plot(
        scale_up_mean["configuration"],
        scale_up_mean["scale_up"],
        marker="o",
        label="Real Scale Up"
    )

    # Linear Scale-Up
    plt.axhline(
        y=1,
        color="black",
        linestyle="--",
        label="Linear Scale Up"
    )

    # Configuration
    plt.xlabel("Cores / Dataset Size")
    plt.ylabel("Scale-Up")
    plt.ylim([0,1.1])

    plt.title("Scale-Up scalability analysis")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()