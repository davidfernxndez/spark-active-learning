"""
Download and prepare the HIGGS dataset.

This script:
    1. Downloads the original HIGGS dataset from the configured URL.
    2. Selects a stratified subset with TARGET_SAMPLES observations.
    3. Generates nested datasets containing 10%, 20%, 40%, 80%
       and 100% of the selected data.
    4. Preserves the original class distribution in all datasets.
    5. Stores all generated datasets as CSV files.

The generated files are:
    higgs-10.csv
    higgs-20.csv
    higgs-40.csv
    higgs-80.csv
    higgs-100.csv

The random seed and dataset-related parameters are defined in
the global config.py file to ensure reproducibility.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import urllib.request

from tqdm import tqdm
import pandas as pd

from config import (
    DATA_DIR,
    COMPRESSED_FILE_PATH,
    DATASET_URL,
    RANDOM_SEED,
    TARGET_SAMPLES,
    DATA_PERCENTAGES,
)


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print()
    print("=" * 60)
    print("HIGGS DATASET PREPARATION")
    print("=" * 60)

    # ==========================================================================
    # Download dataset
    # ==========================================================================

    print()
    print("[1/4] Downloading dataset")
    print("-" * 60)

    if not COMPRESSED_FILE_PATH.exists():

        print("Downloading HIGGS dataset...")
        print(f"Source: {DATASET_URL}")

        # Progress bar for the file download
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc="Download",
        ) as progress_bar:

            def update_progress(
                block_num,
                block_size,
                total_size,
            ):
                # Calculate the number of bytes downloaded
                downloaded = block_num * block_size

                # Update progress bar only with the new bytes
                progress_bar.update(
                    downloaded - progress_bar.n
                )

                # Set the total size once it is known
                if total_size > 0:
                    progress_bar.total = total_size

            urllib.request.urlretrieve(
                DATASET_URL,
                COMPRESSED_FILE_PATH,
                reporthook=update_progress,
            )

        print("Download completed.")

    else:
        print(
            "The compressed HIGGS dataset already exists. "
            "Skipping download."
        )

    # ==========================================================================
    # Load original dataset
    # ==========================================================================

    print()
    print("[2/4] Loading original dataset")
    print("-" * 60)

    df = pd.read_csv(
        COMPRESSED_FILE_PATH,
        compression="gzip",
        header=None,
    )

    # ==========================================================================
    # Assign column names
    # ==========================================================================

    # The first column is the label.
    # All remaining columns are features.

    n_features = len(df.columns) - 1

    df.columns = (
        ["label"]
        + [
            f"feature_{i}"
            for i in range(1, n_features + 1)
        ]
    )

    # ==========================================================================
    # Display original dataset information
    # ==========================================================================

    original_samples = len(df)

    print(f"Samples:  {original_samples:,}")
    print(f"Features: {n_features}")
    print(f"Columns:  {len(df.columns)}")

    original_distribution = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print()
    print("Class distribution:")

    for label, proportion in original_distribution.items():
        print(
            f"  Class {int(label)}: "
            f"{proportion:.2%}"
        )

    # ==========================================================================
    # Check target size
    # ==========================================================================

    if TARGET_SAMPLES > len(df):
        raise ValueError(
            f"TARGET_SAMPLES ({TARGET_SAMPLES:,}) is larger than "
            f"the original dataset ({len(df):,})."
        )

    # ==========================================================================
    # Create 100% experimental reference dataset
    # ==========================================================================

    print()
    print("[3/4] Creating 100% experimental reference dataset")
    print("-" * 60)

    if TARGET_SAMPLES < len(df):

        print(f"Original size:   {original_samples:,}")
        print(f"Target size:     {TARGET_SAMPLES:,}")
        print("Sampling method: Stratified sampling")
        print(f"Random seed:     {RANDOM_SEED}")

        stratified_parts = []

        for label, group in df.groupby("label"):

            # Get class distribution of the original dataframe
            class_proportion = len(group) / len(df)

            class_samples = round(
                TARGET_SAMPLES * class_proportion
            )

            sampled_group = group.sample(
                n=class_samples,
                random_state=RANDOM_SEED,
            )

            stratified_parts.append(sampled_group)

        df = pd.concat(
            stratified_parts,
            ignore_index=True,
        )

    else:

        print()
        print(
            "The complete original dataset will be used "
            "as the 100% reference dataset."
        )

    # ==========================================================================
    # Shuffle the selected dataset
    # ==========================================================================

    # Shuffle all samples so that the different classes are
    # randomly distributed throughout the dataset.

    df = (
        df
        .sample(
            frac=1,
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    # ==========================================================================
    # Assign unique sample identifiers
    # ==========================================================================

    # Create a unique integer identifier for each sample.
    # The identifier is independent of the class label and features.

    df.insert(
        0,
        "id_sample",
        range(len(df)),
    )

    # ==========================================================================
    # Display 100% reference dataset information
    # ==========================================================================

    reference_distribution = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print()
    print("100% reference dataset:")
    print(f"  Samples: {len(df):,}")

    print()
    print("Class distribution:")

    for label, proportion in reference_distribution.items():
        print(
            f"  Class {int(label)}: "
            f"{proportion:.2%}"
        )

    # ==========================================================================
    # Generate datasets
    # ==========================================================================

    print()
    print("[4/4] Generating datasets")
    print("-" * 60)

    n_total = len(df)

    for percentage in DATA_PERCENTAGES:

        # Calculate the number of samples corresponding to
        # the current percentage of the reference dataset.

        n_samples = round(
            n_total * percentage / 100
        )

        # ----------------------------------------------------------------------
        # Select the samples
        # ----------------------------------------------------------------------

        # Because df is already shuffled, taking the first
        # n_samples produces nested datasets:
        #
        #   10% ⊂ 20% ⊂ 40% ⊂ 80% ⊂ 100%

        subset = df.iloc[:n_samples].copy()

        # ----------------------------------------------------------------------
        # Output path
        # ----------------------------------------------------------------------

        output_path = (DATA_DIR / f"higgs-{percentage}.csv")

        # ----------------------------------------------------------------------
        # Save dataset
        # ----------------------------------------------------------------------

        subset.to_csv(
            output_path,
            index=False,
            header=True,
        )

        # ----------------------------------------------------------------------
        # Display information
        # ----------------------------------------------------------------------

        distribution = (
            subset["label"]
            .value_counts(normalize=True)
            .sort_index()
        )

        print()
        print(
            f"  {percentage:>3}% | "
            f"{len(subset):>10,} samples | "
            f"Class 0: {distribution.get(0, 0):.2%} | "
            f"Class 1: {distribution.get(1, 0):.2%}"
        )

        print( f"  Saved to: {output_path}")


    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()