"""
This script provides the main entry point for evaluating the scalability of
the implemented Active Learning strategies in Spark.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import pandas as pd
import time

# PySpark SQL
from pyspark.sql import SparkSession
import pyspark.sql.functions as sql_f

# PySpark ML
from pyspark.ml.classification import LogisticRegression

# Config variables
from config import (
    RESULTS_DIR,
    DATA_PERCENTAGES,
    TRAIN_TEST_SPLIT,
    INITIAL_LABELED_FRACTION,
    QUERY_BATCH_FRACTION,
)

# Import the Spark-based methods used to implement the Active Learning workflow.
from AL_methods import *

def main(argv):
    """
    Run an Active Learning scalability experiment using Apache Spark.

    The experiment loads and preprocesses the specified dataset, initializes
    the labeled and unlabeled training sets, and applies one iteration of the
    selected instance sampling strategy.
    
    The experiment supports two instance selection strategies:
    - random: randomly selects instances from the unlabeled pool.
    - smart: first filters instances based on model uncertainty and
      then selects a diverse batch using K-Means clustering.

    Parameters
    ----------
    argv : list
        Command-line arguments excluding the program name.

        Expected arguments:
            argv[0] : int
                Number of Spark cores to use.

            argv[1] : int
                Percentage of the dataset to use. Must be one of the values
                defined in ``DATA_PERCENTAGES``.

            argv[2] : str
                Instance selection strategy. Supported values are
                ''random'' and ''smart''. If an unsupported value is
                provided, ''smart'' is used by default.


            argv[4] : str, optional
                Output CSV filename. If omitted, ''AL_scalability.csv''
                is used.

    Returns
    -------
    None
        The experiment results are written to the specified output CSV file.
    """

    # Check that the required arguments are provided.
    if len(argv) < 2:
        print(
            "Usage: python script.py <cores> <data_percentage> "
            "[selection_method] [output_file]"
        )
        sys.exit(1)

    # Parse the number of cores.
    try:
        num_cores = int(argv[0])
    except ValueError:
        print("Error: number of cores must be an integer.")
        sys.exit(1)

    # Validate the requested number of cores against the available cores.
    available_cores = os.cpu_count()

    if num_cores <= 0:
        print("Error: number of cores must be greater than 0.")
        sys.exit(1)

    if num_cores > available_cores:
        print(
            f"Error: requested {num_cores} cores, but only "
            f"{available_cores} cores are available."
        )
        sys.exit(1)

    # Parse and validate the dataset size.
    try:
        percentage_set = int(argv[1])
    except ValueError:
        print("Error: dataset percentage must be an integer.")
        sys.exit(1)

    if percentage_set not in DATA_PERCENTAGES:
        print(
            f"Error: invalid dataset percentage {percentage_set}. "
            f"Expected one of {DATA_PERCENTAGES}."
        )
        sys.exit(1)

    # Use the smart selection strategy by default.
    selection_method = "smart"

    if len(argv) >= 3:
        if argv[2] in ("random", "smart"):
            selection_method = argv[2]
        else:
            print(
                f"Warning: unknown selection method '{argv[2]}'. "
                "Using 'smart' instead."
            )


    # Use the default output filename if none is provided.
    output_filename = argv[3] if len(argv) >= 4 else "AL_scalability.csv"
    output_file = RESULTS_DIR / output_filename

    print("\n" + "=" * 60)
    print(" STARTING SPARK ACTIVE LEARNING")
    print("=" * 60)
    spark = (
        SparkSession.builder.master(f"local[{num_cores}]")
        .appName(f"Active learning with {num_cores} partitions")
        .config("spark.driver.memory", "16g")
        .config("spark.executor.memory", "16g")
        .getOrCreate()
    )

    # Get SparkContext associated to SparkSession
    sc = spark.sparkContext
    print(f"Spark session initialized successfully on '{sc.master}'")

    print("\n" + "=" * 60)
    print(" LOAD AND PREPROCESS DATASET")
    print("=" * 60)
    train_df, test_df = load_and_preprocess_data(spark, percentage_set)

    # Cache both DataFrames because they are repeatedly accessed throughout the
    # iterative Active Learning process. The test set remains unchanged and is
    # reused for evaluation at every iteration, while the training set is updated
    # after each iteration but is repeatedly accessed by the selection, training,
    # and state-update operations.
    test_df.cache()
    train_df.cache()

    # Compute the initial number of labeled and unlabeled training instances (materialize cache)
    stats = train_df.select(
        sql_f.count(sql_f.when(sql_f.col("state") == "L", True)).alias("labeled"),
        sql_f.count(sql_f.when(sql_f.col("state") == "U", True)).alias("unlabeled")
    ).first()

    labeled_size = stats["labeled"]
    unlabeled_size = stats["unlabeled"]
    train_size = labeled_size + unlabeled_size

    # Compute the test (materialize cache) and total dataset sizes.
    test_size = test_df.count()
    total_size = train_size + test_size

    # Compute the number of instances queried from the oracle at each iteration.
    query_batch = int(total_size * TRAIN_TEST_SPLIT[0] * QUERY_BATCH_FRACTION)

    print("\n" + "=" * 60)
    print("EXPERIMENT SETUP")
    print("=" * 60)
    print("Dataset configuration:")
    print(f"  Train/Test split: {TRAIN_TEST_SPLIT}")
    print(f"  Total dataset size: {total_size:,} instances")
    print(f"  Training set size: {train_size:,} instances")
    print(f"  Test set size: {test_size:,} instances")

    print("\nInitial training set state:")
    print(
        f"  Labeled (L):   {labeled_size:,} instances "
        f"({labeled_size / train_size * 100:.2f}%)"
    )
    print(
        f"  Unlabeled (U): {unlabeled_size:,} instances "
        f"({unlabeled_size / train_size * 100:.2f}%)"
    )
    print(f"  Initial labeled fraction: {INITIAL_LABELED_FRACTION * 100}% of training set size.")


    print("\nActive Learning configuration:")
    print(f"  Number of iterations: 1")
    print(f"  Query budget per iteration: {query_batch:,} instances")
    print(f"  Query batch fraction: {QUERY_BATCH_FRACTION * 100}% of training set size.")

    print("\nInstance selection strategy:")
    if selection_method == "random":
        print("  Random selection")
    elif selection_method == "smart":
        print("  Uncertainty filtering + K-Means diversity selection")

    print("\n" + "=" * 60)
    print("INITIAL MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    # Start the timer after data loading and preprocessing. The measured execution
    # time covers one Active Learning iteration, including model training on the
    # labeled data, prediction on the unlabeled pool, uncertainty-based candidate
    # filtering, and K-Means diversity selection.
    start = time.time()

    # Initialize the Logistic Regression estimator used as the classification model.
    lr_model = LogisticRegression(
        labelCol="label",
        featuresCol="features"
    )

    # Train the model on the initially labeled training instances and evaluate
    # its performance on the independent test set
    lr_model_fit, _ = train_and_evaluate(train_df, test_df, model=lr_model, metric_name="accuracy")

    print("\n" + "=" * 60)
    print("ACTIVE LEARNING ITERATION")
    print("=" * 60)

    # Limit the query batch to the number of instances currently available
    # in the unlabeled pool.
    query_batch = min(query_batch, unlabeled_size)

    if selection_method == "random":
        print("Random selection method started.")
        new_train_df = random_selection(train_df, query_batch)
    else:
        print("Uncertainty filtering + K-Means diversity selection method started.")
        # Phase 1: Filter a candidate pool containing the most uncertain
        # unlabeled instances.
        print("\nPhase 1/2: Filtering candidates by uncertainty...")
        uncertainty_candidates_df = get_uncertainty_candidates(
            train_df, lr_model_fit, query_batch, unlabeled_size
        )

        # Phase 2: select a diverse batch from the uncertainty-based
        # candidate pool using K-Means clustering.
        print(
            f"\nPhase 2/2: Selecting {query_batch:,} diverse instances "
            "using K-Means..."
        )

    new_train_df = diversity_k_means_selection(uncertainty_candidates_df, train_df, query_batch, sc)

    # Release the previous training DataFrame from cache and cache the updated
    # version so it can be efficiently reused in the next Active Learning iteration.
    train_df.unpersist()
    train_df = new_train_df.cache()


    # Recompute the number of labeled and unlabeled instances after the
    # oracle-labeling step (Materialize cache)
    stats = train_df.select(
        sql_f.count(sql_f.when(sql_f.col("state") == "L", True)).alias("labeled"),
        sql_f.count(sql_f.when(sql_f.col("state") == "U", True)).alias("unlabeled")
    ).first()
    labeled_size = stats["labeled"]
    unlabeled_size = stats["unlabeled"]
    train_size = labeled_size + unlabeled_size

    print("\nTraining set state after labeling:")
    print(
        f"  Labeled (L):   {labeled_size:,} instances "
        f"({labeled_size / train_size * 100:.2f}%)"
    )
    print(
        f"  Unlabeled (U): {unlabeled_size:,} instances "
        f"({unlabeled_size / train_size * 100:.2f}%)"
    )

    # Stop the timer after completing the Active Learning iteration
    end = time.time()

    # Store results
    results_df = pd.DataFrame([{
        "method": selection_method,
        "percentage_set": percentage_set,
        "cores": num_cores,
        "time": end - start
    }])
    file_exists = os.path.exists(output_file)
    results_df.to_csv(
        output_file,
        mode = "a" if file_exists else "w",
        header = not file_exists,
        index = False
    )

    # Close SparkSession to release resources
    spark.stop()

if __name__ == "__main__":
    main(sys.argv[1:])