"""
This module implements the methods used in the Spark Active Learning experiments
for data loading, model training and evaluation, and instance selection.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# PySpark SQL
import pyspark.sql.functions as sql_f

# PySpark ML - Features, Models & Evaluation
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors

# Config variables
from config import (
    DATA_DIR,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
    INITIAL_LABELED_FRACTION,
    UNCERTAINTY_CANDIDATE_FACTOR,
    UNCERTAINTY_QUANTILE_EPSILON
)


def load_and_preprocess_data(
    spark,
    percentage_set,
    train_test_split = TRAIN_TEST_SPLIT,
    inital_labeled_fraction = INITIAL_LABELED_FRACTION,
    data_dir = DATA_DIR,
    seed = RANDOM_SEED
):
    """
    Loads and preprocesses the dataset for the active learning experiment.

    The preprocessing pipeline consists of:
    - Loading the selected dataset.
    - Casting columns to the required data types.
    - Assembling the feature vector.
    - Splitting the data into training and test sets.
    - Standardizing the features using a Standardscaler fitted exclusively on 
        the training set and then applied to both training and test data 
        to prevent data leakage.
    - Finally, each training instance is randomly initialized as either labeled
        (L) or unlabeled (U) according to the specified initial labeled fraction.
        The test set remains independent of the active learning labeling process.

    Parameters:
    ----------
    spark : pyspark.sql.SparkSession
        Active SparkSession used to load and preprocess the dataset.
    
    percentage_set : int
        Dataset size to load, expressed as a percentage of the original
        dataset. Supported values are 10, 20, 40, 80, and 100.
    
    train_test_split : list, default=TRAIN_TEST_SPLIT
        Two-element list specifying the proportions used to split the dataset
        into training and test sets.
    
    inital_labeled_fraction : float, default=INITIAL_LABELED_FRACTION
        Fraction of training instances initially assigned to the labeled
        state (L). The remaining training instances are assigned to the
        unlabeled state (U).
    
    seed : int, default=RANDOM_SEED
        Random seed used for the train/test split and the initial assignment
        of labeled and unlabeled instances.

    Returns:
    ----------
    train_df : pyspark.sql.DataFrame
        Preprocessed training DataFrame containing the ``id_sample``,
        ``label``, standardized ``features`` columns, and ``state`` 
        column indicating whether each instance is labeled (L) or
        unlabeled (U).

    test_df:  pyspark.sql.DataFrame
        Preprocessed test DataFrame containing the ``id_sample``, ``label``,
        and standardized ``features`` columns. This set is kept separate from
        the active learning labeling process.
    """
    print("load_and_preprocess_data():")
    # Load the selected dataset. 
    dataset_path = str(data_dir / f"higgs-{percentage_set}.csv")
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("sep", ",")
        .csv(dataset_path)
    )
    print(f"\t- Dataset loaded successfully from '{dataset_path}'")


    # Identify feature columns.
    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    # Build expressions to cast identifiers and features to the required types.
    cast_expressions = [
        sql_f.col("id_sample").cast("int"),
        sql_f.col("label").cast("int")
    ] + [sql_f.col(c).cast("float") for c in feature_cols]

    # Apply the required data types while preserving any remaining columns.
    remaining_cols = [
        c for c in df.columns
        if c not in ["id_sample", "label"] + feature_cols
    ]
    df = df.select(*cast_expressions, *remaining_cols)

    print("\t- Data types converted:\n\t\t-ID and label to INT\n\t\t-features to FLOAT.")

    # Assemble the individual feature columns into a single Spark ML vector.
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    df = assembler.transform(df).drop(*feature_cols)
    print("\t- Feature columns assembled into 'features' vector column")

    # Split Train/Test
    train_df, test_df = df.randomSplit(train_test_split, seed=seed)
    print("\t- Dataset split into training and test sets.") 

    # Configure feature standardization.
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True
    )

    # Fit the scaler exclusively on the training data to prevent data leakage.
    scaler_model = scaler.fit(train_df)

    # Standardize the training features.
    train_df = (
        scaler_model
        .transform(train_df)
        .drop("features")
        .withColumnRenamed("scaled_features", "features")
    )

    # Apply the same transformation to the test features.
    test_df = (
        scaler_model
        .transform(test_df)
        .drop("features")
        .withColumnRenamed("scaled_features", "features")
    )
    print("\t- Features standardized using statistics computed from the training set.") 

    # Randomly initialize the active learning state of each training instance.
    train_df = train_df.withColumn(
        "state",
        sql_f.when(
            sql_f.rand(seed) < inital_labeled_fraction,
            "L"
        ).otherwise("U")
    )
    print(
        f"\t- Training set initialized with approximately "
        f"{inital_labeled_fraction:.1%} labeled instances."
    )
    return train_df, test_df


def train_and_evaluate(train_df, test_df, model, metric_name="accuracy"):
    """
    Trains a classification model using the labeled training instances and
    evaluates its performance on the test set.

    Parameters
    ----------
    train_df : pyspark.sql.DataFrame
        DataFrame containing the training instances. Instances are expected to have a
        ``state`` column indicating whether they are labeled (L) or unlabeled (U).
    
    test_df : pyspark.sql.DataFrame
        DataFrame containing the test instances used to evaluate the trained model.
    
    model : pyspark.ml.Model
        Spark ML estimator to be trained on the labeled instances from train_df.
    
    metric_name : str, default="accuracy"
        Name of the classification metric used by MulticlassClassificationEvaluator
        to evaluate the predictions.

    Returns
    ------
    model_fit: pyspark.ml.Model
        Fitted model trained exclusively on the labeled instances from train_df.
    
    metric_value: float
        Value of the specified evaluation metric on the test set.
    """
    print("train_and_evaluate():")
    print("\t- Training model on labeled instances...")

    # Keep only instances currently marked as labeled.
    labeled_train_df = train_df.filter(sql_f.col("state") == "L")

    # Train the model using the labeled subset
    model_fit = model.fit(labeled_train_df)

    # Generate predictions for the test set.
    test_pred_df = model_fit.transform(test_df)

    # Evaluate the predictions using the specified classification metric.
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric_name
    )
    metric_value = evaluator.evaluate(test_pred_df)
    
    print(f"\t- Model evaluation completed -> {metric_name} = {metric_value:.4f} ") 

    return model_fit, metric_value


def get_uncertainty_candidates(
    train_df,
    model_fit,
    B,
    N_U,
    alpha=UNCERTAINTY_CANDIDATE_FACTOR,
    epsilon=UNCERTAINTY_QUANTILE_EPSILON 
):
    """
    Identifies a candidate pool of unlabeled instances with the highest
    uncertainty according to the predictions of a fitted model.

    First, the fitted model is applied to all unlabeled instances to obtain
    their class probabilities. An uncertainty score is then computed for each
    instance, where values closer to 1 indicate greater uncertainty.

    The candidate pool is obtained by applying a quantile-based threshold.
    The proportion of instances retained is defined as:

        p = min(1.0, (alpha * B) / |U_t|)

    and the corresponding uncertainty threshold is the (1 - p)-quantile.
    Spark's ``approxQuantile`` method is used to estimate this quantile in a
    distributed manner using the Greenwald-Khanna algorithm.
     
    Parameters
    ----------
    train_df : pyspark.sql.DataFrame
        DataFrame containing the training instances. Instances are expected to have a
        ``state`` column indicating whether they are labeled (L) or unlabeled (U).
    
    model_fit: pyspark.ml.Model
        Fitted model trained exclusively on the labeled instances from train_df
        and used to obtain class probabilities for the unlabeled instances.
    
    B: int
        Number of instances selected for labeling in each iteration (Active learning budget).
    
    N_U: int
        Number of currently unlabeled instances in train_df.

    alpha : int, default=UNCERTAINTY_CANDIDATE_FACTOR
        Multiplier over the budget B that defines the target size ratio of the
        filtered candidate pool

    epsilon : float, default=UNCERTAINTY_QUANTILE_EPSILON
        Relative error tolerance used by ``approxQuantile`` for estimating
        the uncertainty quantile.

    Returns
    ------
    candidates_df : pyspark.sql.DataFrame
        DataFrame containing the unlabeled candidate instances with the columns
        ``id_sample``, ``features``, and ``uncertainty``. The candidates
        correspond to the unlabeled instances whose uncertainty is greater
        than or equal to the estimated quantile threshold.
    """
    print("get_uncertainty_candidates():")
    print("\t- Computing uncertainty scores for unlabeled instances...")

    # Keep only unlabeled instances
    unlabeled_df = train_df.filter(sql_f.col("state") == "U")

    # Apply the fitted model to obtain class probabilities for each instance.
    unlabeled_pred_df = model_fit.transform(unlabeled_df) 

    # Compute the uncertainty score from the positive-class probability.
    # Values closer to 1 correspond to predictions closer to the decision
    # boundary and therefore to higher model uncertainty.
    unlabeled_pred_df = unlabeled_pred_df.withColumn(
        "uncertainty",
        1 - 2 * sql_f.abs(
            vector_to_array("probability")[1] - 0.5
        )
    )
    print("\t- Computing threshold using approxQuantile...")
    # Determine the proportion of unlabeled instances to retain.
    p = min(1.0, (alpha * B) / N_U)

    # Estimate the uncertainty threshold using a distributed approximate
    # quantile computation.
    threshold_val = unlabeled_pred_df.stat.approxQuantile(
        "uncertainty", [1.0 - p], epsilon
    )[0]

    print(f"\t- Filtering unlabeled instances with uncertainty >= {threshold_val:.4f}.")

    # Filter unlabeled instances to get the candidates set
    candidates_df = (
        unlabeled_pred_df
        .select("id_sample", "features", "uncertainty")
        .filter(sql_f.col("uncertainty") >= threshold_val)
    )

    return candidates_df


def diversity_k_means_selection(candidates_df, train_df, B, sc, seed = RANDOM_SEED):
    """
	Selects a diverse batch of B unlabeled instances from the uncertainty-based
	candidate pool using BisectingK-Means, selecting the instance closest to each cluster
	centroid. 

    Cluster centroids are broadcast to the worker nodes so that distances can
    be computed locally without repeatedly transferring the centroid data.
    The closest instance within each cluster is then obtained using a
    distributed reduceByKey operation, which requires only a shuffle of
    the compact (cluster_id, candidate) pairs.


    Finally, the IDs of the B selected instances are broadcast to all worker 
    nodes, while the full train_df dataset remains in-place without being 
    shuffled across the network. 

    A Broadcast Hash Join is performed to identify the target instances, 
    simulating the active learning oracle labeling step by updating their 
    state from ``U`` (unlabeled) to ``L`` (labeled). Broadcasting the small 
    set of selected IDs  instead of performing a full Shuffle Hash 
    or Sort-Merge Join reduces network I/O  and execution latency.

    Parameters
    ----------
    candidates_df : pyspark.sql.DataFrame
        DataFrame containing the unlabeled candidates instances with the columns
        ``id_sample``, ``features``, and ``uncertainty``.

    train_df : pyspark.sql.DataFrame
        DataFrame containing the training instances. Instances are expected to have a
        ``state`` column indicating whether they are labeled (L) or unlabeled (U).

    B: int
        Number of instances selected for labeling in each iteration (Active learning budget).
        K-Means is configured with B clusters.

    sc : pyspark.SparkContext
        Active Spark context used to broadcast the cluster centroids to the
        worker nodes.

    seed: int, default = RANDOM_SEED
        Random seed used to initialize the K-Means clustering algorithm.

    Returns:
    ----------
    updated_train_df : pyspark.sql.DataFrame
        Updated train_df in which the selected instances have their
        ``state`` set to ``"L"``. All other instance states remain unchanged.
    """
    print("diversity_k_means_selection():")
    print(f"\t- Running K-Means diversity selection with k={B}...")

    # Fit K-Means on the uncertainty-based candidate pool.
    kmeans = BisectingKMeans(
        k=B,
        seed=seed,
        featuresCol="features",
        predictionCol="cluster_id",
        maxIter=20,
    )
    kmeans_model = kmeans.fit(candidates_df)
    clustered_candidates_df = kmeans_model.transform(candidates_df)

    # Broadcast the cluster centroids so that each worker can access them
    # locally when computing the distance of a candidate to its centroid.
    # This avoids repeatedly transferring the small centroid array during
    # the distributed distance computation.
    centers_broadcast = sc.broadcast(kmeans_model.clusterCenters())

    print(f"\t- Selecting the candidate closest to each cluster centroid....")

    # Convert the clustered DataFrame to an RDD containing:
    #   key   -> cluster ID
    #   value -> (sample ID, squared distance to the cluster centroid)
    #
    # The broadcast centroids can be accessed locally on each worker through
    # centers_broadcast.value.
    rdd_mapped = clustered_candidates_df.rdd.map(
        lambda row: (
            row.cluster_id,
            (
                row.id_sample,
                float(Vectors.squared_distance(row.features, centers_broadcast.value[row.cluster_id]))
            )
        )
    )

    # For each cluster, retain the candidate with the minimum distance
    # to the centroid. reduceByKey performs this aggregation in a distributed
    # manner and can combine candidates locally before the shuffle, reducing
    # the amount of data transferred between workers.
    closest_per_cluster_rdd = rdd_mapped.reduceByKey(
        lambda candidate1, candidate2:
            candidate1 if candidate1[1] < candidate2[1] else candidate2
    )

    print(f"\t- Marking {B} selected instances as labeled in train_df...")
    
    # Create a compact DataFrame containing only the IDs of the selected instances.
    selected_ids_df = closest_per_cluster_rdd.map(lambda x: (x[1][0],)).toDF(["selected_id"])

    # Broadcast the small set of selected IDs to avoid shuffling the large
    # training DataFrame during the join.
    updated_train_df = (
        train_df.join(
            sql_f.broadcast(selected_ids_df),
            train_df["id_sample"] == selected_ids_df["selected_id"],
            how="left"
        )
        .withColumn(
            "state",
            sql_f.when(
                sql_f.col("selected_id").isNotNull(),
                sql_f.lit("L")
            ).otherwise(sql_f.col("state"))
        )
        .drop("selected_id")
    )

    # Release the broadcast variable once it is no longer required.
    centers_broadcast.destroy()

    return updated_train_df


def random_selection(train_df, B, seed= RANDOM_SEED):
    """
    Randomly selects a batch of unlabeled instances and marks them as labeled.

    A Broadcast Hash Join is performed to identify the target instances, 
    simulating the active learning oracle labeling step by updating their 
    state from ``U`` (unlabeled) to ``L`` (labeled). Broadcasting the small 
    set of selected IDs  instead of performing a full Shuffle Hash 
    or Sort-Merge Join reduces network I/O  and execution latency.

    Parameters
    ----------
    train_df : pyspark.sql.DataFrame
        DataFrame containing the training instances. Instances are expected to have a
        ``state`` column indicating whether they are labeled (L) or unlabeled (U).

    B : int
        Number of instances selected for labeling in each iteration (Active learning budget).

    Returns
    -------
    updated_train_df : pyspark.sql.DataFrame
        Updated training DataFrame in which the selected instances have their
        ``state`` set to ``L``. All other instance states remain unchanged.
    """

    print("random_selection():")

    # Randomly select the required number of instances from the unlabeled pool.
    selected_ids = (train_df
        .filter(sql_f.col("state") == "U")
        .orderBy(sql_f.rand(seed=seed))
        .limit(B)
        .select("id_sample")
        .withColumn("selected", sql_f.lit(1))
    )
    print(f"\t- Marking {B} random selected instances as labeled in train_df...")
    
    # Broadcast the small set of selected IDs to avoid shuffling the larger
    # training DataFrame during the join.
    #
    # The join identifies the instances selected by the active learning
    # strategy and simulates the oracle labeling them by changing their state
    # from "U" (unlabeled) to "L" (labeled).
    updated_train_df = (
        train_df
        .join(
            sql_f.broadcast(selected_ids),
            on="id_sample",
            how="left"
        )
        .withColumn(
            "state",
            sql_f.when(
                sql_f.col("selected") == 1,
                sql_f.lit("L")
            ).otherwise(sql_f.col("state"))
        )
        .drop("selected")
    )


    return updated_train_df