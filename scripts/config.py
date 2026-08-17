from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
COMPRESSED_FILE_PATH = DATA_DIR / "HIGGS.csv.gz"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# Dataset
# ============================================================
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00280/HIGGS.csv.gz"
)
TARGET_SAMPLES = 1_000_000
DATA_PERCENTAGES = [10, 20, 40, 80, 100]

# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Active Learning configuration
# ============================================================

# Train/Test percentages split
TRAIN_TEST_SPLIT = [0.7, 0.3]

# Fraction of the total dataset assigned to the initial labeled set (N_L).
#
# Consequently, the initial unlabeled set (N_U) will contain the remaining
# fraction (1 - INITIAL_LABELED_FRACTION).
#
# Example:
#   For a total dataset N = 700,000:
#   - INITIAL_LABELED_FRACTION = 0.05
#   - Initial labeled set size (N_L) = 35,000
#   - Initial unlabeled set size (N_U) = 665,000
INITIAL_LABELED_FRACTION = 0.05

# Fraction used to compute the active learning iteration budget (B):
#  B = N * QUERY_BATCH_FRACTION
# Represents the proportion of samples queried from the oracle relative to the total dataset size (N).
#
# Example:
#   For a total dataset N= 700,000 and QUERY_BATCH_FRACTION = 0.01:
#   - Annotation budget per iteration (B) = 7000 samples.
QUERY_BATCH_FRACTION = 0.01


# Multiplier over the budget B that defines the target size ratio of the
# filtered candidate pool  before diversity sampling.
#
# Determines the quantile filtering threshold q = 1 - p, where:
# p = min(1.0, (UNCERTAINTY_CANDIDATE_FACTOR * B / N_U))
UNCERTAINTY_CANDIDATE_FACTOR = 10.0

# Relative error tolerance (epsilon) for approximate quantile estimation (q).
UNCERTAINTY_QUANTILE_EPSILON = 0.001

# ============================================================
# Spark
# ============================================================
APP_NAME = "ActiveLearning"
MASTER = "local[*]"