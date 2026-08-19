# Data Directory

This directory contains the datasets used in the Active Learning experiments.

The HIGGS datasets are **not included in the repository** due to their large size. They must be generated locally by executing the data preparation script provided in the repository.^

## Dataset Information

* **Dataset name:** HIGGS
* **Source:** UCI Machine Learning Repository
* **Dataset identifier:** HIGGS
* **Task:** Binary classification
* **Number of instances:** 11,000,000
* **Number of features:** 28
* **Feature type:** Numerical
* **Target variable:** Binary class label (`0` or `1`)
* **Original file:** `HIGGS.csv.gz`

The dataset was originally introduced in the following publication:
> Baldi, P., Sadowski, P., & Whiteson, D. (2014). Searching for exotic particles in high-energy physics with deep learning. *Nature Communications*, 5, 4308.

The dataset is available through the UCI Machine Learning Repository under the **HIGGS** dataset identifier. The download URL used by this project is stored in the `DATASET_URL` variable in `scripts/config.py`.

## Dataset Generation

To download and prepare the HIGGS dataset, execute:

```bash
python scripts/download_data.py
```

The script downloads the original HIGGS dataset from the **UCI Machine Learning Repository**.  After downloading the original dataset, the script generates the dataset subsets required by the experiments and stores them in this `data/` directory.

The generated datasets include:

* `higgs-10.csv`
* `higgs-20.csv`
* `higgs-40.csv`
* `higgs-80.csv`
* `higgs-100.csv`

The 100% dataset consists of a subsample of the original dataset, with the number of samples determined by the TARGET_SAMPLES variable defined in config.py. 
