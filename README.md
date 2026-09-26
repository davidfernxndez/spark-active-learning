# Distributed Active Learning with Apache Spark

<p align="left">
  <strong>
    A distributed Active Learning framework in PySpark that combines
    uncertainty filtering and diversity-based instance selection,
    with experimental evaluation of predictive performance and scalability.
  </strong>
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.8.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/Apache%20Spark-3.3.0-orange" alt="Apache Spark">
  <img src="https://img.shields.io/badge/PySpark-3.3.0-orange" alt="PySpark">
  <img src="https://img.shields.io/badge/MLlib-Spark-orange" alt="Spark MLlib">
</p>

## 📑 Table of Contents

* [Overview](#-overview)
* [Scalable Design](#-scalable-design)
* [Key Results](#-key-results)
* [Project Structure](#-project-structure)
* [Dataset](#-dataset)
* [Environment and Installation](#-environment-and-installation)
* [Experimental Setup](#-experimental-setup)


## 📌 Overview

This project implements a **distributed Active Learning solution in PySpark** for binary classification, combining **uncertainty-based filtering** and **diversity-based instance selection**.

The objective is to reduce the number of instances that need to be labeled while maintaining or improving predictive performance. The proposed strategy first identifies the most uncertain instances and subsequently applies a clustering-based selection method to obtain a diverse set of representative samples.

The solution is designed with distributed data processing in mind, using **Spark DataFrames, RDDs, broadcast variables, and distributed aggregation** to process the instance-selection pipeline.

The experimental evaluation focuses on two complementary aspects:

* **Performance** — evolution of classification accuracy as the percentage of labeled data increases, compared with a random-selection baseline.
* **Scalability** — evaluation of the computational behavior of the solution through *Speed-up*, *Size-up*, and *Scale-up* experiments.


## ⚙️ Scalable Design

The solution is designed to keep the Active Learning instance-selection process scalable as the dataset size increases.

* **Distributed uncertainty filtering.** Model predictions and uncertainty estimation are performed on the distributed unlabeled pool, while `approxQuantile()` is used to estimate the uncertainty threshold without collecting the dataset on the Driver.

* **Candidate reduction before clustering.** Only the most uncertain instances are retained for the diversity stage, substantially reducing the amount of data that needs to be clustered.

* **Distributed diversity selection.** Candidates are clustered using **Bisecting K-Means** through Spark MLlib, keeping model fitting and cluster assignment distributed.

* **Local computation and distributed aggregation.** Distances to cluster centroids are computed locally on each worker, while `reduceByKey()` performs the selection of the closest candidate to each cluster in a distributed manner.

* **Minimizing data movement.** Cluster centroids and the small set of selected instance identifiers are distributed using **broadcast**, avoiding unnecessary shuffles of the larger datasets.

These design decisions allow the instance-selection pipeline to exploit Spark's distributed execution while limiting the amount of data exchanged between workers.


## 📈 Key Results

### Active Learning Performance

The proposed uncertainty- and diversity-based selection strategy achieves higher predictive performance with fewer labeled instances than the random-selection baseline in the evaluated experiment.

* **0.6415** — maximum accuracy achieved by random selection with **15%** of the data labeled.
* The proposed strategy surpasses this value with only **7%** of the data labeled.

### Scalability

The distributed implementation was evaluated using **Speed-up, Size-up, and Scale-up** experiments.

* **3.87× Speed-up** with 4 cores on the 1-million-instance dataset.
* **2.69× Size-up** when increasing the dataset from 100,000 to 1,000,000 instances.
* **2.30× execution-time increase** when both the dataset size and number of cores are increased by a factor of 10.

> The scalability results were obtained on a local workstation using Apache Spark in local mode and should therefore be interpreted within the evaluated hardware and dataset-size range.


## 📁 Project Structure

```text
.
├── data/
│   └── data_info.md
│
├── scripts/
│   ├── config.py
│   ├── download_data.py
│   ├── AL_methods.py
│   ├── AL_performance.py
│   ├── AL_scalability.py
│   └── plot_results.py
│
├── results/
│
├── main.ipynb
├── performance_experiments.ipynb
├── scalability_experiments.ipynb
│
├── environment.yml
└── README.md
```

* **`main.ipynb`** — Main project notebook, presenting the problem formulation, solution design, experimental methodology, and discussion of the results.
* **`performance_experiments.ipynb`** — Notebook containing the execution of experiments to measure the performance of the proposed solution against a random baseline in a number of Active Learning iterations.
* **`scalability_experiments.ipynb`** — Notebook containing the execution of the scalability experiments for *Speed-Up*, *Size-Up*, and *Scale-Up*.
* **`scripts/`** — Python modules and scripts implementing the Active Learning methods, data preparation, experiment execution, configuration, and result visualization.
* **`data/`** — Directory containing the datasets generated for the experiments. The datasets themselves are not included in the repository due to their size.
* **`results/`** — Directory containing the results generated by the experiments.


## 📊 Dataset

The experiments use the **HIGGS** dataset from the **UCI Machine Learning Repository**.

Due to its size, the dataset files are not included in the repository. Instead, they can be downloaded and prepared following the instructions provided in [`data/data_info.md`](data/data_info.md).


## ⚙️ Environment and Installation

The project was developed and evaluated using **Python 3.8.10** and **Apache Spark 3.3.0** within a Conda environment. The complete environment specification is provided in [`environment.yml`](environment.yml).

The Conda environment is based on the environment recommended in the book *Large-Scale Data Analytics with Python and Spark: A Hands-on Guide to Implementing Machine Learning Solutions by Isaac Triguero and Mikel Galar*. The book also provides different alternatives for installing and configuring Apache Spark and Java.

### Prerequisites

*Java* is required to run Apache Spark. The experiments were conducted using *OpenJDK 11*.

Verify the Java installation with:

```bash
java -version
```

A compatible Java installation must be available and correctly configured before running the Spark applications.

### Creating the Environment

From the root directory of the repository, create the Conda environment with:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate spark-active-learning
```

Although Conda is the recommended option, it is not strictly required. The project can also be executed in another Python environment provided that the required Python version, Apache Spark version, Java installation, and dependencies are available.


## 🖥️ Experimental Setup

The experiments were conducted on a local workstation with a **12th Gen Intel Core i5-12500H processor with 12 physical cores and 16 logical processors, 15.67 GB of RAM, running Microsoft Windows 11 Home 64-bit**.

All scalability experiments were performed on this device using **Apache Spark in local mode**.

---
