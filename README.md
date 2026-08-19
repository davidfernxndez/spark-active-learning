# Distributed Active Learning with Spark

## Project structure

- `config.py`: Global project configuration.
- `scripts/`: Python scripts used to generate data and run experiments.
- `data/`: Datasets used in the experiments.
- `results/`: Experimental results.
- `plots/`: Generated figures.

## Setup

Activate the required Conda environment:

```bash
conda activate pymlspark_book

## Download and prepare the dataset
From the project root directory, run:
python -m scripts.download_data


## Environment and Installation

The project was developed and evaluated using **Python 3.8.10** and **Apache Spark 3.3.0** within a Conda environment. The complete environment specification, including the required dependencies and their versions, is provided in [`environment.yml`](environment.yml).

The Conda environment used in this project is based on the environment recommended in the book *Large-Scale Data Analytics with Python and Spark: A Hands-on Guide to Implementing Machine Learning Solutions* by **Isaac Triguero and Mikel Galar**. The book provides several alternatives for installing and configuring Apache Spark and Java, which can be useful when setting up the required execution environment.

### Prerequisites

In addition to the Python dependencies specified in `environment.yml`, **Java is required to run Apache Spark**. The experiments in this project were developed using **OpenJDK 11**.

The Java installation can be verified with:

```bash
java -version
```

A compatible Java installation must be available and correctly configured for Spark to run.

### Creating the Conda Environment

Conda is the recommended option for reproducing the experimental environment. From the root directory of the repository, create the environment with:

```bash
conda env create -f environment.yml
```

Then, activate it with:

```bash
conda activate spark-active-learning
```

Although Conda is recommended, it is not strictly required. The project can also be executed using another Python environment, provided that **Python 3.8.10**, **Apache Spark 3.3.0**, a compatible **Java installation**, and the dependencies specified in `environment.yml` are available.

## Experimental Environment

The experiments were conducted on a local workstation with the following hardware configuration:

| Component               | Specification                        |
| ----------------------- | ------------------------------------ |
| **CPU**                 | 12th Gen Intel(R) Core(TM) i5-12500H |
| **Physical cores**      | 12                                   |
| **Logical processors**  | 16                                   |
| **Maximum clock speed** | 2.50 GHz                             |
| **RAM**                 | 15.67 GB                             |
| **Operating System**    | Microsoft Windows 11 Home (64-bit)   |

All scalability experiments were performed on this device using Apache Spark in local mode.
