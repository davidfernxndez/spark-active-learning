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