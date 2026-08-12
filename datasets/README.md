# Datasets

This directory contains the scripts required to fetch, verify, and preprocess the datasets used throughout the textbook and companion notebooks.

**Note:** Raw and processed data files (`*.csv`, `*.tar.gz`, etc.) are explicitly ignored in the `.gitignore` to keep the repository lightweight.

## Available Scripts
* `fetch_standard_datasets.py`: Downloads toy datasets (e.g., Iris, California Housing, Breast Cancer) directly via `scikit-learn`.
* `download_external.sh`: A bash script leveraging `wget` and `curl` to pull larger datasets (e.g., ImageNet subsets, Shakespeare corpus for LLM training) from external URLs.
