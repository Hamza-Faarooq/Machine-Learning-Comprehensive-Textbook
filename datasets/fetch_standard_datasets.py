import os
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing

def setup_directories() -> None:
    os.makedirs("raw", exist_ok=True)
    os.makedirs("processed", exist_ok=True)

def fetch_and_save_california_housing() -> None:
    """Fetches the California Housing dataset for regression tasks."""
    print("Fetching California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Save raw data
    df.to_csv("raw/california_housing.csv", index=False)
    print(f"Saved: raw/california_housing.csv ({len(df)} rows)")

def fetch_and_save_breast_cancer() -> None:
    """Fetches the Breast Cancer dataset for binary classification tasks."""
    print("Fetching Breast Cancer dataset...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    # Save raw data
    df.to_csv("raw/breast_cancer.csv", index=False)
    print(f"Saved: raw/breast_cancer.csv ({len(df)} rows)")

if __name__ == "__main__":
    setup_directories()
    fetch_and_save_california_housing()
    fetch_and_save_breast_cancer()
    print("Dataset fetching complete.")
