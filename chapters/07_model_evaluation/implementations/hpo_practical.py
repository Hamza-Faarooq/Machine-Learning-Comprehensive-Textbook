import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

def compare_search_strategies() -> None:
    """Compares Grid Search and Random Search for Hyperparameter Optimisation."""
    print("--- Hyperparameter Optimisation ---")
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(random_state=42)
    
    # Define a wide hyperparameter space
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 1. Grid Search (Exhaustive)
    # Total fits = 4 * 4 * 3 * 3 * 3 (cv) = 432 fits
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_distributions,
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    
    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    print(f"Grid Search Best CV Score: {grid_search.best_score_ * 100.0:.1f}%")
    
    # 2. Randomized Search
    # Much more sample-efficient; tries distinct values across important dimensions[cite: 1]
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=20, # Only 20 * 3 = 60 fits
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    print("Starting Random Search...")
    random_search.fit(X_train, y_train)
    print(f"Random Search Best CV Score: {random_search.best_score_ * 100.0:.1f}%")

if __name__ == "__main__":
    compare_search_strategies()
