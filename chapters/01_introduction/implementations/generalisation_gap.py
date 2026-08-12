import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generate_data(n_samples: int = 30) -> tuple:
    """Generates synthetic data from a true sine wave with Gaussian noise."""
    np.random.seed(42)
    X = np.sort(np.random.rand(n_samples, 1) * 2.0 * np.pi, axis=0)
    # True function is a sine wave
    y_true = np.sin(X).ravel()
    # Add noise with a variance of 0.5
    y_noisy = y_true + np.random.randn(n_samples) * 0.5
    return X, y_noisy, y_true

def evaluate_capacity() -> None:
    """
    Demonstrates the fundamental problem of generalization by fitting
    polynomials of increasing capacity to noisy data[cite: 1].
    """
    X_train, y_train, _ = generate_data(n_samples=20)
    X_test, y_test, _ = generate_data(n_samples=20) # Held-out test set
    
    degrees = [1, 3, 15]
    
    print("--- Generalization Gap Analysis ---")
    
    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        X_poly_train = poly_features.fit_transform(X_train)
        X_poly_test = poly_features.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        
        train_preds = model.predict(X_poly_train)
        test_preds = model.predict(X_poly_test)
        
        train_error = mean_squared_error(y_train, train_preds)
        test_error = mean_squared_error(y_test, test_preds)
        generalization_gap = test_error - train_error
        
        print(f"Degree {degree:02d} | Train MSE: {train_error:7.1f} | Test MSE: {test_error:7.1f} | Gap: {generalization_gap:7.1f}")

if __name__ == "__main__":
    evaluate_capacity()
