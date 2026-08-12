import numpy as np
from typing import Tuple, Optional

class LinearRegressionScratch:
    """
    Ordinary Least Squares (OLS) Regression implemented from scratch.
    """
    def __init__(self) -> None:
        self.theta: Optional[np.ndarray] = None

    def add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Adds a column of ones to the design matrix X."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model using the Normal Equations.
        Equation: \hat{\theta} = (X^T X)^{-1} X^T y[cite: 1]
        """
        X_b = self.add_intercept(X)
        # Using pseudoinverse for numerical stability in case of singular matrices[cite: 1]
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def fit_gradient_descent(
        self, X: np.ndarray, y: np.ndarray, 
        learning_rate: float = 0.01, epochs: int = 1000
    ) -> list:
        """
        Fits the model using Batch Gradient Descent.
        Gradient: \nabla_\theta J = \frac{1}{n} X^T (X\theta - y)[cite: 1]
        """
        X_b = self.add_intercept(X)
        n = X_b.shape[0]
        self.theta = np.zeros(X_b.shape[1])
        loss_history = []

        for _ in range(epochs):
            predictions = X_b @ self.theta
            errors = predictions - y
            
            # Compute Mean Squared Error
            loss = (1.0 / (2.0 * n)) * np.sum(errors ** 2)
            loss_history.append(loss)
            
            # Compute gradients and update
            gradients = (1.0 / n) * (X_b.T @ errors)
            self.theta -= learning_rate * gradients
            
        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target values for given input features."""
        if self.theta is None:
            raise ValueError("Model must be fitted before predicting.")
        X_b = self.add_intercept(X)
        return X_b @ self.theta

class RidgeRegressionScratch(LinearRegressionScratch):
    """
    Ridge Regression (L2 Regularization) implemented from scratch.
    """
    def __init__(self, lambda_param: float = 1.0) -> None:
        super().__init__()
        self.lambda_param = lambda_param

    def fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model using the regularised Normal Equations.
        Equation: \hat{\theta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y[cite: 1]
        """
        X_b = self.add_intercept(X)
        d = X_b.shape[1]
        
        # Identity matrix, but we do not regularize the intercept term[cite: 1]
        I = np.eye(d)
        I[0, 0] = 0.0 
        
        self.theta = np.linalg.inv(X_b.T @ X_b + self.lambda_param * I) @ X_b.T @ y

if __name__ == "__main__":
    # Simple test to ensure numerical formatting aligns with rigorous standards
    np.random.seed(42)
    X_dummy = np.array([[1.0], [2.0], [3.0]])
    y_dummy = np.array([2.0, 4.0, 6.0])
    
    model = LinearRegressionScratch()
    model.fit_closed_form(X_dummy, y_dummy)
    
    print("Test Predictions (Closed Form):")
    for pred in model.predict(X_dummy):
        print(f"{pred:.1f}")
