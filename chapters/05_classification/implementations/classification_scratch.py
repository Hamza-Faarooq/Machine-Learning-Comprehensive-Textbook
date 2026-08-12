import numpy as np
from typing import Tuple, Optional

class LogisticRegressionScratch:
    """
    Binary Logistic Regression implemented from scratch.
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        self.theta: Optional[np.ndarray] = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Maps real numbers to (0, 1).
        Equation: \sigma(z) = \frac{1}{1 + e^{-z}}[cite: 1]
        """
        return 1.0 / (1.0 + np.exp(-z))

    def add_intercept(self, X: np.ndarray) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> list:
        """
        Optimizes the binary cross-entropy loss using gradient descent.
        Gradient: \nabla_\theta J = \frac{1}{n} X^T (\hat{p} - y)[cite: 1]
        """
        X_b = self.add_intercept(X)
        n = X_b.shape[0]
        self.theta = np.zeros(X_b.shape[1])
        loss_history = []

        for _ in range(self.epochs):
            z = X_b @ self.theta
            p_hat = self._sigmoid(z)
            
            # Binary Cross-Entropy Loss
            # Adding a small epsilon to prevent log(0)
            epsilon = 1e-9
            loss = -(1.0 / n) * np.sum(
                y * np.log(p_hat + epsilon) + (1.0 - y) * np.log(1.0 - p_hat + epsilon)
            )
            loss_history.append(loss)
            
            # Gradient Update
            gradients = (1.0 / n) * (X_b.T @ (p_hat - y))
            self.theta -= self.lr * gradients
            
        return loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_b = self.add_intercept(X)
        return self._sigmoid(X_b @ self.theta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

class LinearSVMScratch:
    """
    Linear Support Vector Machine optimized via SGD on the Hinge Loss.
    """
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, epochs: int = 1000) -> None:
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the SVM using the Soft-Margin Hinge Loss formulation[cite: 1].
        Labels must be strictly {-1, 1}.
        """
        # Ensure labels are -1 and 1
        y_ = np.where(y <= 0.0, -1.0, 1.0)
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1.0
                if condition:
                    # Point is strictly outside the margin; apply only regularisation
                    self.w -= self.lr * (2.0 * self.lambda_param * self.w)
                else:
                    # Point violates the margin; update weights and bias
                    self.w -= self.lr * (2.0 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

if __name__ == "__main__":
    np.random.seed(42)
    # Toy dataset for binary classification
    X_dummy = np.array([[0.1, 0.2], [1.5, 1.8], [0.2, 0.1], [2.0, 2.1]])
    y_dummy = np.array([0.0, 1.0, 0.0, 1.0])
    
    log_reg = LogisticRegressionScratch(epochs=5000)
    log_reg.fit(X_dummy, y_dummy)
    preds = log_reg.predict(X_dummy)
    
    print("Logistic Regression Predictions:")
    for p in preds:
        print(f"{float(p):.1f}")
