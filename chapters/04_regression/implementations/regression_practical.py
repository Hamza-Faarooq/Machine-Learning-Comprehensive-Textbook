import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def run_sklearn_pipeline() -> None:
    """End-to-end regression pipeline using scikit-learn."""
    print("--- Scikit-Learn Regression Pipeline ---")
    
    # Dataset Loading and Preprocessing
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Ordinary Least Squares
    ols_model = LinearRegression()
    ols_model.fit(X_train_scaled, y_train)
    ols_preds = ols_model.predict(X_test_scaled)
    print(f"OLS MSE: {mean_squared_error(y_test, ols_preds):.1f}")
    
    # 2. Ridge Regression (L2)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_preds = ridge_model.predict(X_test_scaled)
    print(f"Ridge MSE: {mean_squared_error(y_test, ridge_preds):.1f}")
    
    # 3. LASSO Regression (L1)
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train_scaled, y_train)
    lasso_preds = lasso_model.predict(X_test_scaled)
    print(f"LASSO MSE: {mean_squared_error(y_test, lasso_preds):.1f}")
    print("\n")

def run_pytorch_pipeline() -> None:
    """Linear regression implemented as a single-layer PyTorch network."""
    print("--- PyTorch Regression Pipeline ---")
    
    # Dataset Loading and Preprocessing
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Model Definition
    model = nn.Linear(in_features=20, out_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training Loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_scaled)
        loss = criterion(predictions, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_scaled)
        test_loss = criterion(test_preds, y_test_tensor)
        print(f"PyTorch Model Test MSE: {test_loss.item():.1f}")

if __name__ == "__main__":
    run_sklearn_pipeline()
    run_pytorch_pipeline()
