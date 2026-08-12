import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def run_sklearn_classification() -> None:
    """Evaluates various classical classifiers using scikit-learn."""
    print("--- Scikit-Learn Classification Pipeline ---")
    
    # Dataset Loading and Preprocessing
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Support Vector Machine (RBF Kernel)": SVC(kernel='rbf', probability=True)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds) * 100.0
        print(f"{name} Accuracy: {acc:.1f}%")
    print("\n")

def run_pytorch_logistic_regression() -> None:
    """Logistic Regression implemented as a PyTorch neural network."""
    print("--- PyTorch Logistic Regression ---")
    
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_t = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # A single linear layer followed by a sigmoid is equivalent to Logistic Regression
    model = nn.Sequential(
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    
    criterion = nn.BCELoss() # Binary Cross-Entropy
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_preds_class = (test_preds >= 0.5).float()
        acc = (test_preds_class == y_test_t).float().mean() * 100.0
        print(f"PyTorch LR Accuracy: {acc.item():.1f}%")

if __name__ == "__main__":
    run_sklearn_classification()
    run_pytorch_logistic_regression()
