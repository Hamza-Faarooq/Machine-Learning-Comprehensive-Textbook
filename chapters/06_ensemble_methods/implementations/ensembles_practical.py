import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

np.random.seed(42)

def run_tree_ensembles() -> None:
    """Evaluates various tree-based ensemble methods."""
    print("--- Tree-Based Ensemble Comparison ---")
    
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Random Forest (Bagging + Feature Subsampling)
    # Reduces variance by averaging deep, decorrelated trees[cite: 1]
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test)) * 100.0
    print(f"Random Forest Accuracy: {rf_acc:.1f}%")
    
    # 2. AdaBoost (Adaptive Boosting)
    # Reduces bias by upweighting misclassified examples[cite: 1]
    base_tree = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostClassifier(estimator=base_tree, n_estimators=100, algorithm="SAMME", random_state=42)
    ada.fit(X_train, y_train)
    ada_acc = accuracy_score(y_test, ada.predict(X_test)) * 100.0
    print(f"AdaBoost Accuracy: {ada_acc:.1f}%")
    
    # 3. Gradient Boosting
    # Fits trees sequentially to the pseudo-residuals of the loss function[cite: 1]
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test)) * 100.0
    print(f"Gradient Boosting Accuracy: {gb_acc:.1f}%")
    
    # 4. XGBoost (Extreme Gradient Boosting)
    # Utilizes a second-order Taylor expansion of the loss function[cite: 1]
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test)) * 100.0
    print(f"XGBoost Accuracy: {xgb_acc:.1f}%")
    print("\n")

def run_model_stacking() -> None:
    """Demonstrates cross-validated model stacking to prevent data leakage."""
    print("--- Model Stacking ---")
    
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Base learners should be diverse to correct shared systematic errors[cite: 1]
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ]
    
    # Meta-learner is typically a simple, regularised model[cite: 1]
    # StackingClassifier automatically handles the out-of-fold (OOF) predictions
    stacking_clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    stacking_clf.fit(X_train, y_train)
    stack_acc = accuracy_score(y_test, stacking_clf.predict(X_test)) * 100.0
    print(f"Stacked Ensemble Accuracy: {stack_acc:.1f}%")

if __name__ == "__main__":
    run_tree_ensembles()
    run_model_stacking()
