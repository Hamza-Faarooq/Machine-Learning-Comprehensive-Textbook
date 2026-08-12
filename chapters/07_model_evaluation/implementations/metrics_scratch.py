import numpy as np

class ClassificationMetrics:
    """
    Binary classification metrics implemented from mathematical first principles.
    """
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """Computes TP, FP, TN, FN."""
        tp = np.sum((y_true == 1.0) & (y_pred == 1.0))
        fp = np.sum((y_true == 0.0) & (y_pred == 1.0))
        tn = np.sum((y_true == 0.0) & (y_pred == 0.0))
        fn = np.sum((y_true == 1.0) & (y_pred == 0.0))
        return tp, fp, tn, fn

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp, fp, tn, fn = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        return (tp + tn) / (tp + fp + tn + fn)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """P(y=1 | \hat{y}=1)[cite: 1]"""
        tp, fp, tn, fn = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        if (tp + fp) == 0.0:
            return 0.0
        return tp / (tp + fp)

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """P(\hat{y}=1 | y=1)[cite: 1]"""
        tp, fp, tn, fn = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        if (tp + fn) == 0.0:
            return 0.0
        return tp / (tp + fn)

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Harmonic mean of precision and recall[cite: 1]."""
        p = ClassificationMetrics.precision(y_true, y_pred)
        r = ClassificationMetrics.recall(y_true, y_pred)
        if (p + r) == 0.0:
            return 0.0
        return 2.0 * (p * r) / (p + r)

class RegressionMetrics:
    """
    Regression metrics implemented from mathematical first principles.
    """
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2.0)

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fraction of variance explained[cite: 1]."""
        ss_res = np.sum((y_true - y_pred) ** 2.0)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2.0)
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)

if __name__ == "__main__":
    y_true_clf = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    y_pred_clf = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    
    print("--- Classification Metrics ---")
    print(f"Accuracy:  {ClassificationMetrics.accuracy(y_true_clf, y_pred_clf):.1f}")
    print(f"Precision: {ClassificationMetrics.precision(y_true_clf, y_pred_clf):.1f}")
    print(f"Recall:    {ClassificationMetrics.recall(y_true_clf, y_pred_clf):.1f}")
    print(f"F1 Score:  {ClassificationMetrics.f1_score(y_true_clf, y_pred_clf):.1f}")
