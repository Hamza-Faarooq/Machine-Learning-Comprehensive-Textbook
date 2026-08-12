import numpy as np

class FairnessMetrics:
    """
    Computes statistical fairness metrics for binary classification models[cite: 1].
    """
    @staticmethod
    def demographic_parity_difference(y_pred: np.ndarray, sensitive_attribute: np.ndarray) -> float:
        """
        Measures whether the prediction rate is independent of the sensitive attribute[cite: 1].
        Ideal value is 0.0.
        """
        prob_group_1 = np.mean(y_pred[sensitive_attribute == 1.0])
        prob_group_0 = np.mean(y_pred[sensitive_attribute == 0.0])
        return float(np.abs(prob_group_1 - prob_group_0))

    @staticmethod
    def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attribute: np.ndarray) -> float:
        """
        Measures the difference in True Positive Rates (Recall) between groups[cite: 1].
        Ideal value is 0.0.
        """
        mask_1 = (sensitive_attribute == 1.0) & (y_true == 1.0)
        mask_0 = (sensitive_attribute == 0.0) & (y_true == 1.0)
        
        tpr_1 = np.sum(y_pred[mask_1]) / np.sum(mask_1) if np.sum(mask_1) > 0.0 else 0.0
        tpr_0 = np.sum(y_pred[mask_0]) / np.sum(mask_0) if np.sum(mask_0) > 0.0 else 0.0
        
        return float(np.abs(tpr_1 - tpr_0))

if __name__ == "__main__":
    # Dummy data
    y_true_dummy = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    y_pred_dummy = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    sensitive_attr = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) # 1: Minority, 0: Majority
    
    dp_diff = FairnessMetrics.demographic_parity_difference(y_pred_dummy, sensitive_attr)
    eo_diff = FairnessMetrics.equal_opportunity_difference(y_true_dummy, y_pred_dummy, sensitive_attr)
    
    print("--- Fairness Metrics ---")
    print(f"Demographic Parity Difference: {dp_diff:.1f}")
    print(f"Equal Opportunity Difference:  {eo_diff:.1f}")
