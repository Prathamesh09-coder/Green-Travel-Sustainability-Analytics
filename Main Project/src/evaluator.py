import logging
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

class Evaluator:
    """Computes classifier evaluation metrics and optimizes decision thresholds."""
    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

    def evaluate(self, y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = None) -> Dict[str, Any]:
        """Calculates performance metrics for given predictions and targets."""
        # Find optimal threshold if not provided
        if threshold is None:
            threshold, best_f1 = self.find_optimal_threshold(y_true, y_pred_prob)
            self.logger.info(f"Using optimized threshold: {threshold:.3f} (Validation F1: {best_f1:.4f})")
        else:
            self.logger.info(f"Using manual threshold: {threshold:.3f}")
            
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        auc = roc_auc_score(y_true, y_pred_prob)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            "roc_auc": float(auc),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "threshold": float(threshold)
        }
        
        self.logger.info(f"--- Evaluation Metrics ---")
        self.logger.info(f"ROC-AUC:   {auc:.5f}")
        self.logger.info(f"Accuracy:  {accuracy:.5f}")
        self.logger.info(f"Precision: {precision:.5f}")
        self.logger.info(f"Recall:    {recall:.5f}")
        self.logger.info(f"F1-Score:  {f1:.5f}")
        self.logger.info(f"Confusion Matrix:\n{cm}")
        
        return metrics

    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> Tuple[float, float]:
        """Grid searches the classification threshold to maximize validation F1-score."""
        best_threshold = 0.5
        best_f1 = 0.0
        
        # Test thresholds from 0.01 to 0.99
        for threshold in np.linspace(0.01, 0.99, 99):
            y_pred = (y_pred_prob >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
                
        return float(best_threshold), float(best_f1)
