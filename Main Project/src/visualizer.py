import os
import logging
from typing import Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

class Visualizer:
    """Generates evaluation plots and feature importance charts."""
    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.plots_dir = self.config.outputs["plots_dir"]
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_prob: np.ndarray, filename: str = "roc_curve.png") -> None:
        """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        out_path = os.path.join(self.plots_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        self.logger.info(f"Saved ROC curve plot to {out_path}")

    def plot_pr_curve(self, y_true: np.ndarray, y_pred_prob: np.ndarray, filename: str = "precision_recall_curve.png") -> None:
        """Plots and saves the Precision-Recall (PR) curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        out_path = os.path.join(self.plots_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        self.logger.info(f"Saved Precision-Recall curve plot to {out_path}")

    def plot_feature_importance(self, model: Any, feature_names: List[str], model_name: str, top_n: int = 20, filename: str = "feature_importance.png") -> None:
        """Plots and saves tabular feature importance for trees/linear models."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])
            else:
                self.logger.warning(f"Model {model_name} does not have importances/coefficients.")
                return
                
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 8))
            plt.title(f"Top {top_n} Feature Importances ({model_name})")
            plt.barh(range(top_n), importances[indices][::-1], align="center", color="teal")
            plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
            plt.xlabel("Relative Importance")
            plt.tight_layout()
            
            out_path = os.path.join(self.plots_dir, f"{model_name.lower()}_{filename}")
            plt.savefig(out_path, dpi=300)
            plt.close()
            self.logger.info(f"Saved feature importance plot to {out_path}")
        except Exception as e:
            self.logger.error(f"Error plotting feature importance for {model_name}: {e}")

    def plot_shap_summary(self, model: Any, X: pd.DataFrame, filename: str = "shap_summary.png") -> None:
        """Plots SHAP summary values for a subset of the dataset."""
        try:
            import shap
            self.logger.info("Computing SHAP values for summary plot...")
            
            # Use sample subset to avoid long runtimes
            sample_size = min(500, len(X))
            X_sample = X.sample(sample_size, random_state=42)
            
            # Simple TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(10, 8))
            # Handle list output for multiclass/binary in shap vs single array
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Binary classification list output
                shap.summary_plot(shap_values[1], X_sample, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, show=False)
                
            plt.tight_layout()
            out_path = os.path.join(self.plots_dir, filename)
            plt.savefig(out_path, dpi=300)
            plt.close()
            self.logger.info(f"Saved SHAP summary plot to {out_path}")
        except ImportError:
            self.logger.warning("SHAP library not found. Skipping SHAP plot generation.")
        except Exception as e:
            self.logger.error(f"Error generating SHAP plot: {e}")
