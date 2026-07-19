import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

# Disable Optuna log verbosity by default to keep logs clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelTrainer:
    """Manages cross-validated training, tuning, and ensembling for tabular classifiers."""
    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.trained_models: Dict[str, List[Any]] = {}
        self.validation_predictions: Dict[str, np.ndarray] = {}
        self.val_scores: Dict[str, float] = {}
        self.ensemble_weights: Dict[str, float] = {}

    def _get_lgb_early_stopping(self):
        try:
            import lightgbm as lgb
            return [lgb.early_stopping(stopping_rounds=self.config.training["early_stopping_rounds"], verbose=False)]
        except Exception:
            return None

    def tune_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Runs Bayesian optimization using Optuna to tune LightGBM parameters."""
        self.logger.info("Starting hyperparameter tuning for LightGBM...")
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "random_state": self.config.training["random_state"],
                "verbosity": -1
            }
            
            skf = StratifiedKFold(
                n_splits=3, # Use 3 splits for speed during tuning
                shuffle=True,
                random_state=self.config.training["random_state"]
            )
            
            scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                model = LGBMClassifier(**params)
                callbacks = self._get_lgb_early_stopping()
                
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks
                )
                
                preds = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, preds))
                
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.config.training["optuna_trials"],
            timeout=self.config.training["optuna_timeout"]
        )
        
        self.logger.info(f"Tuning complete. Best ROC-AUC: {study.best_value:.5f}")
        return study.best_params

    def train_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any] = None) -> float:
        """Trains a single model type using 5-Fold Stratified Cross-Validation."""
        self.logger.info(f"Training cross-validated {model_name}...")
        
        # Load default parameters from config
        params = self.config.models.get(model_name, {}).copy()
        if best_params:
            params.update(best_params)
            
        params["random_state"] = self.config.training["random_state"]
        
        skf = StratifiedKFold(
            n_splits=self.config.training["n_splits"],
            shuffle=True,
            random_state=self.config.training["random_state"]
        )
        
        models_list = []
        oof_preds = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            if model_name == "lightgbm":
                model = LGBMClassifier(**params, verbosity=-1)
                callbacks = self._get_lgb_early_stopping()
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            elif model_name == "xgboost":
                model = XGBClassifier(**params, early_stopping_rounds=self.config.training["early_stopping_rounds"])
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif model_name == "catboost":
                model = CatBoostClassifier(**params, early_stopping_rounds=self.config.training["early_stopping_rounds"])
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], use_best_model=True, verbose=False)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
                
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            models_list.append(model)
            
            # Save checkpoints
            model_dir = self.config.outputs["models_dir"]
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(model, os.path.join(model_dir, f"{model_name}_fold_{fold}.pkl"))
            
        score = roc_auc_score(y, oof_preds)
        self.logger.info(f"{model_name} Out-Of-Fold ROC-AUC: {score:.5f}")
        
        self.trained_models[model_name] = models_list
        self.validation_predictions[model_name] = oof_preds
        self.val_scores[model_name] = score
        
        return score

    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Finds optimal ensembling weights for validation predictions using soft voting."""
        self.logger.info("Building Weighted Ensemble...")
        
        # Verify we have trained models
        model_names = list(self.validation_predictions.keys())
        if not model_names:
            raise RuntimeError("Train at least one model before building the ensemble.")
            
        # If only one model, assign weight 1.0
        if len(model_names) == 1:
            self.ensemble_weights = {model_names[0]: 1.0}
            return self.val_scores[model_names[0]]
            
        # Optimization loop to find optimal weights maximizing ROC-AUC
        best_score = 0.0
        best_weights = {}
        
        # Grid search over simple weight combinations
        from itertools import product
        # Generate combinations summing up to 1.0
        weight_grids = []
        for p in product(np.linspace(0, 1, 11), repeat=len(model_names)):
            if np.isclose(sum(p), 1.0):
                weight_grids.append(p)
                
        for w in weight_grids:
            ensemble_preds = np.zeros(len(y))
            for i, model in enumerate(model_names):
                ensemble_preds += w[i] * self.validation_predictions[model]
                
            score = roc_auc_score(y, ensemble_preds)
            if score > best_score:
                best_score = score
                best_weights = {name: w[idx] for idx, name in enumerate(model_names)}
                
        self.ensemble_weights = best_weights
        self.logger.info(f"Optimal Ensemble Weights: {self.ensemble_weights}")
        self.logger.info(f"Ensemble Out-Of-Fold ROC-AUC: {best_score:.5f}")
        
        return best_score

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Generates ensembled probability predictions for unseen private test set."""
        ensemble_preds = np.zeros(len(X))
        
        for model_name, weight in self.ensemble_weights.items():
            if weight == 0:
                continue
            models = self.trained_models[model_name]
            model_preds = np.zeros(len(X))
            
            for model in models:
                model_preds += model.predict_proba(X)[:, 1] / len(models)
                
            ensemble_preds += weight * model_preds
            
        return ensemble_preds
