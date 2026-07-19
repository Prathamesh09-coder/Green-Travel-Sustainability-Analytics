import os
import argparse
import json
import logging
from typing import Dict, Any
import pandas as pd
from src.config import Config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
from src.visualizer import Visualizer

def setup_logging(logs_dir: str) -> None:
    """Configures system logs to output both to console and a log file."""
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "pipeline_execution.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ]
    )

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for pipeline customization."""
    parser = argparse.ArgumentParser(description="End-to-End GreenTravel ML Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="project/configs/config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Bayesian optimization (Optuna) to tune hyperparameters"
    )
    return parser.parse_args()

def main() -> None:
    """Orchestrates the entire ML pipeline from loading data to saving predictions."""
    args = parse_args()
    
    # Load settings
    config = Config(args.config)
    setup_logging(config.outputs["logs_dir"])
    logger = logging.getLogger("Pipeline")
    logger.info("Initializing GreenTravel Sustainability ML Pipeline...")
    
    # Instantiate modules
    loader = DataLoader(config)
    engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    evaluator = Evaluator(config)
    visualizer = Visualizer(config)
    
    try:
        # 1. Load Data
        trip_train, attr_train, log_train = loader.load_public_data()
        trip_test, attr_test, log_test = loader.load_private_data()
        
        # 2. Feature Engineering
        X_train, y_train = engineer.build_features(trip_train, attr_train, log_train, is_train=True)
        X_test, _ = engineer.build_features(trip_test, attr_test, log_test, is_train=False)
        
        # Verify no column leakage
        leakage_overlap = set(X_train.columns).intersection(set(config.features["prohibited_leakage_cols"]))
        if leakage_overlap:
            raise ValueError(f"CRITICAL: Prohibited target/leakage features found in X_train: {leakage_overlap}")
            
        # 3. Hyperparameter Tuning (Optional)
        best_params = None
        if args.tune:
            best_params = trainer.tune_lightgbm(X_train, y_train)
            
        # 4. Model Training
        logger.info("Training individual classifiers...")
        trainer.train_single_model("lightgbm", X_train, y_train, best_params=best_params)
        trainer.train_single_model("xgboost", X_train, y_train)
        trainer.train_single_model("catboost", X_train, y_train)
        
        # 5. Ensemble Training
        trainer.train_ensemble(X_train, y_train)
        
        # 6. Evaluation
        logger.info("Evaluating Out-Of-Fold predictions on training set...")
        ensemble_oof_preds = trainer.validation_predictions["lightgbm"] * trainer.ensemble_weights.get("lightgbm", 0.0) + \
                             trainer.validation_predictions["xgboost"] * trainer.ensemble_weights.get("xgboost", 0.0) + \
                             trainer.validation_predictions["catboost"] * trainer.ensemble_weights.get("catboost", 0.0)
                             
        metrics = evaluator.evaluate(y_train, ensemble_oof_preds)
        
        # Save metrics JSON
        metrics_path = os.path.join(config.outputs["plots_dir"], "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved evaluation metrics to {metrics_path}")
        
        # 7. Visualization
        logger.info("Generating evaluation and feature importance charts...")
        visualizer.plot_roc_curve(y_train, ensemble_oof_preds)
        visualizer.plot_pr_curve(y_train, ensemble_oof_preds)
        
        # Feature importances for LightGBM fold 0
        lgb_model = trainer.trained_models["lightgbm"][0]
        visualizer.plot_feature_importance(lgb_model, X_train.columns.tolist(), "LightGBM")
        
        # SHAP summary plot (uses subset inside)
        visualizer.plot_shap_summary(lgb_model, X_train)
        
        # 8. Inference on Test Set
        logger.info("Generating predictions for the private test set...")
        test_preds = trainer.predict_probability(X_test)
        
        # Save submission
        submission = pd.read_csv(config.data["sample_submission_path"])
        submission["HighCarbon"] = test_preds
        
        os.makedirs(config.outputs["submission_dir"], exist_ok=True)
        sub_path = os.path.join(config.outputs["submission_dir"], "submission.csv")
        submission.to_csv(sub_path, index=False)
        logger.info(f"Generated submission file successfully at: {sub_path}")
        logger.info(f"Submission Shape: {submission.shape}")
        logger.info(f"First 10 Rows:\n{submission.head(10)}")
        
        logger.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.exception(f"CRITICAL ERROR in pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()
