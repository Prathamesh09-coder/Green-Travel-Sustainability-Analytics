import pandas as pd
import logging
from typing import Tuple
from src.config import Config

class DataLoader:
    """Class to load, clean, and validate training and test datasets."""
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix string encoding issues such as 'SÃ£o Paulo'."""
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.replace("SÃ£o Paulo", "São Paulo", regex=False)
            # Remove any trailing/leading whitespaces
            df[col] = df[col].str.strip()
        return df

    def load_public_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads and cleans the training (public) dataset files."""
        self.logger.info("Loading public trip details, attributes, and event logs...")
        
        trip_data = pd.read_csv(self.config.data["train_data_path"])
        attributes = pd.read_csv(self.config.data["train_attributes_path"])
        log = pd.read_csv(self.config.data["train_log_path"])

        # Clean text encodings
        trip_data = self._clean_text(trip_data)
        attributes = self._clean_text(attributes)
        log = self._clean_text(log)

        self.logger.info(
            f"Loaded successfully. Trip details shape: {trip_data.shape}, "
            f"Attributes shape: {attributes.shape}, Log shape: {log.shape}"
        )
        
        # Log target balance
        target_col = self.config.features["target_col"]
        if target_col in trip_data.columns:
            balance = trip_data[target_col].value_counts(normalize=True).to_dict()
            self.logger.info(f"Target variable '{target_col}' balance: {balance}")
            
        return trip_data, attributes, log

    def load_private_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads and cleans the test (private) dataset files."""
        self.logger.info("Loading private trip details, attributes, and event logs...")
        
        trip_data = pd.read_csv(self.config.data["test_data_path"])
        attributes = pd.read_csv(self.config.data["test_attributes_path"])
        log = pd.read_csv(self.config.data["test_log_path"])

        trip_data = self._clean_text(trip_data)
        attributes = self._clean_text(attributes)
        log = self._clean_text(log)

        self.logger.info(
            f"Loaded successfully. Private details shape: {trip_data.shape}, "
            f"Private attributes shape: {attributes.shape}, Private log shape: {log.shape}"
        )
        return trip_data, attributes, log
