import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Any
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineer:
    """Performs feature engineering on trip details, event attributes, and event logs."""
    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.fitted = False
        self.numerical_cols: List[str] = []
        self.categorical_cols_to_encode: List[str] = []
        self.ohe_features: List[str] = []
        
    def _extract_event_log_features(self, log_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates the event log table to trip-level (Case ID) features."""
        self.logger.info("Extracting features from event log...")
        
        # Parse timestamps
        log_df = log_df.copy()
        log_df["EventTimestamp"] = pd.to_datetime(log_df["EventTimestamp"])
        
        # Sort log to guarantee StepOrder alignment
        log_df = log_df.sort_values(by=["TripID", "StepOrder"])
        
        # Aggregate base statistics
        log_grouped = log_df.groupby("TripID")
        
        # Start and end timestamps
        timestamps = log_grouped["EventTimestamp"].agg(["min", "max", "count"])
        timestamps["Process_Duration_Hours"] = (timestamps["max"] - timestamps["min"]).dt.total_seconds() / 3600.0
        timestamps = timestamps.rename(columns={"count": "Total_Events"})
        
        # Unique events
        unique_events = log_grouped["EventName"].nunique().rename("Unique_Events_Count")
        
        # Specific event counts
        event_names = log_df["EventName"].unique()
        event_counts = {}
        for event in event_names:
            col_name = f"Count_Event_{event.replace(' ', '_')}"
            event_counts[col_name] = log_grouped["EventName"].apply(lambda s: (s == event).sum())
            
        event_counts_df = pd.DataFrame(event_counts)
        
        # Extract temporal features from the start of the trip process
        first_timestamps = timestamps["min"]
        temporal_df = pd.DataFrame(index=timestamps.index)
        temporal_df["Start_Hour"] = first_timestamps.dt.hour
        temporal_df["Start_Month"] = first_timestamps.dt.month
        temporal_df["Start_DayOfWeek"] = first_timestamps.dt.dayofweek
        temporal_df["Start_IsWeekend"] = (first_timestamps.dt.dayofweek >= 5).astype(int)
        temporal_df["Start_Quarter"] = first_timestamps.dt.quarter
        
        # Compute specific lead times/delays
        # We find the first timestamp of key activities
        def get_activity_time(activity: str) -> pd.Series:
            filtered = log_df[log_df["EventName"] == activity]
            first_occ = filtered.groupby("TripID")["EventTimestamp"].min()
            return first_occ

        t_book = get_activity_time("Book Mode of Transportation")
        t_submit = get_activity_time("Submit Travel Request")
        t_approve = get_activity_time("Travel Request Approved")
        t_depart = get_activity_time("Take Departure Flight")
        if t_depart.isnull().all():
            t_depart = get_activity_time("Take Departure Train")
            
        t_exp_submit = get_activity_time("Submit Expense Request")
        t_exp_reimburse = get_activity_time("Expense Reimbursement")

        lead_times = pd.DataFrame(index=timestamps.index)
        lead_times["LeadTime_Book_to_Submit_Days"] = (t_submit - t_book).dt.total_seconds() / (24 * 3600.0)
        lead_times["LeadTime_Submit_to_Approve_Days"] = (t_approve - t_submit).dt.total_seconds() / (24 * 3600.0)
        lead_times["LeadTime_Book_to_Depart_Days"] = (t_depart - t_book).dt.total_seconds() / (24 * 3600.0)
        lead_times["LeadTime_Expense_Submit_to_Reimburse_Days"] = (t_exp_reimburse - t_exp_submit).dt.total_seconds() / (24 * 3600.0)

        # Impute missing lead times with -1 and add indicator flags
        for col in lead_times.columns:
            lead_times[f"{col}_Missing"] = lead_times[col].isnull().astype(int)
            lead_times[col] = lead_times[col].fillna(-1.0)
            
        # Combine log features
        log_features = pd.concat([
            timestamps[["Process_Duration_Hours", "Total_Events"]],
            unique_events,
            event_counts_df,
            temporal_df,
            lead_times
        ], axis=1)
        
        return log_features

    def _extract_event_attributes_features(self, attr_df: pd.DataFrame) -> pd.DataFrame:
        """Parses and cleans the event attributes table."""
        self.logger.info("Extracting features from event attributes...")
        attr_df = attr_df.copy()
        
        # Drop columns not used or leaking info
        drop_cols = ["TravelRequestID", "ExpenseRequestID"]
        # Also drop EmployeeNumber if present, since it is missing in the private test set
        if "EmployeeNumber" in attr_df.columns:
            drop_cols.append("EmployeeNumber")
            
        attr_df = attr_df.drop(columns=[c for c in drop_cols if c in attr_df.columns])
        
        # Replace NaN values for numeric attribute fields
        attr_df["ExpenseReimbursementAmount"] = attr_df["ExpenseReimbursementAmount"].fillna(0.0)
        attr_df["TransportationPriceDifference"] = attr_df["TransportationPriceDifference"].fillna(0.0)
        attr_df["ExtensionLength"] = attr_df["ExtensionLength"].fillna(0)
        attr_df["DaysPreapproved"] = attr_df["DaysPreapproved"].fillna(0)
        
        # Create cancellation and hotel change indicator flags
        attr_df["Has_Transport_Cancellation"] = attr_df["ReasonForTransportCancellation"].notnull().astype(int)
        attr_df["Has_Hotel_Change"] = attr_df["NewHotelSelection"].notnull().astype(int)
        attr_df["Has_Transportation_Change"] = attr_df["NewModeOfTransportation"].notnull().astype(int)
        attr_df["Has_Delay"] = attr_df["ReasonForDelay"].notnull().astype(int)
        
        # Impute missing categorical attributes with a string constant 'None'
        cat_cols = [
            "ExpenseDenialReason", "ExpenseReimbursementReason", "ReasonForTransportCancellation",
            "NewTransportSelection", "ReasonForNewTransport", "NewHotelSelection", "ReasonForNewHotel",
            "NewModeOfTransportation", "ReasonForTransportationChange", "ReasonForDelay", "ProcessCode"
        ]
        for col in cat_cols:
            if col in attr_df.columns:
                attr_df[col] = attr_df[col].fillna("None")
                
        # Set TripID as index
        attr_df = attr_df.set_index("TripID")
        return attr_df

    def build_features(self, trip_df: pd.DataFrame, attr_df: pd.DataFrame, log_df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Combines all tables and builds final feature matrices for ML training/inference."""
        self.logger.info(f"Building features (is_train={is_train})...")
        trip_df = trip_df.copy()
        
        # Separate target label if training
        target_col = self.config.features["target_col"]
        y = pd.Series()
        if is_train and target_col in trip_df.columns:
            y = trip_df[target_col]
            
        # Drop leakage/prohibited columns
        leakage_cols = self.config.features["prohibited_leakage_cols"]
        trip_df = trip_df.drop(columns=[c for c in leakage_cols if c in trip_df.columns])
        
        # Drop EmployeeNumber if present (not in test set)
        if "EmployeeNumber" in trip_df.columns:
            trip_df = trip_df.drop(columns=["EmployeeNumber"])
            
        # Set TripID as index
        trip_df = trip_df.set_index("TripID")
        
        # Extract features from Event Log and Event Attributes
        log_feats = self._extract_event_log_features(log_df)
        attr_feats = self._extract_event_attributes_features(attr_df)
        
        # Join all three tables
        features_df = trip_df.join(log_feats, how="left").join(attr_feats, how="left")
        
        # Handle new route metrics (Route Popularity, City-Country mismatches)
        features_df["Route"] = features_df["DepartureLocationCity"] + " -> " + features_df["ArrivalLocationCity"]
        features_df["Is_International"] = (features_df["DepartureLocationCountry"] != features_df["ArrivalLocationCountry"]).astype(int)
        
        # Fill any remaining NaNs in features
        num_cols_to_fill = features_df.select_dtypes(include=[np.number]).columns
        features_df[num_cols_to_fill] = features_df[num_cols_to_fill].fillna(0.0)
        
        cat_cols_to_fill = features_df.select_dtypes(include=["object"]).columns
        features_df[cat_cols_to_fill] = features_df[cat_cols_to_fill].fillna("Unknown")

        # Set up list of numerical and categorical columns
        if not self.fitted:
            self.numerical_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude target/ID if they are here
            for col in ["TripID", target_col]:
                if col in self.numerical_cols:
                    self.numerical_cols.remove(col)
                    
            self.categorical_cols_to_encode = features_df.select_dtypes(include=["object"]).columns.tolist()
            self.logger.info(f"Numerical columns for scaling: {len(self.numerical_cols)}")
            self.logger.info(f"Categorical columns for encoding: {self.categorical_cols_to_encode}")

        # Scale numerical features
        if is_train:
            features_df[self.numerical_cols] = self.scaler.fit_transform(features_df[self.numerical_cols])
        else:
            features_df[self.numerical_cols] = self.scaler.transform(features_df[self.numerical_cols])
            
        # Encode categorical features
        if is_train:
            encoded_cats = self.ohe.fit_transform(features_df[self.categorical_cols_to_encode])
            self.ohe_features = self.ohe.get_feature_names_out(self.categorical_cols_to_encode).tolist()
            self.fitted = True
        else:
            encoded_cats = self.ohe.transform(features_df[self.categorical_cols_to_encode])
            
        encoded_cats_df = pd.DataFrame(encoded_cats, columns=self.ohe_features, index=features_df.index)
        
        # Merge scaled numerical and encoded categorical features
        final_features_df = pd.concat([features_df[self.numerical_cols], encoded_cats_df], axis=1)
        self.logger.info(f"Final feature matrix shape: {final_features_df.shape}")
        
        return final_features_df, y
