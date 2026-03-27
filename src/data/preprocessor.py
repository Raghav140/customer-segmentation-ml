"""Data preprocessing utilities for customer segmentation system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

from src.config.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for customer segmentation."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.preprocessing_info = {}
        
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (numeric_columns, categorical_columns)
        """
        # Numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Categorical columns (object and boolean types)
        categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        
        # Remove customer_id from feature columns if present
        if "customer_id" in numeric_columns:
            numeric_columns.remove("customer_id")
        if "customer_id" in categorical_columns:
            categorical_columns.remove("customer_id")
            
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.feature_columns = numeric_columns + categorical_columns
        
        logger.info(f"Identified {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns")
        
        return numeric_columns, categorical_columns
        
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent"
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            numeric_strategy: Strategy for numeric columns (mean, median, most_frequent, constant)
            categorical_strategy: Strategy for categorical columns (most_frequent, constant)
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        df_clean = df.copy()
        
        # Handle numeric columns
        if self.numeric_columns:
            numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            df_clean[self.numeric_columns] = numeric_imputer.fit_transform(df_clean[self.numeric_columns])
            
            # Store imputer for later use
            self.preprocessing_info["numeric_imputer"] = numeric_imputer
            
        # Handle categorical columns
        if self.categorical_columns:
            categorical_imputer = SimpleImputer(strategy=categorical_strategy)
            df_clean[self.categorical_columns] = categorical_imputer.fit_transform(df_clean[self.categorical_columns])
            
            # Store imputer for later use
            self.preprocessing_info["categorical_imputer"] = categorical_imputer
            
        # Log missing value information
        missing_before = df.isnull().sum().sum()
        missing_after = df_clean.isnull().sum().sum()
        
        logger.info(f"Missing values reduced from {missing_before} to {missing_after}")
        
        return df_clean
        
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Handle outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            method: Method for outlier detection (iqr, zscore)
            factor: Factor for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Handling outliers using {method} method")
        
        df_clean = df.copy()
        outliers_info = {}
        
        for col in self.numeric_columns:
            if col not in df_clean.columns:
                continue
                
            original_values = df_clean[col].copy()
            
            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outliers = z_scores > factor
                
            else:
                logger.warning(f"Unknown outlier method: {method}")
                continue
                
            # Cap outliers instead of removing them
            if method == "iqr":
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            elif method == "zscore":
                df_clean[col] = df_clean[col].clip(
                    df_clean[col].mean() - factor * df_clean[col].std(),
                    df_clean[col].mean() + factor * df_clean[col].std()
                )
                
            outlier_count = outliers.sum()
            outliers_info[col] = outlier_count
            
            if outlier_count > 0:
                logger.info(f"Capped {outlier_count} outliers in column '{col}'")
                
        self.preprocessing_info["outliers_info"] = outliers_info
        
        return df_clean
        
    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            method: Encoding method (onehot, label)
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info(f"Encoding categorical features using {method} method")
        
        if not self.categorical_columns:
            logger.info("No categorical columns to encode")
            return df
            
        df_encoded = df.copy()
        
        if method == "onehot":
            # Use OneHotEncoder for categorical columns
            self.column_transformer = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", self.numeric_columns),
                    ("cat", OneHotEncoder(drop="first", sparse_output=False), self.categorical_columns)
                ],
                remainder="drop"
            )
            
            # Fit and transform
            encoded_data = self.column_transformer.fit_transform(df_encoded[self.feature_columns])
            
            # Get feature names after encoding
            numeric_features = self.numeric_columns
            categorical_features = self.column_transformer.named_transformers_["cat"].get_feature_names_out(self.categorical_columns)
            all_features = list(numeric_features) + list(categorical_features)
            
            # Create new DataFrame
            df_encoded = pd.DataFrame(encoded_data, columns=all_features, index=df_encoded.index)
            
            # Add back non-feature columns (like customer_id)
            non_feature_columns = [col for col in df.columns if col not in self.feature_columns]
            for col in non_feature_columns:
                df_encoded[col] = df[col]
                
        elif method == "label":
            # Use LabelEncoder for categorical columns
            for col in self.categorical_columns:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
                    
        else:
            logger.warning(f"Unknown encoding method: {method}")
            
        self.preprocessing_info["encoding_method"] = method
        
        return df_encoded
        
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Scaling numeric features")
        
        if not self.numeric_columns:
            logger.info("No numeric columns to scale")
            return df
            
        df_scaled = df.copy()
        
        # Scale only numeric columns
        df_scaled[self.numeric_columns] = self.scaler.fit_transform(df_scaled[self.numeric_columns])
        
        # Store scaling info
        self.preprocessing_info["scaler"] = self.scaler
        self.preprocessing_info["scaled_columns"] = self.numeric_columns
        
        logger.info(f"Scaled {len(self.numeric_columns)} numeric columns")
        
        return df_scaled
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        encode_categorical: bool = True,
        scale_features: bool = True,
        encoding_method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to handle outliers
            encode_categorical: Whether to encode categorical features
            scale_features: Whether to scale features
            encoding_method: Method for categorical encoding
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting complete preprocessing pipeline")
        
        df_processed = df.copy()
        
        # Identify column types
        self.identify_column_types(df_processed)
        
        # Handle missing values
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed)
            
        # Handle outliers
        if handle_outliers:
            df_processed = self.handle_outliers(df_processed)
            
        # Encode categorical features
        if encode_categorical and self.categorical_columns:
            df_processed = self.encode_categorical_features(df_processed, encoding_method)
            
        # Scale features
        if scale_features and self.numeric_columns:
            df_processed = self.scale_features(df_processed)
            
        logger.info("Preprocessing pipeline completed")
        
        return df_processed
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessing pipeline.
        
        Args:
            df: New DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming new data using fitted pipeline")
        
        df_transformed = df.copy()
        
        # Handle missing values using fitted imputers
        if "numeric_imputer" in self.preprocessing_info:
            df_transformed[self.numeric_columns] = self.preprocessing_info["numeric_imputer"].transform(
                df_transformed[self.numeric_columns]
            )
            
        if "categorical_imputer" in self.preprocessing_info:
            df_transformed[self.categorical_columns] = self.preprocessing_info["categorical_imputer"].transform(
                df_transformed[self.categorical_columns]
            )
            
        # Handle outliers (using same bounds as training)
        if "outliers_info" in self.preprocessing_info:
            df_transformed = self.handle_outliers(df_transformed, method="iqr")
            
        # Encode categorical features
        if self.column_transformer:
            encoded_data = self.column_transformer.transform(df_transformed[self.feature_columns])
            
            # Get feature names
            numeric_features = self.numeric_columns
            categorical_features = self.column_transformer.named_transformers_["cat"].get_feature_names_out(self.categorical_columns)
            all_features = list(numeric_features) + list(categorical_features)
            
            df_transformed = pd.DataFrame(encoded_data, columns=all_features, index=df_transformed.index)
            
            # Add back non-feature columns
            non_feature_columns = [col for col in df.columns if col not in self.feature_columns]
            for col in non_feature_columns:
                df_transformed[col] = df[col]
                
        elif self.label_encoders:
            for col, le in self.label_encoders.items():
                if col in df_transformed.columns:
                    df_transformed[col] = le.transform(df_transformed[col])
                    
        # Scale features
        if "scaler" in self.preprocessing_info:
            df_transformed[self.numeric_columns] = self.preprocessing_info["scaler"].transform(
                df_transformed[self.numeric_columns]
            )
            
        logger.info("Data transformation completed")
        
        return df_transformed
        
    def save_preprocessor(self, file_path: Union[str, Path]) -> None:
        """
        Save the fitted preprocessor to disk.
        
        Args:
            file_path: Path to save the preprocessor
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        preprocessor_data = {
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "column_transformer": self.column_transformer,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "feature_columns": self.feature_columns,
            "preprocessing_info": self.preprocessing_info,
        }
        
        joblib.dump(preprocessor_data, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
        
    def load_preprocessor(self, file_path: Union[str, Path]) -> None:
        """
        Load a fitted preprocessor from disk.
        
        Args:
            file_path: Path to the saved preprocessor
        """
        preprocessor_data = joblib.load(file_path)
        
        self.scaler = preprocessor_data["scaler"]
        self.label_encoders = preprocessor_data["label_encoders"]
        self.column_transformer = preprocessor_data["column_transformer"]
        self.numeric_columns = preprocessor_data["numeric_columns"]
        self.categorical_columns = preprocessor_data["categorical_columns"]
        self.feature_columns = preprocessor_data["feature_columns"]
        self.preprocessing_info = preprocessor_data["preprocessing_info"]
        
        logger.info(f"Preprocessor loaded from {file_path}")
        
    def get_preprocessing_summary(self) -> Dict:
        """
        Get a summary of the preprocessing steps applied.
        
        Returns:
            Dictionary with preprocessing summary
        """
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "feature_columns": self.feature_columns,
            "preprocessing_steps": list(self.preprocessing_info.keys()),
            "encoding_method": self.preprocessing_info.get("encoding_method", "none"),
            "outliers_handled": self.preprocessing_info.get("outliers_info", {}),
        }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            # Make a copy to avoid modifying original
            df_clean = df.copy()
            
            # Handle missing values
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            categorical_columns = df_clean.select_dtypes(include=['object']).columns
            
            # Fill missing numeric values with median
            for col in numeric_columns:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
            
            # Fill missing categorical values with mode
            for col in categorical_columns:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(mode_val, inplace=True)
            
            logger.info(f"Data cleaned: {len(df_clean)} records, {df_clean.isnull().sum().sum()} missing values handled")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        try:
            # Make a copy to avoid modifying original
            df_scaled = df.copy()
            
            # Get numeric columns only (exclude categorical and ID columns)
            numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove customer_id if it exists (it might be numeric)
            if 'customer_id' in numeric_columns:
                numeric_columns.remove('customer_id')
            
            if numeric_columns:
                # Fit and transform numeric columns
                df_scaled[numeric_columns] = self.scaler.fit_transform(df_scaled[numeric_columns])
                logger.info(f"Scaled {len(numeric_columns)} numeric features")
            
            return df_scaled
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise
