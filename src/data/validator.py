"""Data validation utilities for customer segmentation system."""

import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass

from src.config.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Union[int, float]]


class DataValidator:
    """Validates customer data for quality and completeness."""
    
    def __init__(self):
        """Initialize the data validator."""
        self.required_columns = settings.data.required_columns
        
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Perform comprehensive validation on the DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        # Basic structure validation
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, {})
            
        # Check for required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")
            
        # Check data types
        type_issues = self._validate_data_types(df)
        if type_issues:
            warnings.extend(type_issues)
            
        # Check missing values
        missing_issues = self._validate_missing_values(df)
        warnings.extend(missing_issues["warnings"])
        if missing_issues["errors"]:
            errors.extend(missing_issues["errors"])
            
        # Check for outliers
        outlier_issues = self._validate_outliers(df)
        warnings.extend(outlier_issues)
        
        # Check for unrealistic values
        unrealistic_issues = self._validate_realistic_values(df)
        warnings.extend(unrealistic_issues)
        
        # Create summary
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "duplicate_rows": duplicate_count,
            "missing_value_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "error_count": len(errors),
            "warning_count": len(warnings),
        }
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation failed with {len(errors)} errors and {len(warnings)} warnings")
            
        return ValidationResult(is_valid, errors, warnings, summary)
        
    def _validate_data_types(self, df: pd.DataFrame) -> List[str]:
        """Validate and suggest corrections for data types."""
        issues = []
        
        # Common numeric columns that should be numeric
        numeric_columns = {
            "age": "numeric",
            "annual_income": "numeric", 
            "spending_score": "numeric",
            "purchase_frequency": "numeric",
            "avg_transaction_value": "numeric",
            "customer_since_years": "numeric",
            "last_purchase_days_ago": "numeric",
            "total_purchases": "numeric",
        }
        
        for col, expected_type in numeric_columns.items():
            if col in df.columns:
                if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"Column '{col}' should be numeric but is {df[col].dtype}")
                    
        return issues
        
    def _validate_missing_values(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate missing values in the DataFrame."""
        warnings = []
        errors = []
        
        # Calculate missing value percentages
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        
        for col, percentage in missing_percentages.items():
            if percentage > 50:
                errors.append(f"Column '{col}' has {percentage:.1f}% missing values (too high)")
            elif percentage > 20:
                warnings.append(f"Column '{col}' has {percentage:.1f}% missing values")
            elif percentage > 0:
                logger.debug(f"Column '{col}' has {percentage:.1f}% missing values")
                
        return {"errors": errors, "warnings": warnings}
        
    def _validate_outliers(self, df: pd.DataFrame) -> List[str]:
        """Validate outliers in numeric columns using IQR method."""
        warnings = []
        
        numeric_columns = df.select_dtypes(include=["number"]).columns
        
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_percentage = (len(outliers) / len(df)) * 100
                
                if outlier_percentage > 5:
                    warnings.append(
                        f"Column '{col}' has {outlier_percentage:.1f}% outliers "
                        f"({len(outliers)} values)"
                    )
                    
        return warnings
        
    def _validate_realistic_values(self, df: pd.DataFrame) -> List[str]:
        """Validate that values are realistic for customer data."""
        warnings = []
        
        # Age validation
        if "age" in df.columns:
            invalid_ages = df[(df["age"] < 18) | (df["age"] > 100)]
            if len(invalid_ages) > 0:
                warnings.append(f"Found {len(invalid_ages)} customers with invalid age (<18 or >100)")
                
        # Annual income validation
        if "annual_income" in df.columns:
            negative_income = df[df["annual_income"] < 0]
            if len(negative_income) > 0:
                warnings.append(f"Found {len(negative_income)} customers with negative annual income")
                
        # Spending score validation
        if "spending_score" in df.columns:
            invalid_scores = df[(df["spending_score"] < 1) | (df["spending_score"] > 100)]
            if len(invalid_scores) > 0:
                warnings.append(f"Found {len(invalid_scores)} customers with spending score outside 1-100 range")
                
        # Purchase frequency validation
        if "purchase_frequency" in df.columns:
            negative_freq = df[df["purchase_frequency"] < 0]
            if len(negative_freq) > 0:
                warnings.append(f"Found {len(negative_freq)} customers with negative purchase frequency")
                
        return warnings
        
    def validate_customer_id_uniqueness(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate that customer IDs are unique and properly formatted.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult for customer ID validation
        """
        errors = []
        warnings = []
        
        if "customer_id" not in df.columns:
            errors.append("Missing 'customer_id' column")
            return ValidationResult(False, errors, warnings, {})
            
        # Check for duplicates
        duplicate_ids = df["customer_id"].duplicated().sum()
        if duplicate_ids > 0:
            errors.append(f"Found {duplicate_ids} duplicate customer IDs")
            
        # Check for null values
        null_ids = df["customer_id"].isnull().sum()
        if null_ids > 0:
            errors.append(f"Found {null_ids} null customer IDs")
            
        # Check for negative IDs
        if pd.api.types.is_numeric_dtype(df["customer_id"]):
            negative_ids = (df["customer_id"] < 0).sum()
            if negative_ids > 0:
                warnings.append(f"Found {negative_ids} negative customer IDs")
                
        summary = {
            "total_customers": len(df),
            "duplicate_ids": duplicate_ids,
            "null_ids": null_ids,
        }
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, summary)
        
    def suggest_data_cleaning(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Suggest data cleaning operations based on validation results.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of suggested cleaning operations
        """
        suggestions = {
            "drop_duplicates": [],
            "fill_missing": [],
            "handle_outliers": [],
            "convert_types": [],
            "remove_invalid": [],
        }
        
        # Check for duplicates
        if df.duplicated().sum() > 0:
            suggestions["drop_duplicates"].append("Remove duplicate rows")
            
        # Check missing values
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        for col, percentage in missing_percentages.items():
            if 0 < percentage < 20:
                if pd.api.types.is_numeric_dtype(df[col]):
                    suggestions["fill_missing"].append(f"Fill missing values in '{col}' with median")
                else:
                    suggestions["fill_missing"].append(f"Fill missing values in '{col}' with mode")
            elif percentage >= 20:
                suggestions["remove_invalid"].append(f"Consider dropping column '{col}' (too many missing values)")
                
        # Check outliers
        numeric_columns = df.select_dtypes(include=["number"]).columns
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percentage = (len(outliers) / len(df)) * 100
            
            if 0 < outlier_percentage <= 5:
                suggestions["handle_outliers"].append(f"Cap outliers in '{col}' at IQR bounds")
                
        return suggestions
