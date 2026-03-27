"""Data loading utilities for customer segmentation system."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class DataLoader:
    """Handles loading of customer data from various file formats."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.supported_formats = settings.data.supported_formats
        self.max_file_size_mb = settings.data.max_file_size_mb
        
    def load_data(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from the specified file path.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            MemoryError: If file is too large
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise MemoryError(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum "
                f"allowed size ({self.max_file_size_mb} MB)"
            )
            
        # Get file extension
        file_extension = file_path.suffix.lower().lstrip('.')
        
        # Check if format is supported
        if file_extension not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
            
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Load data based on file format
            if file_extension == "csv":
                df = pd.read_csv(file_path, **kwargs)
            elif file_extension == "parquet":
                df = pd.read_parquet(file_path, **kwargs)
            elif file_extension == "json":
                df = pd.read_json(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {file_extension}")
                
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
            
    def load_sample_data(self) -> pd.DataFrame:
        """
        Generate sample customer data for testing and demonstration.
        
        Returns:
            Sample customer DataFrame
        """
        logger.info("Generating sample customer data")
        
        import numpy as np
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate sample data
        n_customers = 1000
        
        data = {
            "customer_id": range(1, n_customers + 1),
            "age": np.random.normal(35, 12, n_customers).astype(int),
            "gender": np.random.choice(["Male", "Female"], n_customers),
            "annual_income": np.random.normal(50000, 15000, n_customers),
            "spending_score": np.random.uniform(1, 100, n_customers),
            "purchase_frequency": np.random.poisson(3, n_customers),
            "avg_transaction_value": np.random.normal(150, 50, n_customers),
            "customer_since_years": np.random.uniform(0.5, 10, n_customers),
            "last_purchase_days_ago": np.random.exponential(30, n_customers),
            "total_purchases": np.random.negative_binomial(10, 0.5, n_customers),
            "preferred_category": np.random.choice(
                ["Electronics", "Clothing", "Home", "Sports", "Books"], 
                n_customers
            ),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic constraints
        df["age"] = df["age"].clip(18, 80)
        df["annual_income"] = df["annual_income"].clip(15000, 150000)
        df["avg_transaction_value"] = df["avg_transaction_value"].clip(10, 500)
        df["customer_since_years"] = df["customer_since_years"].clip(0.5, 10)
        
        logger.info(f"Generated sample data with {len(df)} customers")
        return df
        
    def save_data(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Save DataFrame to the specified file path.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            format: File format (csv, parquet, json). If None, inferred from extension.
            **kwargs: Additional arguments for pandas write functions
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format
        if format is None:
            format = file_path.suffix.lower().lstrip('.')
            
        if format not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
            
        logger.info(f"Saving data to {file_path}")
        
        try:
            if format == "csv":
                df.to_csv(file_path, index=False, **kwargs)
            elif format == "parquet":
                df.to_parquet(file_path, index=False, **kwargs)
            elif format == "json":
                df.to_json(file_path, **kwargs)
                
            logger.info(f"Successfully saved data to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            raise
    
    def load_from_dict(self, data_list: List[Dict]) -> pd.DataFrame:
        """
        Load data from a list of dictionaries.
        
        Args:
            data_list: List of dictionaries containing customer data
            
        Returns:
            pd.DataFrame: Loaded data as DataFrame
        """
        try:
            df = pd.DataFrame(data_list)
            logger.info(f"Loaded {len(df)} records from dictionary list")
            return df
        except Exception as e:
            logger.error(f"Error loading data from dict: {str(e)}")
            raise
