"""Logging utilities for customer segmentation system."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from src.config.logging_config import get_logger


class PerformanceLogger:
    """Logger for performance monitoring and timing."""
    
    def __init__(self, logger_name: str = __name__):
        """Initialize performance logger."""
        self.logger = get_logger(logger_name)
        
    def log_execution_time(self, func: Callable) -> Callable:
        """
        Decorator to log function execution time.
        
        Args:
            func: Function to time
            
        Returns:
            Wrapped function with timing
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            self.logger.info(f"Starting execution of {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
                raise
                
        return wrapper
        
    def log_memory_usage(self, func: Callable) -> Callable:
        """
        Decorator to log memory usage before and after function execution.
        
        Args:
            func: Function to monitor
            
        Returns:
            Wrapped function with memory logging
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import psutil
                process = psutil.Process()
                
                # Memory before
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                self.logger.debug(f"Memory before {func.__name__}: {memory_before:.2f} MB")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before
                self.logger.debug(f"Memory after {func.__name__}: {memory_after:.2f} MB (Δ{memory_diff:+.2f} MB)")
                
                return result
                
            except ImportError:
                self.logger.warning("psutil not available for memory monitoring")
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in memory monitoring for {func.__name__}: {str(e)}")
                return func(*args, **kwargs)
                
        return wrapper


class ExperimentLogger:
    """Logger for ML experiments and model training."""
    
    def __init__(self, experiment_name: str):
        """Initialize experiment logger."""
        self.experiment_name = experiment_name
        self.logger = get_logger(f"experiment.{experiment_name}")
        self.metrics = {}
        self.parameters = {}
        self.artifacts = []
        
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log experiment parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.parameters.update(params)
        self.logger.info(f"Parameters logged: {params}")
        
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log experiment metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        self.metrics.update(metrics)
        self.logger.info(f"Metrics logged: {metrics}")
        
    def log_artifact(self, artifact_path: str, description: str = "") -> None:
        """
        Log experiment artifact.
        
        Args:
            artifact_path: Path to artifact
            description: Description of artifact
        """
        artifact_info = {
            "path": artifact_path,
            "description": description,
            "timestamp": time.time()
        }
        self.artifacts.append(artifact_info)
        self.logger.info(f"Artifact logged: {artifact_path} - {description}")
        
    def log_model_info(self, model_name: str, model_type: str, **kwargs) -> None:
        """
        Log model information.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            **kwargs: Additional model information
        """
        model_info = {
            "name": model_name,
            "type": model_type,
            **kwargs
        }
        self.logger.info(f"Model info logged: {model_info}")
        
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary.
        
        Returns:
            Dictionary with experiment summary
        """
        return {
            "experiment_name": self.experiment_name,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "timestamp": time.time()
        }


def setup_function_logging(func: Callable) -> Callable:
    """
    Decorator to set up comprehensive logging for functions.
    
    Args:
        func: Function to log
        
    Returns:
        Wrapped function with comprehensive logging
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function start
        logger.info(f"=== {func.__name__.upper()} START ===")
        
        # Log input parameters (safely)
        if args:
            logger.debug(f"Positional args: {len(args)} arguments")
        if kwargs:
            logger.debug(f"Keyword args: {list(kwargs.keys())}")
            
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Log successful completion
            logger.info(f"=== {func.__name__.upper()} COMPLETED SUCCESSFULLY ===")
            return result
            
        except Exception as e:
            # Log error
            logger.error(f"=== {func.__name__.upper()} FAILED ===")
            logger.error(f"Error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
            
    return wrapper


class DataLogger:
    """Logger for data operations."""
    
    def __init__(self, operation_name: str):
        """Initialize data logger."""
        self.operation_name = operation_name
        self.logger = get_logger(f"data.{operation_name}")
        
    def log_data_info(self, df, operation: str) -> None:
        """
        Log DataFrame information.
        
        Args:
            df: DataFrame to log info about
            operation: Operation being performed
        """
        try:
            self.logger.info(f"Data {operation}: Shape {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Log column info
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            self.logger.debug(f"Columns: {len(df.columns)} total, {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
            
            # Log missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / (df.shape[0] * df.shape[1])) * 100
                self.logger.warning(f"Missing values: {missing_count} ({missing_percentage:.2f}%)")
            else:
                self.logger.info("No missing values found")
                
        except Exception as e:
            self.logger.error(f"Error logging data info: {str(e)}")
            
    def log_transformation(self, input_shape: tuple, output_shape: tuple, transformation: str) -> None:
        """
        Log data transformation.
        
        Args:
            input_shape: Shape of input data
            output_shape: Shape of output data
            transformation: Type of transformation
        """
        self.logger.info(f"Transformation '{transformation}': {input_shape} → {output_shape}")
        
        if input_shape != output_shape:
            self.logger.info(f"Shape change: Rows {input_shape[0]}→{output_shape[0]}, Columns {input_shape[1]}→{output_shape[1]}")


# Convenience functions for common logging patterns
def log_model_training(model_name: str, params: Dict[str, Any], metrics: Dict[str, float]) -> None:
    """
    Log model training information.
    
    Args:
        model_name: Name of the model
        params: Model parameters
        metrics: Training metrics
    """
    logger = get_logger("model_training")
    logger.info(f"Model '{model_name}' training completed")
    logger.info(f"Parameters: {params}")
    logger.info(f"Metrics: {metrics}")


def log_data_split(train_shape: tuple, test_shape: tuple, validation_shape: Optional[tuple] = None) -> None:
    """
    Log data split information.
    
    Args:
        train_shape: Shape of training set
        test_shape: Shape of test set
        validation_shape: Shape of validation set (optional)
    """
    logger = get_logger("data_split")
    total_samples = train_shape[0] + test_shape[0] + (validation_shape[0] if validation_shape else 0)
    
    logger.info(f"Data split completed - Total samples: {total_samples}")
    logger.info(f"Train: {train_shape[0]} ({train_shape[0]/total_samples*100:.1f}%)")
    logger.info(f"Test: {test_shape[0]} ({test_shape[0]/total_samples*100:.1f}%)")
    
    if validation_shape:
        logger.info(f"Validation: {validation_shape[0]} ({validation_shape[0]/total_samples*100:.1f}%)")


def log_feature_engineering(input_features: list, output_features: list, operations: list) -> None:
    """
    Log feature engineering operations.
    
    Args:
        input_features: List of input features
        output_features: List of output features
        operations: List of operations performed
    """
    logger = get_logger("feature_engineering")
    logger.info(f"Feature engineering completed")
    logger.info(f"Input features: {len(input_features)}")
    logger.info(f"Output features: {len(output_features)}")
    logger.info(f"Operations: {', '.join(operations)}")
    
    if len(input_features) != len(output_features):
        logger.info(f"Feature count changed: {len(input_features)} → {len(output_features)}")
