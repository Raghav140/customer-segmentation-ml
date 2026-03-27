"""Helper utilities for customer segmentation system."""

import json
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_model(model: Any, file_path: Union[str, Path]) -> None:
    """
    Save model using joblib.
    
    Args:
        model: Model to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    joblib.dump(model, file_path)


def load_model(file_path: Union[str, Path]) -> Any:
    """
    Load model using joblib.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded model
    """
    return joblib.load(file_path)


def generate_file_hash(file_path: Union[str, Path]) -> str:
    """
    Generate MD5 hash for a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string in format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def format_number(number: float, decimal_places: int = 2) -> str:
    """
    Format number with thousand separators and decimal places.
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{number:,.{decimal_places}f}"


def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate memory usage of DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    memory_usage = df.memory_usage(deep=True)
    
    return {
        "total_memory_mb": memory_usage.sum() / 1024**2,
        "memory_per_column_mb": (memory_usage / 1024**2).to_dict(),
        "average_memory_per_column_mb": memory_usage.mean() / 1024**2,
        "largest_column": memory_usage.idxmax(),
        "largest_column_memory_mb": memory_usage.max() / 1024**2
    }


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    df_optimized = df.copy()
    
    # Downcast numeric columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
    # Convert object columns to category if they have low cardinality
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Less than 50% unique values
            df_optimized[col] = df_optimized[col].astype('category')
            
    return df_optimized


def create_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create comprehensive data profile.
    
    Args:
        df: DataFrame to profile
        
    Returns:
        Dictionary with data profile information
    """
    profile = {
        "basic_info": {
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "column_count": len(df.columns),
            "row_count": len(df)
        },
        "column_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": {},
        "categorical_summary": {}
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        profile["numeric_summary"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "q25": float(df[col].quantile(0.25)),
            "q50": float(df[col].quantile(0.50)),
            "q75": float(df[col].quantile(0.75)),
            "zeros": int((df[col] == 0).sum()),
            "missing": int(df[col].isnull().sum())
        }
        
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        profile["categorical_summary"][col] = {
            "unique_count": int(df[col].nunique()),
            "most_frequent": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
            "most_frequent_count": int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0,
            "missing": int(df[col].isnull().sum())
        }
        
    return profile


def validate_feature_importance(importance_scores: Dict[str, float], threshold: float = 0.01) -> Dict[str, float]:
    """
    Validate and filter feature importance scores.
    
    Args:
        importance_scores: Dictionary of feature importance scores
        threshold: Minimum importance threshold
        
    Returns:
        Filtered importance scores
    """
    # Normalize scores
    total_score = sum(importance_scores.values())
    if total_score > 0:
        normalized_scores = {k: v/total_score for k, v in importance_scores.items()}
    else:
        normalized_scores = importance_scores.copy()
        
    # Filter by threshold
    filtered_scores = {k: v for k, v in normalized_scores.items() if v >= threshold}
    
    return filtered_scores


def create_cluster_labels(cluster_centers: np.ndarray, feature_names: List[str]) -> Dict[int, str]:
    """
    Create human-readable cluster labels based on cluster centers.
    
    Args:
        cluster_centers: Cluster center coordinates
        feature_names: Names of features
        
    Returns:
        Dictionary mapping cluster numbers to labels
    """
    labels = {}
    
    for i, center in enumerate(cluster_centers):
        # Find features with highest and lowest values for this cluster
        feature_contributions = list(zip(feature_names, center))
        feature_contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top contributing features
        top_features = feature_contributions[:3]
        bottom_features = feature_contributions[-3:]
        
        # Create label based on dominant characteristics
        high_aspects = [f"High {feat}" for feat, val in top_features if val > 0]
        low_aspects = [f"Low {feat}" for feat, val in bottom_features if val < 0]
        
        label_parts = high_aspects[:2] + low_aspects[:1]  # Limit label length
        label = f"Cluster {i}: " + ", ".join(label_parts) if label_parts else f"Cluster {i}"
        
        labels[i] = label
        
    return labels


def calculate_cluster_statistics(df: pd.DataFrame, cluster_column: str) -> Dict[int, Dict[str, float]]:
    """
    Calculate statistics for each cluster.
    
    Args:
        df: DataFrame with cluster assignments
        cluster_column: Name of cluster column
        
    Returns:
        Dictionary with cluster statistics
    """
    cluster_stats = {}
    
    for cluster_id in df[cluster_column].unique():
        cluster_data = df[df[cluster_column] == cluster_id]
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        
        stats = {
            "size": len(cluster_data),
            "percentage": (len(cluster_data) / len(df)) * 100
        }
        
        # Add statistics for numeric columns
        for col in numeric_cols:
            if col != cluster_column:
                stats[f"{col}_mean"] = float(cluster_data[col].mean())
                stats[f"{col}_std"] = float(cluster_data[col].std())
                stats[f"{col}_min"] = float(cluster_data[col].min())
                stats[f"{col}_max"] = float(cluster_data[col].max())
                
        cluster_stats[int(cluster_id)] = stats
        
    return cluster_stats


def export_results_to_excel(
    data: Dict[str, pd.DataFrame],
    output_path: Union[str, Path],
    include_charts: bool = False
) -> None:
    """
    Export multiple DataFrames to Excel with multiple sheets.
    
    Args:
        data: Dictionary of sheet_name -> DataFrame
        output_path: Output Excel file path
        include_charts: Whether to include basic charts
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in data.items():
            # Truncate sheet name if too long
            sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
            
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Add basic formatting if charts requested
            if include_charts:
                try:
                    worksheet = writer.sheets[sheet_name]
                    
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                                
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                        
                except Exception:
                    # If formatting fails, continue without it
                    pass


def merge_dictionaries_on_key(dicts: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """
    Merge multiple dictionaries on a common key.
    
    Args:
        dicts: List of dictionaries to merge
        key: Key to merge on
        
    Returns:
        Merged dictionary
    """
    merged = {}
    
    for d in dicts:
        if key in d:
            merged.update(d[key])
            
    return merged


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Result of division or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested items
        sep: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            
    return dict(items)
