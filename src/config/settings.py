"""
Settings configuration for customer segmentation system.
"""

import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    'data': {
        'supported_formats': ['.csv', '.xlsx', '.json'],
        'max_file_size_mb': 100,
        'required_columns': ['customer_id', 'age', 'annual_income', 'spending_score'],
        'optional_columns': ['purchase_frequency', 'last_purchase_days', 'customer_years'],
        'sample_size': 1000,
        'random_state': 42
    },
    'model': {
        'default_clusters': 4,
        'max_clusters': 10,
        'kmeans_max_iter': 300,
        'kmeans_n_init': 10,
        'kmeans_random_state': 42,
        'hierarchical_linkage': 'ward',
        'hierarchical_metric': 'euclidean'
    },
    'visualization': {
        'color_scheme': 'viridis',
        'chart_height': 600,
        'chart_width': 800,
        'show_legend': True
    }
}

# Environment-specific settings
settings = DEFAULT_CONFIG.copy()

class Settings:
    """Settings class for attribute access."""
    def __init__(self, config):
        self._config = config
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    def __getattr__(self, name):
        if name in self._config:
            if isinstance(self._config[name], dict):
                return Settings(self._config[name])
            return self._config[name]
        raise AttributeError(f"'Settings' object has no attribute '{name}'")

# Create settings object for attribute access
settings_obj = Settings(settings)

# For backward compatibility
settings = settings_obj

def get_setting(key_path: str, default: Any = None) -> Any:
    """
    Get a setting value by key path.
    
    Args:
        key_path: Dot-separated path to the setting (e.g., 'data.sample_size')
        default: Default value if setting not found
        
    Returns:
        Setting value or default
    """
    keys = key_path.split('.')
    value = settings
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

def get_settings() -> Settings:
    """
    Get the settings object.
    
    Returns:
        Settings object with all configuration
    """
    return settings_obj

def update_settings(new_settings: Dict[str, Any]):
    """
    Update settings with new values.
    
    Args:
        new_settings: Dictionary of settings to update
    """
    settings.update(new_settings)
