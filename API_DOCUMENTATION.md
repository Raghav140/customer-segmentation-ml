# 📚 API Documentation

## 🔌 Overview

The Customer Segmentation AI provides both an interactive Streamlit interface and programmatic API access for integration with other systems.

## 🌐 Streamlit Web Interface

### Main Application Endpoint
- **URL**: `http://localhost:8501` (default)
- **Method**: Web interface
- **Authentication**: None (local development)

### Key Features
- **File Upload**: CSV data upload via web interface
- **Interactive Controls**: Real-time parameter adjustment
- **Live Visualization**: Interactive charts and plots
- **Results Export**: Download clustering results

## 🔧 Python API

### Core Modules

#### Data Processing (`src.data`)

```python
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator

# Load customer data
loader = DataLoader()
data = loader.load_csv('customers.csv')

# Validate data
validator = DataValidator()
if validator.validate(data):
    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process(data)
```

#### Feature Engineering (`src.features`)

```python
from src.features.engineer import FeatureEngineer
from src.features.selector import FeatureSelector

# Create features
engineer = FeatureEngineer()
features = engineer.create_rfm_features(data)

# Select best features
selector = FeatureSelector()
selected_features = selector.select(features, method='variance')
```

#### Clustering Models (`src.models`)

```python
from src.models.kmeans_model import KMeansModel
from src.models.hierarchical_model import HierarchicalModel

# K-Means clustering
kmeans = KMeansModel(n_clusters=4)
kmeans.fit(data)
predictions = kmeans.predict(data)

# Hierarchical clustering
hierarchical = HierarchicalModel(n_clusters=4)
hierarchical.fit(data)
predictions = hierarchical.predict(data)
```

#### Business Insights (`src.insights`)

```python
from src.insights.profiler import SegmentProfiler
from src.insights.recommendations import RecommendationEngine

# Profile customer segments
profiler = SegmentProfiler()
profiles = profiler.analyze_segments(data, predictions)

# Generate business recommendations
engineer = RecommendationEngine()
recommendations = engineer.generate_recommendations(profiles)
```

#### Visualization (`src.visualization`)

```python
from src.visualization.storytelling import BusinessStoryteller

# Create business-focused visualizations
storyteller = BusinessStoryteller()
chart = storyteller.create_segment_overview(data, predictions)
```

## 📊 Complete Workflow Example

```python
import pandas as pd
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.kmeans_model import KMeansModel
from src.insights.profiler import SegmentProfiler
from src.visualization.storytelling import BusinessStoryteller

def analyze_customers(csv_file_path: str):
    """Complete customer segmentation analysis."""
    
    # 1. Load and validate data
    loader = DataLoader()
    data = loader.load_csv(csv_file_path)
    
    # 2. Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process(data)
    
    # 3. Engineer features
    engineer = FeatureEngineer()
    features = engineer.create_features(processed_data)
    
    # 4. Cluster customers
    model = KMeansModel()
    model.fit(features)
    predictions = model.predict(features)
    
    # 5. Generate insights
    profiler = SegmentProfiler()
    profiles = profiler.analyze_segments(features, predictions)
    
    # 6. Create visualizations
    storyteller = BusinessStoryteller()
    chart = storyteller.create_segment_overview(features, predictions)
    
    return {
        'predictions': predictions,
        'profiles': profiles,
        'visualization': chart,
        'model': model
    }

# Usage
results = analyze_customers('customer_data.csv')
print(f"Found {len(set(results['predictions']))} customer segments")
```

## 🔍 Model Parameters

### KMeansModel Parameters

```python
KMeansModel(
    n_clusters: int = 4,           # Number of clusters
    init: str = 'k-means++',       # Initialization method
    max_iter: int = 300,           # Maximum iterations
    random_state: int = 42,        # Random seed
    n_init: int = 10               # Number of restarts
)
```

### HierarchicalModel Parameters

```python
HierarchicalModel(
    n_clusters: int = 4,           # Number of clusters
    linkage: str = 'ward',         # Linkage method
    distance_threshold: float = None,  # Distance threshold
    compute_full_tree: bool = True  # Compute full tree
)
```

## 📈 Evaluation Metrics

```python
from src.utils.metrics import ClusteringMetrics

# Evaluate clustering performance
metrics = ClusteringMetrics()
scores = metrics.evaluate_all(data, predictions)

print(f"Silhouette Score: {scores['silhouette']:.3f}")
print(f"Davies-Bouldin Index: {scores['davies_bouldin']:.3f}")
print(f"Calinski-Harabasz Index: {scores['calinski_harabasz']:.3f}")
```

## 🎨 Custom Visualization

```python
import plotly.graph_objects as go
from src.visualization.storytelling import BusinessStoryteller

# Create custom segment visualization
storyteller = BusinessStoryteller()

# Segment overview chart
overview_chart = storyteller.create_segment_overview(
    data=features, 
    labels=predictions,
    title="Customer Segments Analysis"
)

# Feature importance chart
feature_chart = storyteller.create_feature_importance(
    features=features,
    labels=predictions
)

# Business recommendations
recommendations_chart = storyteller.create_recommendations_chart(
    recommendations=profiles
)
```

## 🔧 Configuration

### Environment Variables

```python
import os
from src.config.settings import Config

# Load configuration
config = Config()

# Access settings
model_params = config.get_model_params()
viz_settings = config.get_visualization_settings()
data_config = config.get_data_config()
```

### Custom Configuration

```python
# src/config/custom_config.py
CUSTOM_CONFIG = {
    'model': {
        'default_clusters': 4,
        'max_clusters': 10,
        'random_state': 42
    },
    'visualization': {
        'color_scheme': 'viridis',
        'chart_height': 600,
        'chart_width': 800
    },
    'data': {
        'required_columns': ['customer_id', 'age', 'annual_income', 'spending_score'],
        'optional_columns': ['purchase_frequency', 'last_purchase_days', 'customer_years']
    }
}
```

## 🧪 Testing API Components

```python
import pytest
from src.models.kmeans_model import KMeansModel
from src.data.loader import DataLoader

def test_clustering_workflow():
    """Test complete clustering workflow."""
    # Load test data
    loader = DataLoader()
    data = loader.load_sample_data(n_samples=100)
    
    # Test clustering
    model = KMeansModel(n_clusters=3)
    model.fit(data)
    predictions = model.predict(data)
    
    # Assertions
    assert len(predictions) == len(data)
    assert len(set(predictions)) == 3
    assert model.is_fitted
```

## 📦 Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile
from src.data.loader import DataLoader
from src.models.kmeans_model import KMeansModel

app = FastAPI()

@app.post("/analyze")
async def analyze_customers(file: UploadFile):
    """API endpoint for customer analysis."""
    
    # Load uploaded data
    loader = DataLoader()
    data = await loader.load_upload(file)
    
    # Run clustering
    model = KMeansModel()
    model.fit(data)
    predictions = model.predict(data)
    
    return {
        "status": "success",
        "predictions": predictions.tolist(),
        "n_clusters": len(set(predictions))
    }
```

### Jupyter Notebook Integration

```python
# In Jupyter notebook
%load_ext autoreload
%autoreload 2

from src.data.loader import DataLoader
from src.models.kmeans_model import KMeansModel
from src.visualization.storytelling import BusinessStoryteller

# Interactive analysis
loader = DataLoader()
data = loader.load_csv('your_data.csv')

model = KMeansModel()
model.fit(data)
predictions = model.predict(data)

# Visualize results
storyteller = BusinessStoryteller()
chart = storyteller.create_segment_overview(data, predictions)
chart.show()
```

## 🚀 Performance Considerations

### Memory Optimization
```python
# Process large datasets in chunks
def process_large_dataset(file_path, chunk_size=1000):
    loader = DataLoader()
    
    for chunk in loader.load_chunks(file_path, chunk_size):
        # Process chunk
        processed_chunk = preprocessor.process(chunk)
        yield processed_chunk
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model_predictions(data_hash):
    """Cache model predictions."""
    model = KMeansModel()
    model.fit(data)
    return model.predict(data)
```

## 📞 API Support

### Error Handling
```python
from src.exceptions import DataValidationError, ModelNotFittedError

try:
    model = KMeansModel()
    predictions = model.predict(data)  # Error: model not fitted
except ModelNotFittedError as e:
    print(f"Model error: {e}")
```

### Logging
```python
import logging
from src.utils.logger import setup_logger

logger = setup_logger('customer_segmentation')

logger.info("Starting customer segmentation analysis")
model = KMeansModel()
model.fit(data)
logger.info(f"Clustering completed with {model.n_clusters} segments")
```

This API documentation provides comprehensive guidance for integrating the Customer Segmentation AI into various workflows and applications.
