# 🏗️ Project Architecture

## 📁 Overall Structure

```
customer-segmentation-ml/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── .env.example             # Environment variables template
├── README.md                # Main documentation
│
├── src/                     # Core ML logic
│   ├── __init__.py
│   ├── data/               # Data processing utilities
│   │   ├── __init__.py
│   │   ├── loader.py       # Data loading functions
│   │   ├── preprocessor.py # Data cleaning and preprocessing
│   │   └── validator.py    # Data validation
│   │
│   ├── features/           # Feature engineering
│   │   ├── __init__.py
│   │   ├── engineer.py     # Feature creation
│   │   └── selector.py     # Feature selection
│   │
│   ├── models/             # ML models
│   │   ├── __init__.py
│   │   ├── base.py         # Base model class
│   │   ├── kmeans_model.py # K-Means implementation
│   │   └── hierarchical_model.py # Hierarchical clustering
│   │
│   ├── insights/           # Business intelligence
│   │   ├── __init__.py
│   │   ├── profiler.py     # Customer segment profiling
│   │   ├── recommendations.py # Business recommendations
│   │   └── business_storyteller.py # Narrative generation
│   │
│   ├── visualization/      # Data visualization
│   │   ├── __init__.py
│   │   └── storytelling.py # Business-focused charts
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── metrics.py      # Evaluation metrics
│       └── helpers.py      # Helper functions
│
├── notebooks/              # Analysis and experiments
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   └── 03_clustering_experiments.ipynb # Clustering experiments
│
└── tests/                  # Test suite
    └── test_basic.py       # Basic functionality tests
```

## 🔄 Data Flow

### 1. Data Input
- **Streamlit UI**: File upload or sample data selection
- **Data Validation**: Schema and quality checks
- **Preprocessing**: Cleaning, scaling, missing value handling

### 2. Feature Engineering
- **RFM Analysis**: Recency, Frequency, Monetary features
- **Behavioral Features**: Purchase patterns, engagement metrics
- **Demographic Features**: Age-based segments, income categories

### 3. Clustering Pipeline
- **Dimensionality Reduction**: PCA for visualization
- **Model Selection**: K-Means vs Hierarchical comparison
- **Optimal K Selection**: Silhouette analysis, elbow method
- **Quality Assessment**: Multiple validation metrics

### 4. Insights Generation
- **Segment Profiling**: Statistical characteristics per cluster
- **Business Naming**: Human-readable segment names
- **Recommendations**: Actionable business strategies
- **Visualization**: Interactive charts and dashboards

## 🧠 Core Components

### Streamlit Application (`streamlit_app.py`)
- **Frontend Interface**: Interactive web UI
- **Session Management**: State handling across interactions
- **Visualization**: Plotly charts and animations
- **User Experience**: Progress indicators, error handling

### Data Processing (`src/data/`)
- **Data Loader**: Multiple data source support
- **Preprocessor**: Cleaning, scaling, encoding
- **Validator**: Schema validation, quality checks

### ML Models (`src/models/`)
- **Base Model**: Abstract interface for all models
- **K-Means**: Primary clustering algorithm
- **Hierarchical**: Alternative clustering approach

### Business Intelligence (`src/insights/`)
- **Profiler**: Segment characteristic analysis
- **Recommendations**: Business action suggestions
- **Storyteller**: Narrative generation for insights

### Visualization (`src/visualization/`)
- **Charts**: Interactive Plotly visualizations
- **Storytelling**: Business-focused presentation
- **Animations**: Engaging visual effects

## 🔧 Technical Decisions

### **K-Means as Primary Algorithm**
- **Why**: Fast, scalable, interpretable
- **Use Case**: Large datasets, spherical clusters
- **Validation**: Silhouette score, inertia analysis

### **PCA for Visualization**
- **Why**: Preserves variance while reducing dimensions
- **Benefit**: 2D/3D visualization of high-dimensional data
- **Trade-off**: Some information loss for interpretability

### **Streamlit for UI**
- **Why**: Rapid prototyping, Python-native
- **Benefits**: Easy deployment, interactive widgets
- **Limitations**: Less customizable than web frameworks

### **Modular Architecture**
- **Why**: Maintainability, testability, reusability
- **Benefits**: Easy to extend, clear separation of concerns
- **Pattern**: Dependency injection, abstract base classes

## 📊 Quality Assurance

### **Model Validation**
- **Silhouette Score**: Cluster separation quality
- **Davies-Bouldin Index**: Cluster compactness
- **Visual Inspection**: Manual validation of results

### **Data Quality**
- **Completeness**: Missing value analysis
- **Consistency**: Data type validation
- **Outlier Detection**: Statistical outlier handling

### **Code Quality**
- **Type Hints**: Python type annotations
- **Documentation**: Docstrings and comments
- **Testing**: Unit tests for core functionality

## 🚀 Deployment Architecture

### **Development**
- **Local Environment**: Python virtual environment
- **Streamlit Dev Server`: Hot reload, debugging
- **Jupyter Notebooks**: Analysis and experimentation

### **Production**
- **Docker Container**: Consistent deployment environment
- **Streamlit Cloud**: Easy hosting option
- **Cloud Services**: AWS, GCP, Azure deployment options

## 🔮 Future Enhancements

### **Model Improvements**
- **Advanced Algorithms**: DBSCAN, Gaussian Mixture Models
- **Deep Learning**: Autoencoders for feature learning
- **Time Series**: Temporal pattern analysis

### **Feature Expansion**
- **Real-time Data**: Streaming data processing
- **External Data**: Social media, web analytics
- **Advanced Features**: Customer lifetime value, churn prediction

### **UI/UX Enhancements**
- **Custom Styling**: Advanced CSS themes
- **Mobile Responsive**: Mobile-optimized interface
- **Advanced Interactions**: Drag-and-drop, real-time updates

This architecture ensures the project is maintainable, scalable, and professional while delivering immediate business value.
