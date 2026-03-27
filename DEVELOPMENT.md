# 🛠️ Development Guide

## 🚀 Getting Started for Development

### Prerequisites
- Python 3.9+
- Git
- Code editor (VS Code recommended)

### Setup Development Environment

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/customer-segmentation-ml.git
cd customer-segmentation-ml
```

2. **Create virtual environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install development dependencies**
```bash
pip install pytest jupyter notebook black flake8
```

## 📁 Project Structure for Developers

### Core Components
- **`streamlit_app.py`**: Main application entry point
- **`src/`**: All ML and business logic
- **`notebooks/`**: Analysis and experimentation
- **`tests/`**: Test suite

### Development Workflow

#### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes
# ...

# Run tests
pytest tests/

# Run app locally
streamlit run streamlit_app.py
```

#### 2. Data Analysis
```bash
# Start Jupyter for analysis
jupyter notebook

# Open notebooks/01_eda.ipynb for exploratory analysis
# Open notebooks/03_clustering_experiments.ipynb for ML experiments
```

#### 3. Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_basic.py::test_data_loading
```

## 🔧 Code Style and Standards

### Python Code Style
- **Formatter**: Black
- **Linter**: Flake8
- **Type Hints**: Required for all functions
- **Docstrings**: Google style preferred

### Example Code Style
```python
from typing import Optional, List
import pandas as pd

def process_customer_data(
    data: pd.DataFrame, 
    features: List[str],
    scaling: bool = True
) -> Optional[pd.DataFrame]:
    """Process customer data for clustering.
    
    Args:
        data: Raw customer data
        features: List of feature columns to use
        scaling: Whether to scale the data
        
    Returns:
        Processed data or None if processing fails
        
    Raises:
        ValueError: If required columns are missing
    """
    # Implementation here
    pass
```

### File Organization
- **Imports**: Standard library → Third-party → Local imports
- **Constants**: UPPER_CASE at module level
- **Classes**: PascalCase
- **Functions**: snake_case
- **Variables**: snake_case

## 🧪 Testing Strategy

### Test Categories
1. **Unit Tests**: Individual function testing
2. **Integration Tests**: Component interaction testing
3. **UI Tests**: Streamlit component testing
4. **Data Tests**: Data validation and quality

### Test Structure
```python
# tests/test_data_processing.py
import pytest
import pandas as pd
from src.data.preprocessor import DataPreprocessor

class TestDataPreprocessor:
    def test_clean_data_success(self):
        """Test successful data cleaning."""
        # Arrange
        preprocessor = DataPreprocessor()
        dirty_data = pd.DataFrame({...})
        
        # Act
        clean_data = preprocessor.clean(dirty_data)
        
        # Assert
        assert clean_data is not None
        assert len(clean_data) > 0
```

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=src

# Specific file
pytest tests/test_data_processing.py

# With verbose output
pytest -v
```

## 📊 Data Management

### Sample Data
- **Location**: Generated dynamically in `streamlit_app.py`
- **Format**: Pandas DataFrame
- **Size**: 1000 customers by default
- **Features**: Age, income, spending, frequency, etc.

### Data Validation
```python
# src/data/validator.py
def validate_customer_data(data: pd.DataFrame) -> bool:
    """Validate customer data schema."""
    required_columns = ['customer_id', 'age', 'annual_income', 'spending_score']
    
    # Check required columns
    if not all(col in data.columns for col in required_columns):
        return False
    
    # Check data types
    # Check for missing values
    # Check value ranges
    
    return True
```

## 🤖 Model Development

### Adding New Models
1. **Create model class** in `src/models/`
2. **Inherit from base class**:
```python
from src.models.base import BaseModel

class NewClusteringModel(BaseModel):
    def fit(self, data: pd.DataFrame) -> 'NewClusteringModel':
        # Implementation
        pass
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        # Implementation
        pass
```

3. **Add to model registry** in `streamlit_app.py`
4. **Write tests** in `tests/`

### Model Evaluation
```python
# src/utils/metrics.py
def evaluate_clustering(data, labels, predictions):
    """Evaluate clustering performance."""
    silhouette = silhouette_score(data, predictions)
    davies_bouldin = davies_bouldin_score(data, predictions)
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin
    }
```

## 🎨 UI Development

### Streamlit Components
- **Widgets**: `st.button()`, `st.selectbox()`, `st.file_uploader()`
- **Layout**: `st.columns()`, `st.expander()`, `st.container()`
- **Visualization**: `st.plotly_chart()`, `st.dataframe()`

### Custom CSS
```python
# In streamlit_app.py
st.markdown("""
<style>
    .custom-class {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)
```

### State Management
```python
# Session state for persistence
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Update state
st.session_state.processed_data = processed_data
```

## 📝 Documentation

### Code Documentation
- **Docstrings**: Required for all public functions
- **Type Hints**: Required for all function parameters
- **Comments**: Explain complex logic only

### README Updates
- **Features**: Add new features to the list
- **Installation**: Update dependency requirements
- **Usage**: Document new functionality

### Architecture Documentation
- **ARCHITECTURE.md**: Update for new components
- **DEVELOPMENT.md**: Keep development guides current
- **API docs**: Document new endpoints/functions

## 🚀 Deployment

### Local Development
```bash
# Run development server
streamlit run streamlit_app.py --server.port 8501

# With auto-reload
streamlit run streamlit_app.py --server.runOnSave true
```

### Docker Development
```bash
# Build development image
docker build -t customer-segmentation-dev .

# Run with volume mounting
docker run -v $(pwd):/app -p 8501:8501 customer-segmentation-dev
```

### Production Deployment
```bash
# Build production image
docker build -f Dockerfile -t customer-segmentation-prod .

# Run production container
docker run -p 8501:8501 customer-segmentation-prod
```

## 🐛 Debugging

### Common Issues
1. **Import Errors**: Check virtual environment activation
2. **Data Loading**: Verify file paths and formats
3. **Model Training**: Check data preprocessing
4. **UI Rendering**: Inspect browser console for errors

### Debugging Tools
```python
# Debug prints
print(f"Data shape: {data.shape}")

# Streamlit debugging
st.write("Debug info:", debug_variable)

# Exception handling
try:
    risky_operation()
except Exception as e:
    st.error(f"Error: {str(e)}")
```

## 📈 Performance Optimization

### Data Processing
- **Vectorization**: Use NumPy/Pandas operations
- **Memory Management**: Process data in chunks
- **Caching**: Cache expensive computations

### UI Performance
- **Lazy Loading**: Load data only when needed
- **Caching**: Use `@st.cache_data` for expensive functions
- **Optimization**: Minimize widget updates

### Example Caching
```python
@st.cache_data
def expensive_computation(data):
    """Cache expensive data processing."""
    # Complex computation here
    return result
```

## 🤝 Contributing

### Pull Request Process
1. **Fork** the repository
2. **Create feature branch**
3. **Make changes** with tests
4. **Run tests** and ensure they pass
5. **Submit pull request** with description

### Code Review Checklist
- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] No breaking changes
- [ ] Performance impact considered

This development guide ensures consistent, high-quality contributions to the project.
