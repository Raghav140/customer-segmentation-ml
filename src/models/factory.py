"""Model factory for creating clustering models."""

from typing import Dict, Any, Optional, Union
from enum import Enum

from src.config.logging_config import get_logger
from src.models.base import BaseClusteringModel
from src.models.kmeans_model import KMeansModel
from src.models.hierarchical_model import HierarchicalModel

logger = get_logger(__name__)


class ModelType(Enum):
    """Enumeration of available model types."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"


class ClusteringModelFactory:
    """Factory class for creating clustering models."""
    
    _models = {
        ModelType.KMEANS: KMeansModel,
        ModelType.HIERARCHICAL: HierarchicalModel
    }
    
    @classmethod
    def create_model(cls, model_type: Union[str, ModelType], 
                    random_state: Optional[int] = None,
                    **kwargs) -> BaseClusteringModel:
        """
        Create a clustering model instance.
        
        Args:
            model_type: Type of model to create
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for model initialization
            
        Returns:
            Clustering model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        # Convert string to ModelType enum
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError:
                available_types = [t.value for t in ModelType]
                raise ValueError(f"Unknown model type: {model_type}. "
                               f"Available types: {available_types}")
        
        # Check if model type is supported
        if model_type not in cls._models:
            available_types = [t.value for t in ModelType]
            raise ValueError(f"Model type {model_type} not supported. "
                           f"Available types: {available_types}")
        
        # Create model instance
        model_class = cls._models[model_type]
        
        if random_state is not None:
            model = model_class(random_state=random_state)
        else:
            model = model_class()
            
        logger.info(f"Created {model_type.value} model")
        
        return model
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """
        Get information about available models.
        
        Returns:
            Dictionary with model type and description
        """
        return {
            ModelType.KMEANS.value: "K-Means clustering - partitions data into K clusters",
            ModelType.HIERARCHICAL.value: "Hierarchical clustering - builds cluster hierarchy"
        }
    
    @classmethod
    def register_model(cls, model_type: ModelType, model_class: type) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Model type enum
            model_class: Model class to register
        """
        if not issubclass(model_class, BaseClusteringModel):
            raise ValueError("Model class must inherit from BaseClusteringModel")
        
        cls._models[model_type] = model_class
        logger.info(f"Registered new model type: {model_type.value}")
    
    @classmethod
    def create_optimal_model(cls, model_type: Union[str, ModelType],
                           X, **fit_kwargs) -> BaseClusteringModel:
        """
        Create and fit a model with optimal parameters.
        
        Args:
            model_type: Type of model to create
            X: Training data
            **fit_kwargs: Additional parameters for fitting
            
        Returns:
            Fitted clustering model
        """
        model = cls.create_model(model_type)
        
        # Find optimal parameters based on model type
        if isinstance(model, KMeansModel):
            # Find optimal K for K-Means
            optimal_k = model.find_optimal_k(X, plot_results=False)
            model.fit(X, n_clusters=optimal_k, **fit_kwargs)
            
        elif isinstance(model, HierarchicalModel):
            # Find optimal number of clusters for Hierarchical
            optimal_k = model.find_optimal_clusters(X, plot_results=False)
            model.fit(X, n_clusters=optimal_k, **fit_kwargs)
            
        else:
            # Default fitting for unknown models
            model.fit(X, **fit_kwargs)
            
        logger.info(f"Created and fitted optimal {model_type.value} model")
        
        return model


def get_model_factory() -> ClusteringModelFactory:
    """Get the model factory instance."""
    return ClusteringModelFactory()


# Convenience functions
def create_kmeans_model(random_state: int = 42, **kwargs) -> KMeansModel:
    """Create a K-Means model."""
    return ClusteringModelFactory.create_model(ModelType.KMEANS, random_state=random_state, **kwargs)


def create_hierarchical_model(random_state: int = 42, **kwargs) -> HierarchicalModel:
    """Create a Hierarchical model."""
    return ClusteringModelFactory.create_model(ModelType.HIERARCHICAL, random_state=random_state, **kwargs)


def create_model(model_type: str, **kwargs) -> BaseClusteringModel:
    """Create a model by type string."""
    return ClusteringModelFactory.create_model(model_type, **kwargs)
