"""
Dependency injection and component management for the API.
"""

from functools import lru_cache
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ComponentManager:
    """Manages API components and their lifecycle."""
    
    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        try:
            import sys
            from pathlib import Path
            
            # Add src to path
            sys.path.append(str(Path(__file__).parent.parent / "src"))
            
            # Import and initialize components
            from src.data.loader import DataLoader
            from src.data.preprocessor import DataPreprocessor
            from src.features.engineer import FeatureEngineer
            from src.models.kmeans_model import KMeansModel
            from src.models.hierarchical_model import HierarchicalModel
            from src.insights.profiler import SegmentProfiler
            from src.utils.metrics import ClusteringMetrics
            from src.config.settings import get_settings
            
            # Initialize components
            self._components['data_loader'] = DataLoader()
            self._components['preprocessor'] = DataPreprocessor()
            self._components['feature_engineer'] = FeatureEngineer()
            self._components['kmeans_model'] = KMeansModel()
            self._components['hierarchical_model'] = HierarchicalModel()
            self._components['profiler'] = SegmentProfiler()
            self._components['metrics_calculator'] = ClusteringMetrics()
            self._components['settings'] = get_settings()
            
            self._initialized = True
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def get_component(self, name: str) -> Any:
        """Get a specific component."""
        if not self._initialized:
            self.initialize()
        return self._components.get(name)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all components."""
        if not self._initialized:
            self.initialize()
        return self._components.copy()
    
    def health_check(self) -> Dict[str, str]:
        """Check health of all components."""
        status = {}
        for name, component in self._components.items():
            try:
                # Basic health check - component exists and is not None
                if component is not None:
                    status[name] = "healthy"
                else:
                    status[name] = "unhealthy"
            except Exception as e:
                status[name] = f"error: {str(e)}"
        return status


# Global component manager instance
component_manager = ComponentManager()


@lru_cache()
def get_component_manager() -> ComponentManager:
    """Get cached component manager instance."""
    return component_manager


def get_data_loader():
    """Get data loader component."""
    return component_manager.get_component('data_loader')


def get_preprocessor():
    """Get preprocessor component."""
    return component_manager.get_component('preprocessor')


def get_feature_engineer():
    """Get feature engineer component."""
    return component_manager.get_component('feature_engineer')


def get_kmeans_model():
    """Get KMeans model component."""
    return component_manager.get_component('kmeans_model')


def get_hierarchical_model():
    """Get hierarchical model component."""
    return component_manager.get_component('hierarchical_model')


def get_profiler():
    """Get profiler component."""
    return component_manager.get_component('profiler')


def get_metrics_calculator():
    """Get metrics calculator component."""
    return component_manager.get_component('metrics_calculator')


def get_settings():
    """Get settings component."""
    return component_manager.get_component('settings')
