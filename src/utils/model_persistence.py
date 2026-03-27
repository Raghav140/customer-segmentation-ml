"""
Model persistence utilities for saving and loading trained models.
"""

import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelPersistence:
    """Handles saving and loading of ML models and related artifacts."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model persistence.
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "production").mkdir(exist_ok=True)
        (self.models_dir / "staging").mkdir(exist_ok=True)
        (self.models_dir / "development").mkdir(exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, environment: str = "production", 
                   metadata: Optional[Dict] = None) -> str:
        """
        Save a trained model with metadata.
        
        Args:
            model: Trained model object
            model_name: Name for the model file
            environment: Environment (production, staging, development)
            metadata: Additional metadata to save
            
        Returns:
            str: Path to saved model
        """
        try:
            # Create environment directory
            env_dir = self.models_dir / environment
            env_dir.mkdir(exist_ok=True)
            
            # Generate model path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = env_dir / f"{model_name}_{timestamp}.pkl"
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata
            if metadata:
                metadata_path = env_dir / f"{model_name}_{timestamp}_metadata.json"
                metadata.update({
                    'saved_at': datetime.now().isoformat(),
                    'model_path': str(model_path),
                    'environment': environment
                })
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    
    def load_latest_model(self, model_name: str, environment: str = "production") -> Optional[Any]:
        """
        Load the latest version of a model.
        
        Args:
            model_name: Base name of the model
            environment: Environment to load from
            
        Returns:
            Latest model or None if not found
        """
        try:
            env_dir = self.models_dir / environment
            model_files = list(env_dir.glob(f"{model_name}_*.pkl"))
            
            if not model_files:
                logger.warning(f"No models found for {model_name} in {environment}")
                return None
            
            # Get the latest model file
            latest_model = max(model_files, key=os.path.getctime)
            return self.load_model(str(latest_model))
            
        except Exception as e:
            logger.error(f"Failed to load latest model {model_name}: {str(e)}")
            return None
    
    def get_model_metadata(self, model_path: str) -> Optional[Dict]:
        """
        Get metadata for a model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            # Generate metadata path
            model_path = Path(model_path)
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return None
            
        except Exception as e:
            logger.error(f"Failed to load metadata for {model_path}: {str(e)}")
            return None
    
    def list_models(self, environment: str = "production") -> Dict[str, list]:
        """
        List all models in an environment.
        
        Args:
            environment: Environment to list models from
            
        Returns:
            Dictionary of model types and their versions
        """
        try:
            env_dir = self.models_dir / environment
            models = {}
            
            for model_file in env_dir.glob("*.pkl"):
                # Extract base name (remove timestamp and extension)
                base_name = "_".join(model_file.stem.split("_")[:-2])
                
                if base_name not in models:
                    models[base_name] = []
                
                models[base_name].append({
                    'path': str(model_file),
                    'created': datetime.fromtimestamp(os.path.getctime(model_file)),
                    'size': model_file.stat().st_size
                })
            
            # Sort each model list by creation time (newest first)
            for model_name in models:
                models[model_name].sort(key=lambda x: x['created'], reverse=True)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models in {environment}: {str(e)}")
            return {}
    
    def delete_model(self, model_path: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if deleted successfully
        """
        try:
            model_path = Path(model_path)
            
            # Delete model file
            if model_path.exists():
                model_path.unlink()
            
            # Delete metadata file
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Model deleted: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            return False
    
    def get_model_registry(self, environment: str = "production") -> Dict[str, List[Dict]]:
        """
        Get registry of all models in an environment.
        
        Args:
            environment: Environment to check (production, staging, development)
            
        Returns:
            Dictionary of model types and their versions
        """
        try:
            env_dir = self.models_dir / environment
            
            if not env_dir.exists():
                return {}
            
            registry = {}
            
            # Find all model files
            for model_file in env_dir.glob("*.pkl"):
                # Extract model type and version info
                model_name = model_file.stem
                
                # Determine model type from filename
                if 'kmeans' in model_name.lower():
                    model_type = 'kmeans'
                elif 'hierarchical' in model_name.lower():
                    model_type = 'hierarchical'
                else:
                    model_type = 'unknown'
                
                # Get metadata if available
                metadata_path = model_file.parent / f"{model_name}_metadata.json"
                metadata = {}
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                model_info = {
                    'path': str(model_file),
                    'name': model_name,
                    'metadata': metadata,
                    'created_at': metadata.get('training_date', 'unknown'),
                    'clusters': metadata.get('n_clusters', 'unknown'),
                    'algorithm': metadata.get('algorithm', 'unknown')
                }
                
                if model_type not in registry:
                    registry[model_type] = []
                
                registry[model_type].append(model_info)
            
            # Sort models by creation time (newest first)
            for model_type in registry:
                registry[model_type].sort(
                    key=lambda x: x['created_at'], 
                    reverse=True
                )
            
            return registry
            
        except Exception as e:
            logger.error(f"Failed to get model registry: {str(e)}")
            return {}
    
    def promote_model(self, model_path: str, target_environment: str) -> str:
        """
        Promote a model to a different environment.
        
        Args:
            model_path: Path to the model file
            target_environment: Target environment (production, staging, development)
            
        Returns:
            Path to promoted model
        """
        try:
            model_path = Path(model_path)
            target_dir = self.models_dir / target_environment
            target_dir.mkdir(exist_ok=True)
            
            # Copy model file
            target_path = target_dir / model_path.name
            import shutil
            shutil.copy2(model_path, target_path)
            
            # Copy metadata file if exists
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                target_metadata = target_dir / metadata_path.name
                shutil.copy2(metadata_path, target_metadata)
            
            logger.info(f"Model promoted to {target_environment}: {target_path}")
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Failed to promote model {model_path}: {str(e)}")
            raise


# Global model persistence instance
model_persistence = ModelPersistence()


def save_model(model: Any, model_name: str, environment: str = "production", 
               metadata: Optional[Dict] = None) -> str:
    """Convenience function to save a model."""
    return model_persistence.save_model(model, model_name, environment, metadata)


def load_model(model_path: str) -> Any:
    """Convenience function to load a model."""
    return model_persistence.load_model(model_path)


def load_latest_model(model_name: str, environment: str = "production") -> Optional[Any]:
    """Convenience function to load the latest model."""
    return model_persistence.load_latest_model(model_name, environment)
