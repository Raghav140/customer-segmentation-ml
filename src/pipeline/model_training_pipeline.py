"""
Model Training Pipeline with Persistence

Production-ready pipeline for training and saving customer segmentation models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from pathlib import Path
import json

from ..data.loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from ..features.engineer import FeatureEngineer
from ..models.kmeans_model import KMeansModel
from ..models.hierarchical_model import HierarchicalModel
from ..insights.profiler import SegmentProfiler
from ..utils.metrics import ClusteringMetrics
from ..utils.model_persistence import model_persistence
from ..utils.auto_cluster_selection import AutoClusterSelector
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Production-ready pipeline for training customer segmentation models."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the training pipeline.
        
        Args:
            models_dir: Directory to store trained models
        """
        self.settings = get_settings()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.kmeans_model = KMeansModel()
        self.hierarchical_model = HierarchicalModel()
        self.profiler = SegmentProfiler()
        self.metrics_calculator = ClusteringMetrics()
        self.auto_selector = AutoClusterSelector()
        
        # Training history
        self.training_history = []
    
    def load_and_prepare_data(self, data_source: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for training.
        
        Args:
            data_source: Path to data file (optional, uses sample data if None)
            
        Returns:
            Tuple of (raw_data, processed_features)
        """
        try:
            # Load data
            if data_source:
                raw_data = self.data_loader.load_from_file(data_source)
            else:
                raw_data = self.data_loader.load_sample_data()
            
            logger.info(f"Loaded {len(raw_data)} records")
            
            # Preprocess data
            processed_data = self.preprocessor.fit_transform(raw_data)
            
            # Feature engineering
            features = self.feature_engineer.create_features(processed_data)
            
            # Keep only numeric features for clustering
            numeric_features = features.select_dtypes(include=[np.number])
            
            # Remove any remaining NaN values
            numeric_features = numeric_features.fillna(numeric_features.mean())
            
            logger.info(f"Prepared {len(numeric_features)} features with {numeric_features.shape[1]} columns")
            
            return raw_data, numeric_features
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def auto_select_clusters(self, features: pd.DataFrame, 
                            methods: List[str] = None,
                            business_constraints: Dict[str, Any] = None,
                            save_plots: bool = True) -> Dict[str, Any]:
        """
        Automatically select optimal number of clusters using multiple methods.
        
        Args:
            features: Feature matrix for analysis
            methods: List of methods to use
            business_constraints: Business-specific constraints
            save_plots: Whether to save analysis plots
            
        Returns:
            Dictionary with auto selection results
        """
        try:
            logger.info("Starting automatic cluster selection...")
            
            # Run auto cluster selection
            selection_results = self.auto_selector.find_optimal_clusters(
                features, methods, business_constraints
            )
            
            # Generate plots if requested
            if save_plots:
                plot_path = self.models_dir / "reports" / f"cluster_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plot_path.parent.mkdir(exist_ok=True)
                
                saved_path = self.auto_selector.plot_cluster_analysis(
                    features, selection_results, str(plot_path)
                )
                selection_results['plot_path'] = saved_path
            
            logger.info(f"Auto cluster selection completed: optimal_k={selection_results['optimal_clusters']}")
            
            return selection_results
            
        except Exception as e:
            logger.error(f"Auto cluster selection failed: {str(e)}")
            raise
    
    def train_kmeans_model(self, features: pd.DataFrame, n_clusters: Optional[int] = None,
                          save_model: bool = True, environment: str = "production") -> Dict[str, Any]:
        """
        Train K-Means clustering model.
        
        Args:
            features: Feature matrix for training
            n_clusters: Number of clusters (auto-detect if None)
            save_model: Whether to save the trained model
            environment: Environment to save model in
            
        Returns:
            Dictionary with training results
        """
        try:
            # Find optimal clusters if not specified
            if n_clusters is None:
                logger.info("Running auto cluster selection for K-Means...")
                selection_results = self.auto_select_clusters(features, methods=['elbow', 'silhouette', 'gap'])
                n_clusters = selection_results['optimal_clusters']
                logger.info(f"Auto-detected optimal clusters: {n_clusters}")
                logger.info(f"Selection confidence: {selection_results['confidence_score']:.3f}")
                logger.info(f"Reasoning: {selection_results['recommendation']['reasoning']}")
            else:
                logger.info(f"Using specified clusters: {n_clusters}")
            
            # Train model
            self.kmeans_model.fit(features, n_clusters=n_clusters)
            predictions = self.kmeans_model.predict(features)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(features, predictions, n_clusters)
            
            # Generate insights
            features_with_clusters = features.copy()
            features_with_clusters['cluster'] = predictions
            insights = self.profiler.profile_clusters(
                X=features,
                cluster_labels=predictions,
                feature_names=features.columns.tolist()
            )
            
            # Prepare training results
            training_results = {
                'model_type': 'KMeans',
                'n_clusters': n_clusters,
                'predictions': predictions,
                'metrics': metrics,
                'insights': insights,
                'training_time': datetime.now().isoformat(),
                'data_shape': features.shape,
                'model_summary': self.kmeans_model.get_model_summary()
            }
            
            # Save model if requested
            if save_model:
                model_metadata = {
                    'training_metrics': metrics,
                    'data_shape': features.shape,
                    'insights_summary': {k: v.get('characteristics', {}) for k, v in insights.items()}
                }
                
                model_path = self.kmeans_model.save_model(environment, model_metadata)
                training_results['model_path'] = model_path
                logger.info(f"K-Means model saved to {model_path}")
            
            # Add to training history
            self.training_history.append(training_results)
            
            return training_results
            
        except Exception as e:
            logger.error(f"K-Means training failed: {str(e)}")
            raise
    
    def train_hierarchical_model(self, features: pd.DataFrame, n_clusters: Optional[int] = None,
                               save_model: bool = True, environment: str = "production") -> Dict[str, Any]:
        """
        Train Hierarchical clustering model.
        
        Args:
            features: Feature matrix for training
            n_clusters: Number of clusters (auto-detect if None)
            save_model: Whether to save the trained model
            environment: Environment to save model in
            
        Returns:
            Dictionary with training results
        """
        try:
            # Find optimal clusters if not specified
            if n_clusters is None:
                logger.info("Running auto cluster selection for Hierarchical...")
                selection_results = self.auto_select_clusters(features, methods=['elbow', 'silhouette', 'hierarchical'])
                n_clusters = selection_results['optimal_clusters']
                logger.info(f"Auto-detected optimal clusters: {n_clusters}")
                logger.info(f"Selection confidence: {selection_results['confidence_score']:.3f}")
                logger.info(f"Reasoning: {selection_results['recommendation']['reasoning']}")
            else:
                logger.info(f"Using specified clusters: {n_clusters}")
            
            # Train model
            self.hierarchical_model.fit(features, n_clusters=n_clusters)
            predictions = self.hierarchical_model.predict(features)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(features, predictions, n_clusters)
            
            # Generate insights
            features_with_clusters = features.copy()
            features_with_clusters['cluster'] = predictions
            insights = self.profiler.profile_clusters(
                X=features,
                cluster_labels=predictions,
                feature_names=features.columns.tolist()
            )
            
            # Prepare training results
            training_results = {
                'model_type': 'Hierarchical',
                'n_clusters': n_clusters,
                'predictions': predictions,
                'metrics': metrics,
                'insights': insights,
                'training_time': datetime.now().isoformat(),
                'data_shape': features.shape,
                'model_summary': self.hierarchical_model.get_model_summary()
            }
            
            # Save model if requested
            if save_model:
                model_metadata = {
                    'training_metrics': metrics,
                    'data_shape': features.shape,
                    'insights_summary': {k: v.get('characteristics', {}) for k, v in insights.items()}
                }
                
                model_path = self.hierarchical_model.save_model(environment, model_metadata)
                training_results['model_path'] = model_path
                logger.info(f"Hierarchical model saved to {model_path}")
            
            # Add to training history
            self.training_history.append(training_results)
            
            return training_results
            
        except Exception as e:
            logger.error(f"Hierarchical training failed: {str(e)}")
            raise
    
    def train_both_models(self, features: pd.DataFrame, n_clusters: Optional[int] = None,
                         save_models: bool = True, environment: str = "production") -> Dict[str, Any]:
        """
        Train both K-Means and Hierarchical models.
        
        Args:
            features: Feature matrix for training
            n_clusters: Number of clusters (auto-detect if None)
            save_models: Whether to save trained models
            environment: Environment to save models in
            
        Returns:
            Dictionary with training results for both models
        """
        try:
            logger.info("Starting training pipeline for both models")
            
            # Train K-Means
            kmeans_results = self.train_kmeans_model(features, n_clusters, save_models, environment)
            
            # Train Hierarchical
            hierarchical_results = self.train_hierarchical_model(features, n_clusters, save_models, environment)
            
            # Compare models
            comparison = self._compare_models(kmeans_results, hierarchical_results)
            
            # Prepare combined results
            combined_results = {
                'training_completed': datetime.now().isoformat(),
                'data_shape': features.shape,
                'kmeans_results': kmeans_results,
                'hierarchical_results': hierarchical_results,
                'model_comparison': comparison,
                'recommendation': self._get_best_model_recommendation(kmeans_results, hierarchical_results)
            }
            
            logger.info("Training pipeline completed successfully")
            return combined_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def _compare_models(self, kmeans_results: Dict, hierarchical_results: Dict) -> Dict[str, Any]:
        """Compare performance of two models."""
        kmeans_metrics = kmeans_results['metrics']
        hierarchical_metrics = hierarchical_results['metrics']
        
        comparison = {
            'silhouette_score': {
                'kmeans': kmeans_metrics.get('silhouette_score', 0),
                'hierarchical': hierarchical_metrics.get('silhouette_score', 0),
                'winner': 'kmeans' if kmeans_metrics.get('silhouette_score', 0) > hierarchical_metrics.get('silhouette_score', 0) else 'hierarchical'
            },
            'davies_bouldin_index': {
                'kmeans': kmeans_metrics.get('davies_bouldin_index', float('inf')),
                'hierarchical': hierarchical_metrics.get('davies_bouldin_index', float('inf')),
                'winner': 'kmeans' if kmeans_metrics.get('davies_bouldin_index', float('inf')) < hierarchical_metrics.get('davies_bouldin_index', float('inf')) else 'hierarchical'
            }
        }
        
        return comparison
    
    def _get_best_model_recommendation(self, kmeans_results: Dict, hierarchical_results: Dict) -> Dict[str, Any]:
        """Get recommendation for best model."""
        comparison = self._compare_models(kmeans_results, hierarchical_results)
        
        # Count wins
        kmeans_wins = sum(1 for metric in comparison.values() if metric['winner'] == 'kmeans')
        hierarchical_wins = sum(1 for metric in comparison.values() if metric['winner'] == 'hierarchical')
        
        if kmeans_wins > hierarchical_wins:
            return {
                'recommended_model': 'KMeans',
                'reason': f'K-Means performs better on {kmeans_wins} out of {kmeans_wins + hierarchical_wins} metrics',
                'confidence': kmeans_wins / (kmeans_wins + hierarchical_wins)
            }
        else:
            return {
                'recommended_model': 'Hierarchical',
                'reason': f'Hierarchical performs better on {hierarchical_wins} out of {kmeans_wins + hierarchical_wins} metrics',
                'confidence': hierarchical_wins / (kmeans_wins + hierarchical_wins)
            }
    
    def save_training_report(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save training results to a JSON report.
        
        Args:
            results: Training results dictionary
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_report_{timestamp}.json"
        
        report_path = self.models_dir / "reports" / filename
        report_path.parent.mkdir(exist_ok=True)
        
        # Prepare report data
        report_data = {
            'report_generated': datetime.now().isoformat(),
            'pipeline_version': '1.0.0',
            'results': results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {report_path}")
        return str(report_path)
    
    def get_model_registry(self) -> Dict[str, List[Dict]]:
        """Get registry of all saved models."""
        registry = {}
        
        for environment in ['production', 'staging', 'development']:
            models = model_persistence.list_models(environment)
            if models:
                registry[environment] = models
        
        return registry
    
    def load_production_model(self, model_type: str = 'kmeans') -> bool:
        """
        Load the latest production model.
        
        Args:
            model_type: Type of model to load ('kmeans' or 'hierarchical')
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_type == 'kmeans':
                return self.kmeans_model.load_latest_model('production')
            elif model_type == 'hierarchical':
                return self.hierarchical_model.load_latest_model('production')
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to load production model: {str(e)}")
            return False
    
    def train_both_models_with_auto_selection(self, features: pd.DataFrame,
                                            business_constraints: Dict[str, Any] = None,
                                            save_models: bool = True, 
                                            environment: str = "production") -> Dict[str, Any]:
        """
        Train both models with automatic cluster selection.
        
        Args:
            features: Feature matrix for training
            business_constraints: Business-specific constraints
            save_models: Whether to save trained models
            environment: Environment to save models in
            
        Returns:
            Dictionary with training results and auto selection analysis
        """
        try:
            logger.info("Starting training pipeline with auto cluster selection for both models")
            
            # Run comprehensive auto cluster selection
            selection_results = self.auto_select_clusters(
                features, 
                methods=['elbow', 'silhouette', 'gap', 'hierarchical', 'business'],
                business_constraints=business_constraints
            )
            
            optimal_k = selection_results['optimal_clusters']
            confidence = selection_results['confidence_score']
            
            logger.info(f"Auto selection completed: k={optimal_k}, confidence={confidence:.3f}")
            logger.info(f"Reasoning: {selection_results['recommendation']['reasoning']}")
            
            # Train K-Means with optimal k
            kmeans_results = self.train_kmeans_model(
                features, n_clusters=optimal_k, save_model=save_models, environment=environment
            )
            
            # Train Hierarchical with optimal k
            hierarchical_results = self.train_hierarchical_model(
                features, n_clusters=optimal_k, save_model=save_models, environment=environment
            )
            
            # Compare models
            comparison = self._compare_models(kmeans_results, hierarchical_results)
            
            # Prepare combined results
            combined_results = {
                'training_completed': datetime.now().isoformat(),
                'data_shape': features.shape,
                'auto_selection': selection_results,
                'kmeans_results': kmeans_results,
                'hierarchical_results': hierarchical_results,
                'model_comparison': comparison,
                'recommendation': self._get_best_model_recommendation(kmeans_results, hierarchical_results)
            }
            
            logger.info("Training pipeline with auto selection completed successfully")
            return combined_results
            
        except Exception as e:
            logger.error(f"Training pipeline with auto selection failed: {str(e)}")
            raise
