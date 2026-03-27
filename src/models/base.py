"""Base clustering model for customer segmentation system."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
from pathlib import Path

from config.logging_config import get_logger
from utils.helpers import save_model, load_model, ensure_directory

logger = get_logger(__name__)


class BaseClusteringModel(ABC):
    """Abstract base class for clustering models."""
    
    def __init__(self, model_name: str, random_state: int = 42):
        """
        Initialize base clustering model.
        
        Args:
            model_name: Name of the clustering model
            random_state: Random state for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.cluster_labels = None
        self.cluster_centers = None
        self.n_clusters = None
        self.feature_names = []
        self.training_data = None
        self.metrics = {}
        self.model_params = {}
        
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> 'BaseClusteringModel':
        """
        Fit the clustering model to the data.
        
        Args:
            X: Input data
            **kwargs: Additional parameters for fitting
            
        Returns:
            Fitted model instance
        """
        pass
        
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        pass
        
    def fit_predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """
        Fit model and predict cluster labels in one step.
        
        Args:
            X: Input data
            **kwargs: Additional parameters
            
        Returns:
            Cluster labels
        """
        self.fit(X, **kwargs)
        return self.predict(X)
        
    def calculate_metrics(self, X: Union[pd.DataFrame, np.ndarray], 
                         labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate clustering evaluation metrics.
        
        Args:
            X: Input data
            labels: Cluster labels (if None, use self.cluster_labels)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if labels is None:
            labels = self.cluster_labels
            
        if labels is None:
            raise ValueError("No cluster labels available. Fit model first or provide labels.")
            
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        metrics = {}
        
        try:
            # Silhouette Score (higher is better, ranges from -1 to 1)
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X):
                metrics['silhouette_score'] = silhouette_score(X, labels)
            else:
                metrics['silhouette_score'] = 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
            metrics['silhouette_score'] = 0.0
            
        try:
            # Davies-Bouldin Score (lower is better)
            if len(np.unique(labels)) > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            else:
                metrics['davies_bouldin_score'] = float('inf')
                
        except Exception as e:
            logger.warning(f"Could not calculate Davies-Bouldin score: {e}")
            metrics['davies_bouldin_score'] = float('inf')
            
        try:
            # Calinski-Harabasz Score (higher is better)
            if len(np.unique(labels)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            else:
                metrics['calinski_harabasz_score'] = 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate Calinski-Harabasz score: {e}")
            metrics['calinski_harabasz_score'] = 0.0
            
        # Additional custom metrics
        metrics['n_clusters'] = len(np.unique(labels))
        metrics['total_samples'] = len(labels)
        
        # Cluster size statistics
        cluster_sizes = pd.Series(labels).value_counts()
        metrics['min_cluster_size'] = cluster_sizes.min()
        metrics['max_cluster_size'] = cluster_sizes.max()
        metrics['avg_cluster_size'] = cluster_sizes.mean()
        metrics['cluster_size_std'] = cluster_sizes.std()
        
        # Balance score (lower is more balanced)
        metrics['balance_score'] = cluster_sizes.std() / cluster_sizes.mean() if cluster_sizes.mean() > 0 else float('inf')
        
        self.metrics = metrics
        return metrics
        
    def get_cluster_statistics(self, X: Union[pd.DataFrame, np.ndarray], 
                             feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get detailed statistics for each cluster.
        
        Args:
            X: Input data
            feature_names: Names of features
            
        Returns:
            DataFrame with cluster statistics
        """
        if self.cluster_labels is None:
            raise ValueError("Model must be fitted before getting cluster statistics")
            
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
            
        # Add cluster labels
        X_with_labels = X.copy()
        X_with_labels['cluster'] = self.cluster_labels
        
        # Calculate statistics for each cluster
        cluster_stats = []
        
        for cluster_id in sorted(X_with_labels['cluster'].unique()):
            cluster_data = X_with_labels[X_with_labels['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(X_with_labels)) * 100
            }
            
            # Add statistics for each feature
            numeric_features = X.select_dtypes(include=[np.number]).columns
            
            for feature in numeric_features:
                stats[f'{feature}_mean'] = cluster_data[feature].mean()
                stats[f'{feature}_std'] = cluster_data[feature].std()
                stats[f'{feature}_min'] = cluster_data[feature].min()
                stats[f'{feature}_max'] = cluster_data[feature].max()
                stats[f'{feature}_median'] = cluster_data[feature].median()
                
            cluster_stats.append(stats)
            
        return pd.DataFrame(cluster_stats)
        
    def plot_clusters(self, X: Union[pd.DataFrame, np.ndarray], 
                     method: str = 'pca', 
                     figsize: Tuple[int, int] = (12, 8),
                     title: Optional[str] = None,
                     save_path: Optional[str] = None) -> None:
        """
        Plot clusters using dimensionality reduction.
        
        Args:
            X: Input data
            method: Dimensionality reduction method ('pca', 'tsne', or 'none')
            figsize: Figure size
            title: Plot title
            save_path: Path to save the plot
        """
        if self.cluster_labels is None:
            raise ValueError("Model must be fitted before plotting clusters")
            
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Apply dimensionality reduction if needed
        if method == 'pca' and X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=self.random_state)
            X_plot = pca.fit_transform(X)
            explained_var = sum(pca.explained_variance_ratio_)
            subtitle = f"PCA (Explained Variance: {explained_var:.3f})"
        elif method == 'tsne' and X.shape[1] > 2:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=self.random_state)
            X_plot = tsne.fit_transform(X)
            subtitle = "t-SNE"
        else:
            X_plot = X[:, :2] if X.shape[1] >= 2 else X
            subtitle = "Original Features"
            
        # Create plot
        plt.figure(figsize=figsize)
        
        unique_labels = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = self.cluster_labels == label
            plt.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                       c=[colors[i]], label=f'Cluster {label}', 
                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if title:
            plot_title = title
        else:
            plot_title = f'{self.model_name} Clusters'
            
        plt.title(f'{plot_title}\\n({subtitle})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster plot saved to {save_path}")
            
        plt.tight_layout()
        plt.show()
        
    def create_interactive_cluster_plot(self, X: Union[pd.DataFrame, np.ndarray],
                                      method: str = 'pca',
                                      hover_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create interactive cluster plot using Plotly.
        
        Args:
            X: Input data
            method: Dimensionality reduction method
            hover_data: Additional data for hover tooltips
            
        Returns:
            Plotly figure
        """
        if self.cluster_labels is None:
            raise ValueError("Model must be fitted before creating interactive plot")
            
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Apply dimensionality reduction if needed
        if method == 'pca' and X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=self.random_state)
            X_plot = pca.fit_transform(X)
            explained_var = sum(pca.explained_variance_ratio_)
            subtitle = f"PCA (Explained Variance: {explained_var:.3f})"
        elif method == 'tsne' and X.shape[1] > 2:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=self.random_state)
            X_plot = tsne.fit_transform(X)
            subtitle = "t-SNE"
        else:
            X_plot = X[:, :2] if X.shape[1] >= 2 else X
            subtitle = "Original Features"
            
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': X_plot[:, 0],
            'y': X_plot[:, 1],
            'cluster': self.cluster_labels
        })
        
        # Add hover data if provided
        if hover_data is not None:
            for col in hover_data.columns:
                plot_df[col] = hover_data[col]
                
        # Create plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='cluster',
            hover_data=hover_data.columns.tolist() if hover_data is not None else None,
            title=f'{self.model_name} Clusters<br><sub>{subtitle}</sub>',
            opacity=0.7,
            color_continuous_scale='viridis' if len(np.unique(self.cluster_labels)) > 10 else None
        )
        
        fig.update_layout(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            width=800,
            height=600
        )
        
        return fig
        
    def plot_cluster_sizes(self, figsize: Tuple[int, int] = (10, 6),
                          save_path: Optional[str] = None) -> None:
        """
        Plot cluster size distribution.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if self.cluster_labels is None:
            raise ValueError("Model must be fitted before plotting cluster sizes")
            
        # Calculate cluster sizes
        cluster_sizes = pd.Series(self.cluster_labels).value_counts().sort_index()
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        cluster_sizes.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Cluster Sizes')
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(True, alpha=0.3)
        
        # Pie chart
        ax2.pie(cluster_sizes.values, labels=[f'Cluster {i}' for i in cluster_sizes.index],
               autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes))))
        ax2.set_title('Cluster Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster size plot saved to {save_path}")
            
        plt.show()
        
    def save_model(self, file_path: Union[str, Path]) -> None:
        """
        Save the fitted model.
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        model_data = {
            'model_name': self.model_name,
            'model': self.model,
            'is_fitted': self.is_fitted,
            'cluster_labels': self.cluster_labels,
            'cluster_centers': self.cluster_centers,
            'n_clusters': self.n_clusters,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'metrics': self.metrics,
            'model_params': self.model_params
        }
        
        save_model(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
        
    def load_model(self, file_path: Union[str, Path]) -> 'BaseClusteringModel':
        """
        Load a fitted model.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        model_data = load_model(file_path)
        
        self.model_name = model_data['model_name']
        self.model = model_data['model']
        self.is_fitted = model_data['is_fitted']
        self.cluster_labels = model_data['cluster_labels']
        self.cluster_centers = model_data['cluster_centers']
        self.n_clusters = model_data['n_clusters']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data['random_state']
        self.metrics = model_data['metrics']
        self.model_params = model_data['model_params']
        
        logger.info(f"Model loaded from {file_path}")
        
        return self
        
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary with model summary
        """
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'n_clusters': self.n_clusters,
            'n_features': len(self.feature_names),
            'total_samples': len(self.cluster_labels) if self.cluster_labels is not None else 0,
            'metrics': self.metrics,
            'model_params': self.model_params
        }
        
    def __str__(self) -> str:
        """String representation of the model."""
        status = "Fitted" if self.is_fitted else "Not fitted"
        return f"{self.model_name} ({status}) - {self.n_clusters or '?'} clusters"
        
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return (f"{self.__class__.__name__}(model_name='{self.model_name}', "
               f"n_clusters={self.n_clusters}, is_fitted={self.is_fitted})")
