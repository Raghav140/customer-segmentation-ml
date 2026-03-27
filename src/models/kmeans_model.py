"""K-Means clustering model for customer segmentation system."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config.logging_config import get_logger
from config.settings import settings
from .base import BaseClusteringModel
from ..utils.model_persistence import model_persistence

logger = get_logger(__name__)


class KMeansModel(BaseClusteringModel):
    """K-Means clustering model with enhanced functionality."""
    
    def __init__(self, random_state: int = None):
        """
        Initialize K-Means model.
        
        Args:
            random_state: Random state for reproducibility
        """
        super().__init__("K-Means", random_state or settings.model.kmeans_random_state)
        self.inertia = None
        self.elbow_data = {}
        self.silhouette_data = {}
        self.optimal_k = None
        self.algorithm = 'lloyd'  # Default algorithm
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            n_clusters: int = 8,
            init: str = 'k-means++',
            n_init: int = None,
            max_iter: int = 300,
            algorithm: str = 'lloyd',
            **kwargs) -> 'KMeansModel':
        """
        Fit K-Means model to the data.
        
        Args:
            X: Input data
            n_clusters: Number of clusters
            init: Initialization method
            n_init: Number of random initializations
            max_iter: Maximum number of iterations
            algorithm: K-Means algorithm
            **kwargs: Additional parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting K-Means with {n_clusters} clusters")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Set default n_init from settings
        if n_init is None:
            n_init = settings.model.kmeans_n_init
            
        # Create and fit K-Means model
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            algorithm=algorithm,
            random_state=self.random_state,
            **kwargs
        )
        
        self.cluster_labels = self.model.fit_predict(X)
        self.cluster_centers = self.model.cluster_centers_
        self.inertia = self.model.inertia_
        self.n_clusters = n_clusters
        self.training_data = X
        self.is_fitted = True
        
        # Store model parameters
        self.model_params = {
            'n_clusters': n_clusters,
            'init': init,
            'n_init': n_init,
            'max_iter': max_iter,
            'algorithm': algorithm
        }
        
        # Calculate metrics
        self.calculate_metrics(X)
        
        logger.info(f"K-Means fitted successfully with inertia: {self.inertia:.2f}")
        logger.info(f"Cluster distribution: {np.bincount(self.cluster_labels)}")
        
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        return self.model.predict(X)
        
    def find_optimal_k(self, X: Union[pd.DataFrame, np.ndarray],
                      k_range: Tuple[int, int] = (2, 15),
                      method: str = 'elbow',
                      plot_results: bool = True,
                      figsize: Tuple[int, int] = (15, 5)) -> int:
        """
        Find optimal number of clusters using various methods.
        
        Args:
            X: Input data
            k_range: Range of k values to test
            method: Method for finding optimal k ('elbow', 'silhouette', 'both')
            plot_results: Whether to plot results
            figsize: Figure size for plotting
            
        Returns:
            Optimal number of clusters
        """
        logger.info(f"Finding optimal k using {method} method")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        k_min, k_max = k_range
        k_values = range(k_min, k_max + 1)
        
        # Calculate metrics for each k
        inertias = []
        silhouette_scores = []
        
        for k in k_values:
            logger.info(f"Testing k={k}")
            
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=settings.model.kmeans_n_init,
                random_state=self.random_state
            )
            
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k > 1 and len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
                
        # Store results
        self.elbow_data = {
            'k_values': list(k_values),
            'inertias': inertias
        }
        
        self.silhouette_data = {
            'k_values': list(k_values),
            'silhouette_scores': silhouette_scores
        }
        
        # Find optimal k
        if method == 'elbow':
            self.optimal_k = self._find_elbow_point(k_values, inertias)
        elif method == 'silhouette':
            self.optimal_k = k_values[np.argmax(silhouette_scores)]
        elif method == 'both':
            elbow_k = self._find_elbow_point(k_values, inertias)
            silhouette_k = k_values[np.argmax(silhouette_scores)]
            self.optimal_k = max(elbow_k, silhouette_k)  # Conservative choice
        else:
            raise ValueError(f"Unknown method: {method}")
            
        logger.info(f"Optimal k found: {self.optimal_k}")
        
        # Plot results
        if plot_results:
            self._plot_optimal_k_analysis(figsize)
            
        return self.optimal_k
        
    def _find_elbow_point(self, k_values: range, inertias: List[float]) -> int:
        """
        Find elbow point using the "knee" detection algorithm.
        
        Args:
            k_values: Range of k values
            inertias: Corresponding inertia values
            
        Returns:
            Optimal k (elbow point)
        """
        if len(inertias) < 3:
            return k_values[0]
            
        # Calculate second derivative to find elbow
        deltas = np.diff(inertias)
        second_deltas = np.diff(deltas)
        
        # Find the point where second derivative is maximum
        if len(second_deltas) > 0:
            elbow_idx = np.argmax(second_deltas) + 2  # +2 because of double diff
            return k_values[min(elbow_idx, len(k_values) - 1)]
        else:
            return k_values[0]
            
    def _plot_optimal_k_analysis(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot optimal k analysis results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Elbow method plot
        ax1.plot(self.elbow_data['k_values'], self.elbow_data['inertias'], 
                'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=self.optimal_k, color='r', linestyle='--', 
                   label=f'Optimal k={self.optimal_k}')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Silhouette analysis plot
        ax2.plot(self.silhouette_data['k_values'], self.silhouette_data['silhouette_scores'], 
                'go-', linewidth=2, markersize=8)
        ax2.axvline(x=self.optimal_k, color='r', linestyle='--', 
                   label=f'Optimal k={self.optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_silhouette_analysis(self, X: Union[pd.DataFrame, np.ndarray],
                               figsize: Tuple[int, int] = (15, 8)) -> None:
        """
        Plot detailed silhouette analysis.
        
        Args:
            X: Input data
            figsize: Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before silhouette analysis")
            
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Calculate silhouette values for each sample
        silhouette_vals = silhouette_samples(X, self.cluster_labels)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Silhouette plot
        y_lower = 10
        for i, cluster in enumerate(sorted(np.unique(self.cluster_labels))):
            cluster_silhouette_vals = silhouette_vals[self.cluster_labels == cluster]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.Set3(i / len(np.unique(self.cluster_labels)))
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
            y_lower = y_upper + 10
            
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster label')
        score = self.metrics.get("silhouette_score", 0)
        ax1.set_title(f'Silhouette Plot (Average Score: {score:.3f})')
        ax1.axvline(x=score, color='red', linestyle='--')
        
        # Cluster visualization
        colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(self.cluster_labels))))
        
        # Use PCA for visualization if needed
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=self.random_state)
            X_plot = pca.fit_transform(X)
        else:
            X_plot = X
            
        for i, cluster in enumerate(sorted(np.unique(self.cluster_labels))):
            mask = self.cluster_labels == cluster
            ax2.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                       c=[colors[i]], label=f'Cluster {cluster}', 
                       alpha=0.7, s=50)
            
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.set_title('Cluster Visualization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_cluster_centers(self, feature_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (15, 8),
                           save_path: Optional[str] = None) -> None:
        """
        Plot cluster centers as heatmap.
        
        Args:
            feature_names: Names of features
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting cluster centers")
            
        if feature_names is None:
            feature_names = self.feature_names
            
        # Create DataFrame of cluster centers
        centers_df = pd.DataFrame(
            self.cluster_centers,
            columns=feature_names,
            index=[f'Cluster {i}' for i in range(self.n_clusters)]
        )
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(centers_df, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', cbar_kws={'label': 'Feature Value'})
        plt.title('K-Means Cluster Centers')
        plt.xlabel('Features')
        plt.ylabel('Clusters')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster centers plot saved to {save_path}")
            
        plt.show()
        
    def create_interactive_cluster_analysis(self, X: Union[pd.DataFrame, np.ndarray],
                                         feature_names: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive cluster analysis dashboard.
        
        Args:
            X: Input data
            feature_names: Names of features
            
        Returns:
            Plotly figure with subplots
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before creating interactive analysis")
            
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            if feature_names is None:
                feature_names = X.columns.tolist()
                
        if feature_names is None:
            feature_names = self.feature_names
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Centers Heatmap', 'Elbow Method', 
                          'Silhouette Scores', 'Cluster Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Cluster centers heatmap
        fig.add_trace(
            go.Heatmap(
                z=self.cluster_centers,
                x=feature_names,
                y=[f'Cluster {i}' for i in range(self.n_clusters)],
                colorscale='RdBu',
                name='Cluster Centers'
            ),
            row=1, col=1
        )
        
        # 2. Elbow method
        if self.elbow_data:
            fig.add_trace(
                go.Scatter(
                    x=self.elbow_data['k_values'],
                    y=self.elbow_data['inertias'],
                    mode='lines+markers',
                    name='Inertia',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
            
        # 3. Silhouette scores
        if self.silhouette_data:
            fig.add_trace(
                go.Scatter(
                    x=self.silhouette_data['k_values'],
                    y=self.silhouette_data['silhouette_scores'],
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
        # 4. Cluster distribution
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {i}' for i in cluster_counts.index],
                y=cluster_counts.values,
                name='Cluster Size',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'K-Means Cluster Analysis (k={self.n_clusters})',
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
        fig.update_yaxes(title_text="Inertia", row=1, col=2)
        fig.update_xaxes(title_text="Number of Clusters", row=2, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Number of Samples", row=2, col=2)
        
        return fig
        
    def get_cluster_characteristics(self, X: Union[pd.DataFrame, np.ndarray],
                                 feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get detailed characteristics of each cluster.
        
        Args:
            X: Input data
            feature_names: Names of features
            
        Returns:
            DataFrame with cluster characteristics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster characteristics")
            
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = self.feature_names
            X = pd.DataFrame(X, columns=feature_names)
            
        # Add cluster labels
        X_with_labels = X.copy()
        X_with_labels['cluster'] = self.cluster_labels
        
        # Calculate characteristics for each cluster
        characteristics = []
        
        for cluster_id in sorted(X_with_labels['cluster'].unique()):
            cluster_data = X_with_labels[X_with_labels['cluster'] == cluster_id]
            
            char = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(X_with_labels)) * 100
            }
            
            # Feature statistics
            for feature in X.columns:
                char[f'{feature}_mean'] = cluster_data[feature].mean()
                char[f'{feature}_std'] = cluster_data[feature].std()
                char[f'{feature}_median'] = cluster_data[feature].median()
                
                # Compare to overall mean
                overall_mean = X[feature].mean()
                char[f'{feature}_vs_overall'] = cluster_data[feature].mean() - overall_mean
                
            characteristics.append(char)
            
        return pd.DataFrame(characteristics)
        
    def get_model_summary(self) -> Dict[str, any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with model summary
        """
        summary = super().get_model_summary()
        
        # Add K-Means specific information
        summary.update({
            'inertia': self.inertia,
            'optimal_k': self.optimal_k,
            'elbow_data': self.elbow_data,
            'silhouette_data': self.silhouette_data
        })
        
        return summary
    
    def save_model(self, environment: str = "production", metadata: Optional[Dict] = None) -> str:
        """
        Save the trained model with metadata.
        
        Args:
            environment: Environment (production, staging, development)
            metadata: Additional metadata to save
            
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            raise ValueError("No trained model to save. Call fit() first.")
        
        # Prepare metadata
        model_metadata = {
            'model_type': 'KMeans',
            'n_clusters': self.n_clusters,
            'inertia': self.inertia,
            'optimal_k': self.optimal_k,
            'algorithm': self.algorithm,
            'random_state': self.random_state,
            'training_date': datetime.now().isoformat()
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        # Save the model
        model_path = model_persistence.save_model(
            self.model, 
            f'kmeans_{self.n_clusters}_clusters', 
            environment, 
            model_metadata
        )
        
        logger.info(f"K-Means model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> 'KMeansModel':
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Self with loaded model
        """
        loaded_model = model_persistence.load_model(model_path)
        
        # Get metadata
        metadata = model_persistence.get_model_metadata(model_path)
        
        # Set model attributes
        self.model = loaded_model
        self.n_clusters = loaded_model.n_clusters if hasattr(loaded_model, 'n_clusters') else metadata.get('n_clusters', 4)
        self.inertia = loaded_model.inertia_ if hasattr(loaded_model, 'inertia_') else metadata.get('inertia')
        self.algorithm = loaded_model.algorithm if hasattr(loaded_model, 'algorithm') else metadata.get('algorithm', 'lloyd')
        
        if metadata:
            self.optimal_k = metadata.get('optimal_k')
            self.random_state = metadata.get('random_state')
        
        logger.info(f"K-Means model loaded from {model_path}")
        return self
    
    def load_latest_model(self, environment: str = "production") -> bool:
        """
        Load the latest trained model.
        
        Args:
            environment: Environment to load from
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            latest_model = model_persistence.load_latest_model('kmeans', environment)
            if latest_model is not None:
                self.model = latest_model
                self.n_clusters = latest_model.n_clusters
                self.inertia = latest_model.inertia_
                logger.info(f"Loaded latest K-Means model from {environment}")
                return True
        except Exception as e:
            logger.error(f"Failed to load latest model: {str(e)}")
        
        return False
