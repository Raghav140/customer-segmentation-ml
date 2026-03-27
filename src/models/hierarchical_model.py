"""Hierarchical clustering model for customer segmentation system."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config.logging_config import get_logger
from config.settings import settings
from .base import BaseClusteringModel
from ..utils.model_persistence import model_persistence

logger = get_logger(__name__)


class HierarchicalModel(BaseClusteringModel):
    """Hierarchical clustering model with enhanced functionality."""
    
    def __init__(self, random_state: int = None):
        """
        Initialize Hierarchical model.
        
        Args:
            random_state: Random state for reproducibility
        """
        super().__init__("Hierarchical", random_state or 42)
        self.linkage_matrix = None
        self.linkage_method = settings.model.hierarchical_linkage
        self.metric = settings.model.hierarchical_metric
        self.cophenet_corr = None
        self.distance_matrix = None
        self.dendrogram_data = {}
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            n_clusters: int = 5,
            linkage_method: str = None,
            metric: str = None,
            distance_threshold: Optional[float] = None,
            **kwargs) -> 'HierarchicalModel':
        """
        Fit Hierarchical clustering model to the data.
        
        Args:
            X: Input data
            n_clusters: Number of clusters (or None for distance_threshold)
            linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
            metric: Distance metric
            distance_threshold: Distance threshold for clustering
            **kwargs: Additional parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting Hierarchical clustering with {n_clusters} clusters")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Set parameters
        self.linkage_method = linkage_method or self.linkage_method
        self.metric = metric or self.metric
        
        # Special handling for 'ward' linkage
        if self.linkage_method == 'ward' and self.metric != 'euclidean':
            logger.warning("Ward linkage only works with euclidean metric. Switching to euclidean.")
            self.metric = 'euclidean'
            
        # Store training data
        self.training_data = X
        
        # Compute distance matrix
        self.distance_matrix = pdist(X, metric=self.metric)
        
        # Perform hierarchical clustering
        self.linkage_matrix = linkage(X, method=self.linkage_method, metric=self.metric)
        
        # Calculate cophenetic correlation coefficient
        self.cophenet_corr, _ = cophenet(self.linkage_matrix, self.distance_matrix)
        
        # Form clusters
        if distance_threshold is not None:
            self.cluster_labels = fcluster(self.linkage_matrix, 
                                         distance_threshold, 
                                         criterion='distance')
            self.n_clusters = len(np.unique(self.cluster_labels))
        else:
            self.cluster_labels = fcluster(self.linkage_matrix, 
                                         n_clusters, 
                                         criterion='maxclust')
            self.n_clusters = n_clusters
            
        # Adjust labels to start from 0
        self.cluster_labels = self.cluster_labels - 1
        
        # Calculate cluster centers (mean of points in each cluster)
        self.cluster_centers = []
        for i in range(self.n_clusters):
            cluster_points = X[self.cluster_labels == i]
            if len(cluster_points) > 0:
                self.cluster_centers.append(np.mean(cluster_points, axis=0))
            else:
                self.cluster_centers.append(np.zeros(X.shape[1]))
                
        self.cluster_centers = np.array(self.cluster_centers)
        self.is_fitted = True
        
        # Store model parameters
        self.model_params = {
            'n_clusters': self.n_clusters,
            'linkage_method': self.linkage_method,
            'metric': self.metric,
            'distance_threshold': distance_threshold
        }
        
        # Calculate metrics
        self.calculate_metrics(X)
        
        logger.info(f"Hierarchical clustering fitted successfully")
        logger.info(f"Cophenetic correlation: {self.cophenet_corr:.4f}")
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
        
        # For hierarchical clustering, we need to assign new points to existing clusters
        # by finding the nearest cluster center
        labels = []
        
        for point in X:
            # Calculate distance to each cluster center
            distances = []
            for center in self.cluster_centers:
                if self.metric == 'euclidean':
                    dist = np.linalg.norm(point - center)
                elif self.metric == 'cityblock':
                    dist = np.sum(np.abs(point - center))
                elif self.metric == 'cosine':
                    dist = 1 - np.dot(point, center) / (np.linalg.norm(point) * np.linalg.norm(center) + 1e-8)
                else:
                    # Default to euclidean
                    dist = np.linalg.norm(point - center)
                distances.append(dist)
                
            # Assign to nearest cluster
            labels.append(np.argmin(distances))
            
        return np.array(labels)
        
    def plot_dendrogram(self, X: Union[pd.DataFrame, np.ndarray],
                       truncate_mode: Optional[str] = None,
                       p: int = 30,
                       figsize: Tuple[int, int] = (15, 8),
                       color_threshold: Optional[float] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Plot dendrogram for hierarchical clustering.
        
        Args:
            X: Input data
            truncate_mode: Mode for truncating dendrogram
            p: Parameter for truncation
            figsize: Figure size
            color_threshold: Height at which to cut dendrogram
            save_path: Path to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting dendrogram")
            
        plt.figure(figsize=figsize)
        
        # Create dendrogram
        dendrogram_data = dendrogram(
            self.linkage_matrix,
            truncate_mode=truncate_mode,
            p=p,
            color_threshold=color_threshold,
            show_leaf_counts=True,
            leaf_rotation=90,
            leaf_font_size=10,
            show_contracted=True
        )
        
        # Store dendrogram data for later use
        self.dendrogram_data = dendrogram_data
        
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage_method} linkage, {self.metric} distance)')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dendrogram saved to {save_path}")
            
        plt.show()
        
    def create_interactive_dendrogram(self, X: Union[pd.DataFrame, np.ndarray],
                                   sample_labels: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive dendrogram using Plotly.
        
        Args:
            X: Input data
            sample_labels: Labels for samples
            
        Returns:
            Plotly figure
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before creating interactive dendrogram")
            
        # Extract linkage information for plotting
        icoord = np.array(self.linkage_matrix[:, :2].astype(float))
        dcoord = np.array(self.linkage_matrix[:, 2].astype(float))
        
        # Calculate coordinates
        icoord = icoord * 10 + 5  # Scale for better visualization
        dcoord = dcoord
        
        # Create traces for each link
        traces = []
        
        for i in range(len(icoord)):
            # Vertical line
            traces.append(go.Scatter(
                x=[icoord[i, 0], icoord[i, 0]],
                y=[dcoord[i], self.linkage_matrix[i, 3]],
                mode='lines',
                line=dict(color='blue', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
            
            # Horizontal lines
            traces.append(go.Scatter(
                x=[icoord[i, 0], icoord[i, 1]],
                y=[dcoord[i], dcoord[i]],
                mode='lines',
                line=dict(color='blue', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
            
            traces.append(go.Scatter(
                x=[icoord[i, 1], icoord[i, 1]],
                y=[dcoord[i], self.linkage_matrix[i, 3]],
                mode='lines',
                line=dict(color='blue', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
            
        # Create figure
        fig = go.Figure(traces)
        
        fig.update_layout(
            title=f'Interactive Dendrogram ({self.linkage_method} linkage)',
            xaxis_title='Samples',
            yaxis_title='Distance',
            showlegend=False,
            height=600
        )
        
        return fig
        
    def find_optimal_clusters(self, X: Union[pd.DataFrame, np.ndarray],
                            k_range: Tuple[int, int] = (2, 15),
                            method: str = 'silhouette',
                            plot_results: bool = True,
                            figsize: Tuple[int, int] = (15, 5)) -> int:
        """
        Find optimal number of clusters using various methods.
        
        Args:
            X: Input data
            k_range: Range of k values to test
            method: Method for finding optimal k
            plot_results: Whether to plot results
            figsize: Figure size for plotting
            
        Returns:
            Optimal number of clusters
        """
        logger.info(f"Finding optimal number of clusters using {method} method")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        k_min, k_max = k_range
        k_values = range(k_min, k_max + 1)
        
        # Calculate metrics for each k
        silhouette_scores = []
        cophenet_scores = []
        inertias = []
        
        for k in k_values:
            logger.info(f"Testing k={k}")
            
            # Form clusters
            labels = fcluster(self.linkage_matrix, k, criterion='maxclust')
            labels = labels - 1  # Adjust to start from 0
            
            # Calculate silhouette score
            if k > 1 and len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
                
            # Calculate inertia (within-cluster sum of squares)
            inertia = 0
            for cluster_id in range(k):
                cluster_points = X[labels == cluster_id]
                if len(cluster_points) > 0:
                    cluster_center = np.mean(cluster_points, axis=0)
                    inertia += np.sum((cluster_points - cluster_center) ** 2)
            inertias.append(inertia)
            
        # Store results
        self.silhouette_data = {
            'k_values': list(k_values),
            'silhouette_scores': silhouette_scores
        }
        
        # Find optimal k
        if method == 'silhouette':
            optimal_k = k_values[np.argmax(silhouette_scores)]
        elif method == 'inertia':
            # Use elbow method on inertia
            optimal_k = self._find_elbow_point(k_values, inertias)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        logger.info(f"Optimal number of clusters found: {optimal_k}")
        
        # Plot results
        if plot_results:
            self._plot_optimal_clusters_analysis(figsize)
            
        return optimal_k
        
    def _find_elbow_point(self, k_values: range, values: List[float]) -> int:
        """
        Find elbow point using the "knee" detection algorithm.
        
        Args:
            k_values: Range of k values
            values: Corresponding values
            
        Returns:
            Optimal k (elbow point)
        """
        if len(values) < 3:
            return k_values[0]
            
        # Calculate second derivative to find elbow
        deltas = np.diff(values)
        second_deltas = np.diff(deltas)
        
        # Find the point where second derivative is maximum
        if len(second_deltas) > 0:
            elbow_idx = np.argmax(second_deltas) + 2  # +2 because of double diff
            return k_values[min(elbow_idx, len(k_values) - 1)]
        else:
            return k_values[0]
            
    def _plot_optimal_clusters_analysis(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot optimal clusters analysis results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Silhouette scores plot
        if self.silhouette_data:
            ax1.plot(self.silhouette_data['k_values'], self.silhouette_data['silhouette_scores'], 
                    'go-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_title('Silhouette Analysis')
            ax1.grid(True, alpha=0.3)
            
        # Cophenetic correlation
        if self.cophenet_corr is not None:
            ax2.bar(['Cophenetic Correlation'], [self.cophenet_corr], color='orange')
            ax2.set_ylabel('Correlation Coefficient')
            ax2.set_title('Cophenetic Correlation')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    def plot_cluster_heatmap(self, X: Union[pd.DataFrame, np.ndarray],
                           feature_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> None:
        """
        Plot cluster centers as heatmap.
        
        Args:
            X: Input data
            feature_names: Names of features
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting cluster heatmap")
            
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
        plt.title(f'Hierarchical Cluster Centers ({self.linkage_method} linkage)')
        plt.xlabel('Features')
        plt.ylabel('Clusters')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster heatmap saved to {save_path}")
            
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
            subplot_titles=('Cluster Centers Heatmap', 'Silhouette Scores', 
                          'Cophenetic Correlation', 'Cluster Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
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
        
        # 2. Silhouette scores
        if self.silhouette_data:
            fig.add_trace(
                go.Scatter(
                    x=self.silhouette_data['k_values'],
                    y=self.silhouette_data['silhouette_scores'],
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
            
        # 3. Cophenetic correlation
        if self.cophenet_corr is not None:
            fig.add_trace(
                go.Bar(
                    x=['Cophenetic Correlation'],
                    y=[self.cophenet_corr],
                    name='Cophenetic Corr',
                    marker_color='orange'
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
            title=f'Hierarchical Cluster Analysis (k={self.n_clusters})',
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        fig.update_yaxes(title_text="Correlation", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Number of Samples", row=2, col=2)
        
        return fig
        
    def compare_linkage_methods(self, X: Union[pd.DataFrame, np.ndarray],
                              linkage_methods: List[str] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> Dict[str, float]:
        """
        Compare different linkage methods.
        
        Args:
            X: Input data
            linkage_methods: List of linkage methods to compare
            figsize: Figure size
            
        Returns:
            Dictionary with cophenetic correlations for each method
        """
        if linkage_methods is None:
            linkage_methods = ['ward', 'complete', 'average', 'single']
            
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        results = {}
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, method in enumerate(linkage_methods):
            if i >= len(axes):
                break
                
            try:
                # Perform hierarchical clustering with this linkage method
                if method == 'ward':
                    Z = linkage(X, method='ward', metric='euclidean')
                else:
                    Z = linkage(X, method=method, metric=self.metric)
                    
                # Calculate cophenetic correlation
                coph_corr, _ = cophenet(Z, pdist(X, metric=self.metric))
                results[method] = coph_corr
                
                # Plot truncated dendrogram
                dendrogram(
                    Z,
                    ax=axes[i],
                    truncate_mode='lastp',
                    p=12,
                    show_leaf_counts=True,
                    show_contracted=True,
                    leaf_rotation=45
                )
                
                axes[i].set_title(f'{method.title()} Linkage\\n(Cophenetic: {coph_corr:.3f})')
                
            except Exception as e:
                logger.error(f"Error with {method} linkage: {e}")
                axes[i].text(0.5, 0.5, f'Error with {method}\\n{str(e)}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                
        plt.tight_layout()
        plt.show()
        
        # Find best method
        if results:
            best_method = max(results, key=results.get)
            logger.info(f"Best linkage method: {best_method} (cophenetic: {results[best_method]:.3f})")
            
        return results
        
    def get_model_summary(self) -> Dict[str, any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with model summary
        """
        summary = super().get_model_summary()
        
        # Add Hierarchical specific information
        summary.update({
            'linkage_method': self.linkage_method,
            'metric': self.metric,
            'cophenet_corr': self.cophenet_corr,
            'linkage_matrix_shape': self.linkage_matrix.shape if self.linkage_matrix is not None else None
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
        if self.linkage_matrix is None:
            raise ValueError("No trained model to save. Call fit() first.")
        
        # Prepare model data for saving
        model_data = {
            'linkage_matrix': self.linkage_matrix,
            'n_clusters': self.n_clusters,
            'linkage_method': self.linkage_method,
            'metric': self.metric,
            'cophenet_corr': self.cophenet_corr
        }
        
        # Prepare metadata
        model_metadata = {
            'model_type': 'Hierarchical',
            'n_clusters': self.n_clusters,
            'linkage_method': self.linkage_method,
            'metric': self.metric,
            'cophenet_corr': self.cophenet_corr,
            'random_state': self.random_state,
            'training_date': datetime.now().isoformat()
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        # Save the model
        model_path = model_persistence.save_model(
            model_data, 
            f'hierarchical_{self.n_clusters}_clusters', 
            environment, 
            model_metadata
        )
        
        logger.info(f"Hierarchical model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> 'HierarchicalModel':
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
        self.linkage_matrix = loaded_model['linkage_matrix']
        self.n_clusters = loaded_model['n_clusters']
        self.linkage_method = loaded_model['linkage_method']
        self.metric = loaded_model['metric']
        self.cophenet_corr = loaded_model['cophenet_corr']
        
        if metadata:
            self.random_state = metadata.get('random_state')
        
        logger.info(f"Hierarchical model loaded from {model_path}")
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
            latest_model = model_persistence.load_latest_model('hierarchical', environment)
            if latest_model is not None:
                self.linkage_matrix = latest_model['linkage_matrix']
                self.n_clusters = latest_model['n_clusters']
                self.linkage_method = latest_model['linkage_method']
                self.metric = latest_model['metric']
                self.cophenet_corr = latest_model['cophenet_corr']
                logger.info(f"Loaded latest Hierarchical model from {environment}")
                return True
        except Exception as e:
            logger.error(f"Failed to load latest model: {str(e)}")
        
        return False
