"""Feature selection utilities for customer segmentation system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from src.config.logging_config import get_logger
from src.utils.helpers import validate_feature_importance

logger = get_logger(__name__)


class FeatureSelector:
    """Selects the most relevant features for clustering."""
    
    def __init__(self):
        """Initialize feature selector."""
        self.selected_features = []
        self.feature_scores = {}
        self.selection_methods = []
        
    def remove_low_variance_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low variance.
        
        Args:
            df: Input DataFrame
            threshold: Variance threshold
            
        Returns:
            Tuple of (filtered DataFrame, removed features)
        """
        logger.info(f"Removing low variance features (threshold: {threshold})")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['customer_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not feature_cols:
            logger.warning("No numeric features available for variance filtering")
            return df, []
            
        # Calculate variance for each feature
        variances = df[feature_cols].var()
        low_variance_features = variances[variances < threshold].index.tolist()
        
        # Remove low variance features
        df_filtered = df.drop(columns=low_variance_features)
        
        logger.info(f"Removed {len(low_variance_features)} low variance features: {low_variance_features}")
        
        self.selection_methods.append("variance_threshold")
        
        return df_filtered, low_variance_features
        
    def remove_highly_correlated_features(
        self,
        df: pd.DataFrame,
        correlation_threshold: float = 0.95,
        method: str = 'pearson'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.
        
        Args:
            df: Input DataFrame
            correlation_threshold: Correlation threshold for removal
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Tuple of (filtered DataFrame, removed features)
        """
        logger.info(f"Removing highly correlated features (threshold: {correlation_threshold})")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['customer_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) < 2:
            logger.warning("Not enough features for correlation analysis")
            return df, []
            
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr(method=method)
        
        # Find highly correlated feature pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to remove
        to_remove = set()
        for col in upper_triangle.columns:
            highly_correlated = upper_triangle[col][abs(upper_triangle[col]) > correlation_threshold]
            if not highly_correlated.empty:
                # Remove the feature with lower average correlation
                for correlated_col in highly_correlated.index:
                    avg_corr_col1 = upper_triangle[col].abs().mean()
                    avg_corr_col2 = upper_triangle[correlated_col].abs().mean()
                    
                    if avg_corr_col1 <= avg_corr_col2:
                        to_remove.add(col)
                    else:
                        to_remove.add(correlated_col)
                        
        # Remove highly correlated features
        to_remove = list(to_remove)
        df_filtered = df.drop(columns=to_remove)
        
        logger.info(f"Removed {len(to_remove)} highly correlated features: {to_remove}")
        
        self.selection_methods.append("correlation_filter")
        
        return df_filtered, to_remove
        
    def select_features_by_importance(
        self,
        df: pd.DataFrame,
        n_features: int = 20,
        method: str = 'random_forest'
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """
        Select features based on importance scores.
        
        Args:
            df: Input DataFrame
            n_features: Number of features to select
            method: Importance calculation method
            
        Returns:
            Tuple of (filtered DataFrame, selected features, importance scores)
        """
        logger.info(f"Selecting top {n_features} features using {method}")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['customer_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) <= n_features:
            logger.warning(f"Only {len(feature_cols)} features available, returning all")
            return df, feature_cols, {col: 1.0 for col in feature_cols}
            
        # Create a synthetic target for unsupervised importance calculation
        # Use variance-based clustering as pseudo-target
        X = df[feature_cols].fillna(0)
        
        if method == 'random_forest':
            # Use Random Forest for feature importance
            # Create pseudo-targets using k-means clustering
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            pseudo_targets = kmeans.fit_predict(X)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, pseudo_targets)
            
            importance_scores = dict(zip(feature_cols, rf.feature_importances_))
            
        elif method == 'mutual_info':
            # Use mutual information with pseudo-targets
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            pseudo_targets = kmeans.fit_predict(X)
            
            mi_scores = mutual_info_classif(X, pseudo_targets, random_state=42)
            importance_scores = dict(zip(feature_cols, mi_scores))
            
        elif method == 'variance':
            # Use variance as importance score
            importance_scores = {col: X[col].var() for col in feature_cols}
            
        else:
            raise ValueError(f"Unknown importance method: {method}")
            
        # Normalize importance scores
        max_score = max(importance_scores.values()) if importance_scores else 1
        importance_scores = {k: v/max_score for k, v in importance_scores.items()}
        
        # Select top features
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, _ in sorted_features[:n_features]]
        
        # Filter DataFrame
        df_filtered = df[selected_features + [col for col in df.columns if col not in feature_cols]]
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        
        self.selection_methods.append(f"importance_{method}")
        self.feature_scores = importance_scores
        
        return df_filtered, selected_features, importance_scores
        
    def select_features_by_pca(
        self,
        df: pd.DataFrame,
        variance_threshold: float = 0.95,
        max_components: Optional[int] = None
    ) -> Tuple[np.ndarray, PCA, List[str]]:
        """
        Select features using PCA for dimensionality reduction.
        
        Args:
            df: Input DataFrame
            variance_threshold: Minimum variance to preserve
            max_components: Maximum number of components
            
        Returns:
            Tuple of (transformed data, PCA object, original feature names)
        """
        logger.info(f"Applying PCA with variance threshold: {variance_threshold}")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['customer_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) < 2:
            logger.warning("Not enough features for PCA")
            return df.values, None, feature_cols
            
        # Prepare data
        X = df[feature_cols].fillna(0)
        
        # Apply PCA
        n_components = min(len(feature_cols), max_components or len(feature_cols))
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Determine optimal number of components based on variance threshold
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # Keep only optimal components
        X_pca_optimal = X_pca[:, :optimal_components]
        
        logger.info(f"PCA reduced {len(feature_cols)} features to {optimal_components} components")
        logger.info(f"Explained variance ratio: {cumulative_variance[optimal_components-1]:.3f}")
        
        self.selection_methods.append("pca")
        
        return X_pca_optimal, pca, feature_cols
        
    def select_features_statistical(
        self,
        df: pd.DataFrame,
        k_best: int = 20,
        score_func: str = 'f_classif'
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """
        Select features using statistical tests.
        
        Args:
            df: Input DataFrame
            k_best: Number of best features to select
            score_func: Scoring function
            
        Returns:
            Tuple of (filtered DataFrame, selected features, feature scores)
        """
        logger.info(f"Selecting top {k_best} features using {score_func}")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['customer_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) <= k_best:
            logger.warning(f"Only {len(feature_cols)} features available, returning all")
            return df, feature_cols, {col: 1.0 for col in feature_cols}
            
        # Create pseudo-targets for statistical testing
        X = df[feature_cols].fillna(0)
        
        # Use k-means clustering to create pseudo-targets
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        pseudo_targets = kmeans.fit_predict(X)
        
        # Select scoring function
        if score_func == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k_best)
        elif score_func == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        else:
            raise ValueError(f"Unknown scoring function: {score_func}")
            
        # Fit selector
        X_selected = selector.fit_transform(X, pseudo_targets)
        
        # Get selected feature names and scores
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        feature_scores = dict(zip(feature_cols, selector.scores_))
        
        # Filter DataFrame
        df_filtered = df[selected_features + [col for col in df.columns if col not in feature_cols]]
        
        logger.info(f"Selected {len(selected_features)} features using {score_func}")
        
        self.selection_methods.append(f"statistical_{score_func}")
        self.feature_scores = feature_scores
        
        return df_filtered, selected_features, feature_scores
        
    def comprehensive_feature_selection(
        self,
        df: pd.DataFrame,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        max_features: int = 20,
        importance_method: str = 'random_forest'
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Perform comprehensive feature selection using multiple methods.
        
        Args:
            df: Input DataFrame
            variance_threshold: Variance threshold for removal
            correlation_threshold: Correlation threshold for removal
            max_features: Maximum number of features to keep
            importance_method: Method for importance-based selection
            
        Returns:
            Tuple of (filtered DataFrame, selection results)
        """
        logger.info("Starting comprehensive feature selection")
        
        df_filtered = df.copy()
        selection_results = {
            "original_shape": df.shape,
            "removed_features": [],
            "selection_steps": [],
            "final_features": [],
            "feature_scores": {}
        }
        
        # Step 1: Remove low variance features
        df_filtered, low_var_removed = self.remove_low_variance_features(
            df_filtered, variance_threshold
        )
        if low_var_removed:
            selection_results["removed_features"].extend(low_var_removed)
            selection_results["selection_steps"].append({
                "method": "variance_threshold",
                "removed": low_var_removed,
                "remaining_shape": df_filtered.shape
            })
            
        # Step 2: Remove highly correlated features
        df_filtered, corr_removed = self.remove_highly_correlated_features(
            df_filtered, correlation_threshold
        )
        if corr_removed:
            selection_results["removed_features"].extend(corr_removed)
            selection_results["selection_steps"].append({
                "method": "correlation_filter",
                "removed": corr_removed,
                "remaining_shape": df_filtered.shape
            })
            
        # Step 3: Select by importance if we still have too many features
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['customer_id']
        remaining_features = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(remaining_features) > max_features:
            df_filtered, selected_features, importance_scores = self.select_features_by_importance(
                df_filtered, max_features, importance_method
            )
            
            # Determine which features were removed in this step
            importance_removed = [col for col in remaining_features if col not in selected_features]
            selection_results["removed_features"].extend(importance_removed)
            selection_results["selection_steps"].append({
                "method": f"importance_{importance_method}",
                "removed": importance_removed,
                "selected": selected_features,
                "remaining_shape": df_filtered.shape
            })
            selection_results["feature_scores"] = importance_scores
        else:
            selection_results["final_features"] = remaining_features
            
        selection_results["final_shape"] = df_filtered.shape
        selection_results["final_features"] = [col for col in df_filtered.columns 
                                              if col not in exclude_cols]
        
        logger.info(f"Feature selection completed: {df.shape} → {df_filtered.shape}")
        logger.info(f"Total features removed: {len(selection_results['removed_features'])}")
        
        self.selected_features = selection_results["final_features"]
        
        return df_filtered, selection_results
        
    def get_selection_summary(self) -> Dict[str, any]:
        """
        Get summary of feature selection process.
        
        Returns:
            Dictionary with selection summary
        """
        return {
            "selected_features": self.selected_features,
            "feature_scores": self.feature_scores,
            "selection_methods": self.selection_methods,
            "total_selected": len(self.selected_features)
        }
