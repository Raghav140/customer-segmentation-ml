"""
Metrics and Evaluation Utilities for Customer Segmentation

This module provides comprehensive metrics for evaluating clustering
performance and customer segmentation quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import logging

logger = logging.getLogger(__name__)


class ClusteringMetrics:
    """
    Comprehensive clustering evaluation metrics.
    
    Provides various metrics to evaluate clustering quality including:
    - Internal validation metrics
    - External validation metrics (when ground truth available)
    - Business-specific metrics
    - Visual validation metrics
    """
    
    def __init__(self):
        """Initialize clustering metrics evaluator."""
        self.metric_history = {}
        
    def evaluate_internal_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate internal clustering validation metrics.
        
        Args:
            data: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of internal metrics
        """
        logger.info("Calculating internal clustering metrics...")
        
        metrics = {}
        
        try:
            # Silhouette Score (-1 to 1, higher is better)
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(data):
                silhouette = silhouette_score(data, labels)
                metrics['silhouette_score'] = silhouette
            else:
                metrics['silhouette_score'] = 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
            metrics['silhouette_score'] = 0.0
        
        try:
            # Davies-Bouldin Index (lower is better)
            if len(np.unique(labels)) > 1:
                davies_bouldin = davies_bouldin_score(data, labels)
                metrics['davies_bouldin_index'] = davies_bouldin
            else:
                metrics['davies_bouldin_index'] = float('inf')
                
        except Exception as e:
            logger.warning(f"Could not calculate Davies-Bouldin index: {e}")
            metrics['davies_bouldin_index'] = float('inf')
        
        try:
            # Calinski-Harabasz Index (higher is better)
            if len(np.unique(labels)) > 1:
                calinski_harabasz = calinski_harabasz_score(data, labels)
                metrics['calinski_harabasz_index'] = calinski_harabasz
            else:
                metrics['calinski_harabasz_index'] = 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate Calinski-Harabasz index: {e}")
            metrics['calinski_harabasz_index'] = 0.0
        
        # Custom internal metrics
        metrics.update(self._calculate_custom_internal_metrics(data, labels))
        
        logger.info(f"Internal metrics calculated: {list(metrics.keys())}")
        return metrics
    
    def evaluate_external_metrics(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate external clustering validation metrics.
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted cluster labels
            
        Returns:
            Dictionary of external metrics
        """
        logger.info("Calculating external clustering metrics...")
        
        metrics = {}
        
        try:
            # Adjusted Rand Index (-1 to 1, higher is better)
            ari = adjusted_rand_score(true_labels, predicted_labels)
            metrics['adjusted_rand_index'] = ari
        except Exception as e:
            logger.warning(f"Could not calculate Adjusted Rand Index: {e}")
            metrics['adjusted_rand_index'] = 0.0
        
        try:
            # Normalized Mutual Information (0 to 1, higher is better)
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            metrics['normalized_mutual_info'] = nmi
        except Exception as e:
            logger.warning(f"Could not calculate Normalized Mutual Information: {e}")
            metrics['normalized_mutual_info'] = 0.0
        
        logger.info(f"External metrics calculated: {list(metrics.keys())}")
        return metrics
    
    def evaluate_business_metrics(self, data: pd.DataFrame, labels: np.ndarray, 
                                business_columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate business-specific metrics for customer segments.
        
        Args:
            data: Customer data with business metrics
            labels: Cluster labels
            business_columns: List of business metric columns to analyze
            
        Returns:
            Dictionary of business metrics
        """
        logger.info("Calculating business metrics...")
        
        if business_columns is None:
            business_columns = ['annual_income', 'spending_score', 'purchase_frequency', 
                              'customer_years', 'last_purchase_days']
        
        metrics = {}
        df = data.copy()
        df['cluster'] = labels
        
        # Segment size distribution
        segment_sizes = df['cluster'].value_counts()
        metrics['segment_sizes'] = segment_sizes.to_dict()
        metrics['segment_size_std'] = segment_sizes.std()
        metrics['segment_balance_score'] = self._calculate_segment_balance(segment_sizes)
        
        # Business metric analysis per segment
        for col in business_columns:
            if col in df.columns:
                segment_analysis = df.groupby('cluster')[col].agg(['mean', 'std', 'min', 'max'])
                metrics[f'{col}_by_segment'] = segment_analysis.to_dict()
                
                # Calculate separation between segments
                segment_means = segment_analysis['mean']
                if len(segment_means) > 1:
                    metrics[f'{col}_segment_separation'] = segment_means.std()
        
        # Overall business metrics
        metrics.update(self._calculate_overall_business_metrics(df, business_columns))
        
        logger.info(f"Business metrics calculated: {list(metrics.keys())}")
        return metrics
    
    def evaluate_stability_metrics(self, data: np.ndarray, labels_list: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate clustering stability across multiple runs.
        
        Args:
            data: Feature matrix
            labels_list: List of clustering results from multiple runs
            
        Returns:
            Dictionary of stability metrics
        """
        logger.info("Calculating clustering stability metrics...")
        
        if len(labels_list) < 2:
            logger.warning("Need at least 2 clustering results for stability analysis")
            return {}
        
        metrics = {}
        
        # Calculate pairwise stability
        stability_scores = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                # Adjusted Rand Index between runs
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                stability_scores.append(ari)
        
        metrics['mean_stability'] = np.mean(stability_scores)
        metrics['std_stability'] = np.std(stability_scores)
        metrics['min_stability'] = np.min(stability_scores)
        
        logger.info(f"Stability metrics calculated: {list(metrics.keys())}")
        return metrics
    
    def evaluate_all(self, data: np.ndarray, labels: np.ndarray, 
                    true_labels: np.ndarray = None, 
                    business_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Evaluate all available metrics.
        
        Args:
            data: Feature matrix
            labels: Predicted cluster labels
            true_labels: Ground truth labels (optional)
            business_data: Business metrics data (optional)
            
        Returns:
            Dictionary of all metrics
        """
        logger.info("Calculating comprehensive evaluation metrics...")
        
        all_metrics = {}
        
        # Internal metrics
        all_metrics['internal'] = self.evaluate_internal_metrics(data, labels)
        
        # External metrics (if ground truth available)
        if true_labels is not None:
            all_metrics['external'] = self.evaluate_external_metrics(true_labels, labels)
        
        # Business metrics (if business data available)
        if business_data is not None:
            all_metrics['business'] = self.evaluate_business_metrics(business_data, labels)
        
        # Overall quality score
        all_metrics['overall_quality'] = self._calculate_overall_quality(all_metrics)
        
        # Store in history
        self.metric_history[len(self.metric_history)] = all_metrics
        
        logger.info("Comprehensive evaluation complete")
        return all_metrics
    
    def _calculate_custom_internal_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate custom internal validation metrics."""
        metrics = {}
        
        try:
            # Inertia (within-cluster sum of squares)
            unique_labels = np.unique(labels)
            inertia = 0.0
            
            for label in unique_labels:
                cluster_points = data[labels == label]
                if len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0)
                    inertia += np.sum((cluster_points - centroid) ** 2)
            
            metrics['inertia'] = inertia
            
            # Average within-cluster distance
            avg_within_distance = 0.0
            total_points = 0
            
            for label in unique_labels:
                cluster_points = data[labels == label]
                if len(cluster_points) > 1:
                    distances = np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1)
                    avg_within_distance += np.sum(distances)
                    total_points += len(cluster_points)
            
            if total_points > 0:
                metrics['avg_within_cluster_distance'] = avg_within_distance / total_points
            else:
                metrics['avg_within_cluster_distance'] = 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate custom internal metrics: {e}")
        
        return metrics
    
    def _calculate_segment_balance(self, segment_sizes: pd.Series) -> float:
        """Calculate segment balance score (0 to 1, higher is better)."""
        if len(segment_sizes) == 0:
            return 0.0
        
        # Ideal balance is equal sizes
        ideal_size = len(segment_sizes) / len(segment_sizes)
        
        # Calculate coefficient of variation (lower is better)
        cv = segment_sizes.std() / segment_sizes.mean() if segment_sizes.mean() > 0 else float('inf')
        
        # Convert to balance score (higher is better)
        balance_score = 1.0 / (1.0 + cv)
        
        return balance_score
    
    def _calculate_overall_business_metrics(self, df: pd.DataFrame, 
                                          business_columns: List[str]) -> Dict[str, Any]:
        """Calculate overall business metrics."""
        metrics = {}
        
        # Overall customer value distribution
        if 'annual_income' in df.columns:
            metrics['total_annual_income'] = df['annual_income'].sum()
            metrics['avg_annual_income'] = df['annual_income'].mean()
        
        if 'spending_score' in df.columns:
            metrics['avg_spending_score'] = df['spending_score'].mean()
        
        # Customer engagement
        if 'purchase_frequency' in df.columns:
            metrics['avg_purchase_frequency'] = df['purchase_frequency'].mean()
        
        if 'last_purchase_days' in df.columns:
            metrics['avg_days_since_last_purchase'] = df['last_purchase_days'].mean()
            # Percentage of recent customers (purchased in last 30 days)
            recent_customers = (df['last_purchase_days'] <= 30).sum() / len(df)
            metrics['recent_customer_percentage'] = recent_customers
        
        return metrics
    
    def _calculate_overall_quality(self, all_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality scores."""
        quality_scores = {}
        
        if 'internal' in all_metrics:
            internal = all_metrics['internal']
            
            # Silhouette-based quality (0 to 1)
            silhouette_score = internal.get('silhouette_score', 0)
            quality_scores['silhouette_quality'] = max(0, silhouette_score)
            
            # Davies-Bouldin based quality (0 to 1, inverted)
            db_index = internal.get('davies_bouldin_index', float('inf'))
            quality_scores['db_quality'] = max(0, 1.0 / (1.0 + db_index))
            
            # Combined internal quality
            quality_scores['internal_quality'] = (quality_scores['silhouette_quality'] + 
                                                quality_scores['db_quality']) / 2
        
        if 'business' in all_metrics:
            business = all_metrics['business']
            
            # Segment balance quality (0 to 1)
            balance_score = business.get('segment_balance_score', 0)
            quality_scores['balance_quality'] = balance_score
        
        # Overall quality (weighted average)
        if quality_scores:
            weights = {'silhouette_quality': 0.4, 'db_quality': 0.3, 'balance_quality': 0.3}
            overall_quality = sum(quality_scores.get(k, 0) * w for k, w in weights.items())
            quality_scores['overall'] = overall_quality
        
        return quality_scores
    
    def compare_models(self, metrics_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models based on their metrics.
        
        Args:
            metrics_dict: Dictionary of model names to their metrics
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, metrics in metrics_dict.items():
            row = {'model': model_name}
            
            # Add internal metrics
            if 'internal' in metrics:
                internal = metrics['internal']
                row.update({f'internal_{k}': v for k, v in internal.items()})
            
            # Add business metrics
            if 'business' in metrics:
                business = metrics['business']
                row.update({f'business_{k}': v for k, v in business.items() if isinstance(v, (int, float))})
            
            # Add overall quality
            if 'overall_quality' in metrics:
                overall = metrics['overall_quality']
                row.update(overall)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by overall quality if available
        if 'overall' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('overall', ascending=False)
        
        return comparison_df
    
    def get_metric_history(self) -> Dict[int, Dict[str, Any]]:
        """Get history of all calculated metrics."""
        return self.metric_history
    
    def clear_history(self):
        """Clear metrics history."""
        self.metric_history = {}
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a human-readable metrics report.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("# Clustering Evaluation Report\n")
        
        # Internal metrics
        if 'internal' in metrics:
            report.append("## Internal Validation Metrics")
            internal = metrics['internal']
            
            if 'silhouette_score' in internal:
                silhouette = internal['silhouette_score']
                report.append(f"- **Silhouette Score**: {silhouette:.3f}")
                if silhouette > 0.7:
                    report.append("  - *Excellent* clustering structure")
                elif silhouette > 0.5:
                    report.append("  - *Good* clustering structure")
                elif silhouette > 0.25:
                    report.append("  - *Reasonable* clustering structure")
                else:
                    report.append("  - *Poor* clustering structure")
            
            if 'davies_bouldin_index' in internal:
                db = internal['davies_bouldin_index']
                report.append(f"- **Davies-Bouldin Index**: {db:.3f}")
                if db < 0.5:
                    report.append("  - *Excellent* cluster separation")
                elif db < 1.0:
                    report.append("  - *Good* cluster separation")
                else:
                    report.append("  - *Could be improved* cluster separation")
            
            report.append("")
        
        # Business metrics
        if 'business' in metrics:
            report.append("## Business Metrics")
            business = metrics['business']
            
            if 'segment_balance_score' in business:
                balance = business['segment_balance_score']
                report.append(f"- **Segment Balance**: {balance:.3f}")
                if balance > 0.8:
                    report.append("  - *Well-balanced* segments")
                elif balance > 0.6:
                    report.append("  - *Reasonably balanced* segments")
                else:
                    report.append("  - *Imbalanced* segments")
            
            report.append("")
        
        # Overall quality
        if 'overall_quality' in metrics:
            report.append("## Overall Quality")
            overall = metrics['overall_quality']
            
            if 'overall' in overall:
                score = overall['overall']
                report.append(f"- **Overall Quality Score**: {score:.3f}")
                if score > 0.8:
                    report.append("  - *Excellent* overall quality")
                elif score > 0.6:
                    report.append("  - *Good* overall quality")
                elif score > 0.4:
                    report.append("  - *Acceptable* overall quality")
                else:
                    report.append("  - *Needs improvement*")
            
            report.append("")
        
        return "\n".join(report)
    
    def calculate_all_metrics(self, data: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> Dict[str, float]:
        """
        Calculate all clustering metrics.
        
        Args:
            data: Feature DataFrame
            labels: Cluster labels
            n_clusters: Number of clusters
            
        Returns:
            Dictionary of all metrics
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data
        
        # Calculate internal metrics
        internal_metrics = self.evaluate_internal_metrics(X, labels)
        
        # Calculate business metrics with default business columns
        business_columns = ['annual_income', 'spending_score', 'purchase_frequency', 
                           'customer_years', 'last_purchase_days']
        
        # Filter to only include columns that exist in the data
        available_business_columns = [col for col in business_columns if col in data.columns]
        
        if available_business_columns:
            business_metrics = self.evaluate_business_metrics(data, labels, available_business_columns)
        else:
            business_metrics = {}
        
        # Combine all metrics
        all_metrics = {}
        all_metrics.update(internal_metrics)
        all_metrics.update(business_metrics)
        
        return all_metrics
