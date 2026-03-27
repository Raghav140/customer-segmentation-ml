"""
Auto Cluster Selection System

Intelligent automatic determination of optimal cluster count using multiple methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AutoClusterSelector:
    """
    Intelligent automatic cluster selection system.
    
    Uses multiple methods to determine optimal cluster count:
    - Elbow Method (K-Means)
    - Silhouette Analysis
    - Gap Statistic
    - Hierarchical Analysis
    - Business Heuristics
    """
    
    def __init__(self, max_clusters: int = 10, min_clusters: int = 2):
        """
        Initialize auto cluster selector.
        
        Args:
            max_clusters: Maximum number of clusters to consider
            min_clusters: Minimum number of clusters to consider
        """
        self.max_clusters = max_clusters
        self.min_clusters = min_clusters
        self.selection_history = []
        
    def find_optimal_clusters(self, data: pd.DataFrame, 
                            methods: List[str] = None,
                            business_constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Find optimal number of clusters using multiple methods.
        
        Args:
            data: Feature DataFrame
            methods: List of methods to use ['elbow', 'silhouette', 'gap', 'hierarchical', 'business']
            business_constraints: Business-specific constraints
            
        Returns:
            Dictionary with optimal cluster recommendation and analysis
        """
        if methods is None:
            methods = ['elbow', 'silhouette', 'gap', 'hierarchical', 'business']
        
        logger.info(f"Starting auto cluster selection with methods: {methods}")
        
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data
        
        results = {
            'data_shape': X.shape,
            'methods_used': methods,
            'analysis_timestamp': datetime.now().isoformat(),
            'method_results': {},
            'optimal_clusters': None,
            'confidence_score': 0.0,
            'recommendation': {}
        }
        
        # Run each method
        for method in methods:
            try:
                if method == 'elbow':
                    method_result = self._elbow_method(X)
                elif method == 'silhouette':
                    method_result = self._silhouette_method(X)
                elif method == 'gap':
                    method_result = self._gap_statistic(X)
                elif method == 'hierarchical':
                    method_result = self._hierarchical_method(X)
                elif method == 'business':
                    method_result = self._business_heuristics(data, business_constraints)
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                results['method_results'][method] = method_result
                logger.info(f"Method {method}: optimal_k={method_result.get('optimal_k')}")
                
            except Exception as e:
                logger.error(f"Method {method} failed: {str(e)}")
                results['method_results'][method] = {'error': str(e)}
        
        # Aggregate results
        optimal_k, confidence, recommendation = self._aggregate_results(results['method_results'])
        
        results['optimal_clusters'] = optimal_k
        results['confidence_score'] = confidence
        results['recommendation'] = recommendation
        
        # Add to history
        self.selection_history.append(results)
        
        logger.info(f"Auto cluster selection completed: optimal_k={optimal_k}, confidence={confidence:.3f}")
        
        return results
    
    def _elbow_method(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Elbow method using K-Means inertia.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with elbow method results
        """
        logger.info("Running elbow method...")
        
        inertias = []
        k_range = range(self.min_clusters, self.max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point using second derivative
        inertias = np.array(inertias)
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        # Elbow is where second derivative is maximum
        elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
        optimal_k = k_range[elbow_idx] if elbow_idx < len(k_range) else self.min_clusters + 1
        
        return {
            'optimal_k': optimal_k,
            'inertias': inertias.tolist(),
            'k_range': list(k_range),
            'elbow_point': optimal_k,
            'method_confidence': self._calculate_elbow_confidence(inertias, optimal_k)
        }
    
    def _silhouette_method(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Silhouette analysis method.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with silhouette method results
        """
        logger.info("Running silhouette analysis...")
        
        silhouette_scores = []
        k_range = range(self.min_clusters, min(self.max_clusters + 1, len(X)))
        
        for k in k_range:
            if k == 1:
                silhouette_scores.append(-1)
                continue
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Only calculate if we have valid clustering
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
            else:
                score = -1
            
            silhouette_scores.append(score)
        
        # Find k with maximum silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        return {
            'optimal_k': optimal_k,
            'silhouette_scores': silhouette_scores,
            'k_range': list(k_range),
            'max_silhouette_score': max(silhouette_scores),
            'method_confidence': max(silhouette_scores)  # Higher score = higher confidence
        }
    
    def _gap_statistic(self, X: np.ndarray, n_references: int = 10) -> Dict[str, Any]:
        """
        Gap statistic method.
        
        Args:
            X: Feature matrix
            n_references: Number of reference datasets to generate
            
        Returns:
            Dictionary with gap statistic results
        """
        logger.info("Running gap statistic...")
        
        def compute_gap(X, k, n_references):
            # Compute gap for k clusters
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wk = kmeans.inertia_
            
            # Generate reference datasets
            wks_ref = []
            for _ in range(n_references):
                # Generate uniform random data
                X_ref = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=X.shape)
                kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_ref.fit(X_ref)
                wks_ref.append(kmeans_ref.inertia_)
            
            # Gap statistic
            gap = np.mean(np.log(wks_ref)) - np.log(wk)
            
            return gap, wk, wks_ref
        
        gap_values = []
        k_range = range(self.min_clusters, self.max_clusters + 1)
        
        for k in k_range:
            gap, wk, wks_ref = compute_gap(X, k, n_references)
            gap_values.append(gap)
        
        # Find optimal k (first k where gap(k) <= gap(k+1) - s(k+1))
        # Simplified version: find maximum gap
        optimal_k = k_range[np.argmax(gap_values)]
        
        return {
            'optimal_k': optimal_k,
            'gap_values': gap_values,
            'k_range': list(k_range),
            'max_gap': max(gap_values),
            'method_confidence': max(gap_values) / max(gap_values)  # Normalized confidence
        }
    
    def _hierarchical_method(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Hierarchical clustering method using dendrogram analysis.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with hierarchical method results
        """
        logger.info("Running hierarchical analysis...")
        
        # Compute linkage matrix
        linkage_matrix = linkage(X, method='ward')
        
        # Calculate cophenetic correlation
        from scipy.cluster.hierarchy import cophenet
        from scipy.spatial.distance import pdist
        
        cophenet_corr, _ = cophenet(linkage_matrix, pdist(X))
        
        # Find optimal number of clusters using inconsistency method
        # Simplified: use the largest jump in distance
        distances = linkage_matrix[:, 2]
        diffs = np.diff(distances)
        
        # Find significant jumps
        threshold = np.mean(diffs) + 2 * np.std(diffs)
        significant_jumps = np.where(diffs > threshold)[0]
        
        if len(significant_jumps) > 0:
            optimal_k = len(linkage_matrix) - significant_jumps[0]
        else:
            optimal_k = self.min_clusters + 1
        
        # Ensure within bounds
        optimal_k = max(self.min_clusters, min(optimal_k, self.max_clusters))
        
        return {
            'optimal_k': optimal_k,
            'cophenet_correlation': cophenet_corr,
            'linkage_distances': distances.tolist(),
            'significant_jumps': significant_jumps.tolist(),
            'method_confidence': cophenet_corr  # Higher correlation = higher confidence
        }
    
    def _business_heuristics(self, data: pd.DataFrame, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Business heuristics for cluster selection.
        
        Args:
            data: Feature DataFrame
            constraints: Business-specific constraints
            
        Returns:
            Dictionary with business heuristics results
        """
        logger.info("Running business heuristics...")
        
        if constraints is None:
            constraints = {}
        
        # Default business constraints
        max_segments = constraints.get('max_segments', 8)
        min_segment_size = constraints.get('min_segment_size', 50)
        preferred_segments = constraints.get('preferred_segments', [3, 4, 5])
        
        n_samples = len(data)
        
        # Calculate reasonable range based on sample size
        max_by_sample_size = max(2, min(self.max_clusters, n_samples // min_segment_size))
        max_by_business = min(max_segments, max_by_sample_size)
        
        # Score each potential k
        k_scores = {}
        for k in range(self.min_clusters, min(self.max_clusters, max_by_business) + 1):
            score = 0.0
            
            # Prefer preferred segment counts
            if k in preferred_segments:
                score += 0.3
            
            # Penalize too many segments for sample size
            if n_samples // k < min_segment_size:
                score -= 0.5
            
            # Business rule: odd number of segments often better for marketing
            if k % 2 == 1:
                score += 0.1
            
            # Avoid too few segments (under-segmentation)
            if k <= 2:
                score -= 0.2
            
            k_scores[k] = score
        
        # Select k with highest score
        optimal_k = max(k_scores, key=k_scores.get)
        
        return {
            'optimal_k': optimal_k,
            'k_scores': k_scores,
            'constraints_used': constraints,
            'sample_size': n_samples,
            'method_confidence': max(k_scores.values()) - min(k_scores.values())
        }
    
    def _calculate_elbow_confidence(self, inertias: np.ndarray, elbow_k: int) -> float:
        """Calculate confidence score for elbow method."""
        # Simple confidence based on how sharp the elbow is
        if elbow_k <= 2 or elbow_k >= len(inertias) - 1:
            return 0.5
        
        # Calculate angle at elbow
        idx = elbow_k - self.min_clusters
        if idx > 0 and idx < len(inertias) - 1:
            v1 = np.array([idx - 1, inertias[idx - 1] - inertias[idx]])
            v2 = np.array([idx + 1, inertias[idx + 1] - inertias[idx]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Convert to confidence (sharper angle = higher confidence)
            confidence = 1.0 - (angle / np.pi)
            return max(0.3, min(1.0, confidence))
        
        return 0.5
    
    def _aggregate_results(self, method_results: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
        """
        Aggregate results from multiple methods to determine optimal k.
        
        Args:
            method_results: Results from each method
            
        Returns:
            Tuple of (optimal_k, confidence, recommendation)
        """
        # Collect optimal k values and confidences
        k_votes = {}
        total_confidence = 0.0
        method_count = 0
        
        for method, result in method_results.items():
            if 'error' in result:
                continue
                
            optimal_k = result.get('optimal_k')
            confidence = result.get('method_confidence', 0.5)
            
            if optimal_k is not None:
                k_votes[optimal_k] = k_votes.get(optimal_k, 0) + confidence
                total_confidence += confidence
                method_count += 1
        
        if not k_votes:
            return 3, 0.0, {'error': 'No valid method results'}
        
        # Find k with highest weighted vote
        optimal_k = max(k_votes, key=k_votes.get)
        
        # Calculate overall confidence
        if method_count > 0:
            confidence = k_votes[optimal_k] / total_confidence
        else:
            confidence = 0.0
        
        # Create recommendation
        recommendation = {
            'optimal_clusters': optimal_k,
            'confidence': confidence,
            'method_consensus': k_votes,
            'methods_used': method_count,
            'reasoning': self._generate_reasoning(method_results, optimal_k, k_votes)
        }
        
        return optimal_k, confidence, recommendation
    
    def _generate_reasoning(self, method_results: Dict[str, Any], optimal_k: int, k_votes: Dict[int, float]) -> str:
        """Generate human-readable reasoning for the recommendation."""
        reasoning_parts = []
        
        # Method agreement
        agreeing_methods = []
        for method, result in method_results.items():
            if 'error' not in result and result.get('optimal_k') == optimal_k:
                agreeing_methods.append(method)
        
        if agreeing_methods:
            reasoning_parts.append(f"Strong agreement from {', '.join(agreeing_methods)} methods")
        
        # Vote distribution
        total_votes = sum(k_votes.values())
        vote_percentage = (k_votes[optimal_k] / total_votes) * 100
        reasoning_parts.append(f"Selected by {vote_percentage:.1f}% of weighted votes")
        
        # Method-specific insights
        if 'silhouette' in method_results and 'error' not in method_results['silhouette']:
            sil_score = method_results['silhouette'].get('max_silhouette_score', 0)
            reasoning_parts.append(f"Maximum silhouette score: {sil_score:.3f}")
        
        if 'hierarchical' in method_results and 'error' not in method_results['hierarchical']:
            coph_corr = method_results['hierarchical'].get('cophenet_correlation', 0)
            reasoning_parts.append(f"Good hierarchical structure (cophenetic: {coph_corr:.3f})")
        
        return " | ".join(reasoning_parts)
    
    def get_selection_history(self) -> List[Dict[str, Any]]:
        """Get history of cluster selections."""
        return self.selection_history.copy()
    
    def plot_cluster_analysis(self, data: pd.DataFrame, results: Dict[str, Any], 
                           save_path: Optional[str] = None) -> str:
        """
        Create comprehensive cluster analysis plots.
        
        Args:
            data: Feature DataFrame
            results: Auto cluster selection results
            save_path: Optional path to save plots
            
        Returns:
            Path to saved plot file
        """
        try:
            X = data.values if isinstance(data, pd.DataFrame) else data
            
            # Create subplot grid
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Auto Cluster Selection Analysis', fontsize=16)
            
            # Elbow method plot
            if 'elbow' in results['method_results'] and 'error' not in results['method_results']['elbow']:
                elbow_result = results['method_results']['elbow']
                axes[0, 0].plot(elbow_result['k_range'], elbow_result['inertias'], 'bo-')
                axes[0, 0].axvline(x=elbow_result['optimal_k'], color='r', linestyle='--', 
                                 label=f'Optimal k={elbow_result["optimal_k"]}')
                axes[0, 0].set_xlabel('Number of clusters')
                axes[0, 0].set_ylabel('Inertia')
                axes[0, 0].set_title('Elbow Method')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Silhouette analysis plot
            if 'silhouette' in results['method_results'] and 'error' not in results['method_results']['silhouette']:
                sil_result = results['method_results']['silhouette']
                axes[0, 1].plot(sil_result['k_range'], sil_result['silhouette_scores'], 'go-')
                axes[0, 1].axvline(x=sil_result['optimal_k'], color='r', linestyle='--',
                                 label=f'Optimal k={sil_result["optimal_k"]}')
                axes[0, 1].set_xlabel('Number of clusters')
                axes[0, 1].set_ylabel('Silhouette Score')
                axes[0, 1].set_title('Silhouette Analysis')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Gap statistic plot
            if 'gap' in results['method_results'] and 'error' not in results['method_results']['gap']:
                gap_result = results['method_results']['gap']
                axes[1, 0].plot(gap_result['k_range'], gap_result['gap_values'], 'mo-')
                axes[1, 0].axvline(x=gap_result['optimal_k'], color='r', linestyle='--',
                                 label=f'Optimal k={gap_result["optimal_k"]}')
                axes[1, 0].set_xlabel('Number of clusters')
                axes[1, 0].set_ylabel('Gap Statistic')
                axes[1, 0].set_title('Gap Statistic')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Method consensus plot
            method_votes = results['recommendation']['method_consensus']
            k_values = sorted(method_votes.keys())
            votes = [method_votes[k] for k in k_values]
            
            axes[1, 1].bar(k_values, votes, alpha=0.7)
            axes[1, 1].axvline(x=results['optimal_clusters'], color='r', linestyle='--',
                             label=f'Selected k={results["optimal_clusters"]}')
            axes[1, 1].set_xlabel('Number of clusters')
            axes[1, 1].set_ylabel('Weighted Votes')
            axes[1, 1].set_title('Method Consensus')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Cluster analysis plot saved to {save_path}")
            
            return save_path or "plot_displayed"
            
        except Exception as e:
            logger.error(f"Failed to create cluster analysis plot: {str(e)}")
            return "plot_failed"


# Global instance for easy access
auto_cluster_selector = AutoClusterSelector()
