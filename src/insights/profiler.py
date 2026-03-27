"""Cluster profiling module for detailed customer segment analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config.logging_config import get_logger
from utils.helpers import safe_divide, create_data_profile

logger = get_logger(__name__)


class ClusterProfiler:
    """Detailed profiler for customer clusters."""
    
    def __init__(self):
        """Initialize cluster profiler."""
        self.cluster_profiles = {}
        self.comparison_metrics = {}
        self.statistical_tests = {}
        
    def profile_clusters(self, 
                        X: Union[pd.DataFrame, np.ndarray],
                        cluster_labels: np.ndarray,
                        feature_names: List[str],
                        cluster_centers: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Create detailed profiles for all clusters.
        
        Args:
            X: Original feature data
            cluster_labels: Cluster assignments
            feature_names: Names of features
            cluster_centers: Cluster centers (if available)
            
        Returns:
            Dictionary with detailed cluster profiles
        """
        logger.info("Creating detailed cluster profiles")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
            
        df['cluster'] = cluster_labels
        
        # Create profiles for each cluster
        profiles = {}
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            
            profile = self._create_single_cluster_profile(
                cluster_data, df, cluster_id, feature_names
            )
            
            profiles[f'cluster_{cluster_id}'] = profile
            
        self.cluster_profiles = profiles
        
        # Create comparison metrics
        self.comparison_metrics = self._create_comparison_metrics(df, feature_names)
        
        # Perform statistical tests
        self.statistical_tests = self._perform_statistical_tests(df, feature_names)
        
        results = {
            'cluster_profiles': profiles,
            'comparison_metrics': self.comparison_metrics,
            'statistical_tests': self.statistical_tests,
            'summary': self._create_profiling_summary(profiles)
        }
        
        logger.info(f"Created profiles for {len(profiles)} clusters")
        
        return results
        
    def _create_single_cluster_profile(self, 
                                    cluster_data: pd.DataFrame,
                                    overall_data: pd.DataFrame,
                                    cluster_id: int,
                                    feature_names: List[str]) -> Dict[str, Any]:
        """Create detailed profile for a single cluster."""
        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(overall_data)) * 100,
            'features': {}
        }
        
        # Analyze each feature
        for feature in feature_names:
            if feature not in cluster_data.columns:
                continue
                
            feature_profile = self._analyze_feature(
                cluster_data[feature], 
                overall_data[feature], 
                feature
            )
            
            profile['features'][feature] = feature_profile
            
        # Add behavioral patterns
        profile['behavioral_patterns'] = self._identify_behavioral_patterns(
            cluster_data, feature_names
        )
        
        # Add demographic insights
        profile['demographic_insights'] = self._extract_demographic_insights(
            cluster_data, feature_names
        )
        
        # Add purchase behavior insights
        profile['purchase_insights'] = self._extract_purchase_insights(
            cluster_data, feature_names
        )
        
        return profile
        
    def _analyze_feature(self, 
                        cluster_feature: pd.Series,
                        overall_feature: pd.Series,
                        feature_name: str) -> Dict[str, Any]:
        """Analyze a single feature for a cluster."""
        cluster_stats = cluster_feature.describe()
        overall_stats = overall_feature.describe()
        
        analysis = {
            'cluster_stats': cluster_stats.to_dict(),
            'overall_stats': overall_stats.to_dict(),
            'difference_analysis': {}
        }
        
        # Calculate differences
        cluster_mean = cluster_stats['mean']
        overall_mean = overall_stats['mean']
        overall_std = overall_stats['std']
        
        # Z-score difference
        z_diff = (cluster_mean - overall_mean) / overall_std if overall_std > 0 else 0
        analysis['difference_analysis']['z_score_difference'] = z_diff
        
        # Percentage difference
        pct_diff = safe_divide(cluster_mean - overall_mean, overall_mean) * 100
        analysis['difference_analysis']['percentage_difference'] = pct_diff
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(cluster_feature) - 1) * cluster_feature.var() + 
                             (len(overall_feature) - 1) * overall_feature.var()) / 
                            (len(cluster_feature) + len(overall_feature) - 2))
        
        cohens_d = (cluster_mean - overall_mean) / pooled_std if pooled_std > 0 else 0
        analysis['difference_analysis']['cohens_d'] = cohens_d
        
        # Classification
        if abs(z_diff) > 2:
            significance = "Very High"
        elif abs(z_diff) > 1:
            significance = "High"
        elif abs(z_diff) > 0.5:
            significance = "Medium"
        else:
            significance = "Low"
            
        analysis['difference_analysis']['significance'] = significance
        analysis['difference_analysis']['direction'] = "Higher" if z_diff > 0 else "Lower"
        
        # Distribution analysis
        analysis['distribution_analysis'] = self._analyze_distribution(
            cluster_feature, overall_feature
        )
        
        return analysis
        
    def _analyze_distribution(self, cluster_feature: pd.Series, 
                            overall_feature: pd.Series) -> Dict[str, Any]:
        """Analyze distribution characteristics."""
        analysis = {}
        
        try:
            # Normality test
            _, cluster_normal_p = stats.shapiro(cluster_feature.sample(min(5000, len(cluster_feature))))
            _, overall_normal_p = stats.shapiro(overall_feature.sample(min(5000, len(overall_feature))))
            
            analysis['cluster_normality_p'] = cluster_normal_p
            analysis['overall_normality_p'] = overall_normal_p
            analysis['cluster_is_normal'] = cluster_normal_p > 0.05
            analysis['overall_is_normal'] = overall_normal_p > 0.05
            
        except Exception:
            analysis['cluster_normality_p'] = None
            analysis['overall_normality_p'] = None
            analysis['cluster_is_normal'] = False
            analysis['overall_is_normal'] = False
            
        # Distribution shape
        analysis['cluster_skewness'] = stats.skew(cluster_feature)
        analysis['cluster_kurtosis'] = stats.kurtosis(cluster_feature)
        analysis['overall_skewness'] = stats.skew(overall_feature)
        analysis['overall_kurtosis'] = stats.kurtosis(overall_feature)
        
        # Percentile analysis
        percentiles = [10, 25, 50, 75, 90]
        analysis['cluster_percentiles'] = {f'p{p}': np.percentile(cluster_feature, p) for p in percentiles}
        analysis['overall_percentiles'] = {f'p{p}': np.percentile(overall_feature, p) for p in percentiles}
        
        return analysis
        
    def _identify_behavioral_patterns(self, 
                                   cluster_data: pd.DataFrame,
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Identify behavioral patterns in the cluster."""
        patterns = {}
        
        # RFM Analysis
        rfm_features = ['recency_score', 'frequency_score', 'monetary_score']
        available_rfm = [f for f in rfm_features if f in cluster_data.columns]
        
        if available_rfm:
            patterns['rfm_profile'] = {}
            for feature in available_rfm:
                patterns['rfm_profile'][feature] = {
                    'mean': cluster_data[feature].mean(),
                    'classification': self._classify_rfm_score(cluster_data[feature].mean())
                }
                
        # Loyalty Analysis
        loyalty_features = ['customer_since_years', 'purchase_frequency', 'engagement_score']
        available_loyalty = [f for f in loyalty_features if f in cluster_data.columns]
        
        if available_loyalty:
            patterns['loyalty_profile'] = {}
            for feature in available_loyalty:
                patterns['loyalty_profile'][feature] = {
                    'mean': cluster_data[feature].mean(),
                    'level': self._classify_loyalty_level(cluster_data[feature].mean(), feature)
                }
                
        # Risk Analysis
        risk_features = ['churn_risk_score', 'last_purchase_days_ago', 'dormancy_score']
        available_risk = [f for f in risk_features if f in cluster_data.columns]
        
        if available_risk:
            patterns['risk_profile'] = {}
            for feature in available_risk:
                patterns['risk_profile'][feature] = {
                    'mean': cluster_data[feature].mean(),
                    'level': self._classify_risk_level(cluster_data[feature].mean(), feature)
                }
                
        return patterns
        
    def _classify_rfm_score(self, score: float) -> str:
        """Classify RFM score."""
        if score >= 4.5:
            return "Excellent"
        elif score >= 3.5:
            return "Good"
        elif score >= 2.5:
            return "Average"
        elif score >= 1.5:
            return "Poor"
        else:
            return "Very Poor"
            
    def _classify_loyalty_level(self, value: float, feature: str) -> str:
        """Classify loyalty level."""
        if feature == 'customer_since_years':
            if value >= 5:
                return "Very Loyal"
            elif value >= 3:
                return "Loyal"
            elif value >= 1:
                return "Moderately Loyal"
            else:
                return "New Customer"
        elif feature in ['purchase_frequency', 'engagement_score']:
            if value >= 0.7:
                return "Highly Engaged"
            elif value >= 0.5:
                return "Moderately Engaged"
            elif value >= 0.3:
                return "Low Engagement"
            else:
                return "Very Low Engagement"
        else:
            return "Unknown"
            
    def _classify_risk_level(self, value: float, feature: str) -> str:
        """Classify risk level."""
        if feature == 'churn_risk_score':
            if value >= 2:
                return "High Risk"
            elif value >= 1:
                return "Medium Risk"
            else:
                return "Low Risk"
        elif feature == 'last_purchase_days_ago':
            if value >= 90:
                return "High Risk"
            elif value >= 60:
                return "Medium Risk"
            elif value >= 30:
                return "Low Risk"
            else:
                return "Very Low Risk"
        elif feature == 'dormancy_score':
            if value >= 0.7:
                return "High Risk"
            elif value >= 0.4:
                return "Medium Risk"
            else:
                return "Low Risk"
        else:
            return "Unknown"
            
    def _extract_demographic_insights(self, 
                                    cluster_data: pd.DataFrame,
                                    feature_names: List[str]) -> Dict[str, Any]:
        """Extract demographic insights."""
        insights = {}
        
        # Age analysis
        if 'age' in cluster_data.columns:
            age_stats = cluster_data['age'].describe()
            insights['age'] = {
                'mean_age': age_stats['mean'],
                'age_range': f"{age_stats['min']:.0f}-{age_stats['max']:.0f}",
                'age_group': self._classify_age_group(age_stats['mean'])
            }
            
        # Income analysis
        if 'annual_income' in cluster_data.columns:
            income_stats = cluster_data['annual_income'].describe()
            insights['income'] = {
                'mean_income': income_stats['mean'],
                'income_range': f"${income_stats['min']:.0f}-${income_stats['max']:.0f}",
                'income_level': self._classify_income_level(income_stats['mean'])
            }
            
        return insights
        
    def _classify_age_group(self, age: float) -> str:
        """Classify age group."""
        if age < 25:
            return "Gen Z"
        elif age < 35:
            return "Millennial"
        elif age < 50:
            return "Gen X"
        elif age < 65:
            return "Boomer"
        else:
            return "Senior"
            
    def _classify_income_level(self, income: float) -> str:
        """Classify income level."""
        if income >= 100000:
            return "High Income"
        elif income >= 70000:
            return "Upper Middle Income"
        elif income >= 50000:
            return "Middle Income"
        elif income >= 35000:
            return "Lower Middle Income"
        else:
            return "Low Income"
            
    def _extract_purchase_insights(self, 
                                 cluster_data: pd.DataFrame,
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Extract purchase behavior insights."""
        insights = {}
        
        # Purchase frequency analysis
        if 'purchase_frequency' in cluster_data.columns:
            freq_stats = cluster_data['purchase_frequency'].describe()
            insights['purchase_frequency'] = {
                'mean_frequency': freq_stats['mean'],
                'frequency_level': self._classify_frequency_level(freq_stats['mean'])
            }
            
        # Transaction value analysis
        if 'avg_transaction_value' in cluster_data.columns:
            value_stats = cluster_data['avg_transaction_value'].describe()
            insights['transaction_value'] = {
                'mean_value': value_stats['mean'],
                'value_level': self._classify_transaction_value_level(value_stats['mean'])
            }
            
        # Spending score analysis
        if 'spending_score' in cluster_data.columns:
            spending_stats = cluster_data['spending_score'].describe()
            insights['spending_behavior'] = {
                'mean_spending_score': spending_stats['mean'],
                'spending_type': self._classify_spending_type(spending_stats['mean'])
            }
            
        return insights
        
    def _classify_frequency_level(self, frequency: float) -> str:
        """Classify purchase frequency level."""
        if frequency >= 10:
            return "Very Frequent"
        elif frequency >= 6:
            return "Frequent"
        elif frequency >= 3:
            return "Moderate"
        elif frequency >= 1:
            return "Occasional"
        else:
            return "Rare"
            
    def _classify_transaction_value_level(self, value: float) -> str:
        """Classify transaction value level."""
        if value >= 200:
            return "High Value"
        elif value >= 100:
            return "Medium-High Value"
        elif value >= 50:
            return "Medium Value"
        elif value >= 25:
            return "Low-Medium Value"
        else:
            return "Low Value"
            
    def _classify_spending_type(self, score: float) -> str:
        """Classify spending type."""
        if score >= 80:
            return "High Spender"
        elif score >= 60:
            return "Moderate-High Spender"
        elif score >= 40:
            return "Moderate Spender"
        elif score >= 20:
            return "Low-Moderate Spender"
        else:
            return "Low Spender"
            
    def _create_comparison_metrics(self, 
                                 df: pd.DataFrame,
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Create comparison metrics between clusters."""
        comparison = {}
        
        # Cluster sizes
        cluster_sizes = df['cluster'].value_counts().sort_index()
        comparison['cluster_sizes'] = cluster_sizes.to_dict()
        comparison['size_percentages'] = (cluster_sizes / len(df) * 100).to_dict()
        
        # Feature comparisons
        comparison['feature_comparisons'] = {}
        
        for feature in feature_names:
            if feature not in df.columns:
                continue
                
            feature_comparison = {}
            
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_data = df[df['cluster'] == cluster_id]
                feature_comparison[f'cluster_{cluster_id}'] = {
                    'mean': cluster_data[feature].mean(),
                    'std': cluster_data[feature].std(),
                    'median': cluster_data[feature].median()
                }
                
            comparison['feature_comparisons'][feature] = feature_comparison
            
        return comparison
        
    def _perform_statistical_tests(self, 
                                 df: pd.DataFrame,
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Perform statistical tests between clusters."""
        tests = {}
        
        for feature in feature_names:
            if feature not in df.columns:
                continue
                
            feature_tests = {}
            
            # ANOVA test for differences between clusters
            clusters = [df[df['cluster'] == cluster_id][feature].values 
                       for cluster_id in sorted(df['cluster'].unique())]
            
            try:
                f_stat, p_value = stats.f_oneway(*clusters)
                feature_tests['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant_difference': p_value < 0.05
                }
            except Exception as e:
                feature_tests['anova'] = {'error': str(e)}
                
            # Kruskal-Wallis test (non-parametric alternative)
            try:
                h_stat, p_value = stats.kruskal(*clusters)
                feature_tests['kruskal_wallis'] = {
                    'h_statistic': h_stat,
                    'p_value': p_value,
                    'significant_difference': p_value < 0.05
                }
            except Exception as e:
                feature_tests['kruskal_wallis'] = {'error': str(e)}
                
            tests[feature] = feature_tests
            
        return tests
        
    def _create_profiling_summary(self, profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of profiling results."""
        summary = {
            'total_clusters': len(profiles),
            'cluster_sizes': {},
            'key_differences': [],
            'statistical_significance': {}
        }
        
        # Cluster sizes
        for cluster_id, profile in profiles.items():
            summary['cluster_sizes'][cluster_id] = profile['size']
            
        # Key differences (features with highest effect sizes)
        all_effects = []
        
        for cluster_id, profile in profiles.items():
            for feature, feature_data in profile['features'].items():
                effect_size = abs(feature_data['difference_analysis']['cohens_d'])
                if effect_size > 0.5:  # Medium effect or higher
                    all_effects.append({
                        'cluster': cluster_id,
                        'feature': feature,
                        'effect_size': effect_size,
                        'direction': feature_data['difference_analysis']['direction'],
                        'significance': feature_data['difference_analysis']['significance']
                    })
                    
        # Sort by effect size
        all_effects.sort(key=lambda x: x['effect_size'], reverse=True)
        summary['key_differences'] = all_effects[:20]  # Top 20 differences
        
        # Statistical significance summary
        significant_features = 0
        total_features = 0
        
        for feature, tests in self.statistical_tests.items():
            total_features += 1
            if 'anova' in tests and tests['anova'].get('significant_difference', False):
                significant_features += 1
                
        summary['statistical_significance'] = {
            'significant_features': significant_features,
            'total_features': total_features,
            'significance_ratio': significant_features / total_features if total_features > 0 else 0
        }
        
        return summary
        
    def plot_cluster_profiles(self, 
                            feature_names: List[str],
                            top_n_features: int = 6,
                            figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot cluster profiles comparison.
        
        Args:
            feature_names: List of feature names
            top_n_features: Number of top features to plot
            figsize: Figure size
        """
        if not self.comparison_metrics:
            logger.warning("No comparison metrics available. Run profile_clusters first.")
            return
            
        # Select features with highest variance across clusters
        feature_variances = {}
        
        for feature in feature_names:
            if feature in self.comparison_metrics['feature_comparisons']:
                means = [data['mean'] for data in self.comparison_metrics['feature_comparisons'][feature].values()]
                feature_variances[feature] = np.var(means)
                
        top_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        cluster_ids = sorted(self.cluster_profiles.keys())
        
        for i, (feature, _) in enumerate(top_features):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Extract data for plotting
            clusters = []
            means = []
            stds = []
            
            for cluster_id in cluster_ids:
                if feature in self.comparison_metrics['feature_comparisons'][cluster_id]:
                    clusters.append(cluster_id.replace('cluster_', 'Cluster '))
                    means.append(self.comparison_metrics['feature_comparisons'][cluster_id][feature]['mean'])
                    stds.append(self.comparison_metrics['feature_comparisons'][cluster_id][feature]['std'])
                    
            # Create bar plot
            bars = ax.bar(clusters, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f'{feature.replace("_", " ").title()}')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(stds)*0.1,
                       f'{mean:.2f}', ha='center', va='bottom')
                       
        # Hide unused subplots
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def create_interactive_profile_dashboard(self, 
                                           feature_names: List[str]) -> go.Figure:
        """
        Create interactive dashboard for cluster profiles.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Plotly figure
        """
        if not self.cluster_profiles:
            logger.warning("No cluster profiles available. Run profile_clusters first.")
            return go.Figure()
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Sizes', 'Feature Comparison', 
                          'Statistical Significance', 'Effect Sizes'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        cluster_ids = sorted(self.cluster_profiles.keys())
        cluster_labels = [f'Cluster {cid.split("_")[1]}' for cid in cluster_ids]
        
        # 1. Cluster sizes pie chart
        sizes = [self.cluster_profiles[cid]['size'] for cid in cluster_ids]
        fig.add_trace(
            go.Pie(labels=cluster_labels, values=sizes, name="Cluster Sizes"),
            row=1, col=1
        )
        
        # 2. Feature comparison (top 5 features)
        if self.comparison_metrics.get('feature_comparisons'):
            # Select top 5 features with highest variance
            feature_variances = {}
            for feature in feature_names:
                if feature in self.comparison_metrics['feature_comparisons']:
                    means = [data['mean'] for data in self.comparison_metrics['feature_comparisons'][feature].values()]
                    feature_variances[feature] = np.var(means)
                    
            top_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (feature, _) in enumerate(top_features):
                means = []
                for cid in cluster_ids:
                    if feature in self.comparison_metrics['feature_comparisons'][cid]:
                        means.append(self.comparison_metrics['feature_comparisons'][cid][feature]['mean'])
                    else:
                        means.append(0)
                        
                fig.add_trace(
                    go.Bar(x=cluster_labels, y=means, name=feature.replace('_', ' ').title()),
                    row=1, col=2
                )
                
        # 3. Statistical significance
        if self.statistical_tests:
            significant_features = []
            p_values = []
            
            for feature, tests in self.statistical_tests.items():
                if 'anova' in tests and 'p_value' in tests['anova']:
                    significant_features.append(feature.replace('_', ' ').title())
                    p_values.append(tests['anova']['p_value'])
                    
            if significant_features:
                fig.add_trace(
                    go.Bar(x=significant_features, y=p_values, name="P-Values"),
                    row=2, col=1
                )
                
        # 4. Effect sizes
        effect_sizes = []
        features = []
        
        for profile in self.cluster_profiles.values():
            for feature, data in profile['features'].items():
                effect_size = abs(data['difference_analysis']['cohens_d'])
                if effect_size > 0.3:  # Small effect or higher
                    effect_sizes.append(effect_size)
                    features.append(f"{feature.replace('_', ' ').title()}\\n(Cluster {profile['cluster_id']})")
                    
        if effect_sizes:
            fig.add_trace(
                go.Scatter(x=features, y=effect_sizes, mode='markers', 
                          name="Effect Sizes", marker=dict(size=10)),
                row=2, col=2
            )
            
        fig.update_layout(
            title="Cluster Profiles Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
        
    def export_profiles(self, output_path: str) -> None:
        """
        Export cluster profiles to JSON file.
        
        Args:
            output_path: Path to save the profiles
        """
        from src.utils.helpers import save_json
        
        export_data = {
            'cluster_profiles': self.cluster_profiles,
            'comparison_metrics': self.comparison_metrics,
            'statistical_tests': self.statistical_tests,
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
        
        save_json(export_data, output_path)
        logger.info(f"Cluster profiles exported to {output_path}")


# Alias for backward compatibility
SegmentProfiler = ClusterProfiler
