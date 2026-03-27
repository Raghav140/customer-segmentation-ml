"""Feature engineering builder for customer segmentation system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

from src.config.logging_config import get_logger
from src.utils.helpers import safe_divide

logger = get_logger(__name__)


class FeatureBuilder:
    """Builds engineered features for customer segmentation."""
    
    def __init__(self):
        """Initialize feature builder."""
        self.feature_mappings = {}
        self.feature_importance = {}
        self.engineered_features = []
        
    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with RFM features
        """
        logger.info("Creating RFM features")
        
        df_rfm = df.copy()
        
        # Recency features (lower is better)
        if 'last_purchase_days_ago' in df.columns:
            df_rfm['recency_score'] = pd.qcut(df_rfm['last_purchase_days_ago'], 
                                            q=5, labels=[5, 4, 3, 2, 1])
            df_rfm['recency_normalized'] = 1 - (df_rfm['last_purchase_days_ago'] / 
                                               df_rfm['last_purchase_days_ago'].max())
            
        # Frequency features
        if 'purchase_frequency' in df.columns:
            df_rfm['frequency_score'] = pd.qcut(df_rfm['purchase_frequency'], 
                                              q=5, labels=[1, 2, 3, 4, 5])
            df_rfm['frequency_normalized'] = df_rfm['purchase_frequency'] / \
                                            df_rfm['purchase_frequency'].max()
            
        if 'total_purchases' in df.columns:
            df_rfm['total_purchases_normalized'] = df_rfm['total_purchases'] / \
                                                  df_rfm['total_purchases'].max()
            
        # Monetary features
        if 'annual_income' in df.columns:
            df_rfm['monetary_score'] = pd.qcut(df_rfm['annual_income'], 
                                             q=5, labels=[1, 2, 3, 4, 5])
            df_rfm['income_normalized'] = df_rfm['annual_income'] / \
                                         df_rfm['annual_income'].max()
            
        if 'avg_transaction_value' in df.columns:
            df_rfm['avg_transaction_normalized'] = df_rfm['avg_transaction_value'] / \
                                                  df_rfm['avg_transaction_value'].max()
            
        # Combined RFM score
        rfm_cols = []
        if 'recency_normalized' in df_rfm.columns:
            rfm_cols.append('recency_normalized')
        if 'frequency_normalized' in df_rfm.columns:
            rfm_cols.append('frequency_normalized')
        if 'income_normalized' in df_rfm.columns:
            rfm_cols.append('income_normalized')
            
        if len(rfm_cols) >= 2:
            df_rfm['rfm_combined_score'] = df_rfm[rfm_cols].mean(axis=1)
            
        self.engineered_features.extend([col for col in df_rfm.columns if col not in df.columns])
        
        logger.info(f"Created {len(self.engineered_features)} RFM features")
        return df_rfm
        
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features based on customer patterns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with behavioral features
        """
        logger.info("Creating behavioral features")
        
        df_behavioral = df.copy()
        
        # Customer loyalty metrics
        if 'customer_since_years' in df.columns and 'total_purchases' in df.columns:
            df_behavioral['loyalty_score'] = df_behavioral['total_purchases'] / \
                                           (df_behavioral['customer_since_years'] + 1)
            df_behavioral['purchase_velocity'] = df_behavioral['total_purchases'] / \
                                               df_behavioral['customer_since_years']
            
        # Spending patterns
        if 'annual_income' in df.columns and 'avg_transaction_value' in df.columns:
            df_behavioral['spending_ratio'] = safe_divide(
                df_behavioral['avg_transaction_value'] * 12,  # Annual estimated spending
                df_behavioral['annual_income']
            )
            df_behavioral['income_to_transaction_ratio'] = safe_divide(
                df_behavioral['annual_income'],
                df_behavioral['avg_transaction_value']
            )
            
        # Engagement metrics
        if 'purchase_frequency' in df.columns and 'customer_since_years' in df.columns:
            df_behavioral['engagement_score'] = df_behavioral['purchase_frequency'] * \
                                              df_behavioral['customer_since_years']
            
        # Value-based features
        if 'annual_income' in df.columns and 'spending_score' in df.columns:
            df_behavioral['value_index'] = (df_behavioral['annual_income'] / 1000) * \
                                         df_behavioral['spending_score']
            df_behavioral['high_value_indicator'] = (df_behavioral['value_index'] > 
                                                    df_behavioral['value_index'].median()).astype(int)
            
        # Risk indicators
        if 'last_purchase_days_ago' in df.columns:
            df_behavioral['churn_risk'] = np.where(df_behavioral['last_purchase_days_ago'] > 90, 
                                                 1, 0)
            df_behavioral['dormancy_score'] = np.where(df_behavioral['last_purchase_days_ago'] > 60, 
                                                     df_behavioral['last_purchase_days_ago'] / 365, 0)
            
        # Age-based segmentation
        if 'age' in df.columns:
            df_behavioral['age_group'] = pd.cut(df_behavioral['age'], 
                                              bins=[0, 25, 35, 50, 65, 100],
                                              labels=['Gen Z', 'Millennial', 'Gen X', 'Boomer', 'Senior'])
            df_behavioral['is_young'] = (df_behavioral['age'] < 35).astype(int)
            df_behavioral['is_senior'] = (df_behavioral['age'] > 50).astype(int)
            
        new_features = [col for col in df_behavioral.columns if col not in df.columns]
        self.engineered_features.extend(new_features)
        
        logger.info(f"Created {len(new_features)} behavioral features")
        return df_behavioral
        
    def create_interaction_features(self, df: pd.DataFrame, max_degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial interaction features.
        
        Args:
            df: Input DataFrame
            max_degree: Maximum polynomial degree
            
        Returns:
            DataFrame with interaction features
        """
        logger.info(f"Creating interaction features (degree {max_degree})")
        
        # Select numeric columns (excluding IDs and target-like columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['customer_id', 'recency_score', 'frequency_score', 'monetary_score']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols and 
                       not col.endswith('_normalized') and not col.endswith('_score')]
        
        if len(feature_cols) < 2:
            logger.warning("Not enough numeric columns for interaction features")
            return df
            
        # Limit to top features to avoid explosion of features
        if len(feature_cols) > 5:
            # Select features with highest variance (likely most informative)
            variances = df[feature_cols].var()
            feature_cols = variances.nlargest(5).index.tolist()
            
        df_interaction = df.copy()
        
        try:
            # Create polynomial features
            poly = PolynomialFeatures(degree=max_degree, interaction_only=True, 
                                    include_bias=False)
            poly_features = poly.fit_transform(df[feature_cols])
            
            # Get feature names
            poly_feature_names = poly.get_feature_names_out(feature_cols)
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, 
                                index=df.index)
            
            # Add only interaction terms (not original features)
            interaction_features = [name for name in poly_feature_names if ' ' in name]
            
            for feature in interaction_features:
                if feature in poly_df.columns:
                    df_interaction[f"interaction_{feature.replace(' ', '_').replace('^', '_')}"] = poly_df[feature]
                    
            new_features = [col for col in df_interaction.columns if col not in df.columns]
            self.engineered_features.extend(new_features)
            
            logger.info(f"Created {len(new_features)} interaction features")
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {e}")
            
        return df_interaction
        
    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregation-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with aggregation features
        """
        logger.info("Creating aggregation features")
        
        df_agg = df.copy()
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['customer_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) < 2:
            logger.warning("Not enough numeric columns for aggregation features")
            return df
            
        # Statistical aggregations
        if len(feature_cols) >= 2:
            # Mean of key features
            key_features = ['annual_income', 'spending_score', 'purchase_frequency']
            available_key_features = [col for col in key_features if col in feature_cols]
            
            if len(available_key_features) >= 2:
                df_agg['key_features_mean'] = df_agg[available_key_features].mean(axis=1)
                df_agg['key_features_std'] = df_agg[available_key_features].std(axis=1)
                df_agg['key_features_sum'] = df_agg[available_key_features].sum(axis=1)
                
            # Percentile-based features
            for col in feature_cols[:3]:  # Limit to first 3 features
                if col in df_agg.columns:
                    df_agg[f'{col}_percentile'] = df_agg[col].rank(pct=True)
                    df_agg[f'{col}_decile'] = pd.qcut(df_agg[col], q=10, labels=False) + 1
                    
            # Ratio features
            if 'annual_income' in df_agg.columns and 'avg_transaction_value' in df_agg.columns:
                df_agg['income_to_avg_transaction_ratio'] = safe_divide(
                    df_agg['annual_income'], df_agg['avg_transaction_value']
                )
                
            if 'total_purchases' in df_agg.columns and 'customer_since_years' in df_agg.columns:
                df_agg['purchases_per_year'] = safe_divide(
                    df_agg['total_purchases'], df_agg['customer_since_years']
                )
                
        new_features = [col for col in df_agg.columns if col not in df.columns]
        self.engineered_features.extend(new_features)
        
        logger.info(f"Created {len(new_features)} aggregation features")
        return df_agg
        
    def create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create business-specific features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with business features
        """
        logger.info("Creating business features")
        
        df_business = df.copy()
        
        # Customer Lifetime Value (CLV) estimation
        if all(col in df.columns for col in ['avg_transaction_value', 'purchase_frequency', 'customer_since_years']):
            # Simple CLV formula: avg_transaction * frequency * lifespan
            df_business['estimated_clv'] = (df_business['avg_transaction_value'] * 
                                           df_business['purchase_frequency'] * 
                                           df_business['customer_since_years'])
            
            # CLV segments
            df_business['clv_segment'] = pd.qcut(df_business['estimated_clv'], 
                                               q=4, labels=['Low', 'Medium', 'High', 'Premium'])
            
        # Profitability indicators
        if 'annual_income' in df.columns and 'spending_score' in df.columns:
            # High income, high spending = most profitable
            df_business['profitability_score'] = (df_business['annual_income'] / 1000) * \
                                               (df_business['spending_score'] / 100)
            
            # Profitability segments
            df_business['profitability_tier'] = pd.qcut(df_business['profitability_score'], 
                                                      q=3, labels=['Bronze', 'Silver', 'Gold'])
            
        # Retention likelihood
        if all(col in df.columns for col in ['customer_since_years', 'purchase_frequency', 'last_purchase_days_ago']):
            # Higher retention for long-term, frequent, recent customers
            retention_factors = []
            
            # Customer duration factor (0-1)
            retention_factors.append(df_business['customer_since_years'] / 
                                   df_business['customer_since_years'].max())
            
            # Frequency factor (0-1)
            retention_factors.append(df_business['purchase_frequency'] / 
                                   df_business['purchase_frequency'].max())
            
            # Recency factor (0-1, inverted for recency)
            retention_factors.append(1 - (df_business['last_purchase_days_ago'] / 
                                        df_business['last_purchase_days_ago'].max()))
            
            df_business['retention_likelihood'] = np.mean(retention_factors, axis=0)
            
        # Market potential
        if 'age' in df.columns and 'annual_income' in df.columns:
            # Younger customers with high income have high potential
            df_business['market_potential'] = np.where(
                (df_business['age'] < 40) & (df_business['annual_income'] > df_business['annual_income'].median()),
                'High',
                np.where(
                    (df_business['age'] < 50) & (df_business['annual_income'] > df_business['annual_income'].quantile(0.25)),
                    'Medium',
                    'Low'
                )
            )
            
        # Risk assessment
        risk_factors = []
        
        if 'last_purchase_days_ago' in df.columns:
            # High recency risk
            risk_factors.append((df_business['last_purchase_days_ago'] > 90).astype(int))
            
        if 'purchase_frequency' in df.columns:
            # Low frequency risk
            risk_factors.append((df_business['purchase_frequency'] < 
                               df_business['purchase_frequency'].quantile(0.25)).astype(int))
            
        if 'spending_score' in df.columns:
            # Low spending risk
            risk_factors.append((df_business['spending_score'] < 
                               df_business['spending_score'].quantile(0.25)).astype(int))
            
        if risk_factors:
            df_business['churn_risk_score'] = np.sum(risk_factors, axis=0)
            df_business['risk_level'] = np.where(
                df_business['churn_risk_score'] >= 2,
                'High',
                np.where(df_business['churn_risk_score'] == 1, 'Medium', 'Low')
            )
            
        new_features = [col for col in df_business.columns if col not in df.columns]
        self.engineered_features.extend(new_features)
        
        logger.info(f"Created {len(new_features)} business features")
        return df_business
        
    def build_all_features(
        self,
        df: pd.DataFrame,
        include_rfm: bool = True,
        include_behavioral: bool = True,
        include_interactions: bool = True,
        include_aggregations: bool = True,
        include_business: bool = True,
        interaction_degree: int = 2
    ) -> pd.DataFrame:
        """
        Build all engineered features.
        
        Args:
            df: Input DataFrame
            include_rfm: Whether to include RFM features
            include_behavioral: Whether to include behavioral features
            include_interactions: Whether to include interaction features
            include_aggregations: Whether to include aggregation features
            include_business: Whether to include business features
            interaction_degree: Degree for interaction features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting comprehensive feature engineering")
        
        df_features = df.copy()
        original_shape = df_features.shape
        
        if include_rfm:
            df_features = self.create_rfm_features(df_features)
            
        if include_behavioral:
            df_features = self.create_behavioral_features(df_features)
            
        if include_aggregations:
            df_features = self.create_aggregation_features(df_features)
            
        if include_interactions:
            df_features = self.create_interaction_features(df_features, interaction_degree)
            
        if include_business:
            df_features = self.create_business_features(df_features)
            
        final_shape = df_features.shape
        new_features_count = final_shape[1] - original_shape[1]
        
        logger.info(f"Feature engineering completed: {original_shape[1]} → {final_shape[1]} columns (+{new_features_count})")
        
        return df_features
        
    def get_feature_summary(self) -> Dict[str, List[str]]:
        """
        Get summary of engineered features.
        
        Returns:
            Dictionary with feature categories and their features
        """
        return {
            "engineered_features": self.engineered_features,
            "total_engineered": len(self.engineered_features)
        }
        
    def suggest_feature_importance(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, float]:
        """
        Suggest feature importance based on variance and correlation.
        
        Args:
            df: DataFrame with features
            target_column: Optional target column for correlation-based importance
            
        Returns:
            Dictionary with feature importance scores
        """
        logger.info("Calculating feature importance suggestions")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['customer_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        importance_scores = {}
        
        for col in feature_cols:
            score = 0
            
            # Variance-based importance (higher variance = more information)
            variance_score = df[col].var() / df[feature_cols].var().max()
            score += variance_score * 0.4
            
            # Correlation-based importance (if target available)
            if target_column and target_column in df.columns:
                try:
                    correlation = abs(df[col].corr(df[target_column]))
                    score += correlation * 0.6
                except:
                    pass
            else:
                # Use correlation with other features as proxy
                correlations = df[feature_cols].corrwith(df[col]).abs()
                avg_correlation = correlations.mean()
                score += avg_correlation * 0.2
                
            importance_scores[col] = score
            
        # Sort by importance
        importance_scores = dict(sorted(importance_scores.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        self.feature_importance = importance_scores
        
        return importance_scores
