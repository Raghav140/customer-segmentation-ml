"""
Feature Engineering Module for Customer Segmentation

This module provides comprehensive feature engineering capabilities
for customer segmentation analysis, including RFM analysis,
behavioral features, and demographic features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for customer segmentation.
    
    Creates meaningful features from raw customer data including:
    - RFM (Recency, Frequency, Monetary) analysis
    - Behavioral patterns
    - Demographic segments
    - Engagement metrics
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
        self.feature_importance = {}
        
    def create_rfm_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        
        Args:
            data: Customer data with transaction information
            
        Returns:
            DataFrame with RFM features added
        """
        logger.info("Creating RFM features...")
        
        df = data.copy()
        
        # Recency: Days since last purchase (lower is better)
        if 'last_purchase_days' in df.columns:
            df['recency'] = df['last_purchase_days']
        else:
            # Generate synthetic recency if not available
            df['recency'] = np.random.randint(1, 365, size=len(df))
        
        # Frequency: Purchase frequency
        if 'purchase_frequency' in df.columns:
            df['frequency'] = df['purchase_frequency']
        else:
            # Generate synthetic frequency if not available
            df['frequency'] = np.random.exponential(scale=2, size=len(df))
        
        # Monetary: Annual income as proxy for monetary value
        if 'annual_income' in df.columns:
            df['monetary'] = df['annual_income']
        else:
            # Generate synthetic monetary if not available
            df['monetary'] = np.random.normal(50000, 20000, size=len(df))
        
        # RFM Scores (1-5 scale)
        df['recency_score'] = pd.qcut(df['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
        df['frequency_score'] = pd.qcut(df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        df['monetary_score'] = pd.qcut(df['monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Combined RFM Score
        df['rfm_score'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']
        
        # RFM Segments
        df['rfm_segment'] = self._create_rfm_segments(df)
        
        self.feature_names.extend(['recency', 'frequency', 'monetary', 'rfm_score', 'recency_score', 
                                 'frequency_score', 'monetary_score', 'rfm_segment'])
        
        logger.info(f"Created RFM features: {['recency', 'frequency', 'monetary', 'rfm_score']}")
        return df
    
    def _create_rfm_segments(self, df: pd.DataFrame) -> pd.Series:
        """Create RFM segment names based on scores."""
        segments = []
        
        for _, row in df.iterrows():
            if row['recency_score'] >= 4 and row['frequency_score'] >= 4 and row['monetary_score'] >= 4:
                segments.append('Champions')
            elif row['recency_score'] >= 3 and row['frequency_score'] >= 3 and row['monetary_score'] >= 3:
                segments.append('Loyal Customers')
            elif row['recency_score'] >= 3 and row['frequency_score'] <= 2:
                segments.append('Potential Loyalists')
            elif row['recency_score'] <= 2 and row['frequency_score'] >= 4:
                segments.append('New Customers')
            elif row['recency_score'] <= 2 and row['frequency_score'] <= 2 and row['monetary_score'] <= 2:
                segments.append('At Risk')
            else:
                segments.append('Others')
        
        return pd.Series(segments, index=df.index)
    
    def create_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features from customer data.
        
        Args:
            data: Customer data with behavioral attributes
            
        Returns:
            DataFrame with behavioral features added
        """
        logger.info("Creating behavioral features...")
        
        df = data.copy()
        
        # Spending behavior
        if 'spending_score' in df.columns:
            df['spending_category'] = pd.cut(df['spending_score'], 
                                           bins=[0, 20, 40, 60, 80, 100], 
                                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Purchase patterns
        if 'purchase_frequency' in df.columns:
            df['purchase_pattern'] = np.where(df['purchase_frequency'] > df['purchase_frequency'].median(), 
                                            'Frequent', 'Infrequent')
        
        # Customer tenure
        if 'customer_years' in df.columns:
            df['tenure_category'] = pd.cut(df['customer_years'], 
                                         bins=[0, 1, 3, 5, 10, float('inf')], 
                                         labels=['New', 'Growing', 'Established', 'Mature', 'Veteran'])
        
        # Engagement score (composite)
        df['engagement_score'] = self._calculate_engagement_score(df)
        
        # Loyalty indicators
        df['loyalty_score'] = self._calculate_loyalty_score(df)
        
        # Risk indicators
        df['churn_risk'] = self._calculate_churn_risk(df)
        
        behavioral_features = ['spending_category', 'purchase_pattern', 'tenure_category', 
                              'engagement_score', 'loyalty_score', 'churn_risk']
        self.feature_names.extend([f for f in behavioral_features if f in df.columns])
        
        logger.info(f"Created behavioral features: {[f for f in behavioral_features if f in df.columns]}")
        return df
    
    def _calculate_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate customer engagement score."""
        score = np.zeros(len(df))
        
        # Factor in spending score
        if 'spending_score' in df.columns:
            score += df['spending_score'] / 100 * 0.4
        
        # Factor in purchase frequency
        if 'purchase_frequency' in df.columns:
            freq_normalized = (df['purchase_frequency'] - df['purchase_frequency'].min()) / \
                           (df['purchase_frequency'].max() - df['purchase_frequency'].min())
            score += freq_normalized * 0.3
        
        # Factor in customer tenure
        if 'customer_years' in df.columns:
            tenure_normalized = (df['customer_years'] - df['customer_years'].min()) / \
                             (df['customer_years'].max() - df['customer_years'].min())
            score += tenure_normalized * 0.3
        
        return score
    
    def _calculate_loyalty_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate customer loyalty score."""
        score = np.zeros(len(df))
        
        # Based on tenure and frequency
        if 'customer_years' in df.columns:
            score += df['customer_years'] * 0.5
        
        if 'purchase_frequency' in df.columns:
            score += df['purchase_frequency'] * 0.3
        
        if 'spending_score' in df.columns:
            score += df['spending_score'] / 100 * 0.2
        
        # Normalize to 0-100 scale
        if score.max() > 0:
            score = (score / score.max()) * 100
        
        return score
    
    def _calculate_churn_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate churn risk score."""
        risk = np.zeros(len(df))
        
        # High recency (days since last purchase) increases risk
        if 'last_purchase_days' in df.columns:
            risk += np.where(df['last_purchase_days'] > 90, 0.4, 0)
            risk += np.where(df['last_purchase_days'] > 180, 0.3, 0)
        
        # Low frequency increases risk
        if 'purchase_frequency' in df.columns:
            risk += np.where(df['purchase_frequency'] < 1, 0.3, 0)
        
        # Low spending increases risk
        if 'spending_score' in df.columns:
            risk += np.where(df['spending_score'] < 30, 0.3, 0)
        
        return risk
    
    def create_demographic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic features from customer data.
        
        Args:
            data: Customer data with demographic information
            
        Returns:
            DataFrame with demographic features added
        """
        logger.info("Creating demographic features...")
        
        df = data.copy()
        
        # Age-based features
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 45, 55, 65, 100], 
                                   labels=['Gen Z', 'Millennial', 'Gen X', 'Young Boomer', 'Boomer', 'Senior'])
            
            df['life_stage'] = self._determine_life_stage(df['age'])
        
        # Income-based features
        if 'annual_income' in df.columns:
            df['income_group'] = pd.cut(df['annual_income'], 
                                       bins=[0, 30000, 50000, 75000, 100000, 150000, float('inf')], 
                                       labels=['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High', 'Very High'])
            
            df['income_to_age_ratio'] = df['annual_income'] / (df['age'] + 1)
        
        # Combined demographic segments
        if 'age_group' in df.columns and 'income_group' in df.columns:
            df['demographic_segment'] = df['age_group'].astype(str) + ' - ' + df['income_group'].astype(str)
        
        demographic_features = ['age_group', 'life_stage', 'income_group', 'income_to_age_ratio', 'demographic_segment']
        self.feature_names.extend([f for f in demographic_features if f in df.columns])
        
        logger.info(f"Created demographic features: {[f for f in demographic_features if f in df.columns]}")
        return df
    
    def _determine_life_stage(self, age_series: pd.Series) -> pd.Series:
        """Determine life stage based on age."""
        life_stages = []
        
        for age in age_series:
            if age < 25:
                life_stages.append('Student/Young Professional')
            elif age < 35:
                life_stages.append('Early Career')
            elif age < 45:
                life_stages.append('Established Professional')
            elif age < 55:
                life_stages.append('Senior Professional')
            elif age < 65:
                life_stages.append('Pre-Retirement')
            else:
                life_stages.append('Retired')
        
        return pd.Series(life_stages, index=age_series.index)
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.
        
        Args:
            data: DataFrame with base features
            
        Returns:
            DataFrame with interaction features added
        """
        logger.info("Creating interaction features...")
        
        df = data.copy()
        
        # Income x Spending interaction
        if 'annual_income' in df.columns and 'spending_score' in df.columns:
            df['income_spending_interaction'] = df['annual_income'] * df['spending_score']
        
        # Age x Income interaction
        if 'age' in df.columns and 'annual_income' in df.columns:
            df['age_income_interaction'] = df['age'] * df['annual_income']
        
        # Frequency x Tenure interaction
        if 'purchase_frequency' in df.columns and 'customer_years' in df.columns:
            df['frequency_tenure_interaction'] = df['purchase_frequency'] * df['customer_years']
        
        # Spending to income ratio
        if 'annual_income' in df.columns and 'spending_score' in df.columns:
            df['spending_to_income_ratio'] = (df['spending_score'] * 1000) / df['annual_income']
        
        interaction_features = ['income_spending_interaction', 'age_income_interaction', 
                               'frequency_tenure_interaction', 'spending_to_income_ratio']
        self.feature_names.extend([f for f in interaction_features if f in df.columns])
        
        logger.info(f"Created interaction features: {[f for f in interaction_features if f in df.columns]}")
        return df
    
    def create_features(self, data: pd.DataFrame, feature_types: List[str] = None) -> pd.DataFrame:
        """
        Create all specified feature types.
        
        Args:
            data: Input customer data
            feature_types: List of feature types to create 
                          ['rfm', 'behavioral', 'demographic', 'interaction']
                          If None, creates all types
        
        Returns:
            DataFrame with all requested features
        """
        if feature_types is None:
            feature_types = ['rfm', 'behavioral', 'demographic', 'interaction']
        
        logger.info(f"Creating features: {feature_types}")
        
        df = data.copy()
        
        if 'rfm' in feature_types:
            df = self.create_rfm_features(df)
        
        if 'behavioral' in feature_types:
            df = self.create_behavioral_features(df)
        
        if 'demographic' in feature_types:
            df = self.create_demographic_features(df)
        
        if 'interaction' in feature_types:
            df = self.create_interaction_features(df)
        
        logger.info(f"Feature engineering complete. Created {len(self.feature_names)} features.")
        return df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance
    
    def get_feature_names(self) -> List[str]:
        """Get list of created feature names."""
        return self.feature_names
    
    def reset_features(self):
        """Reset feature tracking."""
        self.feature_names = []
        self.feature_importance = {}
