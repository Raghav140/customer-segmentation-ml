"""Business insights generator for customer segmentation system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from src.config.logging_config import get_logger
from src.utils.helpers import safe_divide, flatten_dict

logger = get_logger(__name__)


class CustomerTier(Enum):
    """Customer tier classification."""
    PREMIUM = "Premium"
    HIGH_VALUE = "High Value"
    MEDIUM_VALUE = "Medium Value"
    LOW_VALUE = "Low Value"
    BUDGET = "Budget"


class RiskLevel(Enum):
    """Customer risk level classification."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class CustomerSegment:
    """Customer segment definition."""
    segment_id: int
    segment_name: str
    customer_tier: CustomerTier
    risk_level: RiskLevel
    size: int
    percentage: float
    characteristics: List[str]
    key_metrics: Dict[str, float]
    recommendations: List[str]
    business_value: float


@dataclass
class BusinessInsight:
    """Business insight definition."""
    insight_type: str
    title: str
    description: str
    impact: str  # High, Medium, Low
    confidence: float  # 0-1
    actionable_steps: List[str]
    kpi_impact: Dict[str, str]
    target_segments: List[int]


class BusinessInsightsGenerator:
    """Generates comprehensive business insights from clustering results."""
    
    def __init__(self):
        """Initialize business insights generator."""
        self.segments = []
        self.insights = []
        self.feature_importance = {}
        self.benchmarks = {}
        
    def analyze_clustering_results(self, 
                                 X: Union[pd.DataFrame, np.ndarray],
                                 cluster_labels: np.ndarray,
                                 feature_names: List[str],
                                 cluster_centers: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze clustering results and generate business insights.
        
        Args:
            X: Original feature data
            cluster_labels: Cluster assignments
            feature_names: Names of features
            cluster_centers: Cluster centers (if available)
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info("Generating comprehensive business insights")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
            
        df['cluster'] = cluster_labels
        
        # 1. Create customer segments
        self.segments = self._create_customer_segments(df, feature_names)
        
        # 2. Generate business insights
        self.insights = self._generate_business_insights(df, feature_names)
        
        # 3. Calculate feature importance
        self.feature_importance = self._calculate_feature_importance(df, feature_names)
        
        # 4. Set benchmarks
        self.benchmarks = self._calculate_benchmarks(df, feature_names)
        
        # 5. Generate recommendations
        recommendations = self._generate_strategic_recommendations()
        
        results = {
            'segments': [self._segment_to_dict(seg) for seg in self.segments],
            'insights': [self._insight_to_dict(insight) for insight in self.insights],
            'feature_importance': self.feature_importance,
            'benchmarks': self.benchmarks,
            'recommendations': recommendations,
            'summary': self._generate_executive_summary()
        }
        
        logger.info(f"Generated {len(self.segments)} segments and {len(self.insights)} insights")
        
        return results
        
    def _create_customer_segments(self, df: pd.DataFrame, feature_names: List[str]) -> List[CustomerSegment]:
        """Create customer segments from clustering results."""
        segments = []
        
        # Calculate overall statistics for comparison
        overall_stats = df[feature_names].describe()
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            cluster_stats = cluster_data[feature_names].describe()
            
            # Determine segment characteristics
            characteristics = self._identify_characteristics(cluster_stats, overall_stats, feature_names)
            
            # Classify customer tier
            customer_tier = self._classify_customer_tier(cluster_stats, overall_stats, characteristics)
            
            # Assess risk level
            risk_level = self._assess_risk_level(cluster_stats, overall_stats, characteristics)
            
            # Generate segment name
            segment_name = self._generate_segment_name(characteristics, customer_tier, risk_level)
            
            # Calculate key metrics
            key_metrics = self._calculate_key_metrics(cluster_stats, overall_stats)
            
            # Generate recommendations
            recommendations = self._generate_segment_recommendations(characteristics, customer_tier, risk_level)
            
            # Calculate business value
            business_value = self._calculate_business_value(customer_tier, risk_level, key_metrics)
            
            segment = CustomerSegment(
                segment_id=cluster_id,
                segment_name=segment_name,
                customer_tier=customer_tier,
                risk_level=risk_level,
                size=len(cluster_data),
                percentage=(len(cluster_data) / len(df)) * 100,
                characteristics=characteristics,
                key_metrics=key_metrics,
                recommendations=recommendations,
                business_value=business_value
            )
            
            segments.append(segment)
            
        return segments
        
    def _identify_characteristics(self, cluster_stats: pd.DataFrame, 
                                overall_stats: pd.DataFrame, 
                                feature_names: List[str]) -> List[str]:
        """Identify key characteristics of a cluster."""
        characteristics = []
        
        for feature in feature_names:
            if feature not in cluster_stats.columns:
                continue
                
            cluster_mean = cluster_stats[feature]['mean']
            overall_mean = overall_stats[feature]['mean']
            overall_std = overall_stats[feature]['std']
            
            # Calculate z-score difference
            z_diff = (cluster_mean - overall_mean) / overall_std if overall_std > 0 else 0
            
            # Identify significant differences
            if abs(z_diff) > 0.5:
                direction = "High" if z_diff > 0 else "Low"
                
                # Map feature names to business terms
                business_feature = self._map_feature_to_business_term(feature)
                characteristics.append(f"{direction} {business_feature}")
                
        return characteristics
        
    def _map_feature_to_business_term(self, feature: str) -> str:
        """Map technical feature names to business terms."""
        mapping = {
            'annual_income': 'Income',
            'spending_score': 'Spending',
            'age': 'Age',
            'purchase_frequency': 'Purchase Frequency',
            'avg_transaction_value': 'Transaction Value',
            'customer_since_years': 'Customer Loyalty',
            'last_purchase_days_ago': 'Purchase Recency',
            'total_purchases': 'Total Purchases',
            'recency_score': 'Recency',
            'frequency_score': 'Frequency',
            'monetary_score': 'Monetary Value',
            'rfm_combined_score': 'RFM Score',
            'loyalty_score': 'Loyalty',
            'engagement_score': 'Engagement',
            'estimated_clv': 'Customer Lifetime Value',
            'profitability_score': 'Profitability',
            'retention_likelihood': 'Retention Likelihood',
            'churn_risk_score': 'Churn Risk'
        }
        
        return mapping.get(feature, feature.replace('_', ' ').title())
        
    def _classify_customer_tier(self, cluster_stats: pd.DataFrame, 
                              overall_stats: pd.DataFrame, 
                              characteristics: List[str]) -> CustomerTier:
        """Classify customer tier based on characteristics."""
        # Check for premium indicators
        premium_indicators = ['High Income', 'High Spending', 'High Monetary Value', 'High CLV']
        high_indicators = ['High Purchase Frequency', 'High Transaction Value', 'High Loyalty']
        
        premium_count = sum(1 for char in characteristics if any(ind in char for ind in premium_indicators))
        high_count = sum(1 for char in characteristics if any(ind in char for ind in high_indicators))
        
        # Check for budget indicators
        budget_indicators = ['Low Income', 'Low Spending', 'Low Transaction Value']
        budget_count = sum(1 for char in characteristics if any(ind in char for ind in budget_indicators))
        
        if premium_count >= 2:
            return CustomerTier.PREMIUM
        elif premium_count >= 1 or high_count >= 2:
            return CustomerTier.HIGH_VALUE
        elif budget_count >= 2:
            return CustomerTier.BUDGET
        elif budget_count >= 1:
            return CustomerTier.LOW_VALUE
        else:
            return CustomerTier.MEDIUM_VALUE
            
    def _assess_risk_level(self, cluster_stats: pd.DataFrame, 
                         overall_stats: pd.DataFrame, 
                         characteristics: List[str]) -> RiskLevel:
        """Assess risk level of customer segment."""
        # Check for high risk indicators
        high_risk_indicators = ['Low Recency', 'High Churn Risk', 'Low Retention', 'Low Engagement']
        medium_risk_indicators = ['Low Purchase Frequency', 'Low Loyalty']
        
        high_risk_count = sum(1 for char in characteristics if any(ind in char for ind in high_risk_indicators))
        medium_risk_count = sum(1 for char in characteristics if any(ind in char for ind in medium_risk_indicators))
        
        if high_risk_count >= 2:
            return RiskLevel.HIGH
        elif high_risk_count >= 1 or medium_risk_count >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def _generate_segment_name(self, characteristics: List[str], 
                             customer_tier: CustomerTier, 
                             risk_level: RiskLevel) -> str:
        """Generate descriptive segment name."""
        tier_names = {
            CustomerTier.PREMIUM: "Premium",
            CustomerTier.HIGH_VALUE: "High-Value",
            CustomerTier.MEDIUM_VALUE: "Standard",
            CustomerTier.LOW_VALUE: "Budget-Conscious",
            CustomerTier.BUDGET: "Economy"
        }
        
        risk_names = {
            RiskLevel.HIGH: "At-Risk",
            RiskLevel.MEDIUM: "Vulnerable",
            RiskLevel.LOW: "Stable"
        }
        
        # Base name from tier
        base_name = tier_names[customer_tier]
        
        # Add risk qualifier if not low risk
        if risk_level != RiskLevel.LOW:
            base_name += f" {risk_names[risk_level]}"
            
        # Add key characteristic
        key_chars = [char for char in characteristics if 'High' in char or 'Low' in char][:2]
        if key_chars:
            key_char = key_chars[0].replace('High ', '').replace('Low ', '')
            base_name += f" {key_char}"
            
        return base_name
        
    def _calculate_key_metrics(self, cluster_stats: pd.DataFrame, 
                             overall_stats: pd.DataFrame) -> Dict[str, float]:
        """Calculate key business metrics for the segment."""
        metrics = {}
        
        # Calculate relative performance metrics
        for feature in cluster_stats.columns:
            if feature in overall_stats.columns:
                cluster_mean = cluster_stats[feature]['mean']
                overall_mean = overall_stats[feature]['mean']
                
                if overall_mean > 0:
                    metrics[f'{feature}_vs_average'] = safe_divide(cluster_mean, overall_mean)
                else:
                    metrics[f'{feature}_vs_average'] = 1.0
                    
        # Calculate composite scores
        if 'annual_income_vs_average' in metrics and 'spending_score_vs_average' in metrics:
            metrics['value_index'] = (metrics['annual_income_vs_average'] + 
                                    metrics['spending_score_vs_average']) / 2
            
        if 'purchase_frequency_vs_average' in metrics and 'customer_since_years_vs_average' in metrics:
            metrics['loyalty_index'] = (metrics['purchase_frequency_vs_average'] + 
                                      metrics['customer_since_years_vs_average']) / 2
            
        return metrics
        
    def _generate_segment_recommendations(self, characteristics: List[str], 
                                        customer_tier: CustomerTier, 
                                        risk_level: RiskLevel) -> List[str]:
        """Generate actionable recommendations for the segment."""
        recommendations = []
        
        # Tier-based recommendations
        if customer_tier == CustomerTier.PREMIUM:
            recommendations.extend([
                "Offer exclusive VIP programs and personalized services",
                "Provide priority customer support and early access to new products",
                "Create premium loyalty rewards and recognition programs"
            ])
        elif customer_tier == CustomerTier.HIGH_VALUE:
            recommendations.extend([
                "Implement upselling and cross-selling strategies",
                "Provide enhanced customer service and personalized recommendations",
                "Create tiered loyalty programs with increasing benefits"
            ])
        elif customer_tier in [CustomerTier.LOW_VALUE, CustomerTier.BUDGET]:
            recommendations.extend([
                "Focus on value-based promotions and discounts",
                "Provide budget-friendly product recommendations",
                "Implement cost-effective communication channels"
            ])
            
        # Risk-based recommendations
        if risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Implement immediate retention campaigns with special offers",
                "Conduct satisfaction surveys to identify pain points",
                "Provide proactive customer service and support"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Monitor engagement metrics closely",
                "Implement periodic check-ins and relationship building",
                "Provide targeted incentives to increase loyalty"
            ])
            
        # Characteristic-based recommendations
        if any('Low Purchase Frequency' in char for char in characteristics):
            recommendations.append("Increase purchase frequency through subscription programs and regular promotions")
            
        if any('Low Transaction Value' in char for char in characteristics):
            recommendations.append("Implement bundling strategies and volume discounts to increase transaction value")
            
        if any('High Churn Risk' in char for char in characteristics):
            recommendations.append("Launch targeted win-back campaigns with personalized offers")
            
        return recommendations[:5]  # Limit to top 5 recommendations
        
    def _calculate_business_value(self, customer_tier: CustomerTier, 
                                risk_level: RiskLevel, 
                                key_metrics: Dict[str, float]) -> float:
        """Calculate business value score for the segment."""
        tier_scores = {
            CustomerTier.PREMIUM: 1.0,
            CustomerTier.HIGH_VALUE: 0.8,
            CustomerTier.MEDIUM_VALUE: 0.6,
            CustomerTier.LOW_VALUE: 0.4,
            CustomerTier.BUDGET: 0.2
        }
        
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.6
        }
        
        base_score = tier_scores[customer_tier]
        risk_adjusted = base_score * risk_multipliers[risk_level]
        
        # Adjust based on key metrics
        value_index = key_metrics.get('value_index', 1.0)
        loyalty_index = key_metrics.get('loyalty_index', 1.0)
        
        final_score = risk_adjusted * (value_index * 0.6 + loyalty_index * 0.4)
        
        return min(final_score, 1.0)  # Cap at 1.0
        
    def _generate_business_insights(self, df: pd.DataFrame, 
                                  feature_names: List[str]) -> List[BusinessInsight]:
        """Generate high-level business insights."""
        insights = []
        
        # Insight 1: Revenue concentration
        segment_sizes = df['cluster'].value_counts()
        if len(segment_sizes) > 1:
            top_segment_pct = (segment_sizes.iloc[0] / len(df)) * 100
            if top_segment_pct > 40:
                insights.append(BusinessInsight(
                    insight_type="Revenue Concentration",
                    title="High Customer Segment Concentration",
                    description=f"Top segment represents {top_segment_pct:.1f}% of customers, indicating potential revenue concentration risk",
                    impact="High",
                    confidence=0.8,
                    actionable_steps=[
                        "Diversify customer base across segments",
                        "Develop strategies to grow smaller segments",
                        "Monitor dependency on top segment"
                    ],
                    kpi_impact={"Revenue Risk": "Reduction", "Customer Diversity": "Improvement"},
                    target_segments=[segment_sizes.index[0]]
                ))
                
        # Insight 2: Growth opportunities
        growth_segments = [seg for seg in self.segments if seg.customer_tier in [CustomerTier.HIGH_VALUE, CustomerTier.PREMIUM]]
        if growth_segments:
            total_growth_potential = sum(seg.percentage for seg in growth_segments)
            insights.append(BusinessInsight(
                insight_type="Growth Opportunity",
                title="High-Value Segment Growth Potential",
                description=f"High-value segments represent {total_growth_potential:.1f}% of customers with significant growth opportunities",
                impact="High",
                confidence=0.9,
                actionable_steps=[
                    "Invest in premium product development",
                    "Enhance VIP customer programs",
                    "Focus on retention of high-value customers"
                ],
                kpi_impact={"Revenue": "Increase", "Customer Lifetime Value": "Increase"},
                target_segments=[seg.segment_id for seg in growth_segments]
            ))
            
        # Insight 3: Risk mitigation
        at_risk_segments = [seg for seg in self.segments if seg.risk_level == RiskLevel.HIGH]
        if at_risk_segments:
            total_risk_percentage = sum(seg.percentage for seg in at_risk_segments)
            insights.append(BusinessInsight(
                insight_type="Risk Mitigation",
                title="Customer Churn Risk Alert",
                description=f"At-risk segments represent {total_risk_percentage:.1f}% of customers requiring immediate attention",
                impact="High",
                confidence=0.85,
                actionable_steps=[
                    "Launch immediate retention campaigns",
                    "Identify root causes of dissatisfaction",
                    "Provide personalized offers and support"
                ],
                kpi_impact={"Churn Rate": "Reduction", "Customer Satisfaction": "Improvement"},
                target_segments=[seg.segment_id for seg in at_risk_segments]
            ))
            
        return insights
        
    def _calculate_feature_importance(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for business decisions."""
        importance = {}
        
        # Calculate variance-based importance
        for feature in feature_names:
            if feature in df.columns:
                # Higher variance indicates more discriminatory power
                variance = df[feature].var()
                importance[feature] = variance
                
        # Normalize importance scores
        if importance:
            max_importance = max(importance.values())
            importance = {k: v/max_importance for k, v in importance.items()}
            
        return importance
        
    def _calculate_benchmarks(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate industry benchmarks and KPIs."""
        benchmarks = {}
        
        # Overall metrics
        benchmarks['overall'] = {
            'total_customers': len(df),
            'avg_income': df['annual_income'].mean() if 'annual_income' in df.columns else 0,
            'avg_spending_score': df['spending_score'].mean() if 'spending_score' in df.columns else 0,
            'avg_purchase_frequency': df['purchase_frequency'].mean() if 'purchase_frequency' in df.columns else 0
        }
        
        # Segment benchmarks
        for segment in self.segments:
            benchmarks[f'segment_{segment.segment_id}'] = {
                'size': segment.size,
                'percentage': segment.percentage,
                'business_value': segment.business_value,
                'risk_score': 1.0 if segment.risk_level == RiskLevel.LOW else 
                             0.7 if segment.risk_level == RiskLevel.MEDIUM else 0.4
            }
            
        return benchmarks
        
    def _generate_strategic_recommendations(self) -> Dict[str, List[str]]:
        """Generate strategic recommendations for the business."""
        recommendations = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_strategy': []
        }
        
        # Immediate actions (next 30 days)
        if any(seg.risk_level == RiskLevel.HIGH for seg in self.segments):
            recommendations['immediate_actions'].append(
                "Launch retention campaigns for at-risk customer segments"
            )
            
        recommendations['immediate_actions'].extend([
            "Set up monitoring dashboard for key segment metrics",
            "Create cross-functional team for segment-based initiatives"
        ])
        
        # Short-term goals (next 90 days)
        premium_segments = [seg for seg in self.segments if seg.customer_tier == CustomerTier.PREMIUM]
        if premium_segments:
            recommendations['short_term_goals'].append(
                "Develop enhanced VIP programs for premium customers"
            )
            
        recommendations['short_term_goals'].extend([
            "Implement personalized marketing campaigns by segment",
            "Optimize product mix based on segment preferences"
        ])
        
        # Long-term strategy (6+ months)
        recommendations['long_term_strategy'].extend([
            "Develop segment-specific product development roadmap",
            "Create predictive models for segment evolution",
            "Establish continuous customer segmentation framework"
        ])
        
        return recommendations
        
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of findings."""
        total_customers = sum(seg.size for seg in self.segments)
        high_value_percentage = sum(seg.percentage for seg in self.segments 
                                  if seg.customer_tier in [CustomerTier.PREMIUM, CustomerTier.HIGH_VALUE])
        at_risk_percentage = sum(seg.percentage for seg in self.segments 
                               if seg.risk_level == RiskLevel.HIGH)
        
        return {
            'total_segments': len(self.segments),
            'total_customers_analyzed': total_customers,
            'high_value_customer_percentage': high_value_percentage,
            'at_risk_customer_percentage': at_risk_percentage,
            'key_insights_count': len(self.insights),
            'top_business_value_segment': max(self.segments, key=lambda s: s.business_value).segment_name,
            'highest_risk_segment': next((s.segment_name for s in self.segments if s.risk_level == RiskLevel.HIGH), None),
            'recommended_focus_areas': [
                "Retain high-value customers through enhanced programs",
                "Mitigate churn risk in vulnerable segments",
                "Develop growth strategies for emerging segments"
            ]
        }
        
    def _segment_to_dict(self, segment: CustomerSegment) -> Dict[str, Any]:
        """Convert segment to dictionary for JSON serialization."""
        return {
            'segment_id': segment.segment_id,
            'segment_name': segment.segment_name,
            'customer_tier': segment.customer_tier.value,
            'risk_level': segment.risk_level.value,
            'size': segment.size,
            'percentage': segment.percentage,
            'characteristics': segment.characteristics,
            'key_metrics': segment.key_metrics,
            'recommendations': segment.recommendations,
            'business_value': segment.business_value
        }
        
    def _insight_to_dict(self, insight: BusinessInsight) -> Dict[str, Any]:
        """Convert insight to dictionary for JSON serialization."""
        return {
            'insight_type': insight.insight_type,
            'title': insight.title,
            'description': insight.description,
            'impact': insight.impact,
            'confidence': insight.confidence,
            'actionable_steps': insight.actionable_steps,
            'kpi_impact': insight.kpi_impact,
            'target_segments': insight.target_segments
        }
