"""Recommendation engine for customer segmentation business actions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

from src.config.logging_config import get_logger
from src.utils.helpers import safe_divide

logger = get_logger(__name__)


class ActionType(Enum):
    """Types of recommended actions."""
    MARKETING = "Marketing"
    PRODUCT = "Product"
    SERVICE = "Service"
    PRICING = "Pricing"
    RETENTION = "Retention"
    ACQUISITION = "Acquisition"


class Priority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Timeframe(Enum):
    """Timeframes for implementation."""
    IMMEDIATE = "Immediate (0-30 days)"
    SHORT_TERM = "Short Term (30-90 days)"
    MEDIUM_TERM = "Medium Term (90-180 days)"
    LONG_TERM = "Long Term (180+ days)"


@dataclass
class Recommendation:
    """Business recommendation definition."""
    id: str
    title: str
    description: str
    action_type: ActionType
    priority: Priority
    timeframe: Timeframe
    target_segments: List[int]
    expected_impact: Dict[str, str]
    implementation_steps: List[str]
    required_resources: List[str]
    success_metrics: List[str]
    estimated_cost: str
    estimated_roi: str
    confidence_score: float


class RecommendationEngine:
    """Generates actionable business recommendations from clustering insights."""
    
    def __init__(self):
        """Initialize recommendation engine."""
        self.recommendations = []
        self.segment_strategies = {}
        self.action_templates = self._load_action_templates()
        
    def _load_action_templates(self) -> Dict[str, Dict]:
        """Load predefined action templates."""
        return {
            'premium_retention': {
                'title': 'Premium Customer Retention Program',
                'description': 'Implement exclusive retention program for high-value customers',
                'action_type': ActionType.RETENTION,
                'priority': Priority.HIGH,
                'timeframe': Timeframe.SHORT_TERM,
                'expected_impact': {'Retention Rate': 'Increase 15-20%', 'Customer Lifetime Value': 'Increase 25%'},
                'implementation_steps': [
                    'Identify premium customers based on CLV and spending patterns',
                    'Design exclusive benefits and rewards program',
                    'Assign dedicated account managers',
                    'Create personalized communication strategy'
                ],
                'required_resources': ['Customer Success Team', 'Marketing Budget', 'Technology Platform'],
                'success_metrics': ['Retention Rate', 'CLV', 'Customer Satisfaction Score'],
                'estimated_cost': 'Medium',
                'estimated_roi': 'High (3-5x)'
            },
            'at_risk_intervention': {
                'title': 'At-Risk Customer Intervention',
                'description': 'Proactive intervention for customers with high churn risk',
                'action_type': ActionType.RETENTION,
                'priority': Priority.CRITICAL,
                'timeframe': Timeframe.IMMEDIATE,
                'expected_impact': {'Churn Rate': 'Reduce 30-40%', 'Customer Satisfaction': 'Improve'},
                'implementation_steps': [
                    'Identify at-risk customers using churn prediction models',
                    'Launch targeted win-back campaigns',
                    'Offer personalized incentives and discounts',
                    'Conduct satisfaction surveys to identify issues'
                ],
                'required_resources': ['Marketing Team', 'Customer Service', 'Promotional Budget'],
                'success_metrics': ['Churn Rate', 'Campaign Response Rate', 'Customer Satisfaction'],
                'estimated_cost': 'Medium',
                'estimated_roi': 'High (4-6x)'
            },
            'upselling_program': {
                'title': 'Strategic Upselling Program',
                'description': 'Develop upselling opportunities for high-potential customers',
                'action_type': ActionType.MARKETING,
                'priority': Priority.HIGH,
                'timeframe': Timeframe.MEDIUM_TERM,
                'expected_impact': {'Average Order Value': 'Increase 20%', 'Revenue': 'Increase 15%'},
                'implementation_steps': [
                    'Analyze purchase patterns to identify upselling opportunities',
                    'Train sales team on upselling techniques',
                    'Create product bundles and tiered offerings',
                    'Implement personalized recommendation engine'
                ],
                'required_resources': ['Sales Team', 'Product Team', 'Analytics Tools'],
                'success_metrics': ['Average Order Value', 'Conversion Rate', 'Revenue per Customer'],
                'estimated_cost': 'Medium',
                'estimated_roi': 'Medium-High (2-3x)'
            },
            'budget_segment_growth': {
                'title': 'Budget-Conscious Segment Growth',
                'description': 'Develop strategies to grow budget-conscious customer segment',
                'action_type': ActionType.MARKETING,
                'priority': Priority.MEDIUM,
                'timeframe': Timeframe.MEDIUM_TERM,
                'expected_impact': {'Market Share': 'Increase 10%', 'Customer Base': 'Expand 20%'},
                'implementation_steps': [
                    'Develop value-focused product lines',
                    'Create budget-friendly marketing campaigns',
                    'Implement referral programs',
                    'Optimize pricing for sensitivity'
                ],
                'required_resources': ['Product Development', 'Marketing Team', 'Pricing Team'],
                'success_metrics': ['Customer Acquisition Cost', 'Market Share', 'Segment Growth Rate'],
                'estimated_cost': 'Medium',
                'estimated_roi': 'Medium (2-3x)'
            },
            'loyalty_program_enhancement': {
                'title': 'Loyalty Program Enhancement',
                'description': 'Enhance loyalty program to increase customer engagement',
                'action_type': ActionType.MARKETING,
                'priority': Priority.HIGH,
                'timeframe': Timeframe.SHORT_TERM,
                'expected_impact': {'Customer Loyalty': 'Increase 25%', 'Repeat Purchases': 'Increase 30%'},
                'implementation_steps': [
                    'Analyze current loyalty program effectiveness',
                    'Design tiered rewards system',
                    'Implement gamification elements',
                    'Create personalized loyalty communications'
                ],
                'required_resources': ['Marketing Team', 'Technology Platform', 'Budget for Rewards'],
                'success_metrics': ['Loyalty Program Participation', 'Repeat Purchase Rate', 'Customer Engagement'],
                'estimated_cost': 'Medium',
                'estimated_roi': 'High (3-4x)'
            },
            'personalization_engine': {
                'title': 'Personalization Engine Implementation',
                'description': 'Implement AI-driven personalization for customer experiences',
                'action_type': ActionType.SERVICE,
                'priority': Priority.MEDIUM,
                'timeframe': Timeframe.LONG_TERM,
                'expected_impact': {'Customer Experience': 'Improve 40%', 'Conversion Rate': 'Increase 25%'},
                'implementation_steps': [
                    'Assess current personalization capabilities',
                    'Select and implement personalization platform',
                    'Develop customer journey mapping',
                    'Create personalized content and offers'
                ],
                'required_resources': ['Technology Team', 'Data Science Team', 'Marketing Team'],
                'success_metrics': ['Personalization Accuracy', 'Conversion Rate', 'Customer Satisfaction'],
                'estimated_cost': 'High',
                'estimated_roi': 'High (3-5x)'
            }
        }
        
    def generate_recommendations(self, 
                               cluster_profiles: Dict[str, Any],
                               business_insights: Dict[str, Any]) -> List[Recommendation]:
        """
        Generate actionable recommendations based on clustering results.
        
        Args:
            cluster_profiles: Detailed cluster profiles
            business_insights: Business insights analysis
            
        Returns:
            List of recommendations
        """
        logger.info("Generating business recommendations")
        
        recommendations = []
        
        # Analyze each segment and generate specific recommendations
        for cluster_id, profile in cluster_profiles.items():
            segment_recommendations = self._generate_segment_recommendations(
                cluster_id, profile
            )
            recommendations.extend(segment_recommendations)
            
        # Generate cross-segment strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(
            cluster_profiles, business_insights
        )
        recommendations.extend(strategic_recommendations)
        
        # Prioritize and deduplicate recommendations
        recommendations = self._prioritize_recommendations(recommendations)
        
        self.recommendations = recommendations
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
        
    def _generate_segment_recommendations(self, 
                                        cluster_id: str, 
                                        profile: Dict[str, Any]) -> List[Recommendation]:
        """Generate recommendations specific to a customer segment."""
        recommendations = []
        segment_id = int(cluster_id.split('_')[1])
        
        # Extract key characteristics
        characteristics = profile.get('behavioral_patterns', {})
        demographics = profile.get('demographic_insights', {})
        purchase_insights = profile.get('purchase_insights', {})
        
        # Risk-based recommendations
        if characteristics.get('risk_profile', {}):
            risk_profile = characteristics['risk_profile']
            
            # Check for high churn risk
            for risk_feature, risk_data in risk_profile.items():
                if risk_data.get('level') == 'High Risk':
                    rec = self._create_at_risk_recommendation(segment_id, risk_feature)
                    recommendations.append(rec)
                    
        # Value-based recommendations
        if demographics.get('income', {}):
            income_level = demographics['income'].get('income_level')
            
            if income_level in ['High Income', 'Upper Middle Income']:
                rec = self._create_premium_recommendation(segment_id)
                recommendations.append(rec)
            elif income_level in ['Low Income', 'Lower Middle Income']:
                rec = self._create_budget_recommendation(segment_id)
                recommendations.append(rec)
                
        # Loyalty-based recommendations
        if characteristics.get('loyalty_profile', {}):
            loyalty_profile = characteristics['loyalty_profile']
            
            high_loyalty_features = [f for f, d in loyalty_profile.items() 
                                   if d.get('level') in ['Highly Engaged', 'Very Loyal']]
            
            if high_loyalty_features:
                rec = self._create_loyalty_recommendation(segment_id)
                recommendations.append(rec)
                
        # Purchase behavior recommendations
        if purchase_insights.get('transaction_value', {}):
            value_level = purchase_insights['transaction_value'].get('value_level')
            
            if value_level in ['High Value', 'Medium-High Value']:
                rec = self._create_upsell_recommendation(segment_id)
                recommendations.append(rec)
                
        return recommendations
        
    def _create_at_risk_recommendation(self, segment_id: int, risk_feature: str) -> Recommendation:
        """Create recommendation for at-risk customers."""
        template = self.action_templates['at_risk_intervention']
        
        return Recommendation(
            id=f"at_risk_{segment_id}_{risk_feature}",
            title=f"At-Risk Intervention - Segment {segment_id}",
            description=f"Targeted intervention for Segment {segment_id} customers with high {risk_feature.replace('_', ' ')}",
            action_type=template['action_type'],
            priority=template['priority'],
            timeframe=template['timeframe'],
            target_segments=[segment_id],
            expected_impact=template['expected_impact'],
            implementation_steps=template['implementation_steps'],
            required_resources=template['required_resources'],
            success_metrics=template['success_metrics'],
            estimated_cost=template['estimated_cost'],
            estimated_roi=template['estimated_roi'],
            confidence_score=0.85
        )
        
    def _create_premium_recommendation(self, segment_id: int) -> Recommendation:
        """Create recommendation for premium customers."""
        template = self.action_templates['premium_retention']
        
        return Recommendation(
            id=f"premium_{segment_id}",
            title=f"Premium Retention - Segment {segment_id}",
            description=f"Exclusive retention program for high-value Segment {segment_id} customers",
            action_type=template['action_type'],
            priority=template['priority'],
            timeframe=template['timeframe'],
            target_segments=[segment_id],
            expected_impact=template['expected_impact'],
            implementation_steps=template['implementation_steps'],
            required_resources=template['required_resources'],
            success_metrics=template['success_metrics'],
            estimated_cost=template['estimated_cost'],
            estimated_roi=template['estimated_roi'],
            confidence_score=0.90
        )
        
    def _create_budget_recommendation(self, segment_id: int) -> Recommendation:
        """Create recommendation for budget-conscious customers."""
        template = self.action_templates['budget_segment_growth']
        
        return Recommendation(
            id=f"budget_{segment_id}",
            title=f"Budget Segment Growth - Segment {segment_id}",
            description=f"Growth strategy for budget-conscious Segment {segment_id} customers",
            action_type=template['action_type'],
            priority=template['priority'],
            timeframe=template['timeframe'],
            target_segments=[segment_id],
            expected_impact=template['expected_impact'],
            implementation_steps=template['implementation_steps'],
            required_resources=template['required_resources'],
            success_metrics=template['success_metrics'],
            estimated_cost=template['estimated_cost'],
            estimated_roi=template['estimated_roi'],
            confidence_score=0.75
        )
        
    def _create_loyalty_recommendation(self, segment_id: int) -> Recommendation:
        """Create recommendation for loyal customers."""
        template = self.action_templates['loyalty_program_enhancement']
        
        return Recommendation(
            id=f"loyalty_{segment_id}",
            title=f"Loyalty Enhancement - Segment {segment_id}",
            description=f"Enhanced loyalty program for engaged Segment {segment_id} customers",
            action_type=template['action_type'],
            priority=template['priority'],
            timeframe=template['timeframe'],
            target_segments=[segment_id],
            expected_impact=template['expected_impact'],
            implementation_steps=template['implementation_steps'],
            required_resources=template['required_resources'],
            success_metrics=template['success_metrics'],
            estimated_cost=template['estimated_cost'],
            estimated_roi=template['estimated_roi'],
            confidence_score=0.80
        )
        
    def _create_upsell_recommendation(self, segment_id: int) -> Recommendation:
        """Create recommendation for upselling opportunities."""
        template = self.action_templates['upselling_program']
        
        return Recommendation(
            id=f"upsell_{segment_id}",
            title=f"Upselling Program - Segment {segment_id}",
            description=f"Strategic upselling program for high-value Segment {segment_id} customers",
            action_type=template['action_type'],
            priority=template['priority'],
            timeframe=template['timeframe'],
            target_segments=[segment_id],
            expected_impact=template['expected_impact'],
            implementation_steps=template['implementation_steps'],
            required_resources=template['required_resources'],
            success_metrics=template['success_metrics'],
            estimated_cost=template['estimated_cost'],
            estimated_roi=template['estimated_roi'],
            confidence_score=0.80
        )
        
    def _generate_strategic_recommendations(self, 
                                          cluster_profiles: Dict[str, Any],
                                          business_insights: Dict[str, Any]) -> List[Recommendation]:
        """Generate strategic cross-segment recommendations."""
        recommendations = []
        
        # Analyze overall business insights
        insights = business_insights.get('insights', [])
        summary = business_insights.get('summary', {})
        
        # Personalization recommendation
        if len(cluster_profiles) > 3:  # Multiple segments suggest need for personalization
            template = self.action_templates['personalization_engine']
            rec = Recommendation(
                id="strategic_personalization",
                title=template['title'],
                description=template['description'],
                action_type=template['action_type'],
                priority=template['priority'],
                timeframe=template['timeframe'],
                target_segments=list(range(len(cluster_profiles))),
                expected_impact=template['expected_impact'],
                implementation_steps=template['implementation_steps'],
                required_resources=template['required_resources'],
                success_metrics=template['success_metrics'],
                estimated_cost=template['estimated_cost'],
                estimated_roi=template['estimated_roi'],
                confidence_score=0.85
            )
            recommendations.append(rec)
            
        # Revenue concentration risk
        high_risk_insights = [ins for ins in insights if ins.get('insight_type') == 'Revenue Concentration']
        if high_risk_insights:
            rec = self._create_diversification_recommendation(cluster_profiles)
            recommendations.append(rec)
            
        return recommendations
        
    def _create_diversification_recommendation(self, cluster_profiles: Dict[str, Any]) -> Recommendation:
        """Create recommendation for customer base diversification."""
        return Recommendation(
            id="strategic_diversification",
            title="Customer Base Diversification Strategy",
            description="Reduce dependency on dominant customer segments through diversification",
            action_type=ActionType.ACQUISITION,
            priority=Priority.HIGH,
            timeframe=Timeframe.MEDIUM_TERM,
            target_segments=list(range(len(cluster_profiles))),
            expected_impact={'Revenue Risk': 'Reduce 30%', 'Market Stability': 'Increase'},
            implementation_steps=[
                'Identify underrepresented market segments',
                'Develop targeted acquisition campaigns',
                'Create products for diverse customer needs',
                'Expand into new geographic or demographic markets'
            ],
            required_resources=['Marketing Team', 'Product Development', 'Market Research'],
            success_metrics=['Segment Diversity Index', 'Revenue Concentration Ratio', 'Market Share'],
            estimated_cost='High',
            estimated_roi='Medium (2-3x)',
            confidence_score=0.75
        )
        
    def _prioritize_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Prioritize recommendations based on impact and urgency."""
        # Define priority scores
        priority_scores = {
            Priority.CRITICAL: 4,
            Priority.HIGH: 3,
            Priority.MEDIUM: 2,
            Priority.LOW: 1
        }
        
        # Sort recommendations by priority and confidence
        recommendations.sort(key=lambda x: (
            priority_scores.get(x.priority, 0),
            x.confidence_score
        ), reverse=True)
        
        # Remove duplicates (similar recommendations for different segments)
        unique_recommendations = []
        seen_titles = set()
        
        for rec in recommendations:
            title_key = rec.title.split(' - ')[0]  # Remove segment-specific part
            if title_key not in seen_titles:
                unique_recommendations.append(rec)
                seen_titles.add(title_key)
                
        return unique_recommendations
        
    def create_implementation_roadmap(self) -> Dict[str, Any]:
        """Create implementation roadmap for recommendations."""
        if not self.recommendations:
            logger.warning("No recommendations available. Generate recommendations first.")
            return {}
            
        roadmap = {
            'immediate_actions': [],
            'short_term_initiatives': [],
            'medium_term_projects': [],
            'long_term_strategy': [],
            'resource_requirements': {},
            'success_metrics': {},
            'timeline_summary': {}
        }
        
        # Group recommendations by timeframe
        for rec in self.recommendations:
            rec_dict = {
                'id': rec.id,
                'title': rec.title,
                'priority': rec.priority.value,
                'target_segments': rec.target_segments,
                'estimated_cost': rec.estimated_cost,
                'expected_roi': rec.estimated_roi,
                'confidence_score': rec.confidence_score
            }
            
            if rec.timeframe == Timeframe.IMMEDIATE:
                roadmap['immediate_actions'].append(rec_dict)
            elif rec.timeframe == Timeframe.SHORT_TERM:
                roadmap['short_term_initiatives'].append(rec_dict)
            elif rec.timeframe == Timeframe.MEDIUM_TERM:
                roadmap['medium_term_projects'].append(rec_dict)
            elif rec.timeframe == Timeframe.LONG_TERM:
                roadmap['long_term_strategy'].append(rec_dict)
                
        # Aggregate resource requirements
        resource_counts = {}
        for rec in self.recommendations:
            for resource in rec.required_resources:
                resource_counts[resource] = resource_counts.get(resource, 0) + 1
                
        roadmap['resource_requirements'] = dict(sorted(resource_counts.items(), 
                                                      key=lambda x: x[1], reverse=True))
        
        # Aggregate success metrics
        metric_counts = {}
        for rec in self.recommendations:
            for metric in rec.success_metrics:
                metric_counts[metric] = metric_counts.get(metric, 0) + 1
                
        roadmap['success_metrics'] = dict(sorted(metric_counts.items(), 
                                               key=lambda x: x[1], reverse=True))
        
        # Create timeline summary
        roadmap['timeline_summary'] = {
            'total_recommendations': len(self.recommendations),
            'immediate_actions': len(roadmap['immediate_actions']),
            'short_term_initiatives': len(roadmap['short_term_initiatives']),
            'medium_term_projects': len(roadmap['medium_term_projects']),
            'long_term_strategy': len(roadmap['long_term_strategy'])
        }
        
        return roadmap
        
    def calculate_business_impact(self, 
                               cluster_profiles: Dict[str, Any],
                               baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate expected business impact of implementing recommendations."""
        impact_analysis = {
            'revenue_impact': {},
            'customer_impact': {},
            'operational_impact': {},
            'roi_projections': {}
        }
        
        # Calculate revenue impact by segment
        total_customers = sum(profile['size'] for profile in cluster_profiles.values())
        
        for rec in self.recommendations:
            for segment_id in rec.target_segments:
                cluster_key = f'cluster_{segment_id}'
                if cluster_key in cluster_profiles:
                    segment_size = cluster_profiles[cluster_key]['size']
                    segment_percentage = segment_size / total_customers
                    
                    # Estimate revenue impact based on recommendation type
                    if rec.action_type == ActionType.RETENTION:
                        # Retention programs typically reduce churn by 15-25%
                        revenue_impact = segment_percentage * 0.20  # 20% revenue protection
                    elif rec.action_type == ActionType.MARKETING:
                        # Marketing programs typically increase revenue by 10-20%
                        revenue_impact = segment_percentage * 0.15  # 15% revenue increase
                    else:
                        revenue_impact = segment_percentage * 0.10  # 10% default impact
                        
                    impact_analysis['revenue_impact'][f'segment_{segment_id}'] = revenue_impact
                    
        # Calculate customer impact
        impact_analysis['customer_impact'] = {
            'retention_improvement': '15-25%',
            'satisfaction_increase': '10-20%',
            'loyalty_growth': '20-30%'
        }
        
        # Calculate operational impact
        impact_analysis['operational_impact'] = {
            'resource_requirements': self.create_implementation_roadmap().get('resource_requirements', {}),
            'implementation_complexity': 'Medium',
            'time_to_value': '3-6 months'
        }
        
        # Calculate ROI projections
        total_investment = len(self.recommendations) * 50000  # $50k average per recommendation
        projected_annual_return = total_customers * 100 * 0.15  # $100 avg customer value * 15% improvement
        
        impact_analysis['roi_projections'] = {
            'total_investment': f"${total_investment:,.0f}",
            'projected_annual_return': f"${projected_annual_return:,.0f}",
            'payback_period': '8-12 months',
            'three_year_roi': f"{(projected_annual_return * 3 / total_investment - 1) * 100:.0f}%"
        }
        
        return impact_analysis
        
    def export_recommendations(self, output_path: str) -> None:
        """Export recommendations to JSON file."""
        from src.utils.helpers import save_json
        
        export_data = {
            'recommendations': [self._recommendation_to_dict(rec) for rec in self.recommendations],
            'implementation_roadmap': self.create_implementation_roadmap(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        save_json(export_data, output_path)
        logger.info(f"Recommendations exported to {output_path}")
        
    def _recommendation_to_dict(self, rec: Recommendation) -> Dict[str, Any]:
        """Convert recommendation to dictionary for JSON serialization."""
        return {
            'id': rec.id,
            'title': rec.title,
            'description': rec.description,
            'action_type': rec.action_type.value,
            'priority': rec.priority.value,
            'timeframe': rec.timeframe.value,
            'target_segments': rec.target_segments,
            'expected_impact': rec.expected_impact,
            'implementation_steps': rec.implementation_steps,
            'required_resources': rec.required_resources,
            'success_metrics': rec.success_metrics,
            'estimated_cost': rec.estimated_cost,
            'estimated_roi': rec.estimated_roi,
            'confidence_score': rec.confidence_score
        }
