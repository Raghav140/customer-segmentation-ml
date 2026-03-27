"""
💡 Business Storyteller - Transforms Clusters into Actionable Business Insights
Creates compelling business narratives and recommendations for each customer segment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class BusinessAction(Enum):
    """Types of business actions for customer segments."""
    RETENTION = "retention"
    UPSELLING = "upselling"
    ACQUISITION = "acquisition"
    RE_ENGAGEMENT = "re_engagement"
    LOYALTY = "loyalty"
    COST_OPTIMIZATION = "cost_optimization"

@dataclass
class BusinessInsight:
    """Business insight for a customer segment."""
    segment_name: str
    description: str
    key_characteristics: List[str]
    recommended_actions: List[BusinessAction]
    business_opportunities: List[str]
    risk_factors: List[str]
    success_metrics: List[str]
    estimated_impact: str

class BusinessStoryteller:
    """
    Transforms technical clustering results into compelling business stories.
    
    This module bridges the gap between ML algorithms and business decision-making
    by creating human-readable insights and actionable recommendations.
    """
    
    def __init__(self):
        self.business_rules = self._load_business_rules()
        self.segment_templates = self._load_segment_templates()
        
    def generate_business_story(self, df: pd.DataFrame, labels: np.ndarray, 
                              feature_names: List[str]) -> Dict[str, BusinessInsight]:
        """
        Generate comprehensive business insights for all clusters.
        
        Args:
            df: Original customer data
            labels: Cluster assignments
            feature_names: Names of features used in clustering
            
        Returns:
            Dictionary mapping cluster IDs to business insights
        """
        print("📖 Creating business stories for each segment...")
        
        insights = {}
        
        for cluster_id in np.unique(labels):
            cluster_insight = self._analyze_single_cluster(
                df, labels, cluster_id, feature_names
            )
            insights[cluster_id] = cluster_insight
            
        print(f"✅ Generated business stories for {len(insights)} segments")
        return insights
    
    def _analyze_single_cluster(self, df: pd.DataFrame, labels: np.ndarray, 
                              cluster_id: int, feature_names: List[str]) -> BusinessInsight:
        """Analyze a single cluster and generate business insights."""
        
        # Get cluster data
        cluster_mask = labels == cluster_id
        cluster_data = df[cluster_mask]
        overall_data = df
        
        # Calculate characteristics
        characteristics = self._analyze_characteristics(
            cluster_data, overall_data, feature_names
        )
        
        # Generate segment name and description
        segment_name = self._generate_segment_name(characteristics)
        description = self._generate_segment_description(characteristics)
        
        # Determine business actions
        recommended_actions = self._determine_business_actions(characteristics)
        
        # Identify opportunities and risks
        opportunities = self._identify_opportunities(characteristics)
        risks = self._identify_risks(characteristics)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(recommended_actions)
        
        # Estimate business impact
        impact = self._estimate_business_impact(characteristics, len(cluster_data))
        
        return BusinessInsight(
            segment_name=segment_name,
            description=description,
            key_characteristics=characteristics,
            recommended_actions=recommended_actions,
            business_opportunities=opportunities,
            risk_factors=risks,
            success_metrics=success_metrics,
            estimated_impact=impact
        )
    
    def _analyze_characteristics(self, cluster_data: pd.DataFrame, 
                               overall_data: pd.DataFrame, 
                               feature_names: List[str]) -> List[str]:
        """Analyze key characteristics of a cluster."""
        characteristics = []
        
        for feature in feature_names:
            cluster_mean = cluster_data[feature].mean()
            overall_mean = overall_data[feature].mean()
            
            # Calculate relative difference
            if overall_mean != 0:
                relative_diff = (cluster_mean / overall_mean - 1) * 100
            else:
                relative_diff = 0
            
            # Generate characteristic description
            if relative_diff > 25:
                characteristics.append(f"High {feature.replace('_', ' ')} (+{relative_diff:.0f}%)")
            elif relative_diff < -25:
                characteristics.append(f"Low {feature.replace('_', ' ')} ({relative_diff:.0f}%)")
            elif 10 <= relative_diff <= 25:
                characteristics.append(f"Above-average {feature.replace('_', ' ')} (+{relative_diff:.0f}%)")
            elif -25 <= relative_diff <= -10:
                characteristics.append(f"Below-average {feature.replace('_', ' ')} ({relative_diff:.0f}%)")
        
        return characteristics
    
    def _generate_segment_name(self, characteristics: List[str]) -> str:
        """Generate a business-friendly name for the segment."""
        
        # Define naming rules based on characteristics
        char_str = ' '.join(characteristics).lower()
        
        if 'high annual_income' in char_str and 'high spending_score' in char_str:
            return "💎 Premium Customers"
        elif 'high annual_income' in char_str:
            return "💰 High-Income Potential"
        elif 'high spending_score' in char_str:
            return "🔥 Frequent Spenders"
        elif 'low purchase_frequency' in char_str or 'low last_purchase_days' in char_str:
            return "⚠️ At-Risk Customers"
        elif 'low annual_income' in char_str:
            return "💵 Budget-Conscious"
        elif 'high age' in char_str:
            return "👥 Established Families"
        elif 'low age' in char_str:
            return "🚀 Young Professionals"
        elif 'high customer_years' in char_str:
            return "🌟 Loyal Customers"
        else:
            return "📊 Standard Customers"
    
    def _generate_segment_description(self, characteristics: List[str]) -> str:
        """Generate a detailed description of the segment."""
        
        if not characteristics:
            return "Customers with average behavioral patterns across all metrics."
        
        description = "This segment consists of customers who "
        
        # Group characteristics by type
        income_related = [c for c in characteristics if 'income' in c]
        spending_related = [c for c in characteristics if 'spending' in c or 'purchase' in c]
        engagement_related = [c for c in characteristics if 'last_purchase' in c or 'customer_years' in c]
        age_related = [c for c in characteristics if 'age' in c]
        
        # Build description
        parts = []
        
        if income_related:
            parts.append(f"have {income_related[0].lower()}")
        
        if spending_related:
            if parts:
                parts.append("and")
            parts.append(f"show {spending_related[0].lower()}")
        
        if engagement_related:
            if parts:
                parts.append("with")
            parts.append(f"{engagement_related[0].lower()}")
        
        if age_related:
            if parts:
                parts.append("and are")
            parts.append(f"{age_related[0].lower()}")
        
        description += ' '.join(parts) + "."
        
        # Add behavioral insight
        if 'high' in description.lower() and 'spending' in description.lower():
            description += " They represent a high-value opportunity for targeted marketing."
        elif 'low' in description.lower() and ('purchase' in description.lower() or 'spending' in description.lower()):
            description += " They may require re-engagement strategies to prevent churn."
        
        return description
    
    def _determine_business_actions(self, characteristics: List[str]) -> List[BusinessAction]:
        """Determine recommended business actions for the segment."""
        actions = []
        char_str = ' '.join(characteristics).lower()
        
        # High-value customers - focus on retention
        if 'high annual_income' in char_str and 'high spending_score' in char_str:
            actions.extend([BusinessAction.RETENTION, BusinessAction.UPSELLING, BusinessAction.LOYALTY])
        
        # High income but low spending - upselling opportunity
        elif 'high annual_income' in char_str and 'low spending_score' in char_str:
            actions.extend([BusinessAction.UPSELLING, BusinessAction.RE_ENGAGEMENT])
        
        # Low engagement - re-engagement needed
        elif 'low purchase_frequency' in char_str or 'high last_purchase_days' in char_str:
            actions.extend([BusinessAction.RE_ENGAGEMENT, BusinessAction.RETENTION])
        
        # Budget conscious - cost optimization
        elif 'low annual_income' in char_str:
            actions.extend([BusinessAction.COST_OPTIMIZATION, BusinessAction.UPSELLING])
        
        # Young professionals - acquisition and growth
        elif 'low age' in char_str:
            actions.extend([BusinessAction.ACQUISITION, BusinessAction.UPSELLING])
        
        # Loyal customers - loyalty programs
        elif 'high customer_years' in char_str:
            actions.extend([BusinessAction.LOYALTY, BusinessAction.UPSELLING])
        
        # Default actions
        if not actions:
            actions = [BusinessAction.UPSELLING, BusinessAction.RE_ENGAGEMENT]
        
        return actions
    
    def _identify_opportunities(self, characteristics: List[str]) -> List[str]:
        """Identify business opportunities for the segment."""
        opportunities = []
        char_str = ' '.join(characteristics).lower()
        
        if 'high annual_income' in char_str:
            opportunities.append("Premium product offerings with high-margin potential")
            opportunities.append("Exclusive membership programs and VIP services")
        
        if 'high spending_score' in char_str:
            opportunities.append("Cross-selling complementary products")
            opportunities.append("Early access to new products and features")
        
        if 'low purchase_frequency' in char_str:
            opportunities.append("Re-engagement campaigns with personalized offers")
            opportunities.append("Loyalty program enrollment incentives")
        
        if 'high customer_years' in char_str:
            opportunities.append("Referral program participation")
            opportunities.append("Brand ambassador partnerships")
        
        if 'low age' in char_str:
            opportunities.append("Digital-first engagement strategies")
            opportunities.append("Social media and influencer partnerships")
        
        if 'low annual_income' in char_str:
            opportunities.append("Value-focused product bundles")
            opportunities.append("Flexible payment options and financing")
        
        return opportunities if opportunities else ["Standard customer relationship management"]
    
    def _identify_risks(self, characteristics: List[str]) -> List[str]:
        """Identify risk factors for the segment."""
        risks = []
        char_str = ' '.join(characteristics).lower()
        
        if 'low purchase_frequency' in char_str or 'high last_purchase_days' in char_str:
            risks.append("High churn risk without intervention")
            risks.append("Decreasing customer lifetime value")
        
        if 'low spending_score' in char_str:
            risks.append("Low revenue contribution per customer")
            risks.append("Vulnerability to competitor offers")
        
        if 'high annual_income' in char_str:
            risks.append("High service expectations")
            risks.append("Potential for brand switching to premium alternatives")
        
        if 'low customer_years' in char_str:
            risks.append("Low brand loyalty and attachment")
            risks.append("Higher acquisition costs to maintain relationship")
        
        return risks if risks else ["Standard business risks"]
    
    def _define_success_metrics(self, actions: List[BusinessAction]) -> List[str]:
        """Define success metrics for recommended actions."""
        metrics = []
        
        for action in actions:
            if action == BusinessAction.RETENTION:
                metrics.extend(["Customer retention rate", "Churn reduction percentage"])
            elif action == BusinessAction.UPSELLING:
                metrics.extend(["Average order value", "Cross-sell conversion rate"])
            elif action == BusinessAction.ACQUISITION:
                metrics.extend(["New customer acquisition cost", "Conversion rate"])
            elif action == BusinessAction.RE_ENGAGEMENT:
                metrics.extend(["Re-engagement campaign response rate", "Purchase frequency increase"])
            elif action == BusinessAction.LOYALTY:
                metrics.extend(["Loyalty program participation", "Customer satisfaction score"])
            elif action == BusinessAction.COST_OPTIMIZATION:
                metrics.extend(["Service cost per customer", "Operational efficiency"])
        
        return list(set(metrics))  # Remove duplicates
    
    def _estimate_business_impact(self, characteristics: List[str], segment_size: int) -> str:
        """Estimate the business impact of this segment."""
        char_str = ' '.join(characteristics).lower()
        
        # Calculate impact score
        impact_score = 0
        
        if 'high annual_income' in char_str:
            impact_score += 3
        if 'high spending_score' in char_str:
            impact_score += 3
        if 'low purchase_frequency' in char_str:
            impact_score += 2  # High impact if we can re-engage
        if 'high customer_years' in char_str:
            impact_score += 2
        if 'low annual_income' in char_str:
            impact_score += 1
        
        # Size factor
        if segment_size > 200:
            size_factor = "High"
        elif segment_size > 100:
            size_factor = "Medium"
        else:
            size_factor = "Low"
        
        # Impact level
        if impact_score >= 6:
            impact_level = "High"
        elif impact_score >= 3:
            impact_level = "Medium"
        else:
            impact_level = "Low"
        
        return f"{impact_level} impact ({size_factor} priority segment)"
    
    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business rules for insight generation."""
        return {
            'high_threshold': 1.25,  # 25% above average
            'low_threshold': 0.75,   # 25% below average
            'min_segment_size': 20,  # Minimum customers for meaningful segment
            'max_segments': 8        # Maximum segments for manageability
        }
    
    def _load_segment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load templates for different segment types."""
        return {
            'premium': {
                'actions': [BusinessAction.RETENTION, BusinessAction.UPSELLING],
                'opportunities': ['VIP programs', 'Exclusive access'],
                'risks': ['High expectations', 'Competition sensitivity']
            },
            'at_risk': {
                'actions': [BusinessAction.RE_ENGAGEMENT, BusinessAction.RETENTION],
                'opportunities': ['Win-back campaigns', 'Loyalty incentives'],
                'risks': ['Churn probability', 'Brand switching']
            },
            'growth': {
                'actions': [BusinessAction.UPSELLING, BusinessAction.ACQUISITION],
                'opportunities': ['Market expansion', 'Product development'],
                'risks': ['Investment requirements', 'Competition']
            }
        }
    
    def create_executive_summary(self, insights: Dict[int, BusinessInsight]) -> str:
        """Create C-level executive summary of all segments."""
        
        total_customers = sum(len(insight.segment_name.split()) for insight in insights.values())  # Placeholder
        
        summary = f"""
# 📊 Customer Segmentation Executive Summary

## 🎯 Overview
We've identified {len(insights)} distinct customer segments, each with unique characteristics and opportunities.

## 💼 Key Business Opportunities
"""
        
        # Find highest impact segments
        high_impact_segments = [
            insight for insight in insights.values() 
            if 'High impact' in insight.estimated_impact
        ]
        
        for insight in high_impact_segments[:3]:  # Top 3
            summary += f"""
### {insight.segment_name}
{insight.description}
**Key Actions**: {', '.join([action.value.replace('_', ' ').title() for action in insight.recommended_actions[:3]])}
**Expected Impact**: {insight.estimated_impact}
"""
        
        summary += f"""
## 📈 Strategic Recommendations
1. **Prioritize High-Impact Segments**: Focus resources on segments with highest business potential
2. **Implement Targeted Campaigns**: Develop specific marketing strategies for each segment
3. **Monitor Performance**: Track success metrics for each recommended action
4. **Optimize Continuously**: Refine strategies based on performance data

## 🎯 Next Steps
- Develop detailed action plans for each segment
- Allocate marketing budget based on segment priorities
- Implement tracking and measurement systems
- Review and adjust strategies quarterly

---
*Analysis completed on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return summary
