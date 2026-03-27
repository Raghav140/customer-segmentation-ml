"""Interactive dashboard for business insights visualization."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from datetime import datetime

from src.config.logging_config import get_logger
from src.insights.generator import BusinessInsightsGenerator, CustomerSegment, CustomerTier, RiskLevel
from src.insights.profiler import ClusterProfiler
from src.insights.recommendations import RecommendationEngine, Recommendation

logger = get_logger(__name__)


class InsightsDashboard:
    """Interactive dashboard for visualizing business insights."""
    
    def __init__(self):
        """Initialize insights dashboard."""
        self.insights_generator = BusinessInsightsGenerator()
        self.profiler = ClusterProfiler()
        self.recommendation_engine = RecommendationEngine()
        self.dashboard_data = {}
        
    def create_comprehensive_dashboard(self, 
                                    clustering_results: Dict[str, Any],
                                    feature_names: List[str]) -> go.Figure:
        """
        Create comprehensive dashboard with all insights.
        
        Args:
            clustering_results: Results from clustering analysis
            feature_names: List of feature names
            
        Returns:
            Plotly figure with comprehensive dashboard
        """
        logger.info("Creating comprehensive insights dashboard")
        
        # Generate all insights
        insights_results = self._generate_all_insights(clustering_results, feature_names)
        self.dashboard_data = insights_results
        
        # Create dashboard with multiple subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Customer Segment Distribution',
                'Segment Value vs Risk Matrix',
                'Revenue Concentration Analysis',
                'Feature Importance',
                'Behavioral Patterns',
                'Recommendations Priority',
                'Implementation Timeline',
                'ROI Projections',
                'Success Metrics'
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "gantt"}, {"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        # 1. Customer Segment Distribution
        self._add_segment_distribution(fig, insights_results, row=1, col=1)
        
        # 2. Segment Value vs Risk Matrix
        self._add_value_risk_matrix(fig, insights_results, row=1, col=2)
        
        # 3. Revenue Concentration
        self._add_revenue_concentration(fig, insights_results, row=1, col=3)
        
        # 4. Feature Importance
        self._add_feature_importance(fig, insights_results, row=2, col=1)
        
        # 5. Behavioral Patterns
        self._add_behavioral_patterns(fig, insights_results, row=2, col=2)
        
        # 6. Recommendations Priority
        self._add_recommendations_priority(fig, insights_results, row=2, col=3)
        
        # 7. Implementation Timeline
        self._add_implementation_timeline(fig, insights_results, row=3, col=1)
        
        # 8. ROI Projections
        self._add_roi_projections(fig, insights_results, row=3, col=2)
        
        # 9. Success Metrics
        self._add_success_metrics(fig, insights_results, row=3, col=3)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Customer Segmentation - Business Insights Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=1200,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
        
    def _generate_all_insights(self, 
                             clustering_results: Dict[str, Any], 
                             feature_names: List[str]) -> Dict[str, Any]:
        """Generate all insights for dashboard."""
        # Extract clustering data
        X = clustering_results.get('X')
        cluster_labels = clustering_results.get('cluster_labels')
        cluster_centers = clustering_results.get('cluster_centers')
        
        if X is None or cluster_labels is None:
            logger.warning("Missing clustering data for insights generation")
            return {}
            
        # Generate business insights
        business_insights = self.insights_generator.analyze_clustering_results(
            X, cluster_labels, feature_names, cluster_centers
        )
        
        # Generate cluster profiles
        cluster_profiles = self.profiler.profile_clusters(
            X, cluster_labels, feature_names, cluster_centers
        )
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            cluster_profiles['cluster_profiles'], business_insights
        )
        
        # Create implementation roadmap
        roadmap = self.recommendation_engine.create_implementation_roadmap()
        
        # Calculate business impact
        baseline_metrics = {
            'total_customers': len(cluster_labels),
            'avg_revenue_per_customer': 1000,  # Placeholder
            'current_retention_rate': 0.85  # Placeholder
        }
        
        business_impact = self.recommendation_engine.calculate_business_impact(
            cluster_profiles['cluster_profiles'], baseline_metrics
        )
        
        return {
            'business_insights': business_insights,
            'cluster_profiles': cluster_profiles,
            'recommendations': recommendations,
            'roadmap': roadmap,
            'business_impact': business_impact,
            'segments': business_insights.get('segments', [])
        }
        
    def _add_segment_distribution(self, fig: go.Figure, data: Dict[str, Any], 
                                row: int, col: int) -> None:
        """Add segment distribution pie chart."""
        segments = data.get('segments', [])
        
        if not segments:
            return
            
        labels = [seg['segment_name'] for seg in segments]
        values = [seg['percentage'] for seg in segments]
        
        # Create color map based on customer tier
        colors = []
        for seg in segments:
            if seg['customer_tier'] == 'Premium':
                colors.append('#FF6B6B')
            elif seg['customer_tier'] == 'High Value':
                colors.append('#4ECDC4')
            elif seg['customer_tier'] == 'Medium Value':
                colors.append('#45B7D1')
            elif seg['customer_tier'] == 'Low Value':
                colors.append('#96CEB4')
            else:
                colors.append('#FFEAA7')
                
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name="Segment Distribution",
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='outside'
            ),
            row=row, col=col
        )
        
    def _add_value_risk_matrix(self, fig: go.Figure, data: Dict[str, Any], 
                             row: int, col: int) -> None:
        """Add value vs risk matrix scatter plot."""
        segments = data.get('segments', [])
        
        if not segments:
            return
            
        # Prepare data
        segment_names = [seg['segment_name'] for seg in segments]
        business_values = [seg['business_value'] for seg in segments]
        
        # Convert risk level to numeric
        risk_scores = []
        for seg in segments:
            if seg['risk_level'] == 'Low':
                risk_scores.append(1)
            elif seg['risk_level'] == 'Medium':
                risk_scores.append(2)
            else:  # High
                risk_scores.append(3)
                
        # Create colors based on tier
        colors = []
        for seg in segments:
            if seg['customer_tier'] == 'Premium':
                colors.append('red')
            elif seg['customer_tier'] == 'High Value':
                colors.append('orange')
            elif seg['customer_tier'] == 'Medium Value':
                colors.append('blue')
            else:
                colors.append('green')
                
        fig.add_trace(
            go.Scatter(
                x=business_values,
                y=risk_scores,
                mode='markers+text',
                text=segment_names,
                textposition='top center',
                marker=dict(
                    size=[seg['percentage'] for seg in segments],
                    color=colors,
                    opacity=0.7,
                    line=dict(width=2, color='white')
                ),
                name="Value-Risk Matrix"
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Business Value", row=row, col=col)
        fig.update_yaxes(
            title_text="Risk Level",
            ticktext=['Low', 'Medium', 'High'],
            tickvals=[1, 2, 3],
            row=row, col=col
        )
        
    def _add_revenue_concentration(self, fig: go.Figure, data: Dict[str, Any], 
                                 row: int, col: int) -> None:
        """Add revenue concentration analysis."""
        segments = data.get('segments', [])
        
        if not segments:
            return
            
        # Sort segments by size
        sorted_segments = sorted(segments, key=lambda x: x['percentage'], reverse=True)
        
        segment_names = [seg['segment_name'] for seg in sorted_segments]
        percentages = [seg['percentage'] for seg in sorted_segments]
        
        # Calculate cumulative percentage
        cumulative_percentages = np.cumsum(percentages)
        
        fig.add_trace(
            go.Bar(
                x=segment_names,
                y=percentages,
                name="Segment Size %",
                marker_color='lightblue',
                opacity=0.7
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=segment_names,
                y=cumulative_percentages,
                mode='lines+markers',
                name="Cumulative %",
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=row, col=col
        )
        
        # Add 80% threshold line
        fig.add_hline(
            y=80, line_dash="dash", line_color="orange",
            annotation_text="80% Threshold",
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Customer Segments", row=row, col=col)
        fig.update_yaxes(title_text="Percentage (%)", row=row, col=col)
        
    def _add_feature_importance(self, fig: go.Figure, data: Dict[str, Any], 
                              row: int, col: int) -> None:
        """Add feature importance chart."""
        feature_importance = data.get('business_insights', {}).get('feature_importance', {})
        
        if not feature_importance:
            return
            
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        features = [f[0].replace('_', ' ').title() for f in sorted_features]
        importance_scores = [f[1] for f in sorted_features]
        
        fig.add_trace(
            go.Bar(
                x=importance_scores,
                y=features,
                orientation='h',
                name="Feature Importance",
                marker=dict(
                    color=importance_scores,
                    colorscale='viridis',
                    showscale=True
                )
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Importance Score", row=row, col=col)
        fig.update_yaxes(title_text="Features", row=row, col=col)
        
    def _add_behavioral_patterns(self, fig: go.Figure, data: Dict[str, Any], 
                               row: int, col: int) -> None:
        """Add behavioral patterns analysis."""
        profiles = data.get('cluster_profiles', {}).get('cluster_profiles', {})
        
        if not profiles:
            return
            
        # Extract behavioral data
        behavioral_data = []
        segment_names = []
        
        for cluster_id, profile in profiles.items():
            segment_names.append(f"Segment {cluster_id.split('_')[1]}")
            
            # Calculate behavioral scores
            behavioral_score = 0
            if 'behavioral_patterns' in profile:
                patterns = profile['behavioral_patterns']
                
                # RFM score
                if 'rfm_profile' in patterns:
                    rfm_score = 0
                    for metric, data in patterns['rfm_profile'].items():
                        if data.get('classification') == 'Excellent':
                            rfm_score += 4
                        elif data.get('classification') == 'Good':
                            rfm_score += 3
                        elif data.get('classification') == 'Average':
                            rfm_score += 2
                        elif data.get('classification') == 'Poor':
                            rfm_score += 1
                    behavioral_score += rfm_score / 3  # Average across RFM
                    
            behavioral_data.append(behavioral_score)
            
        if behavioral_data:
            fig.add_trace(
                go.Bar(
                    x=segment_names,
                    y=behavioral_data,
                    name="Behavioral Score",
                    marker=dict(
                        color=behavioral_data,
                        colorscale='plasma',
                        showscale=True
                    )
                ),
                row=row, col=col
            )
            
        fig.update_xaxes(title_text="Segments", row=row, col=col)
        fig.update_yaxes(title_text="Behavioral Score", row=row, col=col)
        
    def _add_recommendations_priority(self, fig: go.Figure, data: Dict[str, Any], 
                                    row: int, col: int) -> None:
        """Add recommendations priority chart."""
        recommendations = data.get('recommendations', [])
        
        if not recommendations:
            return
            
        # Group recommendations by priority
        priority_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        
        for rec in recommendations:
            priority = getattr(rec, 'priority', 'Medium')
            if hasattr(priority, 'value'):
                priority = priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
        priorities = list(priority_counts.keys())
        counts = list(priority_counts.values())
        
        colors = ['red', 'orange', 'yellow', 'green']
        
        fig.add_trace(
            go.Bar(
                x=priorities,
                y=counts,
                name="Recommendations",
                marker=dict(colors=colors),
                text=counts,
                textposition='auto'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Priority Level", row=row, col=col)
        fig.update_yaxes(title_text="Number of Recommendations", row=row, col=col)
        
    def _add_implementation_timeline(self, fig: go.Figure, data: Dict[str, Any], 
                                   row: int, col: int) -> None:
        """Add implementation timeline Gantt chart."""
        roadmap = data.get('roadmap', {})
        
        if not roadmap:
            return
            
        # Create Gantt chart data
        tasks = []
        start_dates = []
        end_dates = []
        colors = []
        
        # Define timeframes
        timeframes = {
            'Immediate (0-30 days)': (0, 30),
            'Short Term (30-90 days)': (30, 90),
            'Medium Term (90-180 days)': (90, 180),
            'Long Term (180+ days)': (180, 365)
        }
        
        color_map = {
            'Immediate (0-30 days)': 'red',
            'Short Term (30-90 days)': 'orange',
            'Medium Term (90-180 days)': 'yellow',
            'Long Term (180+ days)': 'green'
        }
        
        timeframe_counts = {}
        for timeframe, (start, end) in timeframes.items():
            count = 0
            if timeframe == 'Immediate (0-30 days)':
                count = len(roadmap.get('immediate_actions', []))
            elif timeframe == 'Short Term (30-90 days)':
                count = len(roadmap.get('short_term_initiatives', []))
            elif timeframe == 'Medium Term (90-180 days)':
                count = len(roadmap.get('medium_term_projects', []))
            elif timeframe == 'Long Term (180+ days)':
                count = len(roadmap.get('long_term_strategy', []))
                
            if count > 0:
                tasks.append(f"{timeframe.split(' ')[0]} Actions ({count})")
                start_dates.append(start)
                end_dates.append(end)
                colors.append(color_map[timeframe])
                
        if tasks:
            fig.add_trace(
                go.Bar(
                    x=[end - start for start, end in zip(start_dates, end_dates)],
                    y=tasks,
                    orientation='h',
                    name="Timeline",
                    marker=dict(colors=colors),
                    text=[f"{start}-{end} days" for start, end in zip(start_dates, end_dates)],
                    textposition='auto'
                ),
                row=row, col=col
            )
            
        fig.update_xaxes(title_text="Duration (days)", row=row, col=col)
        fig.update_yaxes(title_text="Implementation Phases", row=row, col=col)
        
    def _add_roi_projections(self, fig: go.Figure, data: Dict[str, Any], 
                           row: int, col: int) -> None:
        """Add ROI projections chart."""
        business_impact = data.get('business_impact', {}).get('roi_projections', {})
        
        if not business_impact:
            return
            
        # Extract ROI data
        metrics = list(business_impact.keys())
        values = []
        
        for metric in metrics:
            value = business_impact[metric]
            # Extract numeric values
            if metric == 'total_investment':
                values.append(float(value.replace('$', '').replace(',', '')))
            elif metric == 'projected_annual_return':
                values.append(float(value.replace('$', '').replace(',', '')))
            elif metric == 'payback_period':
                # Convert to months
                if '8-12' in value:
                    values.append(10)
                else:
                    values.append(12)
            elif 'three_year_roi' in metric:
                values.append(float(value.replace('%', '')))
            else:
                values.append(0)
                
        # Create bar chart
        display_metrics = [m.replace('_', ' ').title() for m in metrics]
        
        fig.add_trace(
            go.Bar(
                x=display_metrics,
                y=values,
                name="ROI Metrics",
                marker=dict(
                    color=['red' if 'investment' in m.lower() else 'green' for m in display_metrics]
                ),
                text=[f"{v:,.0f}" if v > 100 else f"{v}" for v in values],
                textposition='auto'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="ROI Metrics", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)
        
    def _add_success_metrics(self, fig: go.Figure, data: Dict[str, Any], 
                           row: int, col: int) -> None:
        """Add success metrics chart."""
        roadmap = data.get('roadmap', {}).get('success_metrics', {})
        
        if not roadmap:
            return
            
        # Get top success metrics
        sorted_metrics = sorted(roadmap.items(), key=lambda x: x[1], reverse=True)[:8]
        
        metrics = [m[0].replace('_', ' ').title() for m in sorted_metrics]
        counts = [m[1] for m in sorted_metrics]
        
        fig.add_trace(
            go.Bar(
                x=counts,
                y=metrics,
                orientation='h',
                name="Success Metrics",
                marker=dict(
                    color=counts,
                    colorscale='blues',
                    showscale=True
                ),
                text=counts,
                textposition='auto'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Frequency", row=row, col=col)
        fig.update_yaxes(title_text="Success Metrics", row=row, col=col)
        
    def create_segment_deep_dive(self, 
                               segment_id: int,
                               data: Dict[str, Any]) -> go.Figure:
        """Create detailed analysis for a specific segment."""
        segments = data.get('segments', [])
        profiles = data.get('cluster_profiles', {}).get('cluster_profiles', {})
        
        # Find the segment
        segment = next((s for s in segments if s['segment_id'] == segment_id), None)
        profile = profiles.get(f'cluster_{segment_id}', {})
        
        if not segment or not profile:
            return go.Figure()
            
        # Create detailed dashboard for the segment
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Segment {segment_id}: {segment["segment_name"]}',
                'Key Characteristics',
                'Behavioral Patterns',
                'Recommendations'
            ),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # 1. Segment overview indicator
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=segment['business_value'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Business Value Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Key characteristics
        characteristics = segment.get('characteristics', [])
        if characteristics:
            fig.add_trace(
                go.Bar(
                    x=[1] * len(characteristics),
                    y=characteristics,
                    orientation='h',
                    name="Characteristics",
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            
        # 3. Behavioral patterns
        if 'behavioral_patterns' in profile:
            patterns = profile['behavioral_patterns']
            pattern_names = []
            pattern_scores = []
            
            for pattern_type, pattern_data in patterns.items():
                for metric, data in pattern_data.items():
                    pattern_names.append(f"{pattern_type}_{metric}")
                    # Convert classification to score
                    classification = data.get('classification', 'Average')
                    if classification == 'Excellent':
                        score = 4
                    elif classification == 'Good':
                        score = 3
                    elif classification == 'Average':
                        score = 2
                    elif classification == 'Poor':
                        score = 1
                    else:
                        score = 2
                    pattern_scores.append(score)
                    
            if pattern_names:
                fig.add_trace(
                    go.Bar(
                        x=pattern_scores,
                        y=[name.replace('_', ' ').title() for name in pattern_names],
                        orientation='h',
                        name="Behavioral Scores",
                        marker_color='lightgreen'
                    ),
                    row=2, col=1
                )
                
        # 4. Recommendations table
        recommendations = data.get('recommendations', [])
        segment_recs = [rec for rec in recommendations if segment_id in rec.target_segments]
        
        if segment_recs:
            rec_data = []
            for rec in segment_recs[:5]:  # Top 5 recommendations
                rec_data.append([
                    rec.title,
                    rec.priority.value if hasattr(rec.priority, 'value') else rec.priority,
                    rec.timeframe.value if hasattr(rec.timeframe, 'value') else rec.timeframe,
                    f"{rec.confidence_score:.1%}"
                ])
                
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Recommendation', 'Priority', 'Timeframe', 'Confidence'],
                        fill_color='lightblue',
                        align='left'
                    ),
                    cells=dict(
                        values=list(zip(*rec_data)) if rec_data else [[], [], [], []],
                        fill_color='white',
                        align='left'
                    )
                ),
                row=2, col=2
            )
            
        fig.update_layout(
            title=f"Segment {segment_id} Deep Dive Analysis",
            height=800,
            showlegend=False
        )
        
        return fig
        
    def export_dashboard_data(self, output_path: str) -> None:
        """Export dashboard data for external visualization."""
        from src.utils.helpers import save_json
        
        export_data = {
            'dashboard_data': self.dashboard_data,
            'export_timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        save_json(export_data, output_path)
        logger.info(f"Dashboard data exported to {output_path}")
