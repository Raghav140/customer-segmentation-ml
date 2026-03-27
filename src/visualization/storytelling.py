"""
🎨 Business Storytelling Visualizations
Creates compelling, business-focused visualizations for customer segmentation.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from typing import Dict, List, Tuple, Any

class BusinessStoryteller:
    """Creates business-focused visualizations that tell compelling stories."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.business_colors = {
            'premium': '#1f77b4',
            'high_value': '#ff7f0e', 
            'standard': '#2ca02c',
            'at_risk': '#d62728',
            'potential': '#9467bd'
        }
    
    def create_cluster_story_plot(self, df: pd.DataFrame, labels: np.ndarray, 
                                cluster_names: Dict[int, str], pca_components: np.ndarray = None) -> go.Figure:
        """
        Create the main cluster visualization with business context.
        
        Args:
            df: Original customer data
            labels: Cluster assignments
            cluster_names: Business-friendly cluster names
            pca_components: PCA components for 2D visualization
            
        Returns:
            Plotly figure with interactive cluster plot
        """
        if pca_components is None:
            # Apply PCA if not provided
            pca = PCA(n_components=2, random_state=42)
            features = df.select_dtypes(include=[np.number]).columns
            X_scaled = (df[features] - df[features].mean()) / df[features].std()
            pca_components = pca.fit_transform(X_scaled)
        
        # Create plot dataframe
        plot_df = pd.DataFrame({
            'x': pca_components[:, 0],
            'y': pca_components[:, 1],
            'cluster': [cluster_names[label] for label in labels],
            'size': np.random.uniform(20, 60, len(labels))  # Varying point sizes
        })
        
        # Create the main plot
        fig = go.Figure()
        
        # Add clusters with different colors and sizes
        for i, cluster_name in enumerate(np.unique(plot_df['cluster'])):
            cluster_data = plot_df[plot_df['cluster'] == cluster_name]
            
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=cluster_name,
                marker=dict(
                    size=cluster_data['size'],
                    color=self.color_palette[i % len(self.color_palette)],
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'PC1: %{x:.2f}<br>' +
                             'PC2: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout for business presentation
        fig.update_layout(
            title={
                'text': '🎯 Customer Segments: Understanding Your Customer Universe',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#1f2937'}
            },
            xaxis_title='Principal Component 1 (Customer Behavior Pattern)',
            yaxis_title='Principal Component 2 (Value & Engagement)',
            width=900,
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font={'size': 12}
            ),
            plot_bgcolor='rgba(248, 250, 252, 0.8)',
            paper_bgcolor='white',
            hovermode='closest'
        )
        
        # Add grid for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_before_after_pca_plot(self, df: pd.DataFrame, pca_components: np.ndarray) -> go.Figure:
        """
        Create before/after PCA comparison to show dimensionality reduction benefits.
        
        Args:
            df: Original customer data
            pca_components: PCA components
            
        Returns:
            Plotly figure showing PCA transformation
        """
        features = df.select_dtypes(include=[np.number]).columns
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                '📊 Original Features (High Dimensional)',
                '✨ After PCA (2D Visualization)'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Before PCA - show correlation heatmap concept
        if len(features) >= 2:
            # Select two most correlated features for visualization
            corr_matrix = df[features].corr()
            max_corr = 0
            feat1, feat2 = features[0], features[1]
            
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    if abs(corr_matrix.iloc[i, j]) > max_corr:
                        max_corr = abs(corr_matrix.iloc[i, j])
                        feat1, feat2 = features[i], features[j]
            
            fig.add_trace(
                go.Scatter(
                    x=df[feat1],
                    y=df[feat2],
                    mode='markers',
                    name='Original Data',
                    marker=dict(
                        size=8,
                        color='lightblue',
                        opacity=0.6
                    ),
                    hovertemplate=f'{feat1}: %{{x:.2f}}<br>{feat2}: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.update_xaxes(title_text=feat1.replace('_', ' ').title(), row=1, col=1)
            fig.update_yaxes(title_text=feat2.replace('_', ' ').title(), row=1, col=1)
        
        # After PCA
        fig.add_trace(
            go.Scatter(
                x=pca_components[:, 0],
                y=pca_components[:, 1],
                mode='markers',
                name='PCA Transformed',
                marker=dict(
                    size=8,
                    color='lightcoral',
                    opacity=0.6
                ),
                hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text='Principal Component 1', row=1, col=2)
        fig.update_yaxes(title_text='Principal Component 2', row=1, col=2)
        
        fig.update_layout(
            title={
                'text': '🔄 Dimensionality Reduction: Making Complex Data Understandable',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            width=1000,
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(248, 250, 252, 0.8)'
        )
        
        return fig
    
    def create_cluster_selection_story(self, silhouette_scores: List[float], 
                                     inertias: List[float], optimal_k: int) -> go.Figure:
        """
        Create an engaging story about how we selected the optimal number of clusters.
        
        Args:
            silhouette_scores: Silhouette scores for different k values
            inertias: Inertia values for different k values
            optimal_k: The selected optimal number of clusters
            
        Returns:
            Plotly figure explaining cluster selection
        """
        k_range = list(range(2, len(silhouette_scores) + 2))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '🎯 Silhouette Score: "How Good Are Our Clusters?"',
                '📊 Elbow Method: "Where Do We Stop Adding Clusters?"',
                '📈 Cluster Quality Metrics',
                '🏆 Our Decision: Why K={optimal_k}'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Silhouette plot
        fig.add_trace(
            go.Scatter(
                x=k_range,
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=10),
                hovertemplate='K: %{x}<br>Silhouette: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Highlight optimal k
        fig.add_trace(
            go.Scatter(
                x=[optimal_k],
                y=[silhouette_scores[optimal_k - 2]],
                mode='markers',
                name=f'Optimal K={optimal_k}',
                marker=dict(size=20, color='#ef4444', symbol='star'),
                hovertemplate=f'Optimal K: {optimal_k}<br>Score: {silhouette_scores[optimal_k - 2]:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Elbow plot
        fig.add_trace(
            go.Scatter(
                x=k_range,
                y=inertias,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='#10b981', width=3),
                marker=dict(size=10),
                hovertemplate='K: %{x}<br>Inertia: %{y:.0f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Metrics comparison
        metrics_df = pd.DataFrame({
            'K': k_range,
            'Silhouette': silhouette_scores,
            'Inertia (normalized)': [(i - min(inertias)) / (max(inertias) - min(inertias)) for i in inertias]
        })
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['K'],
                y=metrics_df['Silhouette'],
                name='Silhouette Score',
                marker_color='#3b82f6',
                opacity=0.7,
                hovertemplate='K: %{x}<br>Silhouette: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Decision indicator
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=optimal_k,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Optimal Clusters"},
                gauge={
                    'axis': {'range': [None, max(k_range)]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, optimal_k], 'color': "lightgray"},
                        {'range': [optimal_k, max(k_range)], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': optimal_k
                    }
                },
                delta={'reference': max(k_range) // 2}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': '🔍 Finding the Sweet Spot: How We Chose the Perfect Number of Customer Segments',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 22}
            },
            width=1200,
            height=800,
            showlegend=False,
            plot_bgcolor='rgba(248, 250, 252, 0.8)'
        )
        
        return fig
    
    def create_business_impact_dashboard(self, df: pd.DataFrame, labels: np.ndarray,
                                       cluster_insights: Dict[str, Any]) -> go.Figure:
        """
        Create a business-focused dashboard showing the impact of customer segmentation.
        
        Args:
            df: Customer data
            labels: Cluster assignments
            cluster_insights: Business insights for each cluster
            
        Returns:
            Plotly figure with business impact visualization
        """
        # Calculate business metrics
        cluster_metrics = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = df[cluster_mask]
            
            # Calculate key business metrics
            avg_income = cluster_data['annual_income'].mean() if 'annual_income' in cluster_data else 0
            avg_spending = cluster_data['spending_score'].mean() if 'spending_score' in cluster_data else 0
            cluster_size = len(cluster_data)
            
            cluster_metrics[cluster_id] = {
                'size': cluster_size,
                'avg_income': avg_income,
                'avg_spending': avg_spending,
                'revenue_potential': avg_income * avg_spending / 100
            }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '👥 Customer Distribution',
                '💰 Revenue Potential by Segment',
                '📊 Income vs Spending Analysis',
                '🎯 Business Opportunity Matrix'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        cluster_names = list(cluster_insights.keys())
        cluster_sizes = [cluster_metrics[i]['size'] for i in range(len(cluster_names))]
        revenue_potential = [cluster_metrics[i]['revenue_potential'] for i in range(len(cluster_names))]
        
        # Customer distribution pie chart
        fig.add_trace(
            go.Pie(
                labels=cluster_names,
                values=cluster_sizes,
                name="Customer Distribution",
                marker=dict(colors=self.color_palette[:len(cluster_names)])
            ),
            row=1, col=1
        )
        
        # Revenue potential bar chart
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=revenue_potential,
                name="Revenue Potential",
                marker=dict(color=self.color_palette[:len(cluster_names)])
            ),
            row=1, col=2
        )
        
        # Income vs spending scatter
        incomes = [cluster_metrics[i]['avg_income'] for i in range(len(cluster_names))]
        spendings = [cluster_metrics[i]['avg_spending'] for i in range(len(cluster_names))]
        
        fig.add_trace(
            go.Scatter(
                x=incomes,
                y=spendings,
                mode='markers+text',
                text=cluster_names,
                textposition="top center",
                marker=dict(
                    size=[cluster_metrics[i]['size']/10 for i in range(len(cluster_names))],
                    color=self.color_palette[:len(cluster_names)],
                    line=dict(width=2, color='white')
                ),
                name="Income vs Spending"
            ),
            row=2, col=1
        )
        
        # Business opportunity matrix
        opportunity_scores = []
        for i in range(len(cluster_names)):
            # Simple opportunity scoring
            size_score = cluster_metrics[i]['size'] / max(cluster_sizes)
            revenue_score = revenue_potential[i] / max(revenue_potential)
            opportunity_scores.append((size_score + revenue_score) * 50)
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=opportunity_scores,
                name="Business Opportunity",
                marker=dict(
                    color=opportunity_scores,
                    colorscale='Viridis',
                    showscale=True
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': '💼 Business Impact: What Customer Segmentation Means for Your Bottom Line',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 22}
            },
            width=1200,
            height=800,
            showlegend=False,
            plot_bgcolor='rgba(248, 250, 252, 0.8)'
        )
        
        return fig
    
    def create_customer_journey_story(self, df: pd.DataFrame, labels: np.ndarray,
                                    cluster_names: Dict[int, str]) -> go.Figure:
        """
        Create a visualization showing the customer journey across different segments.
        
        Args:
            df: Customer data
            labels: Cluster assignments
            cluster_names: Business-friendly cluster names
            
        Returns:
            Plotly figure showing customer journey
        """
        # Create journey stages based on common customer metrics
        if 'customer_years' in df.columns and 'last_purchase_days' in df.columns:
            journey_data = []
            
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_data = df[cluster_mask]
                
                avg_tenure = cluster_data['customer_years'].mean()
                avg_recency = cluster_data['last_purchase_days'].mean()
                
                journey_data.append({
                    'segment': cluster_names[cluster_id],
                    'avg_tenure_years': avg_tenure,
                    'avg_days_since_purchase': avg_recency,
                    'engagement_level': 'High' if avg_recency < 30 else 'Medium' if avg_recency < 90 else 'Low'
                })
            
            journey_df = pd.DataFrame(journey_data)
            
            # Create bubble chart
            fig = go.Figure()
            
            for _, row in journey_df.iterrows():
                color_map = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}
                
                fig.add_trace(go.Scatter(
                    x=[row['avg_tenure_years']],
                    y=[365 - row['avg_days_since_purchase']],  # Invert for better visualization
                    mode='markers+text',
                    name=row['segment'],
                    text=[row['segment']],
                    textposition="top center",
                    marker=dict(
                        size=[30],
                        color=color_map[row['engagement_level']],
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    hovertemplate=f'<b>{row["segment"]}</b><br>' +
                                 'Avg Tenure: %{x:.1f} years<br>' +
                                 'Engagement: {row["engagement_level"]}<br>' +
                                 '<extra></extra>'
                ))
            
            fig.update_layout(
                title='🛤️ Customer Journey: Mapping Engagement Across Segments',
                xaxis_title='Average Customer Tenure (Years)',
                yaxis_title='Purchase Activity (Higher = More Recent)',
                width=900,
                height=600,
                showlegend=True,
                plot_bgcolor='rgba(248, 250, 252, 0.8)'
            )
            
            return fig
        
        # Fallback if journey data not available
        fig = go.Figure()
        fig.add_annotation(
            text="Customer journey data not available.<br>Requires 'customer_years' and 'last_purchase_days' columns.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=16)
        )
        
        return fig
    
    def create_executive_summary_plot(self, cluster_insights: Dict[str, Any]) -> go.Figure:
        """
        Create an executive summary visualization for C-level presentation.
        
        Args:
            cluster_insights: Business insights for each cluster
            
        Returns:
            Plotly figure with executive summary
        """
        # Extract key metrics for executive summary
        exec_data = []
        
        for cluster_name, insights in cluster_insights.items():
            exec_data.append({
                'segment': cluster_name,
                'size_percentage': insights.get('percentage', 0),
                'priority': 'High' if 'Premium' in cluster_name or 'High' in cluster_name else 'Medium',
                'growth_potential': 'High' if 'Potential' in cluster_name else 'Medium',
                'risk_level': 'High' if 'At Risk' in cluster_name else 'Low'
            })
        
        exec_df = pd.DataFrame(exec_data)
        
        # Create executive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '📊 Market Share by Segment',
                '🎯 Strategic Priority Matrix',
                '💰 Growth Opportunity Index',
                '⚠️ Risk Assessment'
            ),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Market share pie chart
        fig.add_trace(
            go.Pie(
                labels=exec_df['segment'],
                values=exec_df['size_percentage'],
                name="Market Share"
            ),
            row=1, col=1
        )
        
        # Strategic priority matrix
        priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
        growth_map = {'High': 3, 'Medium': 2, 'Low': 1}
        
        fig.add_trace(
            go.Scatter(
                x=[priority_map[p] for p in exec_df['priority']],
                y=[growth_map[g] for g in exec_df['growth_potential']],
                mode='markers+text',
                text=exec_df['segment'],
                textposition="top center",
                marker=dict(
                    size=exec_df['size_percentage'] / 2,
                    color=exec_df['size_percentage'],
                    colorscale='Blues',
                    showscale=True,
                    line=dict(width=2, color='white')
                ),
                name="Strategic Priority"
            ),
            row=1, col=2
        )
        
        # Growth opportunity
        fig.add_trace(
            go.Bar(
                x=exec_df['segment'],
                y=[growth_map[g] * 20 for g in exec_df['growth_potential']],
                name="Growth Potential",
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # Risk assessment
        risk_map = {'High': 3, 'Medium': 2, 'Low': 1}
        fig.add_trace(
            go.Bar(
                x=exec_df['segment'],
                y=[risk_map[r] * 15 for r in exec_df['risk_level']],
                name="Risk Level",
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': '📈 Executive Dashboard: Customer Segmentation at a Glance',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 22}
            },
            width=1200,
            height=800,
            showlegend=False,
            plot_bgcolor='rgba(248, 250, 252, 0.8)'
        )
        
        return fig
