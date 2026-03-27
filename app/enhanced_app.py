"""
Customer Segmentation System - Enhanced Dashboard with API Integration
Production-ready customer segmentation with auto cluster selection and API integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.pipeline.model_training_pipeline import ModelTrainingPipeline
from src.utils.auto_cluster_selection import AutoClusterSelector
from src.utils.api_client import CustomerSegmentationAPIClient
from src.data.loader import DataLoader

# Set page config
st.set_page_config(
    page_title="Customer Segmentation AI - Enhanced",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with API integration styling
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.5); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.8), 0 0 30px rgba(59, 130, 246, 0.6); }
        100% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.5); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInFromLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInFromRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Global theme */
    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #111827 100%);
        color: #FFFFFF;
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6 0%, #60A5FA 50%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-out;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
    }
    
    .tagline {
        font-size: 1.4rem;
        color: #9CA3AF;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out 0.2s both;
        font-weight: 300;
    }
    
    /* Glass cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        animation: fadeIn 0.8s ease-out;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    /* Feature pills */
    .feature-pills {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .feature-pill {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        animation: slideIn 0.6s ease-out;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .feature-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3) !important;
        animation: pulse 2s infinite !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.5) !important;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8787 100%) !important;
        animation: glow 1.5s ease-in-out infinite !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        animation: slideInFromLeft 0.8s ease-out;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.9);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        animation: slideInFromRight 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(59, 130, 246, 0.6);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #3B82F6;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #9CA3AF;
        font-weight: 500;
    }
    
    /* Success animation */
    .success-animation {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        animation: slideUp 0.6s ease-out;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
    }
    
    /* API status indicator */
    .api-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .api-status.online {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .api-status.offline {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Hide default elements */
    .stDeployButton, .stHeader {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedCustomerSegmentationApp:
    """Enhanced Customer Segmentation App with API Integration."""
    
    def __init__(self):
        """Initialize the enhanced app."""
        self.pipeline = None
        self.auto_selector = None
        self.data_loader = None
        self.api_client = CustomerSegmentationAPIClient()
        self.api_available = False
        
    def check_api_status(self):
        """Check if API is available."""
        self.api_available = self.api_client.simple_health_check()
        return self.api_available
    
    def initialize_components(self):
        """Initialize ML components."""
        if self.pipeline is None:
            self.pipeline = ModelTrainingPipeline()
            self.auto_selector = AutoClusterSelector()
            self.data_loader = DataLoader()
    
    def show_api_status(self):
        """Display API status indicator."""
        api_status = self.check_api_status()
        status_class = "online" if api_status else "offline"
        status_text = "API Online" if api_status else "API Offline"
        icon = "🟢" if api_status else "🔴"
        
        st.markdown(f"""
        <div class="api-status {status_class}">
            {icon} {status_text}
        </div>
        """, unsafe_allow_html=True)
        
        return api_status
    
    def show_hero_section(self):
        """Display enhanced hero section."""
        st.markdown('<h1 class="main-header">🎯 Customer Segmentation AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="tagline">AI-powered customer segmentation with automatic cluster selection and API integration</p>', unsafe_allow_html=True)
        
        # Feature pills
        st.markdown("""
        <div class="feature-pills">
            <div class="feature-pill">🤖 Auto Cluster Selection</div>
            <div class="feature-pill">📊 Multi-Method Analysis</div>
            <div class="feature-pill">🔗 API Integration</div>
            <div class="feature-pill">💾 Model Persistence</div>
            <div class="feature-pill">🎯 Business Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
    
    def show_data_upload_section(self):
        """Display data upload section with API option."""
        st.markdown('<div class="section-header">📁 Data Source</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📤 Upload Your Data")
            uploaded_file = st.file_uploader(
                "Upload CSV file with customer data",
                type=['csv'],
                help="File should contain: customer_id, age, annual_income, spending_score, purchase_frequency, last_purchase_days, customer_years"
            )
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Data loaded: {len(data)} records")
                    st.dataframe(data.head())
                    st.session_state.uploaded_data = data
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔗 API Integration")
            
            api_status = self.show_api_status()
            
            if api_status:
                st.success("✅ API is available for predictions")
                
                if st.button("🚀 Use API for Analysis", key="use_api_button"):
                    st.session_state.use_api = True
            else:
                st.warning("⚠️ API is not available")
                st.info("📊 Using local ML pipeline instead")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def show_auto_cluster_selection(self):
        """Display auto cluster selection interface."""
        st.markdown('<div class="section-header">🎯 Auto Cluster Selection</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Selection methods
        st.subheader("🔬 Selection Methods")
        
        available_methods = ['elbow', 'silhouette', 'gap', 'hierarchical', 'business']
        selected_methods = st.multiselect(
            "Choose selection methods:",
            available_methods,
            default=['elbow', 'silhouette', 'business'],
            help="Multiple methods will be combined for optimal results"
        )
        
        # Business constraints
        st.subheader("🏢 Business Constraints")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_segments = st.slider("Max Segments", 2, 10, 6)
            min_segment_size = st.slider("Min Segment Size", 10, 500, 100)
        
        with col2:
            preferred_segments = st.multiselect(
                "Preferred Segments",
                list(range(2, 11)),
                default=[3, 4, 5]
            )
        
        with col3:
            use_business_rules = st.checkbox("Apply Business Heuristics", True)
            consider_odd_segments = st.checkbox("Prefer Odd Numbers", True)
        
        # Auto selection button
        if st.button("🎯 Run Auto Cluster Selection", key="auto_select"):
            self.run_auto_cluster_selection(selected_methods, {
                'max_segments': max_segments,
                'min_segment_size': min_segment_size,
                'preferred_segments': preferred_segments,
                'use_business_rules': use_business_rules,
                'consider_odd_segments': consider_odd_segments
            })
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run_auto_cluster_selection(self, methods, constraints):
        """Execute auto cluster selection."""
        try:
            self.initialize_components()
            
            # Get data
            if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
                data = st.session_state.uploaded_data
            else:
                data = self.data_loader.load_sample_data()
            
            # Prepare features
            from src.data.preprocessor import DataPreprocessor
            from src.features.engineer import FeatureEngineer
            
            preprocessor = DataPreprocessor()
            feature_engineer = FeatureEngineer()
            
            processed_data = preprocessor.fit_transform(data)
            features = feature_engineer.create_features(processed_data)
            numeric_features = features.select_dtypes(include=[np.number])
            numeric_features = numeric_features.fillna(numeric_features.mean())
            
            # Show loading
            with st.spinner("🎯 Running intelligent cluster selection..."):
                # Run auto selection
                selection_results = self.auto_selector.find_optimal_clusters(
                    numeric_features, 
                    methods=methods,
                    business_constraints=constraints
                )
                
                # Store results
                st.session_state.selection_results = selection_results
                st.session_state.features = numeric_features
            
            # Display results
            self.display_selection_results(selection_results)
            
        except Exception as e:
            st.error(f"❌ Auto cluster selection failed: {str(e)}")
    
    def display_selection_results(self, results):
        """Display auto cluster selection results."""
        st.markdown('<div class="success-animation">', unsafe_allow_html=True)
        st.success(f"🎉 Auto Selection Complete! Optimal clusters: {results['optimal_clusters']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Main results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['optimal_clusters']}</div>
                <div class="metric-label">Optimal Clusters</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence_pct = results['confidence_score'] * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence_pct:.1f}%</div>
                <div class="metric-label">Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(results['methods_used'])}</div>
                <div class="metric-label">Methods Used</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Method breakdown
        st.subheader("📊 Method Analysis")
        
        method_data = []
        for method, result in results['method_results'].items():
            if 'error' not in result:
                method_data.append({
                    'Method': method.capitalize(),
                    'Optimal K': result['optimal_k'],
                    'Confidence': f"{result.get('method_confidence', 0):.3f}"
                })
        
        if method_data:
            method_df = pd.DataFrame(method_data)
            st.dataframe(method_df, use_container_width=True)
        
        # Reasoning
        st.subheader("🧠 Selection Reasoning")
        st.info(results['recommendation']['reasoning'])
    
    def show_model_training(self):
        """Display model training interface."""
        st.markdown('<div class="section-header">🤖 Model Training</div>', unsafe_allow_html=True)
        
        if 'selection_results' not in st.session_state:
            st.warning("⚠️ Please run auto cluster selection first")
            return
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Check if auto cluster selection has been run
        if st.session_state.selection_results is None:
            st.info("🎯 Please run auto cluster selection first to determine the optimal number of clusters.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        optimal_k = st.session_state.selection_results['optimal_clusters']
        
        st.subheader(f"🎯 Train Models with {optimal_k} Clusters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Train K-Means", key="train_kmeans"):
                self.train_model('kmeans')
        
        with col2:
            if st.button("🌳 Train Hierarchical", key="train_hierarchical"):
                self.train_model('hierarchical')
        
        if st.button("🔄 Train Both Models", key="train_both"):
            self.train_model('both')
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def train_model(self, model_type):
        """Train the specified model."""
        try:
            self.initialize_components()
            
            features = st.session_state.features
            optimal_k = st.session_state.selection_results['optimal_clusters']
            
            with st.spinner(f"🤖 Training {model_type} model..."):
                if model_type == 'kmeans':
                    results = self.pipeline.train_kmeans_model(
                        features, n_clusters=optimal_k, save_model=True
                    )
                elif model_type == 'hierarchical':
                    results = self.pipeline.train_hierarchical_model(
                        features, n_clusters=optimal_k, save_model=True
                    )
                else:  # both
                    results = self.pipeline.train_both_models_with_auto_selection(
                        features, save_models=True
                    )
                
                st.session_state.training_results = results
            
            st.success(f"✅ {model_type.capitalize()} training completed!")
            self.display_training_results(results)
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
    
    def display_training_results(self, results):
        """Display training results."""
        st.markdown('<div class="success-animation">', unsafe_allow_html=True)
        st.success("🎉 Model Training Completed Successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Metrics display
        if 'metrics' in results:
            st.subheader("📊 Model Performance")
            
            metrics = results['metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Silhouette Score", f"{metrics.get('silhouette_score', 0):.3f}")
            
            with col2:
                st.metric("Davies-Bouldin", f"{metrics.get('davies_bouldin_index', 0):.3f}")
            
            with col3:
                st.metric("Calinski-Harabasz", f"{metrics.get('calinski_harabasz_index', 0):.1f}")
        
        # Model info
        if 'model_path' in results:
            st.subheader("💾 Model Information")
            st.info(f"Model saved to: {results['model_path']}")
    
    def show_api_predictions(self):
        """Display API prediction interface."""
        st.markdown('<div class="section-header">🔗 API Predictions</div>', unsafe_allow_html=True)
        
        if not self.check_api_status():
            st.error("❌ API is not available")
            return
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Sample data for API testing
        st.subheader("📝 Test API with Sample Data")
        
        if st.button("🚀 Test API Prediction", key="test_api"):
            self.test_api_prediction()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def test_api_prediction(self):
        """Test API prediction endpoint."""
        try:
            # Sample customer data
            sample_data = [
                {
                    "customer_id": "API_TEST_001",
                    "age": 35,
                    "annual_income": 75000.0,
                    "spending_score": 65.0,
                    "purchase_frequency": 5.0,
                    "last_purchase_days": 30,
                    "customer_years": 3.0
                },
                {
                    "customer_id": "API_TEST_002",
                    "age": 42,
                    "annual_income": 95000.0,
                    "spending_score": 85.0,
                    "purchase_frequency": 8.0,
                    "last_purchase_days": 15,
                    "customer_years": 5.0
                }
            ]
            
            with st.spinner("🔮 Making API prediction..."):
                result = self.api_client.predict_segments(sample_data, algorithm="kmeans", n_clusters=4)
            
            if result.get('success', False):
                st.success("✅ API Prediction Successful!")
                
                # Display results
                st.subheader("📊 API Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Algorithm", result['algorithm_used'])
                    st.metric("Total Customers", result['total_customers'])
                    st.metric("Clusters Found", result['n_clusters'])
                
                with col2:
                    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                    st.metric("Success", "✅" if result['success'] else "❌")
                
                # Cluster information
                if result.get('clusters'):
                    st.subheader("🎯 Cluster Information")
                    
                    for cluster in result['clusters']:
                        with st.expander(f"Cluster {cluster['cluster_id']} ({cluster['percentage']:.1f}%)"):
                            st.write(f"**Customer Count:** {cluster['customer_count']}")
                            st.write("**Characteristics:**")
                            for key, value in cluster['characteristics'].items():
                                st.write(f"- {key}: {value}")
                            
                            st.write("**Business Insights:**")
                            for insight in cluster['business_insights']:
                                st.write(f"• {insight}")
            else:
                st.error(f"❌ API Error: {result.get('error', 'Unknown error')}")
                if 'message' in result:
                    st.error(result['message'])
        
        except Exception as e:
            st.error(f"❌ API test failed: {str(e)}")
    
    def show_insights_dashboard(self):
        """Display insights dashboard."""
        st.markdown('<div class="section-header">💡 Business Insights</div>', unsafe_allow_html=True)
        
        # Check if training results exist
        if st.session_state.training_results is None:
            st.info("🎯 Please train a model first to view insights.")
            return
        
        results = st.session_state.training_results
        
        # Check if results is a dictionary and has insights
        if not isinstance(results, dict) or 'insights' not in results:
            st.info("📊 No insights available yet. Please train a model with insights generation.")
            return
        
        insights = results['insights']
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        st.subheader("🎯 Cluster Profiles")
        
        for cluster_id, insight in insights.items():
            with st.expander(f"Cluster {cluster_id}"):
                if 'characteristics' in insight:
                    st.write("**Characteristics:**")
                    for key, value in insight['characteristics'].items():
                        st.write(f"- {key}: {value}")
                
                if 'insights' in insight:
                    st.write("**Business Insights:**")
                    for insight_text in insight['insights']:
                        st.write(f"• {insight_text}")
                
                if 'actions' in insight:
                    st.write("**Recommended Actions:**")
                    for action in insight['actions']:
                        st.write(f"🎯 {action}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def main(self):
        """Main application flow."""
        # Initialize session state
        if 'use_api' not in st.session_state:
            st.session_state.use_api = False
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
        if 'selection_results' not in st.session_state:
            st.session_state.selection_results = None
        if 'features' not in st.session_state:
            st.session_state.features = None
        if 'training_results' not in st.session_state:
            st.session_state.training_results = None
        
        # Hero Section
        self.show_hero_section()
        
        # API Status
        self.show_api_status()
        
        # Data Upload Section
        self.show_data_upload_section()
        
        # Auto Cluster Selection
        self.show_auto_cluster_selection()
        
        # Model Training
        self.show_model_training()
        
        # API Predictions
        self.show_api_predictions()
        
        # Insights Dashboard
        self.show_insights_dashboard()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #9CA3AF; margin-top: 2rem;'>"
            "🎯 Customer Segmentation AI - Enhanced with Auto Selection & API Integration"
            "</div>",
            unsafe_allow_html=True
        )

# Initialize and run the app
if __name__ == "__main__":
    app = EnhancedCustomerSegmentationApp()
    app.main()
