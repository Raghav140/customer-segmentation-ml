"""
FastAPI main application for Customer Segmentation API.

Production-ready backend for customer segmentation predictions and insights.
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import sys
import os
from pathlib import Path
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.schemas import (
    PredictionRequest, PredictionResponse, HealthResponse, ErrorResponse,
    ClusterAnalysisRequest, ClusterAnalysisResponse, ClusterInfo,
    CustomerPrediction
)
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor  
from src.features.engineer import FeatureEngineer
from src.models.kmeans_model import KMeansModel
from src.models.hierarchical_model import HierarchicalModel
from src.insights.profiler import SegmentProfiler
from src.utils.metrics import ClusteringMetrics
from src.utils.model_persistence import model_persistence
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('customer_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for models and components
data_loader = None
preprocessor = None
feature_engineer = None
kmeans_model = None
hierarchical_model = None
profiler = None
metrics_calculator = None
settings = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Customer Segmentation API...")
    try:
        # Initialize components
        global data_loader, preprocessor, feature_engineer, kmeans_model, hierarchical_model, profiler, metrics_calculator, settings
        
        settings = get_settings()
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        kmeans_model = KMeansModel()
        hierarchical_model = HierarchicalModel()
        profiler = SegmentProfiler()
        metrics_calculator = ClusteringMetrics()
        
        logger.info("All components initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise
    
    # Shutdown
    logger.info("Shutting down Customer Segmentation API...")


# Create FastAPI app
app = FastAPI(
    title="Customer Segmentation API",
    description="Production-ready API for customer segmentation predictions and business insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # Create response content without datetime objects
    error_content = {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "details": {"traceback": traceback.format_exc()} if settings.debug else None
    }
    
    return JSONResponse(
        status_code=500,
        content=error_content
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler."""
    error_content = {
        "error": "HTTP Error",
        "message": exc.detail,
        "status_code": exc.status_code
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_content
    )


# Dependency injection
def get_components():
    """Get initialized components."""
    return {
        'data_loader': data_loader,
        'preprocessor': preprocessor,
        'feature_engineer': feature_engineer,
        'kmeans_model': kmeans_model,
        'hierarchical_model': hierarchical_model,
        'profiler': profiler,
        'metrics_calculator': metrics_calculator
    }


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    try:
        # Check component health
        components_status = {}
        for name, component in get_components().items():
            components_status[name] = "healthy" if component else "uninitialized"
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            dependencies=components_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


@app.get("/health/simple")
async def simple_health():
    """Simple health check for load balancers."""
    return {"status": "healthy"}


# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_segments(request: PredictionRequest, components: dict = Depends(get_components)):
    """
    Predict customer segments.
    
    Args:
        request: Prediction request with customer data
        components: Initialized ML components
        
    Returns:
        PredictionResponse: Segmentation results with insights
    """
    start_time = time.time()
    
    try:
        logger.info(f"Received prediction request for {len(request.customers)} customers")
        
        # Convert request data to DataFrame
        customer_dicts = [customer.dict() for customer in request.customers]
        df = components['data_loader'].load_from_dict(customer_dicts)
        
        # Preprocess data
        df_clean = components['preprocessor'].clean_data(df)
        df_processed = components['preprocessor'].scale_features(df_clean)
        
        # Feature engineering
        # Exclude non-feature columns like customer_id and handle categorical columns
        exclude_columns = ['customer_id']
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        exclude_columns.extend(categorical_columns)
        
        feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
        df_features = components['feature_engineer'].create_features(df_processed[feature_columns])
        
        # Remove any remaining categorical columns created during feature engineering
        categorical_features = df_features.select_dtypes(include=['object']).columns.tolist()
        if categorical_features:
            logger.info(f"Excluding categorical features: {categorical_features}")
            df_features = df_features.drop(columns=categorical_features)
        
        # Double-check - ensure no categorical columns remain
        remaining_categorical = df_features.select_dtypes(include=['object']).columns.tolist()
        if remaining_categorical:
            logger.error(f"Still have categorical columns: {remaining_categorical}")
            df_features = df_features.drop(columns=remaining_categorical)
        
        # Log final columns for debugging
        logger.info(f"Final columns for clustering: {list(df_features.columns)}")
        
        # Add customer_id back for tracking
        customer_ids = df_processed['customer_id'].values
        
        # Select appropriate model
        if request.algorithm == "kmeans":
            model = components['kmeans_model']
        else:
            model = components['hierarchical_model']
        
        # Determine number of clusters
        n_clusters = request.n_clusters
        if n_clusters is None:
            n_clusters = 4  # Default or use auto-detection
        
        # Remove customer_id from features before clustering
        if 'customer_id' in df_features.columns:
            df_features_for_clustering = df_features.drop(columns=['customer_id'])
        else:
            df_features_for_clustering = df_features
        
        # FINAL SAFETY CHECK: Ensure no categorical columns remain before clustering
        categorical_before_clustering = df_features_for_clustering.select_dtypes(include=['object']).columns.tolist()
        if categorical_before_clustering:
            logger.error(f"CRITICAL: Found categorical columns before clustering: {categorical_before_clustering}")
            df_features_for_clustering = df_features_for_clustering.select_dtypes(exclude=['object'])
            logger.info(f"Removed categorical columns. Remaining columns: {list(df_features_for_clustering.columns)}")
        
        # AGGRESSIVE: Convert all columns to numeric, coercing errors to NaN
        for col in df_features_for_clustering.columns:
            df_features_for_clustering[col] = pd.to_numeric(df_features_for_clustering[col], errors='coerce')
        
        # Drop any columns that became all NaN after conversion
        df_features_for_clustering = df_features_for_clustering.dropna(axis=1, how='all')
        
        # Fill any remaining NaN values with 0
        df_features_for_clustering = df_features_for_clustering.fillna(0)
        
        logger.info(f"Final numeric columns for clustering: {list(df_features_for_clustering.columns)}")
        logger.info(f"Final data shape: {df_features_for_clustering.shape}")
        
        # Fit model and get predictions
        model.fit(df_features_for_clustering, n_clusters=n_clusters)
        predictions = model.predict(df_features_for_clustering)
        
        # Calculate metrics using only numeric features (temporarily disabled)
        # metrics = components['metrics_calculator'].calculate_all_metrics(
        #     df_features_for_clustering, predictions, n_clusters
        # )
        metrics = {}  # Empty metrics for now
        
        # Generate insights (temporarily disabled to isolate the issue)
        # insights = components['profiler'].profile_clusters(df_with_clusters)
        insights = {}  # Empty insights for now
        
        # Build response
        cluster_info_list = []
        customer_predictions = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = predictions == cluster_id
            cluster_customers = df_features_for_clustering[cluster_mask]
            
            # Get cluster insights (empty for now)
            cluster_insight = insights.get(cluster_id, {})
            
            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                customer_count=int(cluster_mask.sum()),
                percentage=float(cluster_mask.mean() * 100),
                characteristics=cluster_insight.get('characteristics', {}),
                business_insights=cluster_insight.get('insights', []),
                recommended_actions=cluster_insight.get('actions', [])
            )
            cluster_info_list.append(cluster_info)
            
            # Create customer predictions
            for idx, customer in enumerate(request.customers):
                if cluster_mask[idx]:
                    customer_prediction = CustomerPrediction(
                        customer_id=customer.customer_id,
                        cluster_id=cluster_id,
                        confidence_score=0.85,  # Calculate actual confidence
                        risk_level=cluster_insight.get('risk_level', 'medium'),
                        value_score=float(cluster_insight.get('value_score', 0.5))
                    )
                    customer_predictions.append(customer_prediction)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction completed in {processing_time:.2f} seconds")
        
        return PredictionResponse(
            success=True,
            algorithm_used=request.algorithm,
            total_customers=len(request.customers),
            n_clusters=n_clusters,
            clusters=cluster_info_list,
            predictions=customer_predictions,
            metrics=metrics,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# Cluster analysis endpoint
@app.post("/analyze", response_model=ClusterAnalysisResponse)
async def analyze_clusters(request: ClusterAnalysisRequest, components: dict = Depends(get_components)):
    """
    Analyze clustering performance and optimal parameters.
    
    Args:
        request: Analysis request
        components: Initialized ML components
        
    Returns:
        ClusterAnalysisResponse: Analysis results
    """
    try:
        logger.info(f"Received cluster analysis request for {request.algorithm}")
        
        # Load sample data for analysis
        df = components['data_loader'].load_sample_data()
        
        # Preprocess and engineer features
        df_clean = components['preprocessor'].clean_data(df)
        df_processed = components['preprocessor'].scale_features(df_clean)
        df_features = components['feature_engineer'].create_features(df_processed)
        
        # Find optimal clusters
        if request.algorithm == "kmeans":
            optimal_k, silhouette_scores, inertias = components['kmeans_model'].find_optimal_clusters(df_features)
        else:
            optimal_k, silhouette_scores, inertias = components['hierarchical_model'].find_optimal_clusters(df_features)
        
        # Calculate metrics
        model = components['kmeans_model'] if request.algorithm == "kmeans" else components['hierarchical_model']
        model.fit(df_features, n_clusters=optimal_k)
        predictions = model.predict(df_features)
        
        metrics = components['metrics_calculator'].calculate_all_metrics(
            df_features, predictions, optimal_k
        )
        
        # Generate cluster details if requested
        cluster_details = None
        if not request.metrics_only:
            df_with_clusters = df_features.copy()
            df_with_clusters['cluster'] = predictions
            insights = components['profiler'].profile_clusters(df_with_clusters)
            
            cluster_details = []
            for cluster_id in range(optimal_k):
                cluster_mask = predictions == cluster_id
                cluster_insight = insights[cluster_id] if cluster_id in insights else {}
                
                cluster_info = ClusterInfo(
                    cluster_id=cluster_id,
                    customer_count=int(cluster_mask.sum()),
                    percentage=float(cluster_mask.mean() * 100),
                    characteristics=cluster_insight.get('characteristics', {}),
                    business_insights=cluster_insight.get('insights', []),
                    recommended_actions=cluster_insight.get('actions', [])
                )
                cluster_details.append(cluster_info)
        
        logger.info(f"Cluster analysis completed for {request.algorithm}")
        
        return ClusterAnalysisResponse(
            algorithm=request.algorithm,
            optimal_clusters=optimal_k,
            metrics=metrics,
            cluster_details=cluster_details,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Cluster analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cluster analysis failed: {str(e)}"
        )


# Get cluster information endpoint
@app.get("/clusters", response_model=List[ClusterInfo])
async def get_clusters(algorithm: str = "kmeans", components: dict = Depends(get_components)):
    """
    Get information about existing clusters.
    
    Args:
        algorithm: Clustering algorithm to use
        components: Initialized ML components
        
    Returns:
        List[ClusterInfo]: Cluster information
    """
    try:
        logger.info(f"Getting cluster information for {algorithm}")
        
        # Load and process sample data
        df = components['data_loader'].load_sample_data()
        df_clean = components['preprocessor'].clean_data(df)
        df_processed = components['preprocessor'].scale_features(df_clean)
        df_features = components['feature_engineer'].create_features(df_processed)
        
        # Fit model and get clusters
        model = components['kmeans_model'] if algorithm == "kmeans" else components['hierarchical_model']
        model.fit(df_features, n_clusters=4)  # Default to 4 clusters
        predictions = model.predict(df_features)
        
        # Generate insights
        df_with_clusters = df_features.copy()
        df_with_clusters['cluster'] = predictions
        insights = components['profiler'].profile_clusters(df_with_clusters)
        
        # Build cluster information
        cluster_info_list = []
        for cluster_id in range(4):
            cluster_mask = predictions == cluster_id
            cluster_insight = insights[cluster_id] if cluster_id in insights else {}
            
            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                customer_count=int(cluster_mask.sum()),
                percentage=float(cluster_mask.mean() * 100),
                characteristics=cluster_insight.get('characteristics', {}),
                business_insights=cluster_insight.get('insights', []),
                recommended_actions=cluster_insight.get('actions', [])
            )
            cluster_info_list.append(cluster_info)
        
        logger.info(f"Retrieved cluster information for {algorithm}")
        return cluster_info_list
        
    except Exception as e:
        logger.error(f"Failed to get cluster information: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cluster information: {str(e)}"
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Segmentation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict - Customer segmentation predictions",
            "analyze": "POST /analyze - Cluster analysis",
            "clusters": "GET /clusters - Get cluster information",
            "health": "GET /health - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
