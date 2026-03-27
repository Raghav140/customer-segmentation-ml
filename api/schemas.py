"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np


class CustomerData(BaseModel):
    """Individual customer data schema."""
    customer_id: str = Field(..., description="Unique customer identifier")
    age: int = Field(..., ge=18, le=100, description="Customer age")
    annual_income: float = Field(..., ge=0, description="Annual income in USD")
    spending_score: float = Field(..., ge=0, le=100, description="Spending behavior score (1-100)")
    purchase_frequency: Optional[float] = Field(None, ge=0, description="Purchases per month")
    last_purchase_days: Optional[int] = Field(None, ge=0, description="Days since last purchase")
    customer_years: Optional[float] = Field(None, ge=0, description="Years as customer")
    
    @validator('annual_income')
    def validate_income(cls, v):
        if v <= 0:
            raise ValueError('Annual income must be positive')
        return v
    
    @validator('spending_score')
    def validate_spending_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Spending score must be between 0 and 100')
        return v


class PredictionRequest(BaseModel):
    """Request schema for customer segmentation prediction."""
    customers: List[CustomerData] = Field(..., min_items=1, description="List of customer data")
    algorithm: str = Field(default="kmeans", description="Clustering algorithm to use")
    n_clusters: Optional[int] = Field(None, ge=2, le=10, description="Number of clusters (auto-detect if None)")
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        if v not in ['kmeans', 'hierarchical']:
            raise ValueError('Algorithm must be either "kmeans" or "hierarchical"')
        return v


class ClusterInfo(BaseModel):
    """Cluster information schema."""
    cluster_id: int = Field(..., description="Cluster identifier")
    customer_count: int = Field(..., description="Number of customers in cluster")
    percentage: float = Field(..., description="Percentage of total customers")
    characteristics: Dict[str, Any] = Field(..., description="Cluster characteristics")
    business_insights: List[str] = Field(..., description="Business insights for this cluster")
    recommended_actions: List[str] = Field(..., description="Recommended business actions")


class CustomerPrediction(BaseModel):
    """Prediction result for individual customer."""
    customer_id: str = Field(..., description="Customer identifier")
    cluster_id: int = Field(..., description="Assigned cluster")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")
    risk_level: str = Field(..., description="Customer risk level")
    value_score: float = Field(..., ge=0, description="Customer value score")


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    success: bool = Field(..., description="Prediction success status")
    algorithm_used: str = Field(..., description="Algorithm used for prediction")
    total_customers: int = Field(..., description="Total number of customers processed")
    n_clusters: int = Field(..., description="Number of clusters found")
    clusters: List[ClusterInfo] = Field(..., description="Cluster information")
    predictions: List[CustomerPrediction] = Field(..., description="Individual predictions")
    metrics: Dict[str, float] = Field(..., description="Clustering quality metrics")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class ClusterAnalysisRequest(BaseModel):
    """Request schema for cluster analysis."""
    algorithm: str = Field(default="kmeans", description="Clustering algorithm to analyze")
    metrics_only: bool = Field(default=False, description="Return only metrics without cluster details")


class ClusterAnalysisResponse(BaseModel):
    """Response schema for cluster analysis."""
    algorithm: str = Field(..., description="Algorithm analyzed")
    optimal_clusters: int = Field(..., description="Optimal number of clusters")
    metrics: Dict[str, float] = Field(..., description="Quality metrics")
    cluster_details: Optional[List[ClusterInfo]] = Field(None, description="Cluster details if requested")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
