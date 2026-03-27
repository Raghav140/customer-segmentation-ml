"""
API Client Utility for Customer Segmentation System

Handles communication with the FastAPI backend.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class CustomerSegmentationAPIClient:
    """Client for interacting with the Customer Segmentation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CustomerSegmentationApp/1.0'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health check response
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def simple_health_check(self) -> bool:
        """
        Simple health check.
        
        Returns:
            True if API is healthy
        """
        try:
            response = self.session.get(f"{self.base_url}/health/simple", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def predict_segments(self, customer_data: List[Dict[str, Any]], 
                        algorithm: str = "kmeans",
                        n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict customer segments.
        
        Args:
            customer_data: List of customer records
            algorithm: Clustering algorithm to use
            n_clusters: Number of clusters (auto-detect if None)
            
        Returns:
            Prediction response
        """
        try:
            payload = {
                "customers": customer_data,
                "algorithm": algorithm,
                "n_clusters": n_clusters
            }
            
            logger.info(f"Making prediction request for {len(customer_data)} customers")
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Prediction successful: {result.get('success', False)}")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Prediction request failed: {str(e)}")
            return {
                "success": False,
                "error": "API request failed",
                "message": str(e)
            }
    
    def analyze_clusters(self, algorithm: str = "kmeans",
                       metrics_only: bool = False) -> Dict[str, Any]:
        """
        Analyze clustering performance.
        
        Args:
            algorithm: Clustering algorithm to analyze
            metrics_only: Return only metrics without cluster details
            
        Returns:
            Analysis response
        """
        try:
            payload = {
                "algorithm": algorithm,
                "metrics_only": metrics_only
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Cluster analysis request failed: {str(e)}")
            return {
                "error": "API request failed",
                "message": str(e)
            }
    
    def get_clusters(self, algorithm: str = "kmeans") -> List[Dict[str, Any]]:
        """
        Get cluster information.
        
        Args:
            algorithm: Clustering algorithm
            
        Returns:
            List of cluster information
        """
        try:
            response = self.session.get(
                f"{self.base_url}/clusters",
                params={"algorithm": algorithm},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Get clusters request failed: {str(e)}")
            return []
    
    def predict_from_dataframe(self, df: pd.DataFrame,
                             algorithm: str = "kmeans",
                             n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict segments from pandas DataFrame.
        
        Args:
            df: Customer data DataFrame
            algorithm: Clustering algorithm
            n_clusters: Number of clusters
            
        Returns:
            Prediction response
        """
        try:
            # Convert DataFrame to list of dictionaries
            customer_data = df.to_dict('records')
            
            # Validate required columns
            required_columns = ['customer_id', 'age', 'annual_income', 'spending_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return {
                    "success": False,
                    "error": "Missing required columns",
                    "message": f"Missing columns: {missing_columns}"
                }
            
            return self.predict_segments(customer_data, algorithm, n_clusters)
            
        except Exception as e:
            logger.error(f"DataFrame prediction failed: {str(e)}")
            return {
                "success": False,
                "error": "Data processing failed",
                "message": str(e)
            }
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information and endpoints.
        
        Returns:
            API information
        """
        try:
            response = self.session.get(f"{self.base_url}/", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Get API info failed: {str(e)}")
            return {"error": str(e)}
    
    def batch_predict(self, data_batches: List[List[Dict[str, Any]]],
                     algorithm: str = "kmeans",
                     n_clusters: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            data_batches: List of customer data batches
            algorithm: Clustering algorithm
            n_clusters: Number of clusters
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, batch in enumerate(data_batches):
            try:
                result = self.predict_segments(batch, algorithm, n_clusters)
                result['batch_index'] = i
                result['batch_size'] = len(batch)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch {i} prediction failed: {str(e)}")
                results.append({
                    'batch_index': i,
                    'batch_size': len(batch),
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def validate_api_connection(self) -> Dict[str, Any]:
        """
        Validate API connection and capabilities.
        
        Returns:
            Connection validation results
        """
        validation_results = {
            "api_available": False,
            "health_status": "unknown",
            "available_endpoints": [],
            "response_time": None,
            "error": None
        }
        
        try:
            # Test basic connectivity
            start_time = datetime.now()
            health_response = self.health_check()
            end_time = datetime.now()
            
            validation_results["response_time"] = (end_time - start_time).total_seconds()
            validation_results["health_status"] = health_response.get("status", "unknown")
            validation_results["api_available"] = health_response.get("status") == "healthy"
            
            # Get available endpoints
            api_info = self.get_api_info()
            if "endpoints" in api_info:
                validation_results["available_endpoints"] = list(api_info["endpoints"].keys())
            
        except Exception as e:
            validation_results["error"] = str(e)
            logger.error(f"API validation failed: {str(e)}")
        
        return validation_results


# Global instance for easy access
api_client = CustomerSegmentationAPIClient()
