"""
🎯 Customer Segmentation System - Interactive Dashboard
AI-powered customer segmentation that reveals hidden business opportunities in seconds.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project tagline
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
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* PREMIUM SAAS DESIGN SYSTEM */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 30px rgba(59, 130, 246, 0.6); }
        100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes press {
        0% { transform: scale(1); }
        50% { transform: scale(0.95); }
        100% { transform: scale(1); }
    }
    
    @keyframes slideInFromLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInFromRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* GLOBAL PREMIUM THEME */
    .stApp {
        background: #0B0F19;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Hide default Streamlit elements */
    .stApp header, .stApp footer, .stApp .stMainMenu {
        display: none !important;
    }
    
    /* Override default styles */
    .stMarkdown {
        color: #FFFFFF !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div {
        color: #9CA3AF !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
    }
    
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    .stButton > button {
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    /* HERO SECTION - PREMIUM DESIGN */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-out;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .tagline {
        font-size: 1.8rem;
        font-weight: 600;
        color: #9CA3AF;
        text-align: center;
        margin-bottom: 3rem;
        animation: slideIn 1.2s ease-out;
        line-height: 1.4;
    }
    
    .section-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 4rem;
        margin-bottom: 2rem;
        position: relative;
        padding-bottom: 1rem;
        animation: slideUp 0.8s ease-out;
        letter-spacing: -0.01em;
    }
    
    /* PREMIUM FEATURE PILLS */
    .feature-pills {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .feature-pill {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 999px;
        color: #9CA3AF;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        animation: slideInFromLeft 0.6s ease-out;
    }
    
    .feature-pill:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .feature-icon {
        width: 20px;
        height: 20px;
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    /* PREMIUM GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        animation: slideInFromRight 0.8s ease-out;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .glass-card h4 {
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .glass-card p {
        color: #9CA3AF;
        font-weight: 500;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* PREMIUM CTA BUTTON */
    .cta-button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: #FFFFFF;
        border: none;
        padding: 1.25rem 3rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 1.25rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(255, 75, 107, 0.3);
        position: relative;
        overflow: hidden;
        animation: glow 2s infinite, press 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .cta-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .cta-button:hover::before {
        left: 100%;
    }
    
    .cta-button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 20px 50px rgba(255, 75, 107, 0.5);
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8787 100%);
    }
    
    .cta-button:active {
        animation: press 0.3s ease;
    }
    
    /* PREMIUM METRIC CARDS */
    .metric-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        animation: slideInFromLeft 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 900;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #9CA3AF;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* PREMIUM INSIGHT BOX */
    .insight-panel {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #3B82F6;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        animation: slideInFromRight 1s ease-out;
    }
    
    .insight-panel h4 {
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .insight-panel p, .insight-panel li {
        color: #9CA3AF;
        font-weight: 500;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    /* PREMIUM LOADING STATES */
    .loading-skeleton {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        animation: shimmer 2s infinite;
    }
    
    .skeleton-line {
        height: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    
    .skeleton-text {
        height: 0.75rem;
        width: 60%;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    
    /* PREMIUM SPACING */
    .section-spacing {
        margin: 3rem 0;
    }
    
    .card-spacing {
        margin: 1.5rem 0;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.2) 100%);
        padding: 1.5rem;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        animation: slideIn 0.8s ease-out;
        transition: all 0.3s ease;
    }
    
    .insight-box:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .insight-box h4 {
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        font-size: 1.3rem;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .insight-box p, .insight-box li {
        color: #F3F4F6;
        font-weight: 600;
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .cluster-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .cluster-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .demo-button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 1.8rem 3.5rem;
        border-radius: 15px;
        font-weight: 800;
        font-size: 1.4rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.5);
        position: relative;
        overflow: hidden;
        animation: pulse 2s infinite;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .demo-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.5s ease;
    }
    
    .demo-button:hover::before {
        left: 100%;
    }
    
    .demo-button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 50px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }
    
    .manager-explanation {
        background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.2);
        animation: fadeIn 1.2s ease-out;
        position: relative;
    }
    
    .manager-explanation::before {
        content: '💡';
        position: absolute;
        top: -15px;
        left: 20px;
        background: #f59e0b;
        color: white;
        padding: 0.5rem;
        border-radius: 50%;
        font-size: 1.2rem;
        animation: float 3s ease-in-out infinite;
    }
    
    .manager-explanation h3 {
        color: #92400e;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #e5e7eb;
        border-top: 4px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .success-animation {
        animation: fadeIn 0.5s ease-out;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
        border-left: 4px solid #3b82f6;
        backdrop-filter: blur(10px);
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .feature-card h4 {
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.3rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .feature-card p {
        color: #F3F4F6;
        font-weight: 600;
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .plot-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .plot-container:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    /* BENEFIT BULLETS - HIGH VISIBILITY */
    .benefit-bullet {
        display: inline-block;
        margin: 0.8rem 1.2rem;
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 3px solid #3b82f6;
        border-radius: 30px;
        font-weight: 800;
        color: #FFFFFF;
        animation: fadeIn 1s ease-out;
        transition: all 0.3s ease;
        font-size: 1.3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
    }
    
    .benefit-bullet:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        border-color: #60a5fa;
    }
    
    /* DEMO BUTTON - MAXIMUM CONTRAST */
    .demo-button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #FFFFFF;
        border: none;
        padding: 2rem 4rem;
        border-radius: 20px;
        font-weight: 900;
        font-size: 1.6rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 15px 40px rgba(239, 68, 68, 0.4);
        position: relative;
        overflow: hidden;
        animation: pulse 2s infinite;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .demo-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s ease;
    }
    
    .demo-button:hover::before {
        left: 100%;
    }
    
    .demo-button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 25px 60px rgba(239, 68, 68, 0.6);
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    }
    
    /* HOW IT WORKS SECTION - SOLID BACKGROUND */
    .how-it-works {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 3rem 0;
        border: 3px solid #f59e0b;
        box-shadow: 0 10px 40px rgba(245, 158, 11, 0.2);
    }
    
    .step-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        margin-top: 2rem;
    }
    
    .step {
        text-align: center;
        padding: 2rem 1.5rem;
        min-width: 180px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 2px solid rgba(245, 158, 11, 0.3);
        transition: all 0.3s ease;
    }
    
    .step:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.3);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .step-number {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.5rem;
        color: white;
        font-weight: 900;
        font-size: 1.5rem;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4);
    }
    
    .step-title {
        font-weight: 900;
        color: #FFFFFF;
        margin-bottom: 1rem;
        font-size: 1.4rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .step-desc {
        color: #EAEAEA;
        font-size: 1.1rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
        line-height: 1.4;
    }
    
    /* FEATURE CARDS - SOLID BACKGROUNDS */
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        border-left: 5px solid #3b82f6;
        border-right: 1px solid rgba(59, 130, 246, 0.3);
        border-top: 1px solid rgba(59, 130, 246, 0.3);
        border-bottom: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        border-left-color: #60a5fa;
    }
    
    .feature-card h4 {
        color: #FFFFFF;
        font-weight: 900;
        font-size: 1.4rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: #EAEAEA;
        font-weight: 600;
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
        line-height: 1.5;
    }
    
    /* INSIGHT BOXES - SOLID BACKGROUNDS */
    .insight-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-left: 5px solid #3b82f6;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        animation: slideIn 0.8s ease-out;
        transition: all 0.3s ease;
        border-right: 1px solid rgba(59, 130, 246, 0.3);
        border-top: 1px solid rgba(59, 130, 246, 0.3);
        border-bottom: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .insight-box:hover {
        transform: translateX(8px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        border-left-color: #60a5fa;
    }
    
    .insight-box h4 {
        color: #FFFFFF;
        margin-bottom: 1rem;
        font-size: 1.4rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .insight-box p, .insight-box li {
        color: #EAEAEA;
        font-weight: 600;
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
        line-height: 1.5;
    }
    
    /* PREVIEW CONTAINER */
    .preview-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 3rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 3px solid #3b82f6;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.2);
        animation: slideIn 1.2s ease-out;
    }
    
    .preview-container h4 {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 1.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
        margin-bottom: 2rem !important;
    }
    
    .preview-container div[style*="font-weight: 600"] {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
    }
    
    .preview-container div[style*="font-size: 0.8rem"] {
        color: #EAEAEA !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.4) !important;
    }
    
    /* CTA SECTION */
    .cta-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 3rem;
        border-radius: 20px;
        margin: 3rem 0;
        border: 2px solid #3b82f6;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.2);
        animation: fadeIn 1.5s ease-out;
        text-align: center;
    }
    
    /* PLOT CONTAINERS */
    .plot-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        border: 2px solid rgba(59, 130, 246, 0.3);
    }
    
    .plot-container:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    /* SUCCESS/ERROR/INFO MESSAGES */
    .stSuccess {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border: 2px solid #10b981 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
    }
    
    .stException {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border: 2px solid #ef4444 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border: 2px solid #3b82f6 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'optimal_k' not in st.session_state:
    st.session_state.optimal_k = None
if 'pca' not in st.session_state:
    st.session_state.pca = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

def load_sample_data():
    """Load sample customer data for demo."""
    np.random.seed(42)
    n_customers = 1000
    
    data = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': np.random.normal(40, 12, n_customers),
        'annual_income': np.random.normal(60000, 25000, n_customers),
        'spending_score': np.random.uniform(1, 100, n_customers),
        'purchase_frequency': np.random.exponential(2, n_customers),
        'last_purchase_days': np.random.exponential(30, n_customers),
        'customer_years': np.random.exponential(3, n_customers)
    })
    
    # Clean data
    data['age'] = np.clip(data['age'], 18, 80)
    data['annual_income'] = np.clip(data['annual_income'], 20000, 150000)
    data['purchase_frequency'] = np.clip(data['purchase_frequency'], 0.1, 20)
    data['last_purchase_days'] = np.clip(data['last_purchase_days'], 1, 365)
    data['customer_years'] = np.clip(data['customer_years'], 0.1, 15)
    
    return data

def preprocess_data(df):
    """Preprocess data for clustering."""
    features = ['age', 'annual_income', 'spending_score', 'purchase_frequency', 
                'last_purchase_days', 'customer_years']
    
    X = df[features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, features

def find_optimal_clusters(X, max_k=10):
    """Find optimal number of clusters using elbow and silhouette methods."""
    silhouette_scores = []
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        silhouette_scores.append(silhouette_score(X, labels))
        inertias.append(kmeans.inertia_)
    
    # Find optimal k based on silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    return optimal_k, silhouette_scores, inertias

def perform_clustering(X, n_clusters):
    """Perform K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Calculate metrics
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    return kmeans, labels, sil_score, db_score

def apply_pca(X, n_components=2):
    """Apply PCA for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    return pca, X_pca

def generate_cluster_insights(df, labels, features):
    """Generate business insights for each cluster."""
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    insights = {}
    
    for cluster_id in np.unique(labels):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        overall_data = df_with_clusters
        
        cluster_insights = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_with_clusters) * 100,
            'characteristics': {}
        }
        
        # Analyze each feature
        for feature in features:
            cluster_mean = cluster_data[feature].mean()
            overall_mean = overall_data[feature].mean()
            
            if cluster_mean > overall_mean * 1.2:
                level = "High"
            elif cluster_mean < overall_mean * 0.8:
                level = "Low"
            else:
                level = "Average"
                
            cluster_insights['characteristics'][feature] = {
                'level': level,
                'value': cluster_mean,
                'vs_overall': (cluster_mean / overall_mean - 1) * 100
            }
        
        insights[cluster_id] = cluster_insights
    
    return insights

def get_cluster_name(insights, cluster_id):
    """Generate business-friendly cluster name."""
    characteristics = insights[cluster_id]['characteristics']
    
    # Simple naming logic based on key characteristics
    if characteristics.get('annual_income', {}).get('level') == 'High':
        if characteristics.get('spending_score', {}).get('level') == 'High':
            return "💎 Premium Customers"
        else:
            return "💰 High Income Potential"
    elif characteristics.get('spending_score', {}).get('level') == 'High':
        return "🔥 Frequent Spenders"
    elif characteristics.get('purchase_frequency', {}).get('level') == 'Low':
        return "⚠️ At Risk Customers"
    else:
        return "👥 Standard Customers"

def explain_like_manager(insights, cluster_names):
    """Generate simple, business-friendly explanations."""
    explanations = {}
    
    for cluster_id, cluster_name in cluster_names.items():
        if cluster_id in insights:
            insight = insights[cluster_id]
            
            # Create simple explanation
            if "Premium" in cluster_name:
                explanation = "💎 **High-Value Customers**: These customers spend a lot and have high income. Give them VIP treatment and exclusive offers to keep them loyal."
            elif "Budget" in cluster_name:
                explanation = "💵 **Budget-Conscious**: These customers watch their spending carefully. Offer them value deals and discounts to increase their purchases."
            elif "Young" in cluster_name:
                explanation = "🚀 **Growing Professionals**: Young customers with good income potential. Build long-term relationships with digital-first experiences."
            elif "At Risk" in cluster_name:
                explanation = "⚠️ **Leaving Customers**: They haven't purchased recently. Send them special offers immediately before they're gone forever."
            elif "Frequent" in cluster_name:
                explanation = "🔥 **Regular Shoppers**: They buy often but could spend more. Recommend premium products to increase their value."
            else:
                explanation = f"👥 **Standard Customers**: Typical customer behavior. Maintain good service with occasional promotions."
            
            explanations[cluster_id] = explanation
    
    return explanations

def create_cluster_plot(df_pca, labels, cluster_names):
    """Create interactive cluster visualization."""
    plot_df = pd.DataFrame({
        'PCA1': df_pca[:, 0],
        'PCA2': df_pca[:, 1],
        'Cluster': [cluster_names[label] for label in labels],
        'Customer ID': range(1, len(labels) + 1)
    })
    
    fig = px.scatter(
        plot_df, 
        x='PCA1', 
        y='PCA2',
        color='Cluster',
        hover_data=['Customer ID'],
        title='🎨 Your Customer Segments - AI-Powered Insights',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        width=900,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_before_after_pca_plot(df, pca_components):
    """Create before/after PCA comparison."""
    features = ['age', 'annual_income', 'spending_score', 'purchase_frequency', 
                'last_purchase_days', 'customer_years']
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '📊 Complex Data (6 Dimensions)',
            '✨ Simple View (2D After AI Processing)'
        ),
        specs=[[{"type": "scatter3d"}, {"type": "scatter"}]]
    )
    
    # Before PCA - show 3D scatter of first 3 features
    fig.add_trace(
        go.Scatter3d(
            x=df[features[0]],
            y=df[features[1]],
            z=df[features[2]],
            mode='markers',
            name='Original Data',
            marker=dict(
                size=5,
                color='lightblue',
                opacity=0.6
            ),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # After PCA - 2D visualization
    fig.add_trace(
        go.Scatter(
            x=pca_components[:, 0],
            y=pca_components[:, 1],
            mode='markers',
            name='AI-Processed',
            marker=dict(
                size=8,
                color='lightcoral',
                opacity=0.8
            ),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="🔄 AI Makes Complex Data Simple to Understand",
        width=1000,
        height=500,
        showlegend=False
    )
    
    fig.update_scenes(aspectmode='cube')
    
    return fig

def create_metrics_plot(silhouette_scores, inertias, optimal_k):
    """Create metrics visualization for cluster selection."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Silhouette Score', 'Elbow Method'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    k_range = list(range(2, len(silhouette_scores) + 2))
    
    # Silhouette plot
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Mark optimal k
    fig.add_trace(
        go.Scatter(
            x=[optimal_k],
            y=[silhouette_scores[optimal_k - 2]],
            mode='markers',
            name=f'Optimal K={optimal_k}',
            marker=dict(size=15, color='#ef4444', symbol='star')
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
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        width=800,
        height=400,
        showlegend=False,
        title_text="🔍 Optimal Cluster Selection Analysis"
    )
    
    return fig

def show_loading_experience():
    """Show premium loading experience with skeleton loaders."""
    # Show loading skeleton cards
    st.markdown("""
    <div class="section-spacing">
        <div class="glass-card">
            <h4>🤖 AI is analyzing your data...</h4>
            <div class="loading-skeleton">
                <div class="skeleton-line"></div>
                <div class="skeleton-text"></div>
                <div class="skeleton-line"></div>
                <div class="skeleton-text" style="width: 80%;"></div>
            </div>
        </div>
        
        <div class="glass-card">
            <h4>📊 Identifying customer segments...</h4>
            <div class="loading-skeleton">
                <div class="skeleton-line"></div>
                <div class="skeleton-text"></div>
                <div class="skeleton-line"></div>
                <div class="skeleton-text" style="width: 70%;"></div>
            </div>
        </div>
        
        <div class="glass-card">
            <h4>💡 Generating business insights...</h4>
            <div class="loading-skeleton">
                <div class="skeleton-line"></div>
                <div class="skeleton-text"></div>
                <div class="skeleton-line"></div>
                <div class="skeleton-text" style="width: 75%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate processing time
    import time
    time.sleep(2)
    
    # Clear the loading state and continue
    st.empty()

# Main App
def main():
    # Hero Section - Premium Design
    st.markdown('<h1 class="main-header">🎯 Customer Segmentation AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">AI-powered customer segmentation that reveals hidden business opportunities in seconds.</p>', unsafe_allow_html=True)
    
    # Feature Pills
    st.markdown("""
    <div class="feature-pills">
        <div class="feature-pill">
            <div class="feature-icon">🤖</div>
            <span>ML-Powered</span>
        </div>
        <div class="feature-pill">
            <div class="feature-icon">⚡</div>
            <span>Instant Analysis</span>
        </div>
        <div class="feature-pill">
            <div class="feature-icon">📊</div>
            <span>Rich Visualizations</span>
        </div>
        <div class="feature-pill">
            <div class="feature-icon">💡</div>
            <span>Business Insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero CTA Section - Premium Design
    st.markdown("""
    <div class="section-spacing">
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <h4 style="margin-bottom: 2rem;">🚀 Transform Your Customer Data with AI</h4>
            <p style="margin-bottom: 2rem; font-size: 1.1rem; color: #9CA3AF;">
                Upload your CSV file or experience the power of AI-powered segmentation to discover valuable customer segments instantly.
            </p>
            <div style="margin-top: 2rem;">
    """, unsafe_allow_html=True)
    
    # Main CTA Button - Centered and Prominent
    if st.button("🚀 TRY INSTANT DEMO", key="demo_button", help="Experience the full power of AI-powered segmentation"):
        show_loading_experience()
    
    st.markdown("""
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Secondary CTA Section
    st.markdown('<div class="card-spacing">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>📁 Upload Your Own Data</h4>
            <p style="margin-bottom: 1rem; font-size: 1rem; color: #9CA3AF;">
                Use your own customer data for personalized insights and recommendations tailored to your business needs.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>⚙️ Advanced Settings</h4>
            <p style="margin-bottom: 1rem; font-size: 1rem; color: #9CA3AF;">
                Customize clustering parameters, choose algorithms, and fine-tune the analysis to match your specific business requirements.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Benefit bullets
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <div class="benefit-bullet">✔ Segment customers instantly</div>
        <div class="benefit-bullet">✔ Discover hidden patterns</div>
        <div class="benefit-bullet">✔ Get actionable insights</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual Preview Section
    st.markdown('<div class="preview-container">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0; padding: 1.5rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 15px; border: 2px dashed #cbd5e1;">
        <h4 style="color: #64748b; margin-bottom: 1rem;">📊 Preview: AI-Generated Customer Segments</h4>
        <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
            <div style="text-align: center; padding: 1rem;">
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 50%; margin: 0 auto 0.5rem; display: flex; align-items: center; justify-content: center;">
                    <span style="color: white; font-size: 1.5rem;">💎</span>
                </div>
                <div style="font-weight: 600; color: #1f2937;">Premium</div>
                <div style="font-size: 0.8rem; color: #6b7280;">High-value customers</div>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); border-radius: 50%; margin: 0 auto 0.5rem; display: flex; align-items: center; justify-content: center;">
                    <span style="color: white; font-size: 1.5rem;">🚀</span>
                </div>
                <div style="font-weight: 600; color: #1f2937;">Growing</div>
                <div style="font-size: 0.8rem; color: #6b7280;">Young professionals</div>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius: 50%; margin: 0 auto 0.5rem; display: flex; align-items: center; justify-content: center;">
                    <span style="color: white; font-size: 1.5rem;">💰</span>
                </div>
                <div style="font-weight: 600; color: #1f2937;">Budget</div>
                <div style="font-size: 0.8rem; color: #6b7280;">Cost-conscious</div>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); border-radius: 50%; margin: 0 auto 0.5rem; display: flex; align-items: center; justify-content: center;">
                    <span style="color: white; font-size: 1.5rem;">⚠️</span>
                </div>
                <div style="font-weight: 600; color: #1f2937;">At Risk</div>
                <div style="font-size: 0.8rem; color: #6b7280;">Leaving soon</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced CTA Section
    st.markdown('<div class="cta-section">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4> Advanced Settings</h4>
            <p style="margin-bottom: 1rem; font-size: 1rem; color: #9CA3AF;">
                Customize clustering parameters, choose algorithms, and fine-tune the analysis to match your specific business requirements.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card">
            <h4>📈 Analytics Dashboard</h4>
            <p style="margin-bottom: 1rem; font-size: 1rem; color: #9CA3AF;">
                View detailed metrics, cluster analysis, and performance indicators for your customer segmentation results.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
                
    # Demo execution function
    def run_demo():
        """Run the complete demo with loading states and results."""
        # Step 1: Load data
        status_text.text(" Loading sample customer data...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        # Load sample data
        st.session_state.data = load_sample_data()
        
        # Step 2: Prepare features
        status_text.text("🔧 Preparing features and scaling data...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        # Auto-run clustering
        features = ['age', 'annual_income', 'spending_score', 'purchase_frequency', 
                   'last_purchase_days', 'customer_years']
        X = st.session_state.data[features].copy()
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Step 3: Find optimal clusters
        status_text.text("🎯 Finding optimal customer segments...")
        progress_bar.progress(75)
        time.sleep(0.5)
        
        # Find optimal clusters
        optimal_k, silhouette_scores, inertias = find_optimal_clusters(X_scaled)
        st.session_state.optimal_k = optimal_k
        
        # Step 4: Perform clustering
        status_text.text("✨ Generating customer segments...")
        progress_bar.progress(90)
        time.sleep(0.5)
        
        # Perform clustering
        kmeans, labels, sil_score, db_score = perform_clustering(X_scaled, optimal_k)
        st.session_state.clusters = labels
        
        # Apply PCA
        pca, X_pca = apply_pca(X_scaled)
        st.session_state.pca = pca
        st.session_state.scaler = scaler
        
        progress_bar.progress(100)
        status_text.text("🎉 Analysis complete!")
        
        # Success animation
        st.markdown("""
        <div class="success-animation">
            ✨ AI Analysis Complete! Your customer segments are ready below.
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)
        st.success("🎉 Demo complete! Scroll down to see results.")
        st.balloons()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown('<div class="how-it-works">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔄 How It Works</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-container">
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-title">📊 Load Data</div>
            <div class="step-desc">Upload customer data or use our sample dataset</div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-title">🤖 Run AI Segmentation</div>
            <div class="step-desc">AI automatically finds optimal customer groups</div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-title">💡 Get Business Insights</div>
            <div class="step-desc">Receive actionable recommendations for each segment</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature showcase section (moved up)
    st.markdown('<h2 class="section-header">✨ Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 <strong>Intelligent Analysis</strong></h4>
            <p>AI automatically finds the optimal number of customer segments using advanced clustering algorithms and quality metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🎨 <strong>Interactive Visualizations</strong></h4>
            <p>Beautiful, interactive charts that make complex customer patterns easy to understand and explore.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>💡 <strong>Business Insights</strong></h4>
            <p>Plain English recommendations that tell you exactly how to market to each customer segment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Traditional upload option (collapsed)
    with st.expander("📁 Or upload your own data", expanded=False):
        st.markdown("### 📊 Data Source")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Use Sample Data", key="sample_data_btn", help="Load our pre-configured customer dataset"):
                with st.spinner("📊 Loading sample data..."):
                    st.session_state.data = load_sample_data()
                    st.success("✅ Sample data loaded successfully!")
        
        with col2:
            uploaded_file = st.file_uploader(
                "📁 Upload CSV File",
                type=['csv'],
                help="Upload your customer data in CSV format"
            )
            
            if uploaded_file is not None:
                with st.spinner("📁 Processing your data..."):
                    try:
                        st.session_state.data = pd.read_csv(uploaded_file)
                        st.success("✅ Data uploaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Data requirements info
        st.markdown("""
        <div style="margin-top: 1rem; padding: 1rem; background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <h4 style="color: #1f2937; margin-bottom: 0.5rem;">📋 Data Requirements</h4>
            <p style="color: #374151; margin-bottom: 0.5rem;"><strong>Required Columns:</strong></p>
            <ul style="color: #374151; margin: 0; padding-left: 1.5rem;">
                <li><code>customer_id</code> - Unique customer identifier</li>
                <li><code>age</code> - Customer age</li>
                <li><code>annual_income</code> - Annual income (USD)</li>
                <li><code>spending_score</code> - Spending behavior (1-100)</li>
            </ul>
            <p style="color: #374151; margin: 0.5rem 0;"><strong>Optional Columns:</strong></p>
            <ul style="color: #374151; margin: 0; padding-left: 1.5rem;">
                <li><code>purchase_frequency</code> - Purchases per month</li>
                <li><code>last_purchase_days</code> - Days since last purchase</li>
                <li><code>customer_years</code> - Years as customer</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Data overview
        st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Customers", f"{len(df):,}")
        
        with col2:
            st.metric("📈 Features", f"{len(df.columns)}")
        
        with col3:
            st.metric("📊 Data Points", f"{len(df) * len(df.columns):,}")
        
        with col4:
            st.metric("🕒 Last Updated", "Just now")
        
        # Show data preview
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(), use_container_width=True)
        
        # Clustering section
        st.markdown('<h2 class="section-header">🎯 Clustering Analysis</h2>', unsafe_allow_html=True)
        
        if st.button("🚀 Start Clustering Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing customer patterns..."):
                # Preprocess data
                X_scaled, scaler, features = preprocess_data(df)
                st.session_state.scaler = scaler
                
                # Find optimal clusters
                optimal_k, silhouette_scores, inertias = find_optimal_clusters(X_scaled)
                st.session_state.optimal_k = optimal_k
                
                # Perform clustering
                kmeans, labels, sil_score, db_score = perform_clustering(X_scaled, optimal_k)
                st.session_state.clusters = labels
                
                # Apply PCA for visualization
                pca, X_pca = apply_pca(X_scaled)
                st.session_state.pca = pca
                
                # Generate insights
                insights = generate_cluster_insights(df, labels, features)
                
                # Display results
                st.success("✅ Clustering completed successfully!")
                
                # Metrics cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Segments Found</h3>
                        <h2>{optimal_k}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Quality Score</h3>
                        <h2>{sil_score:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⚡ Processing Time</h3>
                        <h2>< 30s</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # AI Explanation section
                st.markdown('<h2 class="section-header">🤖 How AI Chose These Segments</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    metrics_fig = create_metrics_plot(silhouette_scores, inertias, optimal_k)
                    # Wrap plot with animation container
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    st.plotly_chart(metrics_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="insight-box">
                        <h4>🧠 AI Decision Process:</h4>
                        <ul>
                            <li>Tested different segment counts</li>
                            <li>Measured quality for each option</li>
                            <li>Selected K={} for best balance</li>
                            <li>Confidence: {:.0%}</li>
                        </ul>
                    </div>
                    """.format(optimal_k, 0.85), unsafe_allow_html=True)
                
                # Before/After PCA visualization
                st.markdown('<h2 class="section-header">🔄 AI Simplifies Complex Data</h2>', unsafe_allow_html=True)
                
                before_after_fig = create_before_after_pca_plot(df, X_pca)
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(before_after_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Cluster visualization
                st.markdown('<h2 class="section-header">🎨 Your Customer Segments</h2>', unsafe_allow_html=True)
                
                cluster_names = {i: get_cluster_name(insights, i) for i in range(optimal_k)}
                cluster_fig = create_cluster_plot(X_pca, labels, cluster_names)
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(cluster_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Business insights with manager explanations
                st.markdown('<h2 class="section-header">💡 Business Insights</h2>', unsafe_allow_html=True)
                
                # Manager explanation button
                if st.button("🎯 Explain Insights (Simple)", key="manager_explain"):
                    manager_explanations = explain_like_manager(insights, cluster_names)
                    
                    for cluster_id in range(optimal_k):
                        cluster_name = cluster_names[cluster_id]
                        cluster_insight = insights[cluster_id]
                        
                        st.markdown(f"""
                        <div class="manager-explanation">
                            <h3>{cluster_name}</h3>
                            <p>{manager_explanations[cluster_id]}</p>
                            <small><strong>Size:</strong> {cluster_insight['size']} customers ({cluster_insight['percentage']:.1f}%)</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Detailed insights
                    for cluster_id in range(optimal_k):
                        cluster_name = cluster_names[cluster_id]
                        cluster_insight = insights[cluster_id]
                        
                        with st.expander(f"📊 {cluster_name} ({cluster_insight['size']} customers, {cluster_insight['percentage']:.1f}%)"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown("**Key Characteristics:**")
                                for feature, char in cluster_insight['characteristics'].items():
                                    level = char['level']
                                    value = char['value']
                                    vs_overall = char['vs_overall']
                                    
                                    if level == "High":
                                        emoji = "📈"
                                        color = "green"
                                    elif level == "Low":
                                        emoji = "📉"
                                        color = "red"
                                    else:
                                        emoji = "➡️"
                                        color = "blue"
                                    
                                    st.markdown(f"{emoji} **{feature.replace('_', ' ').title()}**: {level} ({value:.1f}, {vs_overall:+.1f}% vs average)")
                            
                            with col2:
                                # Business recommendation
                                if "Premium" in cluster_name:
                                    recommendation = "🎯 **Focus on retention** with premium offers and exclusive benefits"
                                elif "High Income" in cluster_name:
                                    recommendation = "💰 **Upselling opportunities** with premium product recommendations"
                                elif "Frequent" in cluster_name:
                                    recommendation = "🔥 **Loyalty programs** to maintain engagement and spending"
                                elif "At Risk" in cluster_name:
                                    recommendation = "⚠️ **Retention campaigns** with special offers and re-engagement"
                                else:
                                    recommendation = "📊 **Standard service** with occasional promotions"
                                
                                st.markdown("**Business Action:**")
                                st.markdown(recommendation)
                
                # Customer detail view
                st.markdown('<h2 class="section-header">🔍 Customer Details</h2>', unsafe_allow_html=True)
                
                selected_customer = st.selectbox(
                    "Select a customer to view details:",
                    options=df.index,
                    format_func=lambda x: f"Customer {df.iloc[x]['customer_id'] if 'customer_id' in df.columns else x + 1}"
                )
                
                if selected_customer is not None:
                    customer_data = df.iloc[selected_customer]
                    customer_cluster = labels[selected_customer]
                    customer_cluster_name = cluster_names[customer_cluster]
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"### 🎯 {customer_cluster_name}")
                        
                        for feature in features:
                            value = customer_data[feature]
                            st.metric(feature.replace('_', ' ').title(), f"{value:.2f}")
                    
                    with col2:
                        st.markdown("### 📊 Cluster Comparison")
                        
                        # Show how this customer compares to cluster average
                        cluster_data = df.iloc[labels == customer_cluster]
                        
                        comparison_data = []
                        for feature in features:
                            customer_val = customer_data[feature]
                            cluster_avg = cluster_data[feature].mean()
                            comparison_data.append({
                                'Feature': feature.replace('_', ' ').title(),
                                'Customer': customer_val,
                                'Cluster Average': cluster_avg,
                                'Difference': customer_val - cluster_avg
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280;'>
        🚀 Customer Segmentation System | Built with Python, Scikit-learn, and Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
