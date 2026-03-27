# 🚀 Quick Setup Guide

## 📋 Prerequisites
- Python 3.9+
- pip or conda

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-segmentation-ml.git
cd customer-segmentation-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run streamlit_app.py
```

### 4. Open your browser
Navigate to http://localhost:8501

## 🎯 Try the Demo
Click **"🚀 TRY INSTANT DEMO"** to see the AI analyze customer data in seconds!

## 📁 Upload Your Own Data
You can also upload your own CSV file with customer data. Required columns:
- `customer_id`
- `age` 
- `annual_income`
- `spending_score`

Optional columns:
- `purchase_frequency`
- `last_purchase_days`
- `customer_years`

## 🔧 Environment Variables
Copy `.env.example` to `.env` and configure if needed.

## 📊 Project Structure
```
customer-segmentation-ml/
├── streamlit_app.py    # Main application
├── src/               # ML logic
├── notebooks/         # Analysis notebooks
└── requirements.txt   # Dependencies
```

## 🐳 Docker (Optional)
```bash
docker build -t customer-segmentation .
docker run -p 8501:8501 customer-segmentation
```

## 📞 Support
If you encounter any issues, please check:
1. Python version is 3.9+
2. All dependencies are installed
3. Streamlit is properly configured

Enjoy exploring your customer segments! 🎉
