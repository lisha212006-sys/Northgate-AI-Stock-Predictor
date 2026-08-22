import os
import sys
import base64
import streamlit as st

# Import global dark theme
from theme import apply_dark_theme

# --- Page Setup ---
st.set_page_config(
    page_title="About | Northgate AI",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global dark theme first
apply_dark_theme()

# --- Image Encoding & Background Loader ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Resolve absolute path to assets/stocks.jpg
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, 'assets', 'stocks.jpg')

if os.path.exists(img_path):
    bin_str = get_base64_of_bin_file(img_path)
    # Applied dark overlay (88% opacity) so text stays clear and readable
    bg_css = f'linear-gradient(rgba(14, 17, 23, 0.88), rgba(14, 17, 23, 0.88)), url("data:image/jpeg;base64,{bin_str}")'
    
    st.markdown(f"""
        <style>
            .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
                background: {bg_css} !important;
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}
            
            [data-testid="stMain"] {{
                background: transparent !important;
            }}
        </style>
    """, unsafe_allow_html=True)

# --- About Page Content ---
st.title("🚀 Northgate AI Stock Predictor")
st.subheader("An End-to-End Quantitative Trading & AI Forecasting System")

st.markdown("---")

# Key Value Highlights
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-header">MACHINE LEARNING & DL</div>
            <div class="metric-value">8 Models</div>
            <div style="color: #8B949E; font-size: 0.85rem;">XGBoost, Random Forest, LSTM, BiLSTM, GRU & Transformers</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-header">ALTERNATIVE DATA</div>
            <div class="metric-value">FinBERT NLP</div>
            <div style="color: #8B949E; font-size: 0.85rem;">Financial news sentiment analysis for market regime signals</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-header">PORTFOLIO OPTIMIZATION</div>
            <div class="metric-value">Max-Sharpe MPT</div>
            <div style="color: #8B949E; font-size: 0.85rem;">SLSQP Efficient Frontier & Monte Carlo simulation</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# System Architecture & Pipeline Layers
st.subheader("🏗️ System Architecture (7-Layer Pipeline)")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    * **Layer 1: Data Ingestion & Features**
      * Fetches historical OHLCV data across equity tickers, S&P 500 (`^GSPC`), and Volatility Index (`^VIX`).
      * Computes technical indicators (RSI, MACD, Bollinger Bands, Moving Averages).
    
    * **Layer 2: Alternative Data & NLP**
      * Scraping & processing financial news headlines.
      * Scoring sentiment using **FinBERT** transformer models to extract market tone.

    * **Layer 3: Classical Machine Learning**
      * Trains Ridge, SVR, Random Forest, and XGBoost models using `TimeSeriesSplit` cross-validation.
    
    * **Layer 4: Deep Learning Architectures**
      * Sequence modeling using PyTorch with LSTM, BiLSTM, GRU, and Transformer encoders.
      * EarlyStopping regularization to prevent overfitting time-series noise.
    """)

with col_b:
    st.markdown("""
    * **Layer 5: Modern Portfolio Theory (MPT)**
      * Constrained optimization (SLSQP) to find maximum Sharpe Ratio portfolio weights.
      * Monte Carlo simulation for efficient frontier risk mapping.

    * **Layer 6: Signal Fusion & Trade Execution**
      * Merges AI predicted expected returns with MPT target portfolio allocations.
      * Computes precise buy/sell dollar orders based on a reference portfolio.

    * **Layer 7: Interactive Presentation Layer**
      * Cached interactive Streamlit interface displaying real-time metrics, loss curves, and trade execution orders.
    """)