import sys
import os
import pandas as pd
import streamlit as st
from PIL import Image

# Fix sys.path so Python can find 'src' from within subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Portfolio Optimization", page_icon="⚖️", layout="wide")

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


@st.cache_data
def load_report_csv(file_name):
    path = os.path.join(REPORTS_DIR, file_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


st.title("⚖️ Section 11: Modern Portfolio Theory (MPT)")
st.markdown("Optimization using SLSQP Max-Sharpe allocation, Monte Carlo cross-validation, and baseline backtesting.")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Optimal Max-Sharpe Weights")
    weights_df = load_report_csv("portfolio_weights.csv")
    
    if weights_df is not None:
        st.dataframe(weights_df, use_container_width=True)
        
        st.subheader("Asset Allocations Bar Chart")
        chart_data = weights_df.set_index("Ticker")[["Max-Sharpe Weight (SLSQP)"]]
        st.bar_chart(chart_data)
    else:
        st.warning("`portfolio_weights.csv` not found. Run `python src/portfolio.py` first.")

with col2:
    st.subheader("Efficient Frontier & Monte Carlo Plot")
    ef_img_path = os.path.join(FIGURES_DIR, "efficient_frontier.png")
    
    if os.path.exists(ef_img_path):
        image = Image.open(ef_img_path)
        st.image(image, caption="Efficient Frontier & Simulated Portfolios", use_column_width=True)
    else:
        st.info("Efficient Frontier image not found in `reports/figures/`.")