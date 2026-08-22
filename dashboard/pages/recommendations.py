import sys
import os
import pandas as pd
import streamlit as st

# Fix sys.path so Python can find 'src' from within subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Trade Recommendations", page_icon="🎯", layout="wide")

REPORTS_DIR = "reports"


@st.cache_data
def load_report_csv(file_name):
    path = os.path.join(REPORTS_DIR, file_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


st.title("🎯 Section 12: Recommendation & Rebalancing Engine")
st.markdown("Transparent signal fusion rules combined with exact dollar rebalancing trade orders.")

rec_df = load_report_csv("recommendations.csv")
rebalance_df = load_report_csv("rebalancing_orders.csv")

# 1. Asset Signals Section
st.subheader("Model Signals & Forecasted Returns")
if rec_df is not None:
    cols = st.columns(len(rec_df))
    for idx, row in rec_df.iterrows():
        with cols[idx]:
            signal = row["Recommendation Signal"]
            st.metric(
                label=f"Asset: {row['Ticker']}",
                value=f"{row['Expected Return (%)']}%",
                delta=f"Signal: {signal}",
                delta_color="green" if signal == "BUY" else ("red" if signal == "SELL" else "off")
            )

    st.markdown("---")
    st.subheader("Transparent Rationale Breakdown")
    st.dataframe(rec_df, use_container_width=True)
else:
    st.warning("`recommendations.csv` not found. Run `python src/recommend.py` first.")

st.markdown("---")

# 2. Rebalancing Execution Section
st.subheader("Portfolio Execution Orders ($100,000 Portfolio Base)")
if rebalance_df is not None:
    st.dataframe(rebalance_df, use_container_width=True)
else:
    st.warning("`rebalancing_orders.csv` not found. Run `python src/recommend.py` first.")