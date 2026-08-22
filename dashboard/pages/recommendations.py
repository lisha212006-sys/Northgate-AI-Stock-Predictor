import sys
import os
import pandas as pd
import streamlit as st

# Fix sys.path to find modules in parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import global dark theme module
from theme import apply_dark_theme

# Page Setup
st.set_page_config(page_title="Trade Recommendations", page_icon="🎯", layout="wide")

# Apply Global Dark Mode Styling
apply_dark_theme()

REPORTS_DIR = "reports"


@st.cache_data
def load_report_csv(file_name):
    path = os.path.join(REPORTS_DIR, file_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


st.title("🎯 Recommendation & Rebalancing Engine")
st.markdown("Transparent signal fusion rules combined with exact dollar rebalancing trade orders.")

rec_df = load_report_csv("recommendations.csv")
rebalance_df = load_report_csv("rebalancing_orders.csv")
weights_df = load_report_csv("portfolio_weights.csv")

# ==========================================
# 1. ASSET SIGNALS & METRIC CARDS
# ==========================================
st.subheader("Model Signals & Forecasted Returns")
if rec_df is not None:
    # Responsive Grid Layout: 4 Cards per Row
    cols_per_row = 4
    rows = range(0, len(rec_df), cols_per_row)
    
    for row_idx in rows:
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            item_idx = row_idx + j
            if item_idx < len(rec_df):
                row = rec_df.iloc[item_idx]
                ticker = row["Ticker"]
                exp_return = row["Expected Return (%)"]
                signal = str(row["Recommendation Signal"]).upper()
                
                # Dynamic CSS Color Badge Mapping
                signal_class = "signal-buy" if "BUY" in signal else ("signal-sell" if "SELL" in signal else "signal-hold")
                
                with cols[j]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-header">{ticker}</div>
                            <div class="metric-value">{exp_return}%</div>
                            <div class="{signal_class}">Signal: {signal}</div>
                        </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Transparent Rationale Breakdown")
    st.dataframe(rec_df, use_container_width=True)
else:
    st.warning("`recommendations.csv` not found. Run `python src/recommend.py` first.")

st.markdown("---")

# ==========================================
# 2. INTERACTIVE PORTFOLIO POSITION TRACKER
# ==========================================
st.subheader("📥 Interactive Position Tracker & Dynamic Rebalancer")
st.markdown("Enter your current share holdings to calculate exact personal **BUY / SELL / HOLD** trade quantities.")

if rec_df is not None and weights_df is not None:
    df_merged = pd.merge(rec_df, weights_df, on="Ticker", how="inner")
    
    # Input Grid for Current Shares
    input_cols = st.columns(4)
    current_shares = {}
    
    # Baseline market prices (or fallback defaults)
    sample_prices = {
        "AAPL": 220.0, "CAT": 350.0, "HD": 360.0, "JNJ": 160.0, 
        "JPM": 200.0, "KO": 68.0, "MSFT": 440.0, "NVDA": 125.0, 
        "PG": 165.0, "XOM": 115.0
    }
    
    for idx, row in df_merged.iterrows():
        ticker = row["Ticker"]
        with input_cols[idx % 4]:
            current_shares[ticker] = st.number_input(
                label=f"{ticker} Current Shares",
                min_value=0,
                value=10,  # Default starting shares
                step=1,
                key=f"user_shares_{ticker}"
            )

    # Cash Injection Option
    add_cash = st.number_input("➕ Inject / Withdraw Cash ($)", value=0.0, step=500.0)

    # Dynamic Rebalancing Logic
    total_current_val = sum(current_shares[t] * sample_prices.get(t, 100.0) for t in current_shares)
    total_target_val = total_current_val + add_cash

    rebalance_data = []
    for idx, row in df_merged.iterrows():
        ticker = row["Ticker"]
        price = sample_prices.get(ticker, 100.0)
        curr_qty = current_shares[ticker]
        curr_val = curr_qty * price
        
        target_weight = row.get("Max-Sharpe Weight (SLSQP)", 1.0 / len(df_merged))
        target_val = total_target_val * target_weight
        target_qty = int(target_val // price)
        
        delta_qty = target_qty - curr_qty
        trade_val = abs(delta_qty * price)

        if delta_qty > 0:
            action = "BUY 🟢"
        elif delta_qty < 0:
            action = "SELL 🔴"
        else:
            action = "HOLD ⚪"

        rebalance_data.append({
            "Ticker": ticker,
            "Price ($)": f"${price:.2f}",
            "Current Shares": curr_qty,
            "Current Value ($)": f"${curr_val:,.2f}",
            "Target Weight": f"{target_weight * 100:.1f}%",
            "Target Value ($)": f"${target_val:,.2f}",
            "Action": action,
            "Shares Delta": abs(delta_qty),
            "Trade Value ($)": f"${trade_val:,.2f}"
        })

    interactive_orders_df = pd.DataFrame(rebalance_data)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Portfolio Top-line Summary Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Current Value", f"${total_current_val:,.2f}")
    with m2:
        st.metric("Target Valuation (with Cash)", f"${total_target_val:,.2f}")
    with m3:
        st.metric("Tracked Equities", len(df_merged))

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(interactive_orders_df, use_container_width=True)

st.markdown("---")

# ==========================================
# 3. BASELINE EXECUTION ORDERS TABLE
# ==========================================
st.subheader("Portfolio Baseline Orders ($100,000 Reference Base)")
if rebalance_df is not None:
    st.dataframe(rebalance_df, use_container_width=True)
else:
    st.warning("`rebalancing_orders.csv` not found. Run `python src/recommend.py` first.")