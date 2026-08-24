import os
import glob
import logging
import subprocess
import pandas as pd
import streamlit as st
from PIL import Image
from theme import apply_dark_theme

# Set up basic logging for debugging data loading issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
st.set_page_config(
    page_title="Northgate AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom brand styling
try:
    apply_dark_theme()
except Exception as e:
    # Fallback in case theme.py is missing or throws an import/runtime error
    st.warning(f"Theme engine failed to load: {e}. Falling back to default.")

# Path definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_PATH = os.path.join(BASE_DIR, "reports")
FIGURES_PATH = os.path.join(REPORTS_PATH, "figures")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")


@st.cache_data(show_spinner="Loading dataset...")
def fetch_local_data(filename, directory=REPORTS_PATH):
    """
    Helper to safely pull CSVs from pipeline outputs.
    Returns None if pipeline hasn't generated the run files yet.
    """
    target_path = os.path.join(directory, filename)
    if not os.path.exists(target_path):
        logger.warning(f"File missing from pipeline output directory: {target_path}")
        return None
    try:
        return pd.read_csv(target_path)
    except Exception as err:
        st.error(f"Failed to parse {filename}: {str(err)}")
        return None


# --- Navigation Sidebar ---
st.sidebar.title("Navigation")
view_selection = st.sidebar.radio("Select View:", [
    "Overview & Recommendations",
    "FinBERT Sentiment Analysis",
    "Model Evaluation (ML vs DL)",
    "Portfolio Optimization & Backtest"
])

# Quick debug tool in sidebar (Classic developer utility)
st.sidebar.markdown("---")
show_debug = st.sidebar.checkbox("Show Developer Debug Info", value=False)
if show_debug:
    st.sidebar.write("### Environment Status")
    st.sidebar.json({
        "reports_dir_exists": os.path.exists(REPORTS_PATH),
        "figures_dir_exists": os.path.exists(FIGURES_PATH),
        "processed_dir_exists": os.path.exists(PROCESSED_DATA_PATH),
        "selected_view": view_selection
    })


# 1. OVERVIEW & RECOMMENDATIONS
if view_selection == "Overview & Recommendations":
    st.title("📈 Signal Fusion & Trade Recommendations")
    st.caption("Fuses deep learning price predictions with Max-Sharpe MPT weights.")

    rec_df = fetch_local_data("recommendations.csv")
    rebalance_df = fetch_local_data("rebalancing_orders.csv")

    if rec_df is not None:
        st.subheader("Asset Recommendations & Signals")
        
        # Grid layout: 4-column split to prevent layout squeezing
        col_count = 4
        chunks = [rec_df[i:i + col_count] for i in range(0, len(rec_df), col_count)]
        
        for chunk in chunks:
            cols = st.columns(col_count)
            for idx, (_, row) in enumerate(chunk.iterrows()):
                ticker = row["Ticker"]
                exp_return = row["Expected Return (%)"]
                signal = str(row["Recommendation Signal"]).upper().strip()
                
                # Dynamic CSS assignment based on signal strength
                if "BUY" in signal:
                    signal_style = "signal-buy"
                elif "SELL" in signal:
                    signal_style = "signal-sell"
                else:
                    signal_style = "signal-hold"
                
                with cols[idx]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-header">{ticker}</div>
                            <div class="metric-value">{exp_return}%</div>
                            <div class="{signal_style}">Signal: {signal}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Raw Signal Breakdown")
        st.dataframe(rec_df, width="stretch")
    else:
        st.warning("⚠️ `recommendations.csv` is missing from `reports/`.")
        if st.button("🚀 Run Recommendation Pipeline Now"):
            with st.spinner("Executing recommendation pipeline (`python src/recommend.py`)..."):
                try:
                    subprocess.run(["python", "src/recommend.py"], check=True)
                    st.success("Recommendation pipeline executed successfully!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Execution failed: {ex}")

    if rebalance_df is not None:
        st.subheader("Portfolio Rebalancing Orders ($100k Target)")
        st.dataframe(rebalance_df, width="stretch")


# 2. FINBERT SENTIMENT ANALYSIS
elif view_selection == "FinBERT Sentiment Analysis":
    st.title("📰 FinBERT Financial News Sentiment")
    st.caption("Alternative data sentiment scores extracted using HuggingFace's ProsusAI/finbert model.")

    sentiment_df = fetch_local_data("sentiment_features.csv", directory=PROCESSED_DATA_PATH)

    if sentiment_df is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Ticker Sentiment Summary")
            ticker_avg = sentiment_df.groupby("Ticker")["Sentiment_Score"].mean().reset_index()
            st.dataframe(ticker_avg, width="stretch")

        with col2:
            st.subheader("Net Polarity Scores")
            st.bar_chart(data=ticker_avg, x="Ticker", y="Sentiment_Score")

        st.markdown("---")
        st.subheader("Detailed Sentiment Feature Matrix")
        st.dataframe(sentiment_df, width="stretch")
    else:
        st.warning("⚠️ Sentiment data missing. Run the notebook `02_sentiment_analysis_finbert.ipynb` to process raw news feeds.")


# 3. MODEL EVALUATION
elif view_selection == "Model Evaluation (ML vs DL)":
    st.title("📊 Model Comparison & Loss Curves")
    st.caption("Evaluation metrics comparing traditional ML baselines against our Deep Learning architectures.")

    master_df = fetch_local_data("master_model_comparison.csv")
    sentiment_df = fetch_local_data("sentiment_experiment_results.csv")

    if master_df is not None:
        st.subheader("Master Model Performance Comparison")
        st.dataframe(master_df, width="stretch")
    else:
        st.warning("⚠️ Master comparison results missing.")
        if st.button("🚀 Run Evaluation Pipeline Now"):
            with st.spinner("Evaluating models (`python src/evaluate.py`)..."):
                try:
                    subprocess.run(["python", "src/evaluate.py"], check=True)
                    st.success("Evaluation complete!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Execution failed: {ex}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("FinBERT Sentiment Integration Impact")
        if sentiment_df is not None:
            st.dataframe(sentiment_df, width="stretch")
        else:
            st.info("No sentiment experimental run data detected.")

    with col2:
        st.subheader("Deep Learning Loss Curves")
        curve_files = glob.glob(os.path.join(FIGURES_PATH, "*_loss_curve.png"))
        
        if curve_files:
            curve_options = {os.path.basename(f): f for f in curve_files}
            selected_label = st.selectbox("Select Model Architecture:", list(curve_options.keys()))
            
            try:
                img = Image.open(curve_options[selected_label])
                st.image(img, caption=f"Loss history: {selected_label}", width="stretch")
            except Exception as img_err:
                st.error(f"Failed to display image: {img_err}")
        else:
            st.info("No training loss curves found in reports directory.")


# 4. PORTFOLIO OPTIMIZATION
elif view_selection == "Portfolio Optimization & Backtest":
    st.title("⚖️ Modern Portfolio Theory (MPT)")
    st.caption("Efficient Frontier optimization and strategy backtest vs equal-weight benchmark.")

    weights_df = fetch_local_data("portfolio_weights.csv")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Optimal Max-Sharpe Weights")
        if weights_df is not None:
            st.dataframe(weights_df, width="stretch")
        else:
            st.warning("⚠️ Portfolio weights not found.")
            if st.button("🚀 Run Portfolio Optimization Now"):
                with st.spinner("Calculating MPT weights (`python src/portfolio.py`)..."):
                    try:
                        subprocess.run(["python", "src/portfolio.py"], check=True)
                        st.success("Optimization complete!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Execution failed: {ex}")

    with col2:
        st.subheader("Efficient Frontier & Monte Carlo")
        ef_path = os.path.join(FIGURES_PATH, "efficient_frontier.png")
        if os.path.exists(ef_path):
            try:
                img = Image.open(ef_path)
                st.image(img, caption="Efficient Frontier Plot", width="stretch")
            except Exception as e:
                st.error(f"Failed to render plot image: {e}")
        else:
            st.info("Efficient frontier plot not found in reports/figures/.")