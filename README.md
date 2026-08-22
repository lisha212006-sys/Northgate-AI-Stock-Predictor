# 📈 Northgate AI Stock Predictor 

An end-to-end quantitative finance platform that predicts asset returns, compares ML and deep learning sequence models, processes financial market sentiment using **FinBERT**, and constructs optimized portfolios using **Modern Portfolio Theory (MPT)**.

---

## 🎯 Project Overview

* **Alternative Data Pipeline**: Real-time financial headline processing and sentiment extraction powered by `ProsusAI/finbert`.
* **Multi-Model Framework**: Performance evaluation across ML algorithms (Ridge, SVR, Random Forest, XGBoost) and Deep Learning sequence architectures (LSTM, BiLSTM, GRU, Transformer Encoders).
* **Modern Portfolio Theory (MPT)**: Max-Sharpe asset allocation via constrained optimization and Efficient Frontier visualization.
* **Interactive Dashboard**: Streamlit web interface for trade recommendations, sentiment analysis, and portfolio rebalancing metrics.

---

## 📁 Repository Structure

```text
Northgate-AI-Stock-Predictor/
├── data/
│   ├── processed/                # Engineered feature matrices & sentiment scores
│   └── raw/                      # Unprocessed market data
├── docs/                         # Documentation & project references
├── models/                       # Saved ML/DL model weights & checkpoints
├── notebooks/                    # Interactive research notebooks
│   ├── eda_and_stationarity.ipynb
│   ├── sentiment_analysis_finbert.ipynb
│   ├── ml_experiments.ipynb
│   ├── dl_experiments.ipynb
│   ├── portfolio_optimization_mpt.ipynb
│   ├── signal_fusion_and_trade_recommendations.ipynb
│   ├── portfolio_rebalancing_and_execution.ipynb
│   └── model_evaluation_and_loss_curves.ipynb
├── output/                       # Data exports & intermediate outputs
├── reports/                      # Generated evaluation tables & figures
│   └── figures/                  # Efficient frontier and training loss plots
├── src/                          # Core codebase
│   ├── models_dl.py              # PyTorch neural network architectures
│   ├── models_ml.py              # Classical machine learning models
│   ├── news.py                   # News headline scraping pipeline
│   ├── portfolio.py              # MPT portfolio optimization logic
│   ├── predict.py                # Prediction inference pipeline
│   ├── recommend.py              # Signal fusion & recommendation engine
│   ├── sentiment.py              # FinBERT sentiment scoring engine
│   └── utils.py                  # Data processing helpers
├── app.py                        # Streamlit interactive application
├── theme.py                      # Dashboard styling configuration
├── requirements.txt              # Dependencies
└── README.md

⚙️ Installation & Setup
1. Clone Repository & Set Up Virtual Environment
git clone [https://github.com/lisha212006-sys/Northgate-AI-Stock-Predictor.git](https://github.com/lisha212006-sys/Northgate-AI-Stock-Predictor.git)
cd Northgate-AI-Stock-Predictor

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # On macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

2. Launch Interactive Dashboard
streamlit run dashboard/app.py

🔬 Research Notebook Execution Order
Run the notebooks in notebooks/ sequentially to execute the full end-to-end pipeline:

eda_and_stationarity.ipynb: Exploratory data analysis & stationarity testing.

sentiment_analysis_finbert.ipynb: FinBERT headline scoring.

ml_experiments.ipynb: Baseline ML model training & evaluation.

dl_experiments.ipynb: Deep learning sequence model training.

portfolio_optimization_mpt.ipynb: Max-Sharpe weight calculation.

signal_fusion_and_trade_recommendations.ipynb: Recommendation engine.

portfolio_rebalancing_and_execution.ipynb: Portfolio rebalancing orders.

model_evaluation_and_loss_curves.ipynb: Comparative performance metrics.

🛠️ Tech Stack
Machine Learning & NLP: PyTorch, Scikit-Learn, XGBoost, Transformers (HuggingFace)

Data Processing & Finance: Pandas, NumPy, SciPy, YFinance

Dashboard & Visualization: Streamlit, Matplotlib, Seaborn