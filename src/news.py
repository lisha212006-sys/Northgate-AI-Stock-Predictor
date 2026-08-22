
import torch
import pandas as pd
import numpy as np
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load FinBERT Model globally to avoid reloading on every function call
MODEL_NAME = "ProsusAI/finbert"
_tokenizer = None
_model = None

def _get_finbert():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
        _model.eval()
    return _tokenizer, _model

def fetch_ticker_news(tickers):
    """Fetches real-time headlines for a list of tickers via yfinance."""
    raw_news = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            for item in t.news or []:
                title = item.get("title") or item.get("content", {}).get("title")
                pub_time = item.get("providerPublishTime") or item.get("content", {}).get("pubDate")
                if title:
                    raw_news.append({
                        "Ticker": ticker,
                        "Headline": title,
                        "Date": pd.to_datetime(pub_time, unit='s') if isinstance(pub_time, (int, float)) else pd.to_datetime(pub_time)
                    })
        except Exception:
            continue

    if not raw_news:
        return pd.DataFrame()

    df = pd.DataFrame(raw_news)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df

def analyze_news_sentiment(news_df):
    """Generates FinBERT positive/negative/neutral sentiment scores and net polarity."""
    if news_df.empty or "Headline" not in news_df.columns:
        return news_df

    tokenizer, model = _get_finbert()
    device = next(model.parameters()).device
    
    headlines = news_df["Headline"].tolist()
    inputs = tokenizer(headlines, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    pos = probs[:, 0].cpu().numpy()
    neg = probs[:, 1].cpu().numpy()
    neu = probs[:, 2].cpu().numpy()

    news_df["FinBERT_Pos"] = pos
    news_df["FinBERT_Neg"] = neg
    news_df["FinBERT_Neu"] = neu
    news_df["Sentiment_Score"] = pos - neg

    conditions = [news_df["Sentiment_Score"] > 0.15, news_df["Sentiment_Score"] < -0.15]
    news_df["Sentiment_Label"] = np.select(conditions, ["Positive", "Negative"], default="Neutral")

    return news_df

def get_latest_sentiment_signals(tickers):
    """Pipeline function to fetch news and return daily averaged sentiment features."""
    news_df = fetch_ticker_news(tickers)
    if news_df.empty:
        return pd.DataFrame()
    
    scored_df = analyze_news_sentiment(news_df)
    daily_summary = scored_df.groupby("Ticker")[["Sentiment_Score", "FinBERT_Pos", "FinBERT_Neg"]].mean().reset_index()
    return daily_summary