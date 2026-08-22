import os
import yaml
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def analyze_news_sentiment(news_df, device):
    """
    Scores news headlines/text using ProsusAI/finbert.
    Returns composite sentiment score: P(Positive) - P(Negative).
    """
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    texts = news_df["Headline"].tolist() if "Headline" in news_df.columns else news_df["Text"].tolist()
    batch_size = 32
    scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        # FinBERT labels mapping: [Positive (0), Negative (1), Neutral (2)]
        for prob in probs:
            pos_score, neg_score = prob[0], prob[1]
            # Net daily sentiment polarity (-1.0 to +1.0)
            net_sentiment = pos_score - neg_score
            scores.append(net_sentiment)

    news_df["Sentiment_Score"] = scores
    return news_df


def align_sentiment_no_lookahead(news_df, features_df):
    """
    Aggregates daily news sentiment and shifts by 1 session (Lag 1) 
    to strictly eliminate look-ahead bias before merging into panel features.
    """
    # Ensure datetime formatting
    news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.date
    features_df["Date"] = pd.to_datetime(features_df["Date"]).dt.date

    # Group by Ticker & Date -> Mean Daily Sentiment
    group_cols = ["Ticker", "Date"] if "Ticker" in news_df.columns else ["Date"]
    daily_sentiment = news_df.groupby(group_cols)["Sentiment_Score"].mean().reset_index()

    # Apply 1-day lag per ticker to eliminate look-ahead bias
    if "Ticker" in daily_sentiment.columns:
        daily_sentiment = daily_sentiment.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        daily_sentiment["FinBERT_Sentiment_Lag1"] = daily_sentiment.groupby("Ticker")["Sentiment_Score"].shift(1)
    else:
        daily_sentiment = daily_sentiment.sort_values("Date").reset_index(drop=True)
        daily_sentiment["FinBERT_Sentiment_Lag1"] = daily_sentiment["Sentiment_Score"].shift(1)

    # Drop unlagged column
    daily_sentiment = daily_sentiment.drop(columns=["Sentiment_Score"]).dropna(subset=["FinBERT_Sentiment_Lag1"])

    # Merge into processed panel features
    merged_df = pd.merge(features_df, daily_sentiment, on=group_cols, how="left")

    # Forward fill missing sentiment days (up to 3 days for weekends), then fill remaining with 0 (neutral)
    if "Ticker" in merged_df.columns:
        merged_df["FinBERT_Sentiment_Lag1"] = merged_df.groupby("Ticker")["FinBERT_Sentiment_Lag1"].ffill(limit=3).fillna(0.0)
    else:
        merged_df["FinBERT_Sentiment_Lag1"] = merged_df["FinBERT_Sentiment_Lag1"].ffill(limit=3).fillna(0.0)

    return merged_df


def main():
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    raw_dir = config["data"]["raw_dir"] if "raw_dir" in config["data"] else "data/raw"

    features_path = os.path.join(processed_dir, "features.parquet")
    news_path = os.path.join(raw_dir, "news_data.csv")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing {features_path}. Run features.py first!")

    features_df = pd.read_parquet(features_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load news dataset or synthesize fallback if raw news file is missing
    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path)
    else:
        # Fallback news frame structure for execution testing
        tickers = features_df["Ticker"].unique() if "Ticker" in features_df.columns else ["ASSET"]
        dates = features_df["Date"].unique()
        sample_rows = []
        for d in dates[:50]:
            for t in tickers:
                sample_rows.append({"Date": d, "Ticker": t, "Headline": f"Quarterly earnings for {t} show strong growth and positive outlook."})
        news_df = pd.DataFrame(sample_rows)

    # 1. Run FinBERT Sentiment Scoring
    news_df = analyze_news_sentiment(news_df, device)

    # 2. Align to Market Sessions without Look-Ahead (Lag 1)
    enriched_features_df = align_sentiment_no_lookahead(news_df, features_df)

    # 3. Save Enriched Features Panel back to Parquet
    enriched_features_df.to_parquet(features_path, index=False)


if __name__ == "__main__":
    main()