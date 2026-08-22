
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

PORTFOLIO_DIR = os.path.join(OUTPUT_DIR, "portfolio")

PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")

