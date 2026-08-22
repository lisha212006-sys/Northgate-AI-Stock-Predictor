import os

import pandas as pd
from pandas_datareader import data as web
from loguru import logger

RAW_DATA_DIR = "data/raw"

os.makedirs(RAW_DATA_DIR, exist_ok=True)

INDICATORS = {
    "DGS10": "10Y_Treasury",
    "UNRATE": "Unemployment",
    "CPIAUCSL": "CPI"
}
def download_indicator(series_id, name):

    logger.info(f"Downloading {name}...")

    df = web.DataReader(
        series_id,
        "fred",
        start="2015-01-01"
    )

    file_path = os.path.join(
        RAW_DATA_DIR,
        f"{name}.csv"
    )

    df.to_csv(file_path)

    logger.success(f"{name} saved.")

def main():

    for series_id, name in INDICATORS.items():

        try:
            download_indicator(series_id, name)

        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    main()