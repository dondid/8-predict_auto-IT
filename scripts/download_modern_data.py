import requests
from src.config import RAW_DATA_DIR
from src.utils import get_logger

logger = get_logger(__name__)

URL = "https://raw.githubusercontent.com/lamtong/car_price_analysis/main/Car_price_2024.csv"
DEST = RAW_DATA_DIR / "modern_cars_2024.csv"

def download_modern_data():
    logger.info(f"Downloading modern data from {URL}...")
    try:
        response = requests.get(URL)
        if response.status_code == 200:
            with open(DEST, 'wb') as f:
                f.write(response.content)
            logger.info(f"Modern data saved to {DEST}")
        else:
            logger.error(f"Failed to download. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Download error: {e}")

if __name__ == "__main__":
    download_modern_data()
