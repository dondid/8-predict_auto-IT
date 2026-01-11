import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data Source
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
COLUMN_NAMES = [
    'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
    'wheel-base', 'length', 'width', 'height', 'curb-weight',
    'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
    'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
    'city-mpg', 'highway-mpg', 'price'
]

# Random Seed
RANDOM_STATE = 42

# AI Assistant Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
