import pandas as pd
import os
from src.config import DATA_URL, COLUMN_NAMES, RAW_DATA_DIR
from src.utils import get_logger, timer_decorator

logger = get_logger(__name__)

class DataLoader:
    def __init__(self):
        self.raw_data_path = RAW_DATA_DIR / "imports-85.csv"

    def load_data(self):
        """
        Loads data. Tries SQL > Local CSV > Download.
        """
        # Try SQL
        try:
            from src.data.db_utils import get_connection, DB_PATH
            if DB_PATH.exists():
                logger.info(f"Loading data from SQL Database: {DB_PATH}")
                conn = get_connection()
                df = pd.read_sql("SELECT * FROM automobiles", conn)
                conn.close()
                
                # Rename columns back to original format (underscores to dashes)
                # SQL doesn't like dashes, but our app code expects them.
                df.columns = [c.replace('_', '-') for c in df.columns]
                
                logger.info(f"Data loaded from SQL. Shape: {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"SQL Load failed: {e}. Falling back to CSV.")

        # Fallback to CSV
        if self.raw_data_path.exists():
            logger.info(f"Loading data from local CSV: {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)
        else:
            logger.info(f"Downloading data from {DATA_URL}")
            df = pd.read_csv(DATA_URL, names=COLUMN_NAMES, na_values='?')
            logger.info(f"Saving raw data to {self.raw_data_path}")
            df.to_csv(self.raw_data_path, index=False)
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def load_modern_data(self):
        """
        Loads the modern car dataset (2020-2024) if available.
        Returns a DataFrame with modern car specs and prices.
        """
        modern_path = RAW_DATA_DIR / "modern_cars_2024.csv"
        if not modern_path.exists():
            return None
        
        try:
            df = pd.read_csv(modern_path)
            # Create 'Make' column if not exists
            if 'Make' not in df.columns and 'Model' in df.columns:
                df['Make'] = df['Model'].astype(str).apply(lambda x: x.split(' ')[0] if ' ' in x else x)
            
            # Normalize Make
            if 'Make' in df.columns:
                df['make_norm'] = df['Make'].str.lower()
                
            return df
        except Exception as e:
            logger.error(f"Error loading modern data: {e}")
            return None


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    print(df.head())
