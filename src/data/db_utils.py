import sqlite3
import pandas as pd
from src.config import RAW_DATA_DIR, DATA_DIR
from src.utils import get_logger

logger = get_logger(__name__)
DB_PATH = DATA_DIR / "automobile.db"

def init_db(csv_path=None):
    """
    Initializes the SQLite database from the CSV file.
    """
    if csv_path is None:
        csv_path = RAW_DATA_DIR / "imports-85.csv"
        
    if not csv_path.exists():
        logger.error(f"CSV not found at {csv_path}")
        return

    logger.info(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Sanitize column names for SQL (replace - with _)
    df.columns = [c.replace('-', '_') for c in df.columns]
    
    # Write to SQL
    df.to_sql("automobiles", conn, if_exists="replace", index=False)
    
    # Create Index for speed
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price ON automobiles (price)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_make ON automobiles (make)")
    
    conn.commit()
    conn.close()
    logger.info("Database initialized and data migrated successfully.")

def get_connection():
    return sqlite3.connect(DB_PATH)

if __name__ == "__main__":
    init_db()
