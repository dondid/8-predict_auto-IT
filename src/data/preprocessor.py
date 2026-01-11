import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE
from src.utils import get_logger, timer_decorator

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        
    @timer_decorator
    def clean_data(self, df):
        """
        Handles missing values and basic cleaning.
        """
        logger.info("Starting data cleaning...")
        initial_rows = len(df)
        
        # Remove rows without price
        df = df.dropna(subset=['price'])
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing price.")
        
        # Impute missing values
        # Numeric -> Median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical -> Mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
                
        return df

    @timer_decorator
    def feature_engineering(self, df):
        """
        Creates new features based on domain knowledge.
        """
        logger.info("Performing feature engineering...")
        
        # Avoid division by zero
        df['power-to-weight'] = df['horsepower'] / df['curb-weight'].replace(0, 1)
        df['fuel-efficiency'] = (df['city-mpg'] + df['highway-mpg']) / 2
        
        # Binning size: Use saved bins if available (inference), else compute (training)
        size_metric = df['length'] * df['width']
        
        try:
            # Try loading bins (inference mode)
            bins = joblib.load(MODELS_DIR / "size_bins.pkl")
            df['size-category'] = pd.cut(size_metric, bins=bins, labels=['Small', 'Medium', 'Large'], include_lowest=True)
            logger.info("Loaded size bins for feature engineering.")
        except (FileNotFoundError, ValueError):
            # Compute new bins (training mode)
            try:
                # Use qcut to determine bins, duplicates drop is important for small datasets
                _, bins = pd.qcut(size_metric, q=3, labels=['Small', 'Medium', 'Large'], retbins=True, duplicates='drop')
                
                # Check if we got enough bins (might get fewer if duplicates dropped)
                if len(bins) < 4:
                     # Fallback if quantiles collapse
                     bins = np.linspace(size_metric.min(), size_metric.max(), 4)

                df['size-category'] = pd.cut(size_metric, bins=bins, labels=['Small', 'Medium', 'Large'], include_lowest=True)
                joblib.dump(bins, MODELS_DIR / "size_bins.pkl")
                logger.info("Computed and saved new size bins.")
            except Exception as e:
                # Fallback purely for robustness if everything fails (e.g. single row training attempt)
                logger.warning(f"Quantile binning failed: {e}. Using simple fallback.")
                df['size-category'] = 'Medium' # Placeholder

        # Luxury brand proxy
        premium_makes = ['mercedes-benz', 'bmw', 'porsche', 'jaguar', 'audi', 'volvo']
        df['is-luxury'] = df['make'].apply(lambda x: 1 if x in premium_makes else 0)

        # === SAFETY & RISK ANALYSIS (New Features) ===
        # 1. Safety Score: Heavy cars (curb-weight) generally safer.
        # Normalize weight (approx 1400-4100 lbs)
        norm_weight = (df['curb-weight'] - 1400) / (4100 - 1400)
        
        # 2. Risk Score: Normalized losses (65-256). Invert for "Safety".
        # If normalized-losses is missing (it's imputed in clean_data), we use it.
        # But for new inference data, it might default to something.
        if 'normalized-losses' in df.columns:
             norm_risk = 1 - ((df['normalized-losses'] - 65) / (256 - 65))
        else:
             norm_risk = 0.5 # Default neutral
             
        # Combined Safety Score (0-10)
        # 60% Weight, 40% Low Risk
        df['safety_score'] = (0.6 * norm_weight + 0.4 * norm_risk) * 10
        df['safety_score'] = df['safety_score'].clip(1, 10).round(1)

        return df

    @timer_decorator
    def encode_and_scale(self, df):
        """
        Encodes categorical variables and scales numerical ones.
        """
        logger.info("Encoding and scaling data...")
        
        # Label Encoding for specific ordered/important columns
        le_cols = ['make', 'engine-type', 'fuel-system', 'size-category']
        for col in le_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        # One-Hot Encoding for the rest
        df = pd.get_dummies(df, drop_first=True)
        
        # Save encoders
        joblib.dump(self.encoders, MODELS_DIR / "encoders.pkl")
        
        return df

    @timer_decorator
    def preprocess_pipeline(self, df):
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        df = self.encode_and_scale(df)
        
        X = df.drop('price', axis=1)
        y = df['price']
        
        # Scale features
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # Save scaler
        joblib.dump(self.scaler, MODELS_DIR / "scaler.pkl")
        
        # Split Data
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=RANDOM_STATE)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)
        
        data_dict = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        # Save processed data
        joblib.dump(data_dict, PROCESSED_DATA_DIR / "processed_data.pkl")
        logger.info(f"Processed data saved to {PROCESSED_DATA_DIR}")
        
        return data_dict

    def load_artifacts(self):
        """
        Loads the saved scaler and label encoders.
        """
        try:
            self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
            self.encoders = joblib.load(MODELS_DIR / "encoders.pkl")
            logger.info("Artifacts loaded successfully.")
        except FileNotFoundError:
            logger.error("Artifacts not found. Run training pipeline first.")
            raise

    def transform_new_data(self, input_df):
        """
        Transforms new data for inference using saved artifacts.
        """
        self.load_artifacts()
        
        # 1. Feature Engineering
        input_df = self.feature_engineering(input_df)
        
        # 2. Label Encoding
        for col, le in self.encoders.items():
            if col in input_df.columns:
                # Handle unseen labels by assigning a default or mode
                input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                input_df[col] = le.transform(input_df[col].astype(str))
        
        # 3. One-Hot Encoding (Align with training columns)
        # Load a sample of processed data to get column names
        # For simplicity, we assume the input_df has all necessary columns or we need to align
        # A robust way is to load the training columns. 
        # But here we will try to match columns from scaler.
        
        # Dummy encoding
        input_df = pd.get_dummies(input_df, drop_first=True)
        
        # Reindex to match scaler requirements
        required_cols = self.scaler.feature_names_in_
        
        # Add missing cols with 0
        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Drop extra cols
        input_df = input_df[required_cols]
        
        # 4. Scaling
        input_transformed = self.scaler.transform(input_df)
        
        return input_transformed

if __name__ == "__main__":
    from src.data.loader import DataLoader
    loader = DataLoader()
    df = loader.load_data()
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(df)
    print("Train shape:", data['X_train'].shape)
