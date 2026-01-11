import joblib
import xgboost as xgb
import numpy as np
from src.config import MODELS_DIR
from src.utils import get_logger, timer_decorator
from src.models.evaluation import evaluate_predictions

logger = get_logger(__name__)

class XGBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        self.name = "XGBoost"

    @timer_decorator
    def train(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"Training {self.name}...")
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        logger.info(f"{self.name} trained successfully.")

    @timer_decorator
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions, self.name)
        return metrics, predictions

    def save(self):
        path = MODELS_DIR / f"{self.name.lower()}_model.pkl"
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
