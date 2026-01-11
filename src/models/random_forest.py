import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from src.config import MODELS_DIR
from src.utils import get_logger, timer_decorator
from src.models.evaluation import evaluate_predictions

logger = get_logger(__name__)

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.name = "RandomForest"

    @timer_decorator
    def train(self, X_train, y_train):
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.name} trained successfully.")

    @timer_decorator
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions, self.name)
        return metrics, predictions

    @timer_decorator
    def cross_validate(self, X, y, cv=5):
        logger.info(f"Running Cross-Validation ({cv} folds) for {self.name}...")
        scores = cross_validate(
            self.model, X, y, cv=cv, 
            scoring=['neg_mean_squared_error', 'r2'],
            return_train_score=False
        )
        
        rmse_scores = np.sqrt(-scores['test_neg_mean_squared_error'])
        logger.info(f"CV Average RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
        logger.info(f"CV Average R2: {scores['test_r2'].mean():.4f}")
        return scores

    def save(self):
        path = MODELS_DIR / f"{self.name.lower()}_model.pkl"
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    @property
    def feature_importance(self):
        return self.model.feature_importances_
