import joblib
import numpy as np
from sklearn.svm import SVR
from src.config import MODELS_DIR
from src.utils import get_logger, timer_decorator
from src.models.evaluation import evaluate_predictions

logger = get_logger(__name__)

class SVRModel:
    def __init__(self, kernel='rbf', C=100, epsilon=0.1):
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        self.name = "SVR"

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

    def save(self):
        path = MODELS_DIR / f"{self.name.lower()}_model.pkl"
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
