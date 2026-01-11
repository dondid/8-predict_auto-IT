import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from src.config import MODELS_DIR
from src.utils import get_logger, timer_decorator
from src.models.evaluation import evaluate_predictions

logger = get_logger(__name__)

class NeuralNetworkModel:
    def __init__(self, hidden_layer_sizes=(100, 50, 30), max_iter=1000, random_state=42):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.name = "NeuralNetwork"

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
