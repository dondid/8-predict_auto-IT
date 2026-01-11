import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils import get_logger

logger = get_logger(__name__)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """
    Calculates regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }
    
    logger.info(f"--- {model_name} Performance ---")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"R2:   {r2:.4f}")
    
    return metrics
