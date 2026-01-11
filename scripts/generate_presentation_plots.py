import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from src.config import MODELS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR
from src.utils import get_logger

logger = get_logger(__name__)

def generate_feature_importance():
    """Generates Feature Importance plot for Random Forest or XGBoost."""
    model_path = MODELS_DIR / "xgboost_model.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "random_forest_model.pkl"
    
    if not model_path.exists():
        logger.warning("No tree-based model found for feature importance.")
        return

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load feature names (quick hack: load X_train column names if saved, or simplistic match)
    # Better: If pipeline, extract from step. If raw regressor, we need names.
    # Assuming preprocessor was saved or we know names. 
    # For now, let's try to get feature_importances_ and just number them if names missing,
    # OR better, load the processed data sample to get columns.
    
    try:
        # Try loading processed data column names
        # This part depends on how you saved data. If 'train_data.pkl' exists...
        # Let's assume standard top features for 1985 dataset just for visual if mapping fails.
        # But we want accuracy.
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Dummy names if we can't find real ones easily in this script context
            # In a real pipeline, we'd load the feature list from metadata.
            # Let's create a generic plot if we can't map perfectly, but 
            # given the project structure, 'data/processed' might have a pickle.
            
            indices = np.argsort(importances)[::-1]
            top_n = 10
            
            plt.figure(figsize=(10, 6))
            plt.title("Top 10 Importanță Caracteristici (Ce contează pentru AI?)")
            plt.bar(range(top_n), importances[indices][:top_n], align="center", color='#FF4B4B')
            plt.xticks(range(top_n), [f"Feature {i}" for i in indices[:top_n]]) # Placeholder names
            plt.tight_layout()
            save_path = FIGURES_DIR / "presentation_feature_importance.png"
            plt.savefig(save_path)
            logger.info(f"Saved feature importance to {save_path}")
            
    except Exception as e:
        logger.error(f"Failed to plot importance: {e}")

def generate_comparison_radar():
    """Generates a Radar Chart comparing models (Simulated data if stats missing)."""
    # In production, load from 'outputs/reports/final_metrics.csv'
    from src.config import REPORTS_DIR
    metrics_path = REPORTS_DIR / "final_metrics.csv"
    
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        # Filter if needed. Plot R2 for all models.
        
        models = df['Model'].values
        r2_scores = df['R2'].values
        
        # Simple Bar Chart instead of Radar (easier to read in slides)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=r2_scores, palette="viridis")
        plt.title("Comparație Performanță Modele (R² Score)")
        plt.ylim(0, 1)
        for i, v in enumerate(r2_scores):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
            
        save_path = FIGURES_DIR / "presentation_model_comparison.png"
        plt.savefig(save_path)
        logger.info(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    generate_comparison_radar()
    generate_feature_importance()
