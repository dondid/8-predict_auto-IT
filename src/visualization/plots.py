import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.config import FIGURES_DIR

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_price_distribution(y, title="Price Distribution", filename="price_dist.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(y, kde=True, ax=ax)
    ax.set_title(title)
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def plot_predictions(y_true, y_pred, title="True vs Predicted", filename="pred_vs_true.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predictions")
    ax.set_title(title)
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def plot_residuals(y_true, y_pred, title="Residuals", filename="residuals.png"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def plot_feature_importance(importances, feature_names, title="Feature Importance", filename="feature_importance.png"):
    if importances is None:
        return
        
    indices = np.argsort(importances)[::-1]
    top_n = 20
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=importances[indices][:top_n], y=np.array(feature_names)[indices][:top_n], ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()
