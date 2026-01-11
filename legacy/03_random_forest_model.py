"""
Proiect Machine Learning: Predicția Prețului Automobilelor
Etapa 3: Random Forest Regressor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import shap

# ============================================================================
# 1. ÎNCĂRCAREA DATELOR PROCESATE
# ============================================================================

def load_processed_data():
    """
    Încarcă datele procesate din fișierul pickle
    """
    print("\n" + "="*80)
    print("ÎNCĂRCARE DATE PROCESATE")
    print("="*80)
    
    with open('processed_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"✓ Date încărcate cu succes")
    print(f"  Training:   {data_dict['X_train'].shape}")
    print(f"  Validation: {data_dict['X_val'].shape}")
    print(f"  Test:       {data_dict['X_test'].shape}")
    
    return data_dict

# ============================================================================
# 2. RANDOM FOREST CU HYPERPARAMETER TUNING
# ============================================================================

def train_random_forest(X_train, y_train, X_val, y_val, tune_hyperparams=True):
    """
    Antrenează Random Forest cu tuning opțional
    """
    print("\n" + "="*80)
    print("RANDOM FOREST REGRESSOR")
    print("="*80)
    
    if tune_hyperparams:
        print("\nHyperparameter Tuning cu GridSearchCV...")
        
        # Definire grid de parametri
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Model de bază
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # GridSearch
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        print("Antrenare GridSearchCV (poate dura câteva minute)...")
        grid_search.fit(X_train, y_train)
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score (MSE): {-grid_search.best_score_:.2f}")
        
        # Model final cu best params
        rf_model = grid_search.best_estimator_
        
    else:
        print("\nAntrenare cu parametri default optimizați...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
    
    print("\n✓ Model Random Forest antrenat cu succes!")
    
    return rf_model

# ============================================================================
# 3. EVALUARE MODEL
# ============================================================================

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluează modelul pe toate seturile de date
    """
    print("\n" + "="*80)
    print("EVALUARE MODEL")
    print("="*80)
    
    results = {}
    
    for name, X, y in [('Training', X_train, y_train), 
                       ('Validation', X_val, y_val),
                       ('Test', X_test, y_test)]:
        
        # Predicții
        y_pred = model.predict(X)
        
        # Metrici
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        r2 = r2_score(y, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'y_true': y,
            'y_pred': y_pred
        }
        
        print(f"\n{name} Set:")
        print(f"  MSE:  {mse:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE:  {mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
    
    return results

# ============================================================================
# 4. CROSS-VALIDATION (30 RUNS)
# ============================================================================

def perform_cross_validation(model, X, y, n_runs=30):
    """
    Efectuează cross-validation cu 30 de rulări
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION (30 RUNS)")
    print("="*80)
    
    print("\nEfectuare 30 rulări cu random subsampling...")
    
    mse_scores = []
    rmse_scores = []
    r2_scores = []
    
    for i in range(n_runs):
        # Random split
        indices = np.random.permutation(len(X))
        train_size = int(0.75 * len(X))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        
        X_train_cv = X.iloc[train_idx]
        X_test_cv = X.iloc[test_idx]
        y_train_cv = y.iloc[train_idx]
        y_test_cv = y.iloc[test_idx]
        
        # Antrenare și predicție
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)
        
        # Metrici
        mse = mean_squared_error(y_test_cv, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_cv, y_pred)
        
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
        if (i + 1) % 10 == 0:
            print(f"  Completat: {i+1}/30 rulări")
    
    # Statistici
    print("\n" + "-"*80)
    print("REZULTATE CROSS-VALIDATION:")
    print("-"*80)
    print(f"MSE:  {np.mean(mse_scores):,.2f} ± {np.std(mse_scores):,.2f}")
    print(f"RMSE: {np.mean(rmse_scores):,.2f} ± {np.std(rmse_scores):,.2f}")
    print(f"R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    
    cv_results = {
        'MSE': mse_scores,
        'RMSE': rmse_scores,
        'R2': r2_scores,
        'MSE_mean': np.mean(mse_scores),
        'MSE_std': np.std(mse_scores),
        'RMSE_mean': np.mean(rmse_scores),
        'RMSE_std': np.std(rmse_scores),
        'R2_mean': np.mean(r2_scores),
        'R2_std': np.std(r2_scores)
    }
    
    return cv_results

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plotează importanța features
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    
    # Extrage importanțe
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Creează DataFrame
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    
    print(f"\nTop {top_n} Features:")
    print(importance_df.to_string(index=False))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importance_df['Importance'], color='steelblue', edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(importance_df['Feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Salvat: rf_feature_importance.png")
    plt.show()
    
    return importance_df

# ============================================================================
# 6. SHAP VALUES
# ============================================================================

def compute_shap_values(model, X_train, X_test, max_samples=100):
    """
    Calculează SHAP values pentru interpretabilitate
    """
    print("\n" + "="*80)
    print("SHAP VALUES ANALYSIS")
    print("="*80)
    
    print(f"\nCalculare SHAP values (folosind {max_samples} sample background)...")
    
    # Creăm explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculăm SHAP values pentru test set
    shap_values = explainer.shap_values(X_test[:max_samples])
    
    print("✓ SHAP values calculate")
    
    # Summary plot
    print("\nGenerare SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:max_samples], show=False)
    plt.tight_layout()
    plt.savefig('rf_shap_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: rf_shap_summary.png")
    plt.show()
    
    return shap_values, explainer

# ============================================================================
# 7. VIZUALIZĂRI
# ============================================================================

def create_visualizations(results, cv_results):
    """
    Creează vizualizări pentru rezultate
    """
    print("\n" + "="*80)
    print("GENERARE VIZUALIZĂRI")
    print("="*80)
    
    # Figure 1: Predicted vs Actual
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, ax) in enumerate(zip(['Training', 'Validation', 'Test'], axes)):
        y_true = results[name]['y_true']
        y_pred = results[name]['y_pred']
        r2 = results[name]['R2']
        
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='black')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Price ($)', fontsize=11)
        ax.set_ylabel('Predicted Price ($)', fontsize=11)
        ax.set_title(f'{name} Set (R² = {r2:.4f})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rf_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: rf_predictions.png")
    plt.show()
    
    # Figure 2: Residual plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, ax) in enumerate(zip(['Training', 'Validation', 'Test'], axes)):
        y_true = results[name]['y_true']
        y_pred = results[name]['y_pred']
        residuals = y_true - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolor='black')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Price ($)', fontsize=11)
        ax.set_ylabel('Residuals ($)', fontsize=11)
        ax.set_title(f'{name} Set - Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rf_residuals.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: rf_residuals.png")
    plt.show()
    
    # Figure 3: Cross-validation distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = [('MSE', cv_results['MSE']), 
               ('RMSE', cv_results['RMSE']), 
               ('R²', cv_results['R2'])]
    
    for (metric_name, values), ax in zip(metrics, axes):
        ax.hist(values, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(values), color='r', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(values):.2f}')
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{metric_name} Distribution (30 runs)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('rf_cv_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: rf_cv_distribution.png")
    plt.show()

# ============================================================================
# 8. SALVARE MODEL
# ============================================================================

def save_model(model, results, cv_results):
    """
    Salvează modelul și rezultatele
    """
    print("\n" + "="*80)
    print("SALVARE MODEL ȘI REZULTATE")
    print("="*80)
    
    # Salvare model
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Salvat: rf_model.pkl")
    
    # Salvare rezultate
    results_summary = {
        'model_name': 'Random Forest',
        'test_metrics': {
            'MSE': results['Test']['MSE'],
            'RMSE': results['Test']['RMSE'],
            'MAE': results['Test']['MAE'],
            'MAPE': results['Test']['MAPE'],
            'R2': results['Test']['R2']
        },
        'cv_metrics': {
            'MSE_mean': cv_results['MSE_mean'],
            'MSE_std': cv_results['MSE_std'],
            'RMSE_mean': cv_results['RMSE_mean'],
            'RMSE_std': cv_results['RMSE_std'],
            'R2_mean': cv_results['R2_mean'],
            'R2_std': cv_results['R2_std']
        }
    }
    
    with open('rf_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    print("✓ Salvat: rf_results.pkl")

# ============================================================================
# 9. FUNCȚIA PRINCIPALĂ
# ============================================================================

def main():
    """
    Funcția principală de execuție
    """
    print("\n" + "="*80)
    print(" " * 20 + "RANDOM FOREST REGRESSOR")
    print(" " * 15 + "Predicția Prețului Automobilelor")
    print("="*80)
    
    # 1. Încărcare date
    data_dict = load_processed_data()
    
    # 2. Antrenare model
    rf_model = train_random_forest(
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val'],
        tune_hyperparams=False  # Set True pentru GridSearch (mai lent)
    )
    
    # 3. Evaluare
    results = evaluate_model(
        rf_model,
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val'],
        data_dict['X_test'], data_dict['y_test']
    )
    
    # 4. Cross-validation
    X_combined = pd.concat([data_dict['X_train'], data_dict['X_val']])
    y_combined = pd.concat([data_dict['y_train'], data_dict['y_val']])
    cv_results = perform_cross_validation(rf_model, X_combined, y_combined, n_runs=30)
    
    # 5. Feature importance
    importance_df = plot_feature_importance(
        rf_model, 
        data_dict['X_train'].columns,
        top_n=15
    )
    
    # 6. SHAP values
    shap_values, explainer = compute_shap_values(
        rf_model,
        data_dict['X_train'],
        data_dict['X_test'],
        max_samples=100
    )
    
    # 7. Vizualizări
    create_visualizations(results, cv_results)
    
    # 8. Salvare
    save_model(rf_model, results, cv_results)
    
    print("\n" + "="*80)
    print("RANDOM FOREST - FINALIZAT CU SUCCES!")
    print("="*80)
    
    return rf_model, results, cv_results

# ============================================================================
# EXECUȚIE
# ============================================================================

if __name__ == "__main__":
    rf_model, results, cv_results = main()