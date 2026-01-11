"""
Proiect Machine Learning: Predicția Prețului Automobilelor
Etapa 4: XGBoost Regressor (Gradient Boosting)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# ============================================================================
# 1. ÎNCĂRCAREA DATELOR
# ============================================================================

def load_processed_data():
    """
    Încarcă datele procesate
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
# 2. XGBOOST CU HYPERPARAMETER TUNING
# ============================================================================

def train_xgboost(X_train, y_train, X_val, y_val, tune_hyperparams=True):
    """
    Antrenează XGBoost cu tuning opțional
    """
    print("\n" + "="*80)
    print("XGBOOST REGRESSOR")
    print("="*80)
    
    if tune_hyperparams:
        print("\nHyperparameter Tuning cu GridSearchCV...")
        
        # Grid de parametri
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        # Model de bază
        xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # GridSearch
        grid_search = GridSearchCV(
            estimator=xgb_base,
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
        
        xgb_model = grid_search.best_estimator_
        
    else:
        print("\nAntrenare cu parametri optimizați...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1
        )
        
        # Antrenare cu early stopping
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        print(f"✓ Best iteration: {xgb_model.best_iteration}")
    
    print("\n✓ Model XGBoost antrenat cu succes!")
    
    return xgb_model

# ============================================================================
# 3. EVALUARE MODEL
# ============================================================================

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluează modelul pe toate seturile
    """
    print("\n" + "="*80)
    print("EVALUARE MODEL")
    print("="*80)
    
    results = {}
    
    for name, X, y in [('Training', X_train, y_train), 
                       ('Validation', X_val, y_val),
                       ('Test', X_test, y_test)]:
        
        y_pred = model.predict(X)
        
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

def perform_cross_validation(X, y, n_runs=30):
    """
    Cross-validation cu 30 rulări
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION (30 RUNS)")
    print("="*80)
    
    print("\nEfectuare 30 rulări cu random subsampling...")
    
    mse_scores = []
    rmse_scores = []
    r2_scores = []
    
    for i in range(n_runs):
        indices = np.random.permutation(len(X))
        train_size = int(0.75 * len(X))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        
        X_train_cv = X.iloc[train_idx]
        X_test_cv = X.iloc[test_idx]
        y_train_cv = y.iloc[train_idx]
        y_test_cv = y.iloc[test_idx]
        
        # Model XGBoost simplu pentru CV
        model_cv = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        model_cv.fit(X_train_cv, y_train_cv, verbose=False)
        y_pred = model_cv.predict(X_test_cv)
        
        mse = mean_squared_error(y_test_cv, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_cv, y_pred)
        
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
        if (i + 1) % 10 == 0:
            print(f"  Completat: {i+1}/30 rulări")
    
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
    Plot feature importance pentru XGBoost
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    
    # Extrage importanțe
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    
    print(f"\nTop {top_n} Features:")
    print(importance_df.to_string(index=False))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importance_df['Importance'], color='orange', edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(importance_df['Feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - XGBoost', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('xgb_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Salvat: xgb_feature_importance.png")
    plt.show()
    
    return importance_df

# ============================================================================
# 6. LEARNING CURVES
# ============================================================================

def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    """
    Plot learning curves pentru a vedea evoluția antrenării
    """
    print("\n" + "="*80)
    print("LEARNING CURVES")
    print("="*80)
    
    # Re-antrenăm modelul cu urmărirea erorilor
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    model_temp = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model_temp.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric='rmse',
        verbose=False
    )
    
    # Extrage rezultatele
    results = model_temp.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, results['validation_0']['rmse'], label='Training', linewidth=2)
    ax.plot(x_axis, results['validation_1']['rmse'], label='Validation', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('XGBoost Learning Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgb_learning_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: xgb_learning_curves.png")
    plt.show()

# ============================================================================
# 7. VIZUALIZĂRI PREDICȚII ȘI RESIDUALS
# ============================================================================

def create_visualizations(results, cv_results):
    """
    Creează vizualizări
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
        
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='black', color='orange')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Price ($)', fontsize=11)
        ax.set_ylabel('Predicted Price ($)', fontsize=11)
        ax.set_title(f'{name} Set (R² = {r2:.4f})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgb_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: xgb_predictions.png")
    plt.show()
    
    # Figure 2: Residuals
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, ax) in enumerate(zip(['Training', 'Validation', 'Test'], axes)):
        y_true = results[name]['y_true']
        y_pred = results[name]['y_pred']
        residuals = y_true - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolor='black', color='orange')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Price ($)', fontsize=11)
        ax.set_ylabel('Residuals ($)', fontsize=11)
        ax.set_title(f'{name} Set - Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgb_residuals.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: xgb_residuals.png")
    plt.show()
    
    # Figure 3: CV Distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = [('MSE', cv_results['MSE']), 
               ('RMSE', cv_results['RMSE']), 
               ('R²', cv_results['R2'])]
    
    for (metric_name, values), ax in zip(metrics, axes):
        ax.hist(values, bins=15, edgecolor='black', alpha=0.7, color='orange')
        ax.axvline(np.mean(values), color='r', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(values):.2f}')
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{metric_name} Distribution (30 runs)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('xgb_cv_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: xgb_cv_distribution.png")
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
    
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Salvat: xgb_model.pkl")
    
    results_summary = {
        'model_name': 'XGBoost',
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
    
    with open('xgb_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    print("✓ Salvat: xgb_results.pkl")

# ============================================================================
# 9. FUNCȚIA PRINCIPALĂ
# ============================================================================

def main():
    """
    Funcția principală
    """
    print("\n" + "="*80)
    print(" " * 23 + "XGBOOST REGRESSOR")
    print(" " * 15 + "Predicția Prețului Automobilelor")
    print("="*80)
    
    # 1. Încărcare date
    data_dict = load_processed_data()
    
    # 2. Antrenare
    xgb_model = train_xgboost(
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val'],
        tune_hyperparams=False
    )
    
    # 3. Evaluare
    results = evaluate_model(
        xgb_model,
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val'],
        data_dict['X_test'], data_dict['y_test']
    )
    
    # 4. Cross-validation
    X_combined = pd.concat([data_dict['X_train'], data_dict['X_val']])
    y_combined = pd.concat([data_dict['y_train'], data_dict['y_val']])
    cv_results = perform_cross_validation(X_combined, y_combined, n_runs=30)
    
    # 5. Feature importance
    importance_df = plot_feature_importance(
        xgb_model,
        data_dict['X_train'].columns,
        top_n=15
    )
    
    # 6. Learning curves
    plot_learning_curves(
        xgb_model,
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val']
    )
    
    # 7. Vizualizări
    create_visualizations(results, cv_results)
    
    # 8. Salvare
    save_model(xgb_model, results, cv_results)
    
    print("\n" + "="*80)
    print("XGBOOST - FINALIZAT CU SUCCES!")
    print("="*80)
    
    return xgb_model, results, cv_results

# ============================================================================
# EXECUȚIE
# ============================================================================

if __name__ == "__main__":
    xgb_model, results, cv_results = main()