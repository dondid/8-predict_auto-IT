"""
Proiect Machine Learning: Predicția Prețului Automobilelor
Etapa 5: Support Vector Regression (SVR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import time


# ============================================================================
# 1. ÎNCĂRCAREA DATELOR
# ============================================================================

def load_processed_data():
    """
    Încarcă datele procesate
    """
    print("\n" + "=" * 80)
    print("ÎNCĂRCARE DATE PROCESATE")
    print("=" * 80)

    with open('processed_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    print(f"✓ Date încărcate cu succes")
    print(f"  Training:   {data_dict['X_train'].shape}")
    print(f"  Validation: {data_dict['X_val'].shape}")
    print(f"  Test:       {data_dict['X_test'].shape}")

    return data_dict


# ============================================================================
# 2. SVR CU HYPERPARAMETER TUNING
# ============================================================================

def train_svr(X_train, y_train, X_val, y_val, tune_hyperparams=True):
    """
    Antrenează SVR cu diferite kernels și tuning
    """
    print("\n" + "=" * 80)
    print("SUPPORT VECTOR REGRESSION (SVR)")
    print("=" * 80)

    if tune_hyperparams:
        print("\nHyperparameter Tuning cu GridSearchCV...")
        print("Testare kernels: RBF și Polynomial")

        # Grid de parametri pentru diferite kernels
        param_grid = [
            {
                'kernel': ['rbf'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            },
            {
                'kernel': ['poly'],
                'C': [0.1, 1, 10, 100],
                'degree': [2, 3, 4],
                'epsilon': [0.01, 0.1, 0.2]
            }
        ]

        # Model de bază
        svr_base = SVR()

        # GridSearch
        grid_search = GridSearchCV(
            estimator=svr_base,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )

        print("\nAntrenare GridSearchCV (poate dura mai mult)...")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()

        print(f"\n✓ Timp antrenare: {end_time - start_time:.2f} secunde")
        print(f"✓ Best kernel: {grid_search.best_params_['kernel']}")
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score (MSE): {-grid_search.best_score_:.2f}")

        svr_model = grid_search.best_estimator_

    else:
        print("\nAntrenare cu parametri optimizați (RBF kernel)...")
        svr_model = SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.1
        )

        start_time = time.time()
        svr_model.fit(X_train, y_train)
        end_time = time.time()

        print(f"✓ Timp antrenare: {end_time - start_time:.2f} secunde")

    print("\n✓ Model SVR antrenat cu succes!")

    # Informații despre support vectors
    print(f"\nSupport Vectors: {len(svr_model.support_)} din {len(X_train)} samples")
    print(f"Procent: {len(svr_model.support_) / len(X_train) * 100:.2f}%")

    return svr_model


# ============================================================================
# 3. EVALUARE MODEL
# ============================================================================

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluează modelul pe toate seturile
    """
    print("\n" + "=" * 80)
    print("EVALUARE MODEL")
    print("=" * 80)

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
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION (30 RUNS)")
    print("=" * 80)

    print("\nEfectuare 30 rulări cu random subsampling...")
    print("ATENȚIE: SVR poate dura mai mult decât celelalte modele!")

    mse_scores = []
    rmse_scores = []
    r2_scores = []

    start_time = time.time()

    for i in range(n_runs):
        indices = np.random.permutation(len(X))
        train_size = int(0.75 * len(X))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        X_train_cv = X.iloc[train_idx]
        X_test_cv = X.iloc[test_idx]
        y_train_cv = y.iloc[train_idx]
        y_test_cv = y.iloc[test_idx]

        # Model SVR simplu pentru CV
        model_cv = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred = model_cv.predict(X_test_cv)

        mse = mean_squared_error(y_test_cv, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_cv, y_pred)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Completat: {i + 1}/30 rulări ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    print(f"\n✓ Timp total CV: {total_time:.2f} secunde")

    print("\n" + "-" * 80)
    print("REZULTATE CROSS-VALIDATION:")
    print("-" * 80)
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
# 5. COMPARAȚIE KERNELS
# ============================================================================

def compare_kernels(X_train, y_train, X_test, y_test):
    """
    Compară performanța diferitelor kernels
    """
    print("\n" + "=" * 80)
    print("COMPARAȚIE KERNELS")
    print("=" * 80)

    kernels = {
        'Linear': {'kernel': 'linear', 'C': 1},
        'RBF': {'kernel': 'rbf', 'C': 100, 'gamma': 'scale'},
        'Polynomial (d=2)': {'kernel': 'poly', 'degree': 2, 'C': 10},
        'Polynomial (d=3)': {'kernel': 'poly', 'degree': 3, 'C': 10}
    }

    kernel_results = {}

    print("\nTestare kernels...")
    for name, params in kernels.items():
        print(f"\n  {name}:")

        try:
            model = SVR(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            kernel_results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }

            print(f"    MSE:  {mse:,.2f}")
            print(f"    RMSE: {rmse:,.2f}")
            print(f"    R²:   {r2:.4f}")

        except Exception as e:
            print(f"    Eroare: {e}")
            kernel_results[name] = None

    # Plot comparație
    valid_results = {k: v for k, v in kernel_results.items() if v is not None}

    if valid_results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        metrics = ['MSE', 'RMSE', 'R2']
        colors = ['coral', 'lightblue', 'lightgreen', 'plum']

        for idx, metric in enumerate(metrics):
            values = [valid_results[k][metric] for k in valid_results.keys()]
            axes[idx].bar(range(len(valid_results)), values, color=colors[:len(valid_results)],
                          edgecolor='black')
            axes[idx].set_xticks(range(len(valid_results)))
            axes[idx].set_xticklabels(list(valid_results.keys()), rotation=15, ha='right')
            axes[idx].set_ylabel(metric, fontsize=11)
            axes[idx].set_title(f'{metric} Comparison Across Kernels',
                                fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('svr_kernel_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Salvat: svr_kernel_comparison.png")
        plt.show()

    return kernel_results


# ============================================================================
# 6. VIZUALIZARE SUPPORT VECTORS
# ============================================================================

def visualize_support_vectors(model, X_train, y_train, feature_names):
    """
    Vizualizează distribuția support vectors
    """
    print("\n" + "=" * 80)
    print("VIZUALIZARE SUPPORT VECTORS")
    print("=" * 80)

    # Informații despre support vectors
    sv_indices = model.support_
    print(f"\nTotal Support Vectors: {len(sv_indices)}")
    print(f"Procent din training: {len(sv_indices) / len(X_train) * 100:.2f}%")

    # Extrage cele mai importante 2 features pentru vizualizare
    # (vom folosi primele 2 features numerice pentru simplitate)
    if len(feature_names) >= 2:
        feat1_idx = 0
        feat2_idx = 1

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot toate punctele
        ax.scatter(X_train.iloc[:, feat1_idx], X_train.iloc[:, feat2_idx],
                   c='lightgray', alpha=0.5, s=30, label='Training points', edgecolor='black')

        # Plot support vectors
        ax.scatter(X_train.iloc[sv_indices, feat1_idx],
                   X_train.iloc[sv_indices, feat2_idx],
                   c='red', s=100, marker='X', label='Support Vectors',
                   edgecolor='black', linewidth=1.5)

        ax.set_xlabel(feature_names[feat1_idx], fontsize=12)
        ax.set_ylabel(feature_names[feat2_idx], fontsize=12)
        ax.set_title('Support Vectors Visualization (First 2 Features)',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('svr_support_vectors.png', dpi=300, bbox_inches='tight')
        print("\n✓ Salvat: svr_support_vectors.png")
        plt.show()


# ============================================================================
# 7. VIZUALIZĂRI PREDICȚII ȘI RESIDUALS
# ============================================================================

def create_visualizations(results, cv_results):
    """
    Creează vizualizări
    """
    print("\n" + "=" * 80)
    print("GENERARE VIZUALIZĂRI")
    print("=" * 80)

    # Figure 1: Predicted vs Actual
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, ax) in enumerate(zip(['Training', 'Validation', 'Test'], axes)):
        y_true = results[name]['y_true']
        y_pred = results[name]['y_pred']
        r2 = results[name]['R2']

        ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='black', color='purple')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Price ($)', fontsize=11)
        ax.set_ylabel('Predicted Price ($)', fontsize=11)
        ax.set_title(f'{name} Set (R² = {r2:.4f})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('svr_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: svr_predictions.png")
    plt.show()

    # Figure 2: Residuals
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, ax) in enumerate(zip(['Training', 'Validation', 'Test'], axes)):
        y_true = results[name]['y_true']
        y_pred = results[name]['y_pred']
        residuals = y_true - y_pred

        ax.scatter(y_pred, residuals, alpha=0.6, edgecolor='black', color='purple')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Price ($)', fontsize=11)
        ax.set_ylabel('Residuals ($)', fontsize=11)
        ax.set_title(f'{name} Set - Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('svr_residuals.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: svr_residuals.png")
    plt.show()

    # Figure 3: CV Distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [('MSE', cv_results['MSE']),
               ('RMSE', cv_results['RMSE']),
               ('R²', cv_results['R2'])]

    for (metric_name, values), ax in zip(metrics, axes):
        ax.hist(values, bins=15, edgecolor='black', alpha=0.7, color='purple')
        ax.axvline(np.mean(values), color='r', linestyle='--', lw=2,
                   label=f'Mean: {np.mean(values):.2f}')
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{metric_name} Distribution (30 runs)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('svr_cv_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: svr_cv_distribution.png")
    plt.show()


# ============================================================================
# 8. SALVARE MODEL
# ============================================================================

def save_model(model, results, cv_results):
    """
    Salvează modelul și rezultatele
    """
    print("\n" + "=" * 80)
    print("SALVARE MODEL ȘI REZULTATE")
    print("=" * 80)

    with open('svr_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Salvat: svr_model.pkl")

    results_summary = {
        'model_name': 'SVR',
        'kernel': model.kernel,
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

    with open('svr_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    print("✓ Salvat: svr_results.pkl")


# ============================================================================
# 9. FUNCȚIA PRINCIPALĂ
# ============================================================================

def main():
    """
    Funcția principală
    """
    print("\n" + "=" * 80)
    print(" " * 18 + "SUPPORT VECTOR REGRESSION")
    print(" " * 15 + "Predicția Prețului Automobilelor")
    print("=" * 80)

    # 1. Încărcare date
    data_dict = load_processed_data()

    # 2. Antrenare SVR
    svr_model = train_svr(
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val'],
        tune_hyperparams=False  # Set True pentru GridSearch (foarte lent!)
    )

    # 3. Evaluare
    results = evaluate_model(
        svr_model,
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val'],
        data_dict['X_test'], data_dict['y_test']
    )

    # 4. Comparație kernels
    kernel_results = compare_kernels(
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_test'], data_dict['y_test']
    )

    # 5. Vizualizare support vectors
    visualize_support_vectors(
        svr_model,
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['X_train'].columns
    )

    # 6. Cross-validation
    X_combined = pd.concat([data_dict['X_train'], data_dict['X_val']])
    y_combined = pd.concat([data_dict['y_train'], data_dict['y_val']])
    cv_results = perform_cross_validation(X_combined, y_combined, n_runs=30)

    # 7. Vizualizări
    create_visualizations(results, cv_results)

    # 8. Salvare
    save_model(svr_model, results, cv_results)

    print("\n" + "=" * 80)
    print("SVR - FINALIZAT CU SUCCES!")
    print("=" * 80)

    return svr_model, results, cv_results


# ============================================================================
# EXECUȚIE
# ============================================================================

if __name__ == "__main__":
    svr_model, results, cv_results = main()