"""
Proiect Machine Learning: Predicția Prețului Automobilelor
Etapa 6: Neural Network (MLPRegressor) - Model Bonus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings

warnings.filterwarnings('ignore')


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
# 2. NEURAL NETWORK CU HYPERPARAMETER TUNING
# ============================================================================

def train_neural_network(X_train, y_train, X_val, y_val, tune_hyperparams=True):
    """
    Antrenează Neural Network (MLP) cu tuning opțional
    """
    print("\n" + "=" * 80)
    print("NEURAL NETWORK (MLP REGRESSOR)")
    print("=" * 80)

    if tune_hyperparams:
        print("\nHyperparameter Tuning cu GridSearchCV...")

        # Grid de parametri
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50), (100, 50, 30)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }

        # Model de bază
        mlp_base = MLPRegressor(
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )

        # GridSearch
        grid_search = GridSearchCV(
            estimator=mlp_base,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )

        print("Antrenare GridSearchCV...")
        grid_search.fit(X_train, y_train)

        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score (MSE): {-grid_search.best_score_:.2f}")

        mlp_model = grid_search.best_estimator_

    else:
        print("\nAntrenare cu arhitectură optimizată...")
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 30),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            verbose=False
        )

        mlp_model.fit(X_train, y_train)

        print(f"✓ Număr de iterații: {mlp_model.n_iter_}")
        print(f"✓ Loss final: {mlp_model.loss_:.4f}")

    print("\n✓ Model Neural Network antrenat cu succes!")
    print(f"\nArhitectură:")
    print(f"  Input layer: {X_train.shape[1]} neuroni")
    if hasattr(mlp_model, 'hidden_layer_sizes'):
        for i, size in enumerate(mlp_model.hidden_layer_sizes):
            print(f"  Hidden layer {i + 1}: {size} neuroni")
    print(f"  Output layer: 1 neuron")

    return mlp_model


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

        # Model MLP simplu pentru CV
        model_cv = MLPRegressor(
            hidden_layer_sizes=(100, 50, 30),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=1000,
            early_stopping=True,
            random_state=42,
            verbose=False
        )

        model_cv.fit(X_train_cv, y_train_cv)
        y_pred = model_cv.predict(X_test_cv)

        mse = mean_squared_error(y_test_cv, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_cv, y_pred)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        if (i + 1) % 10 == 0:
            print(f"  Completat: {i + 1}/30 rulări")

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
# 5. PLOT LEARNING CURVES
# ============================================================================

def plot_learning_curves(model):
    """
    Plot learning curves pentru NN
    """
    print("\n" + "=" * 80)
    print("LEARNING CURVES")
    print("=" * 80)

    if hasattr(model, 'loss_curve_'):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(model.loss_curve_, linewidth=2, label='Training Loss')
        if hasattr(model, 'validation_scores_'):
            # Calculăm loss din score (R²)
            val_loss = [1 - score for score in model.validation_scores_]
            ax.plot(val_loss, linewidth=2, label='Validation Loss')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Neural Network Learning Curves', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('nn_learning_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Salvat: nn_learning_curves.png")
        plt.show()
    else:
        print("Model nu are loss_curve_ disponibil")


# ============================================================================
# 6. WEIGHT DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_weights(model):
    """
    Analizează distribuția weights în NN
    """
    print("\n" + "=" * 80)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 80)

    if hasattr(model, 'coefs_'):
        n_layers = len(model.coefs_)

        fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))

        if n_layers == 1:
            axes = [axes]

        for i, (coef, ax) in enumerate(zip(model.coefs_, axes)):
            weights = coef.flatten()

            ax.hist(weights, bins=50, edgecolor='black', alpha=0.7, color='teal')
            ax.set_xlabel('Weight Value', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Layer {i + 1} Weights Distribution', fontsize=12, fontweight='bold')
            ax.axvline(0, color='r', linestyle='--', linewidth=2)
            ax.grid(True, alpha=0.3, axis='y')

            print(f"\nLayer {i + 1}:")
            print(f"  Shape: {coef.shape}")
            print(f"  Mean: {np.mean(weights):.4f}")
            print(f"  Std: {np.std(weights):.4f}")
            print(f"  Min: {np.min(weights):.4f}")
            print(f"  Max: {np.max(weights):.4f}")

        plt.tight_layout()
        plt.savefig('nn_weight_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✓ Salvat: nn_weight_distribution.png")
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

        ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='black', color='teal')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Price ($)', fontsize=11)
        ax.set_ylabel('Predicted Price ($)', fontsize=11)
        ax.set_title(f'{name} Set (R² = {r2:.4f})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nn_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: nn_predictions.png")
    plt.show()

    # Figure 2: Residuals
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, ax) in enumerate(zip(['Training', 'Validation', 'Test'], axes)):
        y_true = results[name]['y_true']
        y_pred = results[name]['y_pred']
        residuals = y_true - y_pred

        ax.scatter(y_pred, residuals, alpha=0.6, edgecolor='black', color='teal')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Price ($)', fontsize=11)
        ax.set_ylabel('Residuals ($)', fontsize=11)
        ax.set_title(f'{name} Set - Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nn_residuals.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: nn_residuals.png")
    plt.show()

    # Figure 3: CV Distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [('MSE', cv_results['MSE']),
               ('RMSE', cv_results['RMSE']),
               ('R²', cv_results['R2'])]

    for (metric_name, values), ax in zip(metrics, axes):
        ax.hist(values, bins=15, edgecolor='black', alpha=0.7, color='teal')
        ax.axvline(np.mean(values), color='r', linestyle='--', lw=2,
                   label=f'Mean: {np.mean(values):.2f}')
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{metric_name} Distribution (30 runs)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('nn_cv_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: nn_cv_distribution.png")
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

    with open('nn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Salvat: nn_model.pkl")

    results_summary = {
        'model_name': 'Neural Network',
        'architecture': model.hidden_layer_sizes if hasattr(model, 'hidden_layer_sizes') else 'N/A',
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

    with open('nn_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    print("✓ Salvat: nn_results.pkl")


# ============================================================================
# 9. FUNCȚIA PRINCIPALĂ
# ============================================================================

def main():
    """
    Funcția principală
    """
    print("\n" + "=" * 80)
    print(" " * 22 + "NEURAL NETWORK (MLP)")
    print(" " * 15 + "Predicția Prețului Automobilelor")
    print("=" * 80)

    # 1. Încărcare date
    data_dict = load_processed_data()

    # 2. Antrenare
    nn_model = train_neural_network(
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val'],
        tune_hyperparams=False
    )

    # 3. Evaluare
    results = evaluate_model(
        nn_model,
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val'],
        data_dict['X_test'], data_dict['y_test']
    )

    # 4. Learning curves
    plot_learning_curves(nn_model)

    # 5. Weight analysis
    analyze_weights(nn_model)

    # 6. Cross-validation
    X_combined = pd.concat([data_dict['X_train'], data_dict['X_val']])
    y_combined = pd.concat([data_dict['y_train'], data_dict['y_val']])
    cv_results = perform_cross_validation(X_combined, y_combined, n_runs=30)

    # 7. Vizualizări
    create_visualizations(results, cv_results)

    # 8. Salvare
    save_model(nn_model, results, cv_results)

    print("\n" + "=" * 80)
    print("NEURAL NETWORK - FINALIZAT CU SUCCES!")
    print("=" * 80)

    return nn_model, results, cv_results


# ============================================================================
# EXECUȚIE
# ============================================================================

if __name__ == "__main__":
    nn_model, results, cv_results = main()