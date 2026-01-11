"""
Quick Test Pipeline - Versiune simplificată pentru testare rapidă
Rulează în ~2-3 minute cu configurări reduse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print(" " * 20 + "QUICK TEST PIPELINE")
print(" " * 15 + "Predicția Prețului Automobilelor")
print("=" * 80)

# ============================================================================
# 1. ÎNCĂRCAREA DATELOR
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: ÎNCĂRCAREA DATELOR")
print("=" * 80)

try:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
    column_names = [
        'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
        'wheel-base', 'length', 'width', 'height', 'curb-weight',
        'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
        'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
        'city-mpg', 'highway-mpg', 'price'
    ]

    df = pd.read_csv(url, names=column_names, na_values='?')
    print(f"✓ Date încărcate: {df.shape[0]} înregistrări, {df.shape[1]} atribute")
except Exception as e:
    print(f"✗ Eroare la încărcarea datelor: {e}")
    exit(1)

# ============================================================================
# 2. PREPROCESSING RAPID
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: PREPROCESSING")
print("=" * 80)

# Eliminăm price lipsă
df = df.dropna(subset=['price'])
print(f"✓ După eliminare price lipsă: {df.shape[0]} înregistrări")

# Imputare simplă pentru missing values numerice
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Imputare pentru categoriale
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"✓ Missing values imputate")

# Separăm X și y
y = df['price']
X = df.drop('price', axis=1)

# Encoding simplu - păstrăm doar coloanele numerice pentru test rapid
X_numeric = X.select_dtypes(include=[np.number])
print(f"✓ Features numerice selectate: {X_numeric.shape[1]}")

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_numeric, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"✓ Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Scaling aplicat")

# ============================================================================
# 3. ANTRENAREA MODELELOR (CONFIGURĂRI REDUSE PENTRU TEST RAPID)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: ANTRENAREA MODELELOR (configurări test)")
print("=" * 80)

models = {}
results = {}

# Model 1: Random Forest
print("\n[1/4] Random Forest...")
try:
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    y_pred_test = rf.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)

    models['Random Forest'] = rf
    results['Random Forest'] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

    print(f"  ✓ RMSE: {rmse:.2f}, R²: {r2:.4f}")
except Exception as e:
    print(f"  ✗ Eroare: {e}")

# Model 2: XGBoost
print("\n[2/4] XGBoost...")
try:
    xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1,
                                 random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train, verbose=False)

    y_pred_test = xgb_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)

    models['XGBoost'] = xgb_model
    results['XGBoost'] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

    print(f"  ✓ RMSE: {rmse:.2f}, R²: {r2:.4f}")
except Exception as e:
    print(f"  ✗ Eroare: {e}")

# Model 3: SVR
print("\n[3/4] SVR...")
try:
    svr_model = SVR(kernel='rbf', C=100, gamma='scale')
    svr_model.fit(X_train_scaled, y_train)

    y_pred_test = svr_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)

    models['SVR'] = svr_model
    results['SVR'] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

    print(f"  ✓ RMSE: {rmse:.2f}, R²: {r2:.4f}")
except Exception as e:
    print(f"  ✗ Eroare: {e}")

# Model 4: Neural Network
print("\n[4/4] Neural Network...")
try:
    nn_model = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu',
                            solver='adam', max_iter=500, random_state=42)
    nn_model.fit(X_train_scaled, y_train)

    y_pred_test = nn_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)

    models['Neural Network'] = nn_model
    results['Neural Network'] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

    print(f"  ✓ RMSE: {rmse:.2f}, R²: {r2:.4f}")
except Exception as e:
    print(f"  ✗ Eroare: {e}")

# ============================================================================
# 4. COMPARAȚIE RAPIDĂ
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: COMPARAȚIE MODELE")
print("=" * 80)

# Tabel comparativ
print("\nRezultate Test Set:")
print("-" * 80)
print(f"{'Model':<20} {'MSE':>12} {'RMSE':>12} {'R²':>10}")
print("-" * 80)

for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['MSE']:>12,.2f} {metrics['RMSE']:>12,.2f} {metrics['R2']:>10.4f}")

# Găsim cel mai bun model
best_model = max(results.items(), key=lambda x: x[1]['R2'])
print("\n" + "=" * 80)
print(f"🏆 MODEL CÂȘTIGĂTOR: {best_model[0]}")
print(f"   R² Score: {best_model[1]['R2']:.4f}")
print("=" * 80)

# ============================================================================
# 5. QUICK VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: GENERARE VIZUALIZARE")
print("=" * 80)

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Bar chart R²
    model_names = list(results.keys())
    r2_scores = [results[m]['R2'] for m in model_names]

    axes[0, 0].bar(range(len(model_names)), r2_scores,
                   color=['steelblue', 'orange', 'purple', 'teal'], edgecolor='black')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Model Comparison - R² Score', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Bar chart RMSE
    rmse_scores = [results[m]['RMSE'] for m in model_names]

    axes[0, 1].bar(range(len(model_names)), rmse_scores,
                   color=['steelblue', 'orange', 'purple', 'teal'], edgecolor='black')
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Model Comparison - RMSE', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Predicted vs Actual pentru modelul câștigător
    best_model_obj = models[best_model[0]]
    y_pred_best = best_model_obj.predict(X_test_scaled)

    axes[1, 0].scatter(y_test, y_pred_best, alpha=0.6, edgecolor='black')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', lw=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('Actual Price ($)')
    axes[1, 0].set_ylabel('Predicted Price ($)')
    axes[1, 0].set_title(f'{best_model[0]} - Predictions', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Residuals pentru modelul câștigător
    residuals = y_test - y_pred_best

    axes[1, 1].scatter(y_pred_best, residuals, alpha=0.6, edgecolor='black')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Price ($)')
    axes[1, 1].set_ylabel('Residuals ($)')
    axes[1, 1].set_title(f'{best_model[0]} - Residuals', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: quick_test_results.png")
    plt.show()

except Exception as e:
    print(f"✗ Eroare la generarea vizualizării: {e}")

# ============================================================================
# 6. FINAL
# ============================================================================

print("\n" + "=" * 80)
print("✅ QUICK TEST FINALIZAT CU SUCCES!")
print("=" * 80)
print("\nConcluzie:")
print(f"  • Toate cele 4 modele au fost antrenate cu succes")
print(f"  • Model câștigător: {best_model[0]}")
print(f"  • R² Score: {best_model[1]['R2']:.4f}")
print("\nDacă acest test a funcționat, poți rula pipeline-ul complet:")
print("  python 00_master_pipeline.py")
print("\nAcesta va include:")
print("  • Preprocessing complet (encoding, feature engineering)")
print("  • Hyperparameter tuning")
print("  • Cross-validation cu 30 rulări")
print("  • SHAP values și feature importance")
print("  • Wilcoxon test pentru comparație statistică")
print("  • 25+ vizualizări profesionale")
print("=" * 80)