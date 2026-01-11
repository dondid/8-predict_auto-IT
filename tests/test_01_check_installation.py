"""
Test: Verificare instalare librării
Rulează acest script înainte de a începe proiectul
"""

import sys

print("=" * 80)
print(" " * 25 + "VERIFICARE LIBRARIES")
print("=" * 80)

# Lista de librării necesare
required_libraries = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'xgboost': 'xgboost',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'shap': 'shap',
    'scipy': 'scipy'
}

print(f"\nPython Version: {sys.version}")
print("\n" + "-" * 80)
print("Checking libraries...")
print("-" * 80)

missing_libraries = []
installed_libraries = []

for import_name, package_name in required_libraries.items():
    try:
        if import_name == 'sklearn':
            import sklearn

            version = sklearn.__version__
        else:
            lib = __import__(import_name)
            version = lib.__version__ if hasattr(lib, '__version__') else 'N/A'

        print(f"✓ {package_name:20s} - Version: {version}")
        installed_libraries.append(package_name)
    except ImportError:
        print(f"✗ {package_name:20s} - NOT INSTALLED")
        missing_libraries.append(package_name)

print("\n" + "=" * 80)

if missing_libraries:
    print("❌ MISSING LIBRARIES!")
    print("\nPentru a instala toate librăriile lipsă, rulează:")
    print("\n  pip install " + " ".join(missing_libraries))
    print("\nSau:")
    print("\n  pip install -r requirements.txt")
else:
    print("✅ TOATE LIBRĂRIILE SUNT INSTALATE!")
    print("\nPoți continua cu rularea proiectului:")
    print("  python 00_master_pipeline.py")

print("=" * 80)

# Test rapid funcționalitate
if not missing_libraries:
    print("\n" + "=" * 80)
    print("QUICK FUNCTIONALITY TEST")
    print("=" * 80)

    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor

        # Test creeare date dummy
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        # Test RF
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)
        pred_rf = rf.predict(X[:5])
        print("✓ Random Forest: functional")

        # Test XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=10, random_state=42)
        xgb_model.fit(X, y, verbose=False)
        pred_xgb = xgb_model.predict(X[:5])
        print("✓ XGBoost: functional")

        # Test SVR
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X, y)
        pred_svr = svr_model.predict(X[:5])
        print("✓ SVR: functional")

        # Test NN
        nn_model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
        nn_model.fit(X, y)
        pred_nn = nn_model.predict(X[:5])
        print("✓ Neural Network: functional")

        print("\n✅ TOATE MODELELE FUNCȚIONEAZĂ CORECT!")

    except Exception as e:
        print(f"\n❌ Eroare la testare funcționalitate: {e}")

print("=" * 80)