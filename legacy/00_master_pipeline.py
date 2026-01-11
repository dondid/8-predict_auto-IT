"""
PROIECT MACHINE LEARNING: Predicția Prețului Automobilelor
MASTER PIPELINE - Execută tot workflow-ul automat

Data: Ianuarie 2025
Curs: Machine Learning
"""

import sys
import time
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURARE
# ============================================================================

# Modele de rulat (setează False pentru a sări peste unele modele)
RUN_MODELS = {
    'data_loading': True,
    'preprocessing': True,
    'random_forest': True,
    'xgboost': True,
    'svr': True,
    'neural_network': True,
    'comparison': True
}


# ============================================================================
# FUNCȚII HELPER
# ============================================================================

def print_header(text):
    """Print header formatat"""
    print("\n" + "=" * 80)
    print(" " * int((80 - len(text)) / 2) + text)
    print("=" * 80)


def print_step(step_num, step_name):
    """Print step număr și nume"""
    print("\n" + "-" * 80)
    print(f"STEP {step_num}: {step_name}")
    print("-" * 80)


def run_module(module_name, description):
    """
    Execută un modul și cronometrează timpul
    """
    print_step(module_name.split('_')[0], description)

    start_time = time.time()

    try:
        # Import dinamic
        if module_name == "01_data_loading":
            from importlib import import_module
            module = import_module('01_data_loading')
            module.main()
        elif module_name == "02_preprocessing":
            from importlib import import_module
            module = import_module('02_data_preprocessing')
            module.main()
        elif module_name == "03_random_forest":
            from importlib import import_module
            module = import_module('03_random_forest_model')
            module.main()
        elif module_name == "04_xgboost":
            from importlib import import_module
            module = import_module('04_xgboost_model')
            module.main()
        elif module_name == "05_svr":
            from importlib import import_module
            module = import_module('05_svr_model')
            module.main()
        elif module_name == "06_neural_network":
            from importlib import import_module
            module = import_module('06_neural_network_model')
            module.main()
        elif module_name == "07_comparison":
            from importlib import import_module
            module = import_module('07_model_comparison_statistical')
            module.main()

        elapsed_time = time.time() - start_time
        print(f"\n✓ {description} - Finalizat în {elapsed_time:.2f} secunde")
        return True

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ EROARE în {description}:")
        print(f"  {str(e)}")
        print(f"  Timp până la eroare: {elapsed_time:.2f} secunde")
        return False


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """
    Funcția principală - execută întregul pipeline
    """
    print_header("MASTER PIPELINE")
    print("\nProiect: Predicția Prețului Automobilelor")
    print("Dataset: UCI Automobile Data Set")
    print("Modele: Random Forest, XGBoost, SVR, Neural Network")
    print("\nAcest script va rula întregul workflow automat.")
    print("Estimare timp total: 15-30 minute (depinde de hardware)")

    # Confirmare
    response = input("\nDoriți să continuați? (da/nu): ").lower()
    if response not in ['da', 'yes', 'y']:
        print("Pipeline anulat.")
        return

    total_start_time = time.time()
    results = {}

    # ========================================================================
    # ETAPA 1: ÎNCĂRCAREA ȘI EXPLORAREA DATELOR
    # ========================================================================

    if RUN_MODELS['data_loading']:
        success = run_module("01_data_loading",
                             "Încărcarea și Explorarea Datelor")
        results['data_loading'] = success
        if not success:
            print("\n⚠ Oprire pipeline din cauza erorii!")
            return

    # ========================================================================
    # ETAPA 2: PREPROCESSING
    # ========================================================================

    if RUN_MODELS['preprocessing']:
        success = run_module("02_preprocessing",
                             "Preprocessing și Feature Engineering")
        results['preprocessing'] = success
        if not success:
            print("\n⚠ Oprire pipeline din cauza erorii!")
            return

    # ========================================================================
    # ETAPA 3: ANTRENAREA MODELELOR
    # ========================================================================

    print_header("ANTRENAREA MODELELOR")

    # Random Forest
    if RUN_MODELS['random_forest']:
        success = run_module("03_random_forest",
                             "Random Forest Regressor")
        results['random_forest'] = success

    # XGBoost
    if RUN_MODELS['xgboost']:
        success = run_module("04_xgboost",
                             "XGBoost Regressor")
        results['xgboost'] = success

    # SVR
    if RUN_MODELS['svr']:
        success = run_module("05_svr",
                             "Support Vector Regression")
        results['svr'] = success

    # Neural Network
    if RUN_MODELS['neural_network']:
        success = run_module("06_neural_network",
                             "Neural Network (MLP)")
        results['neural_network'] = success

    # ========================================================================
    # ETAPA 4: COMPARAȚIE STATISTICĂ
    # ========================================================================

    if RUN_MODELS['comparison']:
        print_header("COMPARAȚIE ȘI ANALIZĂ STATISTICĂ")
        success = run_module("07_comparison",
                             "Comparație Statistică între Modele")
        results['comparison'] = success

    # ========================================================================
    # RAPORT FINAL
    # ========================================================================

    total_elapsed_time = time.time() - total_start_time

    print_header("RAPORT FINAL PIPELINE")

    print(f"\nTimp total execuție: {total_elapsed_time / 60:.2f} minute")
    print("\nRezumat etape:")
    print("-" * 80)

    for step, success in results.items():
        status = "✓ Succes" if success else "✗ Eroare"
        print(f"  {step:25s}: {status}")

    successful_steps = sum(results.values())
    total_steps = len(results)

    print(f"\nTotal: {successful_steps}/{total_steps} etape finalizate cu succes")

    if successful_steps == total_steps:
        print("\n🎉 PIPELINE FINALIZAT CU SUCCES!")
        print("\nFișiere generate:")
        print("  • Plots: *.png")
        print("  • Models: *_model.pkl")
        print("  • Results: *_results.pkl")
        print("  • Data: processed_data.pkl")
        print("  • Reports: *.csv, final_report.txt")
    else:
        print("\n⚠ Pipeline finalizat cu erori!")
        print("Verificați output-ul pentru detalii.")

    print("\n" + "=" * 80)
    print("Pentru prezentare, folosiți:")
    print("  • Slides: Creați PowerPoint cu plot-urile generate")
    print("  • Demo: Arătați codul și vizualizările")
    print("  • Raport: final_report.txt")
    print("=" * 80)


# ============================================================================
# EXECUȚIE
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline întrerupt de utilizator!")
        print("Progresul poate fi reluat rulând modulele individual.")
    except Exception as e:
        print(f"\n\n✗ EROARE CRITICĂ în pipeline:")
        print(f"  {str(e)}")
        import traceback

        traceback.print_exc()