import argparse
import sys
import pandas as pd
from src.utils import setup_logging, get_logger
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.svr import SVRModel
from src.models.neural_network import NeuralNetworkModel
from src.visualization import plots

logger = get_logger("main")

def run_pipeline(args):
    """
    Executes the pipeline steps based on arguments.
    """
    # 1. Load Data
    loader = DataLoader()
    df = loader.load_data()
    
    # 2. Preprocessing
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(df)

    if args.compare:
        logger.info("Running STATISTICAL COMPARISON mode...")
        from src.evaluation.statistical_tests import ModelComparator
        
        # Instantiate wrappers to access underlying models
        models_to_compare = {
            'Random Forest': RandomForestModel().model,
            'XGBoost': XGBoostModel().model,
            'SVR': SVRModel().model,
            'Neural Network': NeuralNetworkModel().model
        }
        
        # Use X_train for repeated CV (it's scaled)
        # We perform CV on the training split to keep test set pure
        comparator = ModelComparator(models_to_compare, data['X_train'], data['y_train'])
        comparator.run_repeated_cv(n_repeats=10, k_folds=5) 
        comparator.perform_wilcoxon_tests()
        comparator.generate_final_report()
        return
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # 3. Model Training & Evaluation
    models_to_run = []
    
    if args.model == 'all' or args.model == 'rf':
        models_to_run.append(RandomForestModel())
    if args.model == 'all' or args.model == 'xgb':
        models_to_run.append(XGBoostModel())
    if args.model == 'all' or args.model == 'svr':
        models_to_run.append(SVRModel())
    if args.model == 'all' or args.model == 'nn':
        models_to_run.append(NeuralNetworkModel())
        
    results = []
    
    for model in models_to_run:
        logger.info(f"Running {model.name}...")
        
        # Train
        model.train(X_train, y_train)
        
        # Evaluate
        metrics, predictions = model.evaluate(X_test, y_test)
        results.append(metrics)
        
        # Visualization
        plots.plot_predictions(y_test, predictions, 
                             title=f"{model.name} Predictions vs True", 
                             filename=f"{model.name.lower()}_pred_vs_true.png")
        plots.plot_residuals(y_test, predictions, 
                           title=f"{model.name} Residuals", 
                           filename=f"{model.name.lower()}_residuals.png")
        
        # Save
        model.save()
        
    # Summary
    if results:
        results_df = pd.DataFrame(results)
        print("\n=== Final Results ===")
        print(results_df.to_string(index=False))
        results_df.to_csv("outputs/reports/final_metrics.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description="Automobile Price Prediction Pipeline")
    parser.add_argument('--run-all', action='store_true', help='Run full pipeline')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', 'rf', 'xgb', 'svr', 'nn'], 
                        help='Specific model to run (default: all)')
    parser.add_argument('--compare', action='store_true', help='Run statistical comparison (Slow: Repeated CV)')
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info("Starting pipeline...")
    
    try:
        run_pipeline(args)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
