import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from pathlib import Path
from src.config import REPORTS_DIR, FIGURES_DIR
from src.utils import get_logger

logger = get_logger(__name__)

class ModelComparator:
    def __init__(self, models_dict, X, y):
        """
        models_dict: dictionary of {name: model_instance}
        X, y: data for cross-validation
        """
        self.models = models_dict
        self.X = X
        self.y = y
        self.results = {}
        self.raw_scores = {} # Store raw scores for Wilcoxon

    def run_repeated_cv(self, n_repeats=10, k_folds=10):
        """
        Runs repeated Cross-Validation to gather a distribution of scores.
        """
        logger.info(f"Starting repeated CV ({n_repeats} repeats, {k_folds} folds)...")
        
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            all_scores = []
            
            # We use the underlying sklearn model if our wrapper has one, 
            # or the object itself if it adheres to sklearn API
            estimator = getattr(model, 'model', model)
            
            for i in range(n_repeats):
                # Different seed for each repeat to get variance
                cv = KFold(n_splits=k_folds, shuffle=True, random_state=42 + i)
                # cross_val_score returns R2 by default for regressors if scoring not specified, 
                # but let's be explicit. We use 'r2'.
                scores = cross_val_score(estimator, self.X, self.y, scoring='r2', cv=cv, n_jobs=-1)
                all_scores.extend(scores)
                
            self.raw_scores[name] = np.array(all_scores)
            self.results[name] = {
                'mean_r2': np.mean(all_scores),
                'std_r2': np.std(all_scores)
            }
            logger.info(f"{name}: Mean R2 = {np.mean(all_scores):.4f} (+/- {np.std(all_scores):.4f})")
            
        return self.results

    def perform_wilcoxon_tests(self):
        """
        Performs pairwise Wilcoxon Signed-Rank tests.
        """
        logger.info("Performing Wilcoxon tests...")
        model_names = list(self.models.keys())
        n_models = len(model_names)
        
        wilcoxon_results = []
        p_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                m1 = model_names[i]
                m2 = model_names[j]
                
                scores1 = self.raw_scores[m1]
                scores2 = self.raw_scores[m2]
                
                # Wilcoxon requires same length, paired samples. 
                # Since we used same seeds for KFold loop order, they are paired by fold/repeat index.
                stat, p_val = stats.wilcoxon(scores1, scores2)
                
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
                
                better = m1 if np.mean(scores1) > np.mean(scores2) else m2
                significance = "Significant" if p_val < 0.05 else "Not Significant"
                
                wilcoxon_results.append({
                    'Model A': m1,
                    'Model B': m2,
                    'p-value': p_val,
                    'Significance': significance,
                    'Winner': better if p_val < 0.05 else "Draw"
                })
                
        # Save results
        df_res = pd.DataFrame(wilcoxon_results)
        df_res.to_csv(REPORTS_DIR / 'wilcoxon_results.csv', index=False)
        logger.info(f"Wilcoxon results saved to {REPORTS_DIR}")
        
        self._plot_heatmap(p_matrix, model_names)
        self._plot_boxplots()
        
        return df_res

    def _plot_heatmap(self, p_matrix, names):
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)
        sns.heatmap(p_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                    xticklabels=names, yticklabels=names, mask=mask,
                    vmin=0, vmax=0.05) # Green for p < 0.05
        plt.title('Wilcoxon Test P-Values (Green = Significant Difference)')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'wilcoxon_heatmap.png', dpi=300)
        plt.close()

    def _plot_boxplots(self):
        plt.figure(figsize=(12, 6))
        df_scores = pd.DataFrame(self.raw_scores)
        sns.boxplot(data=df_scores)
        plt.title('R2 Score Distribution (Repeated CV)')
        plt.ylabel('R2 Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'cv_distribution_boxplot.png', dpi=300)
        plt.close()

    def generate_final_report(self):
        # Generate a text summary aligned with requirements
        best_model = max(self.results, key=lambda x: self.results[x]['mean_r2'])
        
        report = f"""
================================================================================
FINAL STATISTICAL EVALUATION REPORT
================================================================================

1. METHODOLOGY
   - Method: Repeated k-Fold Cross-Validation
   - Repeats: 10
   - Folds: 10
   - Total evaluations per model: 100
   - Metric: R-squared (Coefficient of Determination)

2. MODEL PERFORMANCE (Mean +/- Std Dev)
--------------------------------------------------------------------------------
"""
        for name, res in self.models.items(): # Use models dict for order, but access results
             if name in self.results:
                r = self.results[name]
                report += f"{name:20s}: {r['mean_r2']:.4f} (+/- {r['std_r2']:.4f})\n"

        report += f"""
3. STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test)
--------------------------------------------------------------------------------
   The Wilcoxon test was performed to determine if the differences in performance
   are statistically significant (p < 0.05).

   Best Performing Model (Numerical): {best_model}

   Key Comparisons:
"""
        # Load the df we just saved/created
        df = pd.read_csv(REPORTS_DIR / 'wilcoxon_results.csv')
        for _, row in df.iterrows():
            report += f"   - {row['Model A']} vs {row['Model B']}: p={row['p-value']:.4f} ({row['Significance']})\n"

        with open(REPORTS_DIR / 'statistical_compliance_report.txt', 'w') as f:
            f.write(report)
            
        logger.info(f"Final compliance report generated: {REPORTS_DIR / 'statistical_compliance_report.txt'}")
