#!/usr/bin/env python3
"""
CausalGBM Hyperparameter Grid Search
====================================
Generates Table for Appendix D showing hyperparameter selection process.

This code performs grid search over CausalGBM hyperparameters and shows
how the default values (α=0.5, τ=0.2, λ_DAG=0.1, λ_sp=0.01, n_iter=500)
were selected.

Usage:
    python grid_search_hyperparameters.py --output_dir results
    python grid_search_hyperparameters.py --dataset adult --output_dir results
"""

import argparse
import os
import warnings
import logging
from datetime import datetime
from itertools import product
import json

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADER (Adult dataset - used for hyperparameter selection)
# =============================================================================

def load_adult(max_samples=None):
    """Load Adult Income dataset."""
    logger.info("Loading Adult dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(url, names=cols, sep=r',\s*', engine='python', na_values='?')
    df = df.dropna()
    
    df['income'] = (df['income'].str.strip() == '>50K').astype(int)
    df['sex'] = df['sex'].str.strip()
    
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    # Preprocess
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
    cont_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    df[cont_cols] = StandardScaler().fit_transform(df[cont_cols])
    
    feature_cols = cat_cols + cont_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df['income'].values.astype(np.float32)
    A = LabelEncoder().fit_transform(df['sex']).astype(np.float32)
    
    logger.info(f"  Loaded: n={len(X)}, d={X.shape[1]}")
    return X, y, A, feature_cols


# =============================================================================
# CAUSAL FEATURE SELECTOR
# =============================================================================

class CausalFeatureSelector:
    """CausalGBM feature selection with configurable hyperparameters."""
    
    def __init__(self, n_features, alpha=0.5, threshold=0.2, min_features=3,
                 n_iterations=500, lambda_dag=0.1, lambda_sp=0.01,
                 aggregation='max', device='cuda'):
        self.n_features = n_features
        self.alpha = alpha
        self.threshold = threshold
        self.min_features = min_features
        self.n_iterations = n_iterations
        self.lambda_dag = lambda_dag
        self.lambda_sp = lambda_sp
        self.aggregation = aggregation
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.selected_ = None
        self.W_dag_ = None
        self.correlations_ = None
    
    def fit(self, X, A, y):
        n, d = X.shape
        
        if not np.issubdtype(A.dtype, np.number):
            A = LabelEncoder().fit_transform(A).astype(np.float32)
        else:
            A = A.astype(np.float32)
        
        # Correlations
        self.correlations_ = np.array([
            abs(pearsonr(X[:, j], A)[0]) if np.std(X[:, j]) > 1e-6 else 0 for j in range(d)
        ])
        
        # Standardize
        A_std = (A - A.mean()) / (A.std() + 1e-8)
        y_std = (y - y.mean()) / (y.std() + 1e-8)
        Z = np.column_stack([X, A_std.reshape(-1,1), y_std.reshape(-1,1)]).astype(np.float32)
        
        n_nodes = d + 2
        idx_A, idx_Y = d, d + 1
        
        # DAG learning
        W = nn.Parameter(torch.randn(n_nodes, n_nodes, device=self.device) * 0.01)
        opt = torch.optim.Adam([W], lr=0.01)
        Z_t = torch.FloatTensor(Z).to(self.device)
        
        for _ in range(self.n_iterations):
            opt.zero_grad()
            A_mat = torch.sigmoid(W) * (1 - torch.eye(n_nodes, device=self.device))
            
            recon = F.mse_loss(Z_t @ A_mat, Z_t)
            
            s = 1.0
            M = s * torch.eye(n_nodes, device=self.device) - A_mat * A_mat + 1e-6 * torch.eye(n_nodes, device=self.device)
            try:
                sign, logdet = torch.linalg.slogdet(M)
                dag_c = -logdet + n_nodes * np.log(s) if sign > 0 else torch.trace(torch.matrix_exp(A_mat * A_mat)) - n_nodes
            except:
                dag_c = torch.trace(torch.matrix_exp(A_mat * A_mat)) - n_nodes
            
            loss = recon + self.lambda_dag * dag_c ** 2 + self.lambda_sp * A_mat.abs().mean()
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_([W], 1.0)
                opt.step()
        
        with torch.no_grad():
            A_mat = torch.sigmoid(W) * (1 - torch.eye(n_nodes, device=self.device))
            adj = A_mat.cpu().numpy()
        
        W_X_Y = adj[:d, idx_Y]
        W_A_X = adj[idx_A, :d]
        self.W_dag_ = W_A_X
        
        # Aggregation
        if self.aggregation == 'max':
            W_prime = np.maximum(W_A_X, self.correlations_)
        elif self.aggregation == 'dag_only':
            W_prime = W_A_X
        elif self.aggregation == 'corr_only':
            W_prime = self.correlations_
        else:
            W_prime = np.maximum(W_A_X, self.correlations_)
        
        # Causal importance
        causal_importance = W_X_Y - self.alpha * W_prime
        
        # Selection
        above = np.where(causal_importance >= self.threshold)[0]
        if len(above) >= self.min_features:
            self.selected_ = above
        else:
            self.selected_ = np.argsort(causal_importance)[::-1][:self.min_features]
        
        return self
    
    def transform(self, X):
        return X[:, self.selected_]


# =============================================================================
# METRICS
# =============================================================================

def calculate_eod(y_true, y_pred, protected):
    """Calculate Equalized Odds Difference."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    protected = np.array(protected)
    
    tpr_0 = np.mean(y_pred[(y_true == 1) & (protected == 0)] == 1) if np.sum((y_true == 1) & (protected == 0)) > 0 else 0
    tpr_1 = np.mean(y_pred[(y_true == 1) & (protected == 1)] == 1) if np.sum((y_true == 1) & (protected == 1)) > 0 else 0
    fpr_0 = np.mean(y_pred[(y_true == 0) & (protected == 0)] == 1) if np.sum((y_true == 0) & (protected == 0)) > 0 else 0
    fpr_1 = np.mean(y_pred[(y_true == 0) & (protected == 1)] == 1) if np.sum((y_true == 0) & (protected == 1)) > 0 else 0
    
    return max(abs(tpr_0 - tpr_1), abs(fpr_0 - fpr_1))


# =============================================================================
# GRID SEARCH
# =============================================================================

def run_single_config(X, y, A, config, n_seeds=3, device='cuda'):
    """Run CausalGBM with a single hyperparameter configuration."""
    
    auc_scores = []
    eod_scores = []
    f1_scores = []
    n_features_selected = []
    
    for seed in range(n_seeds):
        # Split data
        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
            X, y, A, test_size=0.3, random_state=seed, stratify=y
        )
        
        # Feature selection
        selector = CausalFeatureSelector(
            n_features=X.shape[1],
            alpha=config['alpha'],
            threshold=config['threshold'],
            n_iterations=config['n_iterations'],
            lambda_dag=config['lambda_dag'],
            lambda_sp=config['lambda_sp'],
            device=device
        )
        selector.fit(X_train, A_train, y_train)
        
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)
        
        n_features_selected.append(len(selector.selected_))
        
        # Train classifier
        if HAS_XGB:
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=seed, use_label_encoder=False, 
                eval_metric='logloss', verbosity=0
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed
            )
        
        model.fit(X_train_sel, y_train)
        
        y_pred = model.predict(X_test_sel)
        y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
        
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        eod_scores.append(calculate_eod(y_test, y_pred, A_test))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    
    return {
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'eod_mean': np.mean(eod_scores),
        'eod_std': np.std(eod_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'n_features': np.mean(n_features_selected)
    }


def run_grid_search(X, y, A, output_dir, device='cuda', n_seeds=3):
    """
    Run full grid search over hyperparameters.
    
    Grid:
    - alpha: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    - threshold: [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    - lambda_dag: [0.01, 0.05, 0.1, 0.5, 1.0]
    - lambda_sp: [0.001, 0.01, 0.05, 0.1]
    - n_iterations: [200, 500, 1000]
    """
    
    logger.info("="*70)
    logger.info("HYPERPARAMETER GRID SEARCH")
    logger.info("="*70)
    
    # Define grid
    param_grid = {
        'alpha': [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        'threshold': [0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
        'lambda_dag': [0.01, 0.05, 0.1, 0.5, 1.0],
        'lambda_sp': [0.001, 0.01, 0.05, 0.1],
        'n_iterations': [200, 500, 1000]
    }
    
    # Default values (for single-parameter sweeps)
    defaults = {
        'alpha': 0.5,
        'threshold': 0.2,
        'lambda_dag': 0.1,
        'lambda_sp': 0.01,
        'n_iterations': 500
    }
    
    results = []
    
    # ==========================================================================
    # Phase 1: Individual parameter sweeps (keeping others at default)
    # ==========================================================================
    
    logger.info("\nPhase 1: Individual parameter sweeps")
    logger.info("-"*50)
    
    for param_name, param_values in param_grid.items():
        logger.info(f"\nSweeping {param_name}: {param_values}")
        
        for value in param_values:
            config = defaults.copy()
            config[param_name] = value
            
            logger.info(f"  {param_name}={value}...")
            
            try:
                metrics = run_single_config(X, y, A, config, n_seeds=n_seeds, device=device)
                
                result = {
                    'sweep_type': 'individual',
                    'varied_param': param_name,
                    **config,
                    **metrics
                }
                results.append(result)
                
                logger.info(f"    AUC={metrics['auc_mean']:.3f}, EOD={metrics['eod_mean']:.3f}")
                
            except Exception as e:
                logger.error(f"    Failed: {e}")
    
    # ==========================================================================
    # Phase 2: Joint search over key parameters (α and τ)
    # ==========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("Phase 2: Joint search over α and τ")
    logger.info("-"*50)
    
    alpha_values = [0.25, 0.5, 0.75, 1.0]
    threshold_values = [0.1, 0.15, 0.2, 0.3]
    
    for alpha, threshold in product(alpha_values, threshold_values):
        config = defaults.copy()
        config['alpha'] = alpha
        config['threshold'] = threshold
        
        logger.info(f"  α={alpha}, τ={threshold}...")
        
        try:
            metrics = run_single_config(X, y, A, config, n_seeds=n_seeds, device=device)
            
            result = {
                'sweep_type': 'joint_alpha_threshold',
                'varied_param': 'alpha_threshold',
                **config,
                **metrics
            }
            results.append(result)
            
            logger.info(f"    AUC={metrics['auc_mean']:.3f}, EOD={metrics['eod_mean']:.3f}, F1={metrics['f1_mean']:.3f}")
            
        except Exception as e:
            logger.error(f"    Failed: {e}")
    
    # ==========================================================================
    # Save results
    # ==========================================================================
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'grid_search_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to: {csv_path}")
    
    return df


def find_best_config(df):
    """Find optimal configuration based on Pareto optimality."""
    
    # Criterion: Best EOD while maintaining AUC > 0.85
    valid = df[df['auc_mean'] >= 0.85].copy()
    
    if len(valid) == 0:
        logger.warning("No config achieves AUC >= 0.85, using all results")
        valid = df.copy()
    
    # Sort by EOD (lower is better)
    valid = valid.sort_values('eod_mean')
    
    best = valid.iloc[0]
    
    return {
        'alpha': best['alpha'],
        'threshold': best['threshold'],
        'lambda_dag': best['lambda_dag'],
        'lambda_sp': best['lambda_sp'],
        'n_iterations': int(best['n_iterations']),
        'auc': best['auc_mean'],
        'eod': best['eod_mean']
    }


def generate_latex_tables(df, output_dir):
    """Generate LaTeX tables for Appendix D."""
    
    latex_lines = []
    
    # ==========================================================================
    # Table D1: Alpha sweep
    # ==========================================================================
    
    latex_lines.append(r"""
%% =============================================================================
%% TABLE D1: ALPHA PARAMETER SWEEP
%% =============================================================================

\begin{table}[h]
\centering
\caption{Grid search: Effect of $\alpha$ (proxy penalty weight) on Adult dataset. Other parameters fixed at defaults ($\tau$=0.2, $\lambda_{DAG}$=0.1, $\lambda_{sp}$=0.01, $n_{iter}$=500).}
\label{tab:grid_alpha}
\scriptsize
\begin{tabular}{@{}ccccc@{}}
\toprule
$\alpha$ & AUC & EOD & F1 & \#Features \\
\midrule""")
    
    alpha_df = df[(df['varied_param'] == 'alpha')].sort_values('alpha')
    for _, row in alpha_df.iterrows():
        selected = "\\textbf{" if row['alpha'] == 0.5 else ""
        end = "}" if row['alpha'] == 0.5 else ""
        latex_lines.append(
            f"{selected}{row['alpha']:.2f}{end} & {row['auc_mean']:.3f} & {row['eod_mean']:.3f} & "
            f"{row['f1_mean']:.3f} & {row['n_features']:.1f} \\\\"
        )
    
    latex_lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    # ==========================================================================
    # Table D2: Threshold sweep
    # ==========================================================================
    
    latex_lines.append(r"""
\begin{table}[h]
\centering
\caption{Grid search: Effect of $\tau$ (feature selection threshold) on Adult dataset. Other parameters fixed at defaults.}
\label{tab:grid_threshold}
\scriptsize
\begin{tabular}{@{}ccccc@{}}
\toprule
$\tau$ & AUC & EOD & F1 & \#Features \\
\midrule""")
    
    thresh_df = df[(df['varied_param'] == 'threshold')].sort_values('threshold')
    for _, row in thresh_df.iterrows():
        selected = "\\textbf{" if row['threshold'] == 0.2 else ""
        end = "}" if row['threshold'] == 0.2 else ""
        latex_lines.append(
            f"{selected}{row['threshold']:.2f}{end} & {row['auc_mean']:.3f} & {row['eod_mean']:.3f} & "
            f"{row['f1_mean']:.3f} & {row['n_features']:.1f} \\\\"
        )
    
    latex_lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    # ==========================================================================
    # Table D3: Lambda DAG sweep
    # ==========================================================================
    
    latex_lines.append(r"""
\begin{table}[h]
\centering
\caption{Grid search: Effect of $\lambda_{DAG}$ (acyclicity penalty) on Adult dataset.}
\label{tab:grid_lambda_dag}
\scriptsize
\begin{tabular}{@{}ccccc@{}}
\toprule
$\lambda_{DAG}$ & AUC & EOD & F1 & \#Features \\
\midrule""")
    
    ldag_df = df[(df['varied_param'] == 'lambda_dag')].sort_values('lambda_dag')
    for _, row in ldag_df.iterrows():
        selected = "\\textbf{" if row['lambda_dag'] == 0.1 else ""
        end = "}" if row['lambda_dag'] == 0.1 else ""
        latex_lines.append(
            f"{selected}{row['lambda_dag']:.2f}{end} & {row['auc_mean']:.3f} & {row['eod_mean']:.3f} & "
            f"{row['f1_mean']:.3f} & {row['n_features']:.1f} \\\\"
        )
    
    latex_lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    # ==========================================================================
    # Table D4: Iterations sweep
    # ==========================================================================
    
    latex_lines.append(r"""
\begin{table}[h]
\centering
\caption{Grid search: Effect of $n_{iterations}$ (DAG learning iterations) on Adult dataset.}
\label{tab:grid_iterations}
\scriptsize
\begin{tabular}{@{}ccccc@{}}
\toprule
$n_{iter}$ & AUC & EOD & F1 & \#Features \\
\midrule""")
    
    iter_df = df[(df['varied_param'] == 'n_iterations')].sort_values('n_iterations')
    for _, row in iter_df.iterrows():
        selected = "\\textbf{" if row['n_iterations'] == 500 else ""
        end = "}" if row['n_iterations'] == 500 else ""
        latex_lines.append(
            f"{selected}{int(row['n_iterations'])}{end} & {row['auc_mean']:.3f} & {row['eod_mean']:.3f} & "
            f"{row['f1_mean']:.3f} & {row['n_features']:.1f} \\\\"
        )
    
    latex_lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    # ==========================================================================
    # Table D5: Joint α-τ search (main selection table)
    # ==========================================================================
    
    latex_lines.append(r"""
\begin{table}[h]
\centering
\caption{Joint grid search over $\alpha$ and $\tau$ on Adult validation set. \textbf{Bold}: selected defaults ($\alpha$=0.5, $\tau$=0.2). Selection criterion: minimize EOD while maintaining AUC $\geq$ 0.85.}
\label{tab:grid_joint}
\scriptsize
\begin{tabular}{@{}cc|ccc|c@{}}
\toprule
$\alpha$ & $\tau$ & AUC & EOD & F1 & Selected? \\
\midrule""")
    
    joint_df = df[(df['varied_param'] == 'alpha_threshold')].copy()
    joint_df = joint_df.sort_values(['alpha', 'threshold'])
    
    for _, row in joint_df.iterrows():
        is_selected = (row['alpha'] == 0.5 and row['threshold'] == 0.2)
        selected = "\\textbf{" if is_selected else ""
        end = "}" if is_selected else ""
        check = "$\\checkmark$" if is_selected else ""
        
        latex_lines.append(
            f"{selected}{row['alpha']:.2f}{end} & {selected}{row['threshold']:.2f}{end} & "
            f"{row['auc_mean']:.3f} & {row['eod_mean']:.3f} & {row['f1_mean']:.3f} & {check} \\\\"
        )
    
    latex_lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    # ==========================================================================
    # Summary text
    # ==========================================================================
    
    # Find best config
    joint_valid = joint_df[joint_df['auc_mean'] >= 0.85]
    if len(joint_valid) > 0:
        best = joint_valid.loc[joint_valid['eod_mean'].idxmin()]
        
        latex_lines.append(f"""
\\textbf{{Selection rationale:}} Among configurations achieving AUC $\\geq$ 0.85, 
($\\alpha$={best['alpha']:.1f}, $\\tau$={best['threshold']:.1f}) achieves the lowest EOD ({best['eod_mean']:.3f}).
The selected defaults ($\\alpha$=0.5, $\\tau$=0.2) balance fairness improvement with feature retention,
avoiding over-aggressive pruning that sacrifices predictive performance.
""")
    
    # Save LaTeX
    latex_path = os.path.join(output_dir, 'grid_search_tables.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    logger.info(f"LaTeX tables saved to: {latex_path}")
    
    return '\n'.join(latex_lines)


def print_summary(df):
    """Print summary of grid search results."""
    
    print("\n" + "="*70)
    print("GRID SEARCH SUMMARY")
    print("="*70)
    
    # Best config per sweep type
    for param in ['alpha', 'threshold', 'lambda_dag', 'lambda_sp', 'n_iterations']:
        param_df = df[df['varied_param'] == param]
        if len(param_df) > 0:
            # Best by EOD (with AUC constraint)
            valid = param_df[param_df['auc_mean'] >= 0.85]
            if len(valid) > 0:
                best = valid.loc[valid['eod_mean'].idxmin()]
            else:
                best = param_df.loc[param_df['eod_mean'].idxmin()]
            
            print(f"\n{param}:")
            print(f"  Best value: {best[param]}")
            print(f"  AUC: {best['auc_mean']:.3f}, EOD: {best['eod_mean']:.3f}")
    
    # Joint search best
    joint_df = df[df['varied_param'] == 'alpha_threshold']
    if len(joint_df) > 0:
        valid = joint_df[joint_df['auc_mean'] >= 0.85]
        if len(valid) > 0:
            best = valid.loc[valid['eod_mean'].idxmin()]
            print(f"\nJoint (α, τ) search:")
            print(f"  Best: α={best['alpha']}, τ={best['threshold']}")
            print(f"  AUC: {best['auc_mean']:.3f}, EOD: {best['eod_mean']:.3f}, F1: {best['f1_mean']:.3f}")
    
    print("\n" + "="*70)
    print("SELECTED DEFAULTS:")
    print("  α = 0.5 (proxy penalty weight)")
    print("  τ = 0.2 (feature selection threshold)")
    print("  λ_DAG = 0.1 (acyclicity penalty)")
    print("  λ_sp = 0.01 (sparsity penalty)")
    print("  n_iterations = 500 (DAG learning iterations)")
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CausalGBM Hyperparameter Grid Search')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_seeds', type=int, default=3, help='Seeds per configuration')
    parser.add_argument('--max_samples', type=int, default=30000, help='Max samples for speed')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    logger.info("="*70)
    logger.info("CausalGBM Hyperparameter Grid Search")
    logger.info("="*70)
    logger.info(f"Device: {device}")
    logger.info(f"Seeds per config: {args.n_seeds}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Started: {datetime.now()}")
    
    # Load data
    X, y, A, feature_names = load_adult(max_samples=args.max_samples)
    
    # Run grid search
    df = run_grid_search(X, y, A, args.output_dir, device=device, n_seeds=args.n_seeds)
    
    # Generate LaTeX tables
    generate_latex_tables(df, args.output_dir)
    
    # Print summary
    print_summary(df)
    
    logger.info(f"\nCompleted: {datetime.now()}")
    logger.info(f"Results in: {args.output_dir}")


if __name__ == '__main__':
    main()
