"""
Indirect Proxy Detection: Extending CausalGBM Beyond Direct Edges
==================================================================

The key limitation of CausalGBM is that it only detects direct proxies
(A → X_j). This script explores approaches to detect INDIRECT proxies
(A → X₂ → X₃ → Y) where X₃ has no direct edge from A.

Core Insight: Proxy-ness is TRANSITIVE in a DAG. If A "taints" X₂,
and X₂ influences X₃, then X₃ is also tainted. The question is HOW
to propagate this taint while avoiding false positives from spurious paths.

Approaches implemented:
  1. Direct-only (original CausalGBM)
  2. Total Effect via (I - W)^{-1} — captures ALL paths
  3. K-hop (K=2): W + W² — captures up to 2-hop indirect effects
  4. K-hop (K=3): W + W² + W³ — captures up to 3-hop
  5. Damped propagation: Σ γ^k W^k — longer paths contribute less
  6. Max-path propagation: max product of edge weights along any A→X path
  7. Causal Taint Propagation (novel): iterative max-propagation on DAG

Usage:
  python acml_indirect_proxy_experiments.py --all
  python acml_indirect_proxy_experiments.py --synthetic_only
  python acml_indirect_proxy_experiments.py --all --n_seeds 10 --device cuda
"""

import os
import sys
import time
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import (
    CausalFeatureSelector, compute_metrics,
    load_adult, load_compas, load_german, load_acs_income,
    load_taiwan_credit, load_bank, load_online_shoppers,
    load_synthetic_loan, load_synthetic_hiring,
)
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    logger.error("PyTorch required")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATASET_LOADERS = {
    'adult': load_adult,
    'acs_income': load_acs_income,
    'compas': load_compas,
    'german': load_german,
    'taiwan_credit': load_taiwan_credit,
    'bank': load_bank,
    'online_shoppers': load_online_shoppers,
    'synthetic_loan': load_synthetic_loan,
    'synthetic_hiring': load_synthetic_hiring,
}


# ============================================================================
# PROXY INFLUENCE COMPUTATION METHODS
# ============================================================================

def compute_direct_influence(adj, idx_A, d):
    """Original CausalGBM: direct edge weights A → X_j only."""
    return np.abs(adj[idx_A, :d])


def compute_total_effect(adj, idx_A, d):
    """
    Total causal effect via matrix inversion: T = (I - W)^{-1}.
    T[A, X_j] captures the sum of ALL directed paths from A to X_j.
    
    In a linear SEM Z = ZW + ε, the total effect of node i on node j
    is given by (I - W)^{-1}[i, j].
    """
    n = adj.shape[0]
    try:
        T = np.linalg.inv(np.eye(n) - adj)
        return np.abs(T[idx_A, :d])
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        T = np.linalg.pinv(np.eye(n) - adj)
        return np.abs(T[idx_A, :d])


def compute_khop_influence(adj, idx_A, d, K=2):
    """
    K-hop influence: sum of W^1 + W^2 + ... + W^K.
    Captures indirect effects up to K hops away from A.
    
    K=1: same as direct
    K=2: captures A → X₂ → X₃ (2-hop indirect)
    K=3: captures A → X₂ → X₃ → X₄ (3-hop indirect)
    """
    n = adj.shape[0]
    influence = np.zeros(d)
    W_power = np.eye(n)
    
    for k in range(1, K + 1):
        W_power = W_power @ adj
        influence += np.abs(W_power[idx_A, :d])
    
    return influence


def compute_damped_influence(adj, idx_A, d, gamma=0.5, K=10):
    """
    Damped propagation: Σ_{k=1}^{K} γ^k · |W^k[A, :]|
    
    Longer paths contribute exponentially less. This prevents
    false positives from long spurious chains while still
    capturing nearby indirect effects.
    
    gamma=1.0: equivalent to total effect (no damping)
    gamma=0.5: 2-hop gets 25% weight, 3-hop gets 12.5%, etc.
    gamma=0.0: equivalent to direct only
    """
    n = adj.shape[0]
    influence = np.zeros(d)
    W_power = np.eye(n)
    
    for k in range(1, K + 1):
        W_power = W_power @ adj
        influence += (gamma ** k) * np.abs(W_power[idx_A, :d])
        # Early termination if contribution is negligible
        if (gamma ** k) < 1e-6:
            break
    
    return influence


def compute_maxpath_influence(adj, idx_A, d):
    """
    Max-path propagation: for each feature X_j, find the path
    from A to X_j with the maximum product of edge weights.
    
    This is solved via dynamic programming on the DAG (topological order).
    Unlike sum-based methods, max-path is less susceptible to
    accumulating many weak spurious paths.
    """
    n = adj.shape[0]
    # Use log-space for numerical stability (max sum of log weights = max product)
    log_adj = np.full((n, n), -np.inf)
    mask = np.abs(adj) > 1e-8
    log_adj[mask] = np.log(np.abs(adj[mask]))
    
    # Bellman-Ford style: find longest path from idx_A to each node
    # (longest in log-space = max product in original space)
    max_influence = np.full(n, -np.inf)
    max_influence[idx_A] = 0  # log(1) = 0
    
    # Iterate n times (guaranteed convergence for DAG)
    for _ in range(n):
        for j in range(n):
            for i in range(n):
                if i != j and log_adj[i, j] > -np.inf:
                    candidate = max_influence[i] + log_adj[i, j]
                    if candidate > max_influence[j]:
                        max_influence[j] = candidate
    
    # Convert back from log-space
    result = np.zeros(d)
    for j in range(d):
        if max_influence[j] > -np.inf:
            result[j] = np.exp(max_influence[j])
    
    return result


def compute_taint_propagation(adj, idx_A, d, threshold=0.1):
    """
    Causal Taint Propagation (CTP) — novel approach.
    
    Idea: propagate "taint" from A through the DAG using iterative
    max-update. A feature is tainted if A can influence it through
    ANY path, weighted by edge strengths.
    
    Algorithm:
      1. Initialize taint(A) = 1.0, taint(others) = 0
      2. For each edge X_i → X_j in the DAG:
         taint(X_j) = max(taint(X_j), taint(X_i) * |W_{X_i → X_j}|)
      3. Repeat until convergence
      4. Features with taint > threshold are potential proxies
    
    Key difference from total effect: uses MAX instead of SUM,
    so a single strong indirect path is sufficient to flag a feature,
    but many weak paths don't accumulate into a false positive.
    
    Key difference from max-path: considers the taint of the parent,
    not just the edge weight. This means taint naturally attenuates
    through weak intermediate nodes.
    """
    n = adj.shape[0]
    taint = np.zeros(n)
    taint[idx_A] = 1.0
    
    # Iterate until convergence
    for iteration in range(n):
        old_taint = taint.copy()
        for i in range(n):
            for j in range(n):
                if i != j and abs(adj[i, j]) > 1e-8:
                    # Propagate taint from i to j
                    propagated = taint[i] * abs(adj[i, j])
                    taint[j] = max(taint[j], propagated)
        
        # Check convergence
        if np.max(np.abs(taint - old_taint)) < 1e-10:
            break
    
    return taint[:d]


# ============================================================================
# EXTENDED CAUSALGBM WITH CONFIGURABLE INFLUENCE METHOD
# ============================================================================

class ExtendedCausalFeatureSelector(CausalFeatureSelector):
    """
    CausalGBM extended with configurable influence propagation.
    
    influence_method options:
      'direct'    — original CausalGBM (W_{A→X_j} only)
      'total'     — total effect via (I-W)^{-1}
      '2hop'      — direct + 2-hop effects
      '3hop'      — direct + 2-hop + 3-hop effects
      'damped'    — damped propagation (γ=0.5)
      'maxpath'   — max-product path from A to X_j
      'taint'     — causal taint propagation (novel)
    """
    
    def __init__(self, n_features, influence_method='direct',
                 gamma=0.5, alpha=0.5, threshold=0.2, min_features=3,
                 n_iterations=500, lambda_dag=0.1, lambda_sp=0.01,
                 aggregation='max', device='cpu'):
        super().__init__(
            n_features, alpha=alpha, threshold=threshold,
            min_features=min_features, n_iterations=n_iterations,
            lambda_dag=lambda_dag, lambda_sp=lambda_sp,
            aggregation=aggregation, device=device
        )
        self.influence_method = influence_method
        self.gamma = gamma
        self.influence_scores_ = None
    
    def fit(self, X, A, y):
        # Run standard DAG learning (from parent class)
        super().fit(X, A, y)
        
        # Now recompute proxy weights using the chosen influence method
        adj = self.learned_adjacency_
        if adj is None:
            return self
        
        n, d = X.shape
        idx_A = d  # A is at position d in the augmented matrix
        
        # Compute influence using chosen method
        if self.influence_method == 'direct':
            W_influence = compute_direct_influence(adj, idx_A, d)
        elif self.influence_method == 'total':
            W_influence = compute_total_effect(adj, idx_A, d)
        elif self.influence_method == '2hop':
            W_influence = compute_khop_influence(adj, idx_A, d, K=2)
        elif self.influence_method == '3hop':
            W_influence = compute_khop_influence(adj, idx_A, d, K=3)
        elif self.influence_method == 'damped':
            W_influence = compute_damped_influence(adj, idx_A, d, gamma=self.gamma)
        elif self.influence_method == 'maxpath':
            W_influence = compute_maxpath_influence(adj, idx_A, d)
        elif self.influence_method == 'taint':
            W_influence = compute_taint_propagation(adj, idx_A, d)
        else:
            raise ValueError(f"Unknown influence method: {self.influence_method}")
        
        self.influence_scores_ = W_influence
        
        # Recompute aggregation with new influence scores
        if self.aggregation == 'max':
            W_prime = np.maximum(W_influence, self.correlations_)
        elif self.aggregation == 'dag_only':
            W_prime = W_influence
        elif self.aggregation == 'corr_only':
            W_prime = self.correlations_
        else:
            W_prime = np.maximum(W_influence, self.correlations_)
        
        # Recompute scoring
        idx_Y = d + 1
        W_X_Y = adj[:d, idx_Y]
        self.causal_importance_ = W_X_Y - self.alpha * W_prime
        
        # Reselect features
        above = np.where(self.causal_importance_ >= self.threshold)[0]
        if len(above) >= self.min_features:
            self.selected_ = above
        else:
            self.selected_ = np.argsort(self.causal_importance_)[::-1][:self.min_features]
        
        return self


# ============================================================================
# EXPERIMENT: COMPARE ALL INFLUENCE METHODS
# ============================================================================

def run_influence_comparison(datasets, output_dir, n_seeds=10, device='cpu'):
    """Compare all influence propagation methods across datasets."""
    
    logger.info("=" * 70)
    logger.info("INDIRECT PROXY DETECTION: INFLUENCE METHOD COMPARISON")
    logger.info("=" * 70)
    
    methods = [
        ('Direct (original)', 'direct'),
        ('Total Effect', 'total'),
        ('2-Hop', '2hop'),
        ('3-Hop', '3hop'),
        ('Damped (γ=0.5)', 'damped'),
        ('Max-Path', 'maxpath'),
        ('Taint Propagation', 'taint'),
    ]
    
    results = []
    
    for ds_name, dataset in datasets.items():
        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name} (n={len(X)}, d={d})")
        logger.info(f"{'='*60}")
        
        for method_label, method_key in methods:
            method_eods = []
            method_aucs = []
            
            for seed in range(n_seeds):
                X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                    X, y, sens, test_size=0.3, random_state=seed, stratify=y
                )
                
                try:
                    selector = ExtendedCausalFeatureSelector(
                        X_tr.shape[1],
                        influence_method=method_key,
                        gamma=0.5,
                        alpha=0.5, threshold=0.2,
                        min_features=max(3, X_tr.shape[1] // 3),
                        n_iterations=500, aggregation='max', device=device
                    )
                    selector.fit(X_tr, s_tr, y_tr)
                    X_tr_sel = selector.transform(X_tr)
                    X_te_sel = selector.transform(X_te)
                    n_feats = len(selector.selected_)
                    
                    model = xgb.XGBClassifier(
                        n_estimators=100, random_state=seed, verbosity=0
                    ) if HAS_XGB else None
                    
                    if model:
                        model.fit(X_tr_sel, y_tr)
                        y_pred = model.predict(X_te_sel)
                        y_prob = model.predict_proba(X_te_sel)[:, 1]
                        metrics = compute_metrics(y_te, y_pred, y_prob, s_te)
                        
                        results.append({
                            'dataset': ds_name,
                            'method': method_label,
                            'method_key': method_key,
                            'seed': seed,
                            'auc': metrics['auc'],
                            'eod': metrics['eod'],
                            'dpd': metrics['dpd'],
                            'wga': metrics['wga'],
                            'f1': metrics['f1'],
                            'n_features': n_feats,
                            'influence_scores': selector.influence_scores_.tolist()
                                if selector.influence_scores_ is not None else [],
                        })
                        method_eods.append(metrics['eod'])
                        method_aucs.append(metrics['auc'])
                
                except Exception as e:
                    logger.warning(f"  {method_label} seed={seed} failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            if method_eods:
                logger.info(
                    f"  {method_label:25s}: AUC={np.mean(method_aucs):.3f}±{np.std(method_aucs):.3f}  "
                    f"EOD={np.mean(method_eods):.3f}±{np.std(method_eods):.3f}  "
                    f"(n={len(method_eods)} seeds)"
                )
        
        # Also run XGBoost baseline (no feature selection)
        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y
            )
            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
            metrics = compute_metrics(y_te, y_pred, y_prob, s_te)
            results.append({
                'dataset': ds_name, 'method': 'XGBoost (no selection)',
                'method_key': 'baseline', 'seed': seed,
                'auc': metrics['auc'], 'eod': metrics['eod'],
                'dpd': metrics['dpd'], 'wga': metrics['wga'],
                'f1': metrics['f1'], 'n_features': d,
                'influence_scores': [],
            })
        
        xgb_eods = [r['eod'] for r in results 
                    if r['dataset'] == ds_name and r['method_key'] == 'baseline']
        logger.info(f"  {'XGBoost (baseline)':25s}: AUC={np.mean([r['auc'] for r in results if r['dataset']==ds_name and r['method_key']=='baseline']):.3f}  EOD={np.mean(xgb_eods):.3f}")
    
    # Save results
    df = pd.DataFrame(results)
    # Drop influence_scores column for CSV (too large)
    df_save = df.drop(columns=['influence_scores'], errors='ignore')
    df_save.to_csv(os.path.join(output_dir, 'influence_comparison_raw.csv'), index=False)
    
    # Summary table
    summary = df.groupby(['dataset', 'method']).agg(
        auc=('auc', 'mean'), auc_std=('auc', 'std'),
        eod=('eod', 'mean'), eod_std=('eod', 'std'),
        n_feats=('n_features', 'mean'),
    ).round(4)
    
    print("\n" + "=" * 70)
    print("FULL RESULTS SUMMARY")
    print("=" * 70)
    for ds in df['dataset'].unique():
        print(f"\n--- {ds} ---")
        if ds in summary.index:
            ds_summary = summary.loc[ds][['auc', 'eod', 'n_feats']]
            print(ds_summary.to_string())
    
    summary.to_csv(os.path.join(output_dir, 'influence_comparison_summary.csv'))
    
    # Compute indirect gap for synthetic datasets
    _compute_indirect_gap(df, output_dir)
    
    return df


def _compute_indirect_gap(df, output_dir):
    """Compute indirect gap metric for synthetic datasets."""
    
    print("\n" + "=" * 70)
    print("INDIRECT GAP ANALYSIS")
    print("=" * 70)
    
    gap_rows = []
    
    for ds_name in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds_name]
        
        baseline_eod = ds_df[ds_df['method_key'] == 'baseline']['eod'].mean()
        
        for method_key in ds_df['method_key'].unique():
            if method_key == 'baseline':
                continue
            method_eod = ds_df[ds_df['method_key'] == method_key]['eod'].mean()
            reduction = (baseline_eod - method_eod) / baseline_eod * 100 if baseline_eod > 0 else 0
            
            gap_rows.append({
                'dataset': ds_name,
                'method': ds_df[ds_df['method_key'] == method_key]['method'].iloc[0],
                'baseline_eod': round(baseline_eod, 4),
                'method_eod': round(method_eod, 4),
                'reduction_%': round(reduction, 1),
            })
    
    gap_df = pd.DataFrame(gap_rows)
    
    for ds in gap_df['dataset'].unique():
        print(f"\n--- {ds} ---")
        ds_gap = gap_df[gap_df['dataset'] == ds].sort_values('method_eod')
        for _, row in ds_gap.iterrows():
            marker = " ★" if 'Taint' in row['method'] or 'Total' in row['method'] else ""
            print(f"  {row['method']:25s}: EOD={row['method_eod']:.4f}  "
                  f"(↓{row['reduction_%']:5.1f}% from {row['baseline_eod']:.3f}){marker}")
    
    gap_df.to_csv(os.path.join(output_dir, 'indirect_gap_analysis.csv'), index=False)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Indirect Proxy Detection Experiments')
    parser.add_argument('--all', action='store_true', help='Run on all datasets')
    parser.add_argument('--synthetic_only', action='store_true',
                       help='Run only on synthetic datasets (faster)')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--output_dir', default='results/acml2026/indirect_proxy')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=200000)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine datasets
    if args.datasets:
        dataset_names = args.datasets
    elif args.synthetic_only:
        dataset_names = ['synthetic_loan', 'synthetic_hiring', 'synthetic_indirect']
    else:
        dataset_names = [
            'adult', 'acs_income', 'compas', 'german',
            'taiwan_credit', 'bank', 'online_shoppers',
            'synthetic_loan', 'synthetic_hiring', 'synthetic_indirect',
        ]
    
    # Load datasets
    logger.info("Loading datasets...")
    datasets = {}
    for name in dataset_names:
        if name == 'synthetic_indirect':
            # Load the indirect proxy synthetic dataset
            try:
                indirect_path = 'synthetic_indirect_proxy_loan.csv'
                if os.path.exists(indirect_path):
                    df = pd.read_csv(indirect_path)
                    feature_cols = [c for c in df.columns 
                                   if c not in ['Race', 'Loan_Approved']]
                    sens_col = 'Race'
                    target_col = 'Loan_Approved'
                    
                    from sklearn.preprocessing import LabelEncoder, StandardScaler
                    X = df[feature_cols].values.astype(np.float32)
                    y = df[target_col].values.astype(np.float32)
                    sens = LabelEncoder().fit_transform(df[sens_col])
                    
                    # Standardize
                    X = StandardScaler().fit_transform(X)
                    
                    from causalgbm_experiments_v2 import DatasetBundle
                    datasets[name] = DatasetBundle(
                        'synthetic_indirect', X, y, sens, sens_col, feature_cols
                    )
                    logger.info(f"  Loaded synthetic_indirect: n={len(X)}, d={X.shape[1]}")
            except Exception as e:
                logger.warning(f"  Could not load synthetic_indirect: {e}")
        elif name in DATASET_LOADERS:
            try:
                ds = DATASET_LOADERS[name](max_samples=args.max_samples)
                datasets[name] = ds
                logger.info(f"  Loaded {name}: n={len(ds.X)}, d={ds.X.shape[1]}")
            except Exception as e:
                logger.warning(f"  Could not load {name}: {e}")
    
    if not datasets:
        logger.error("No datasets loaded!")
        return
    
    # Run comparison
    run_influence_comparison(datasets, args.output_dir,
                           n_seeds=args.n_seeds, device=args.device)
    
    logger.info("\nExperiments complete! Results in: " + args.output_dir)


if __name__ == '__main__':
    main()
