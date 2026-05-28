"""
Indirect Proxy Detection — Self-Contained Implementation
==========================================================
Bypasses CausalFeatureSelector class to directly control DAG learning
and influence computation. Prints debug info to verify the influence
scores are actually different across methods.

Usage:
  python acml_indirect_proxy_v2.py
  python acml_indirect_proxy_v2.py --n_seeds 10
"""

import os
import sys
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as xgb

# Import DAGMA
try:
    from dagma.linear import DagmaLinear
    HAS_DAGMA = True
    logger.info("DAGMA loaded successfully")
except ImportError:
    HAS_DAGMA = False
    logger.warning("DAGMA not installed. Run: pip install dagma")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import compute_metrics


# ============================================================================
# GROUND TRUTH
# ============================================================================

GROUND_TRUTH = {
    'direct_proxies': ['Zip_Code_Risk', 'School_Rating'],
    'indirect_proxies': ['Property_Value', 'Branch_Quality'],
    'legitimate': ['Annual_Income', 'Employment_Years', 'Credit_Score'],
    'spurious': ['Name_Pattern'],
}

ALL_PROXIES = GROUND_TRUTH['direct_proxies'] + GROUND_TRUTH['indirect_proxies']


# ============================================================================
# DAG LEARNING
# ============================================================================

def learn_dag(X, A, y, n_iterations=500):
    """Learn DAG using DAGMA and return adjacency matrix."""
    n, d = X.shape
    # Build augmented matrix [X, A, Y]
    Z = np.column_stack([X, A.reshape(-1, 1), y.reshape(-1, 1)])
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)

    if HAS_DAGMA:
        model = DagmaLinear(loss_type='l2')
        W_est = model.fit(Z, lambda1=0.1)
    else:
        # Fallback: simple correlation-based approximation
        W_est = np.corrcoef(Z.T)
        np.fill_diagonal(W_est, 0)

    return W_est


# ============================================================================
# INFLUENCE COMPUTATION METHODS
# ============================================================================

def influence_direct(adj, idx_A, d):
    """Direct edge weight: W[A, X_j]"""
    return np.abs(adj[idx_A, :d])


def influence_total_effect(adj, idx_A, d):
    """Total effect: (I - W)^{-1}[A, X_j]"""
    n = adj.shape[0]
    try:
        T = np.linalg.inv(np.eye(n) - adj)
    except:
        T = np.linalg.pinv(np.eye(n) - adj)
    return np.abs(T[idx_A, :d])


def influence_2hop(adj, idx_A, d):
    """2-hop: |W| + |W²|"""
    W2 = np.abs(adj) @ np.abs(adj)
    return np.abs(adj[idx_A, :d]) + W2[idx_A, :d]


def influence_3hop(adj, idx_A, d):
    """3-hop: |W| + |W²| + |W³|"""
    absW = np.abs(adj)
    W2 = absW @ absW
    W3 = W2 @ absW
    return np.abs(adj[idx_A, :d]) + W2[idx_A, :d] + W3[idx_A, :d]


def influence_damped(adj, idx_A, d, gamma=0.5, K=10):
    """Damped: Σ γ^k |W^k|"""
    n = adj.shape[0]
    influence = np.zeros(d)
    W_power = np.eye(n)
    absW = np.abs(adj)
    for k in range(1, K + 1):
        W_power = W_power @ absW
        influence += (gamma ** k) * W_power[idx_A, :d]
        if gamma ** k < 1e-8:
            break
    return influence


def influence_maxpath(adj, idx_A, d):
    """Max-path: max product of |edge weights| along any path from A to X_j"""
    n = adj.shape[0]
    absW = np.abs(adj)
    # Dynamic programming on DAG
    max_inf = np.zeros(n)
    max_inf[idx_A] = 1.0

    for _ in range(n):
        updated = False
        for i in range(n):
            if max_inf[i] == 0:
                continue
            for j in range(n):
                if i != j and absW[i, j] > 1e-8:
                    candidate = max_inf[i] * absW[i, j]
                    if candidate > max_inf[j]:
                        max_inf[j] = candidate
                        updated = True
        if not updated:
            break

    return max_inf[:d]


def influence_taint(adj, idx_A, d):
    """Causal Taint Propagation: iterative max-propagation."""
    n = adj.shape[0]
    absW = np.abs(adj)
    taint = np.zeros(n)
    taint[idx_A] = 1.0

    for _ in range(n):
        old = taint.copy()
        for i in range(n):
            if taint[i] < 1e-10:
                continue
            for j in range(n):
                if i != j and absW[i, j] > 1e-8:
                    propagated = taint[i] * absW[i, j]
                    taint[j] = max(taint[j], propagated)
        if np.max(np.abs(taint - old)) < 1e-10:
            break

    return taint[:d]


METHODS = {
    'Direct (original)': influence_direct,
    'Total Effect': influence_total_effect,
    '2-Hop': influence_2hop,
    '3-Hop': influence_3hop,
    'Damped (γ=0.5)': influence_damped,
    'Max-Path': influence_maxpath,
    'Taint Propagation': influence_taint,
}


# ============================================================================
# FEATURE SELECTION WITH CONFIGURABLE INFLUENCE
# ============================================================================

def select_features(adj, corr_with_A, influence_fn, idx_A, idx_Y, d,
                    alpha=0.5, threshold=0.2, min_features=3,
                    aggregation='max'):
    """
    Select features using given influence function.
    Returns: selected feature indices, scores, influence values.
    """
    # Compute influence of A on each feature
    W_influence = influence_fn(adj, idx_A, d)

    # Aggregation with correlation
    if aggregation == 'max':
        W_prime = np.maximum(W_influence, corr_with_A)
    else:
        W_prime = W_influence

    # Predictive value: edge weight X_j → Y
    W_X_Y = np.abs(adj[:d, idx_Y])

    # Scoring: higher = more worth keeping
    scores = W_X_Y - alpha * W_prime

    # Select features above threshold
    selected = np.where(scores >= threshold)[0]
    if len(selected) < min_features:
        selected = np.argsort(scores)[::-1][:min_features]

    return selected, scores, W_influence, W_prime, W_X_Y


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(n_seeds=10, alpha=0.5, threshold=0.2, device='cpu'):
    """Run the full indirect proxy detection experiment."""

    # Load dataset
    csv_path = 'synthetic_indirect_proxy_loan.csv'
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ['Race', 'Loan_Approved']]
    X_raw = df[feature_cols].values.astype(np.float64)
    y = df['Loan_Approved'].values.astype(np.float64)
    sens = df['Race'].values.astype(np.float64)

    n, d = X_raw.shape
    idx_A = d      # A is at position d
    idx_Y = d + 1  # Y is at position d+1
    min_features = max(3, d // 3)

    logger.info(f"Dataset: n={n}, d={d}, features={feature_cols}")
    logger.info(f"Ground truth proxies (direct): {GROUND_TRUTH['direct_proxies']}")
    logger.info(f"Ground truth proxies (indirect): {GROUND_TRUTH['indirect_proxies']}")
    logger.info(f"Legitimate features: {GROUND_TRUTH['legitimate']}")
    logger.info(f"Spurious features: {GROUND_TRUTH['spurious']}")
    logger.info(f"min_features={min_features}, alpha={alpha}, threshold={threshold}")

    all_results = []

    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X_raw, y, sens, test_size=0.3, random_state=seed, stratify=y
        )

        # Standardize
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # Learn DAG once per seed
        adj = learn_dag(X_tr_sc, s_tr, y_tr)

        # Compute correlations
        corr_with_A = np.array([abs(np.corrcoef(X_tr_sc[:, j], s_tr)[0, 1])
                                for j in range(d)])

        # Print debug info for seed 0
        if seed == 0:
            logger.info(f"\n{'='*70}")
            logger.info(f"SEED 0 DEBUG — Learned DAG Analysis")
            logger.info(f"{'='*70}")

            logger.info(f"\nDirect edge weights (A → X_j):")
            for j, fname in enumerate(feature_cols):
                logger.info(f"  {fname:20s}: W_direct={abs(adj[idx_A, j]):.4f}  "
                           f"corr={corr_with_A[j]:.4f}  W_to_Y={abs(adj[j, idx_Y]):.4f}")

            logger.info(f"\nAll influence methods (A → X_j):")
            for method_name, method_fn in METHODS.items():
                inf = method_fn(adj, idx_A, d)
                logger.info(f"\n  {method_name}:")
                for j, fname in enumerate(feature_cols):
                    marker = ""
                    if fname in GROUND_TRUTH['direct_proxies']:
                        marker = " [DIRECT PROXY]"
                    elif fname in GROUND_TRUTH['indirect_proxies']:
                        marker = " [INDIRECT PROXY]"
                    elif fname in GROUND_TRUTH['spurious']:
                        marker = " [SPURIOUS]"
                    logger.info(f"    {fname:20s}: influence={inf[j]:.4f}{marker}")

        # XGBoost baseline (no feature selection)
        model_base = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        model_base.fit(X_tr_sc, y_tr)
        y_pred_base = model_base.predict(X_te_sc)
        y_prob_base = model_base.predict_proba(X_te_sc)[:, 1]
        m_base = compute_metrics(y_te, y_pred_base, y_prob_base, s_te)
        all_results.append({
            'method': 'XGBoost (baseline)', 'seed': seed,
            'auc': m_base['auc'], 'eod': m_base['eod'],
            'n_features': d, 'selected': list(range(d)),
        })

        # Oracle: remove ALL proxies (direct + indirect)
        oracle_keep = [j for j, f in enumerate(feature_cols) if f not in ALL_PROXIES]
        X_tr_oracle = X_tr_sc[:, oracle_keep]
        X_te_oracle = X_te_sc[:, oracle_keep]
        model_oracle = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        model_oracle.fit(X_tr_oracle, y_tr)
        y_pred_oracle = model_oracle.predict(X_te_oracle)
        y_prob_oracle = model_oracle.predict_proba(X_te_oracle)[:, 1]
        m_oracle = compute_metrics(y_te, y_pred_oracle, y_prob_oracle, s_te)
        all_results.append({
            'method': 'Oracle (all proxies removed)', 'seed': seed,
            'auc': m_oracle['auc'], 'eod': m_oracle['eod'],
            'n_features': len(oracle_keep),
            'selected': [feature_cols[j] for j in oracle_keep],
        })

        # Each influence method
        for method_name, method_fn in METHODS.items():
            selected, scores, W_inf, W_prime, W_X_Y = select_features(
                adj, corr_with_A, method_fn, idx_A, idx_Y, d,
                alpha=alpha, threshold=threshold, min_features=min_features,
                aggregation='max'
            )

            X_tr_sel = X_tr_sc[:, selected]
            X_te_sel = X_te_sc[:, selected]

            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            model.fit(X_tr_sel, y_tr)
            y_pred = model.predict(X_te_sel)
            y_prob = model.predict_proba(X_te_sel)[:, 1]
            metrics = compute_metrics(y_te, y_pred, y_prob, s_te)

            selected_names = [feature_cols[j] for j in selected]
            removed_names = [f for f in feature_cols if f not in selected_names]

            all_results.append({
                'method': method_name, 'seed': seed,
                'auc': metrics['auc'], 'eod': metrics['eod'],
                'n_features': len(selected),
                'selected': selected_names,
            })

            if seed == 0:
                direct_removed = [f for f in GROUND_TRUTH['direct_proxies'] if f in removed_names]
                indirect_removed = [f for f in GROUND_TRUTH['indirect_proxies'] if f in removed_names]
                legit_removed = [f for f in GROUND_TRUTH['legitimate'] if f in removed_names]
                spur_removed = [f for f in GROUND_TRUTH['spurious'] if f in removed_names]

                logger.info(f"\n  {method_name}:")
                logger.info(f"    Selected ({len(selected)}): {selected_names}")
                logger.info(f"    Removed: {removed_names}")
                logger.info(f"    Direct proxies removed: {len(direct_removed)}/2 {direct_removed}")
                logger.info(f"    Indirect proxies removed: {len(indirect_removed)}/2 {indirect_removed}")
                logger.info(f"    Legitimate removed: {len(legit_removed)}/3 {legit_removed}")
                logger.info(f"    Spurious removed: {len(spur_removed)}/1 {spur_removed}")

    # Summary
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("SUMMARY (mean over seeds)")
    print("=" * 70)

    summary = results_df.groupby('method').agg(
        auc=('auc', 'mean'), auc_std=('auc', 'std'),
        eod=('eod', 'mean'), eod_std=('eod', 'std'),
        n_feats=('n_features', 'mean'),
    ).round(4)

    # Sort by EOD
    summary = summary.sort_values('eod')

    baseline_eod = summary.loc['XGBoost (baseline)', 'eod']
    oracle_eod = summary.loc['Oracle (all proxies removed)', 'eod']
    oracle_gap = baseline_eod - oracle_eod

    print(f"\nBaseline EOD: {baseline_eod:.4f}")
    print(f"Oracle EOD:   {oracle_eod:.4f}")
    print(f"Max achievable reduction: {oracle_gap:.4f} ({oracle_gap/baseline_eod*100:.1f}%)")
    print()

    for method, row in summary.iterrows():
        reduction = (baseline_eod - row['eod']) / baseline_eod * 100
        gap_closed = (baseline_eod - row['eod']) / oracle_gap * 100 if oracle_gap > 0 else 0
        marker = " ← BEST" if method not in ['XGBoost (baseline)', 'Oracle (all proxies removed)'] \
                 and row['eod'] == summary.drop(['XGBoost (baseline)', 'Oracle (all proxies removed)'])['eod'].min() else ""
        print(f"  {method:30s}: EOD={row['eod']:.4f}±{row['eod_std']:.4f}  "
              f"AUC={row['auc']:.3f}  ↓{reduction:5.1f}%  "
              f"gap_closed={gap_closed:5.1f}%{marker}")

    results_df.to_csv('indirect_proxy_v2_results.csv', index=False)
    print("\nResults saved to indirect_proxy_v2_results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    run_experiment(n_seeds=args.n_seeds, alpha=args.alpha, threshold=args.threshold)
