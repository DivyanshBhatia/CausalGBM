"""
Nonlinear DAG Learning for Indirect Proxy Detection
=====================================================
Tests whether DAGMA-MLP (nonlinear) recovers intermediate edges
that DAGMA-Linear misses (e.g., Zip_Code → Property_Value).

If it works, this solves the indirect proxy limitation at its source.

Usage:
  python acml_nonlinear_dag.py
  python acml_nonlinear_dag.py --n_seeds 10
"""

import os
import sys
import warnings
import logging
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaNonlinear, DagmaMLP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import compute_metrics

import xgboost as xgb

GROUND_TRUTH = {
    'direct_proxies': ['Zip_Code_Risk', 'School_Rating'],
    'indirect_proxies': ['Property_Value', 'Branch_Quality'],
    'legitimate': ['Annual_Income', 'Employment_Years', 'Credit_Score'],
    'spurious': ['Name_Pattern'],
}
ALL_PROXIES = GROUND_TRUTH['direct_proxies'] + GROUND_TRUTH['indirect_proxies']


def learn_dag_linear(Z):
    """Learn DAG using DAGMA linear SEM."""
    model = DagmaLinear(loss_type='l2')
    W = model.fit(Z, lambda1=0.1)
    return W


def learn_dag_nonlinear(Z, hidden_dims=10):
    """Learn DAG using DAGMA nonlinear (MLP)."""
    d = Z.shape[1]
    model = DagmaMLP(dims=[d, hidden_dims, 1], bias=True)
    dagma = DagmaNonlinear(model, verbose=False)
    W = dagma.fit(Z, lambda1=0.02, lambda2=0.005, w_threshold=0.1)
    return W


def print_dag_analysis(W, feature_names, idx_A, idx_Y, label=""):
    """Print detailed DAG edge analysis."""
    d = len(feature_names)
    logger.info(f"\n{'='*60}")
    logger.info(f"DAG Analysis: {label}")
    logger.info(f"{'='*60}")
    
    # A → X edges (proxy detection)
    logger.info(f"\n  A → X_j (proxy influence):")
    for j, fname in enumerate(feature_names):
        w = abs(W[idx_A, j])
        marker = ""
        if fname in GROUND_TRUTH['direct_proxies']: marker = " ◄ DIRECT PROXY"
        elif fname in GROUND_TRUTH['indirect_proxies']: marker = " ◄ INDIRECT PROXY"
        elif fname in GROUND_TRUTH['spurious']: marker = " ◄ SPURIOUS"
        detected = "✓" if w > 0.05 else "✗"
        logger.info(f"    {fname:20s}: {w:.4f} {detected}{marker}")
    
    # X → Y edges (predictive value)
    logger.info(f"\n  X_j → Y (predictive value):")
    for j, fname in enumerate(feature_names):
        w = abs(W[j, idx_Y])
        logger.info(f"    {fname:20s}: {w:.4f}")
    
    # Intermediate edges (the key question!)
    logger.info(f"\n  X_i → X_j (intermediate edges for indirect proxies):")
    for i, fi in enumerate(feature_names):
        for j, fj in enumerate(feature_names):
            if i != j and abs(W[i, j]) > 0.05:
                logger.info(f"    {fi:20s} → {fj:20s}: {abs(W[i,j]):.4f}")
    
    # Specifically check expected intermediate paths
    logger.info(f"\n  Expected intermediate paths:")
    for direct_proxy in GROUND_TRUTH['direct_proxies']:
        dp_idx = feature_names.index(direct_proxy)
        for indirect_proxy in GROUND_TRUTH['indirect_proxies']:
            ip_idx = feature_names.index(indirect_proxy)
            w = abs(W[dp_idx, ip_idx])
            status = "✓ FOUND" if w > 0.05 else "✗ MISSING"
            logger.info(f"    {direct_proxy:20s} → {indirect_proxy:20s}: {w:.4f}  {status}")


def compute_influence_all_methods(W, idx_A, d):
    """Compute influence scores using all propagation methods."""
    absW = np.abs(W)
    n = W.shape[0]
    
    methods = {}
    
    # Direct
    methods['Direct'] = absW[idx_A, :d]
    
    # Total Effect
    try:
        T = np.linalg.inv(np.eye(n) - W)
        methods['Total Effect'] = np.abs(T[idx_A, :d])
    except:
        methods['Total Effect'] = absW[idx_A, :d]
    
    # 2-Hop
    W2 = absW @ absW
    methods['2-Hop'] = absW[idx_A, :d] + W2[idx_A, :d]
    
    # Taint Propagation
    taint = np.zeros(n)
    taint[idx_A] = 1.0
    for _ in range(n):
        old = taint.copy()
        for i in range(n):
            if taint[i] < 1e-10: continue
            for j in range(n):
                if i != j and absW[i, j] > 1e-8:
                    taint[j] = max(taint[j], taint[i] * absW[i, j])
        if np.max(np.abs(taint - old)) < 1e-10:
            break
    methods['Taint'] = taint[:d]
    
    return methods


def select_and_evaluate(W, corr_A, feature_names, influence, idx_A, idx_Y, d,
                        X_tr, X_te, y_tr, y_te, s_te, seed,
                        alpha=0.5, threshold=0.2, min_features=3):
    """Select features using given influence, train XGBoost, evaluate."""
    W_prime = np.maximum(influence, corr_A)
    W_X_Y = np.abs(W[:d, idx_Y])
    scores = W_X_Y - alpha * W_prime
    
    selected = np.where(scores >= threshold)[0]
    if len(selected) < min_features:
        selected = np.argsort(scores)[::-1][:min_features]
    
    selected_names = [feature_names[j] for j in selected]
    removed_names = [f for f in feature_names if f not in selected_names]
    
    model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
    model.fit(X_tr[:, selected], y_tr)
    y_pred = model.predict(X_te[:, selected])
    y_prob = model.predict_proba(X_te[:, selected])[:, 1]
    metrics = compute_metrics(y_te, y_pred, y_prob, s_te)
    
    return metrics, selected_names, removed_names


def run_experiment(n_seeds=10, alpha=0.5, threshold=0.2):
    csv_path = 'synthetic_indirect_proxy_loan.csv'
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ['Race', 'Loan_Approved']]
    X_raw = df[feature_cols].values.astype(np.float64)
    y = df['Loan_Approved'].values.astype(np.float64)
    sens = df['Race'].values.astype(np.float64)
    
    d = len(feature_cols)
    idx_A, idx_Y = d, d + 1
    min_features = max(3, d // 3)
    
    logger.info(f"Features: {feature_cols}")
    logger.info(f"d={d}, min_features={min_features}")
    
    dag_methods = [
        ('DAGMA-Linear', learn_dag_linear),
        ('DAGMA-MLP', learn_dag_nonlinear),
    ]
    
    influence_methods = ['Direct', 'Total Effect', '2-Hop', 'Taint']
    
    results = []
    
    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X_raw, y, sens, test_size=0.3, random_state=seed, stratify=y
        )
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        
        Z = np.column_stack([X_tr_sc, s_tr.reshape(-1,1), y_tr.reshape(-1,1)])
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        
        corr_A = np.array([abs(np.corrcoef(X_tr_sc[:,j], s_tr)[0,1]) for j in range(d)])
        
        # XGBoost baseline
        model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        model.fit(X_tr_sc, y_tr)
        y_pred = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)[:, 1]
        m = compute_metrics(y_te, y_pred, y_prob, s_te)
        results.append({'dag': 'None', 'influence': 'None', 'method': 'XGBoost',
                       'seed': seed, 'auc': m['auc'], 'eod': m['eod'], 'n_feats': d})
        
        # Oracle
        oracle_keep = [j for j, f in enumerate(feature_cols) if f not in ALL_PROXIES]
        model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        model.fit(X_tr_sc[:, oracle_keep], y_tr)
        y_pred = model.predict(X_te_sc[:, oracle_keep])
        y_prob = model.predict_proba(X_te_sc[:, oracle_keep])[:, 1]
        m = compute_metrics(y_te, y_pred, y_prob, s_te)
        results.append({'dag': 'Oracle', 'influence': 'Oracle', 'method': 'Oracle',
                       'seed': seed, 'auc': m['auc'], 'eod': m['eod'], 
                       'n_feats': len(oracle_keep)})
        
        # Each DAG method × each influence method
        for dag_name, dag_fn in dag_methods:
            try:
                W = dag_fn(Z)
                
                if seed == 0:
                    print_dag_analysis(W, feature_cols, idx_A, idx_Y, label=dag_name)
                
                all_influence = compute_influence_all_methods(W, idx_A, d)
                
                for inf_name in influence_methods:
                    influence = all_influence[inf_name]
                    method_label = f"{dag_name} + {inf_name}"
                    
                    m, selected, removed = select_and_evaluate(
                        W, corr_A, feature_cols, influence, idx_A, idx_Y, d,
                        X_tr_sc, X_te_sc, y_tr, y_te, s_te, seed,
                        alpha=alpha, threshold=threshold, min_features=min_features
                    )
                    
                    results.append({
                        'dag': dag_name, 'influence': inf_name,
                        'method': method_label, 'seed': seed,
                        'auc': m['auc'], 'eod': m['eod'],
                        'n_feats': len(selected),
                    })
                    
                    if seed == 0:
                        direct_rm = [f for f in GROUND_TRUTH['direct_proxies'] if f in removed]
                        indirect_rm = [f for f in GROUND_TRUTH['indirect_proxies'] if f in removed]
                        logger.info(f"  {method_label:40s}: selected={selected}, "
                                   f"direct_removed={len(direct_rm)}/2, "
                                   f"indirect_removed={len(indirect_rm)}/2, "
                                   f"EOD={m['eod']:.4f}")
            
            except Exception as e:
                logger.error(f"  {dag_name} failed on seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    rdf = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    summary = rdf.groupby('method').agg(
        auc=('auc', 'mean'), eod=('eod', 'mean'), eod_std=('eod', 'std'),
        n_feats=('n_features' if 'n_features' in rdf.columns else 'n_feats', 'mean'),
    ).round(4).sort_values('eod')
    
    baseline_eod = rdf[rdf['method']=='XGBoost']['eod'].mean()
    oracle_eod = rdf[rdf['method']=='Oracle']['eod'].mean()
    
    print(f"\nBaseline EOD: {baseline_eod:.4f}")
    print(f"Oracle EOD:   {oracle_eod:.4f}\n")
    
    for method, row in summary.iterrows():
        reduction = (baseline_eod - row['eod']) / baseline_eod * 100
        gap = (baseline_eod - row['eod']) / (baseline_eod - oracle_eod) * 100 if baseline_eod != oracle_eod else 0
        is_nonlinear = "★" if "MLP" in method else " "
        print(f"  {is_nonlinear} {method:45s}: EOD={row['eod']:.4f}±{row['eod_std']:.4f}  "
              f"AUC={row['auc']:.3f}  ↓{reduction:5.1f}%  gap={gap:5.1f}%")
    
    rdf.to_csv('nonlinear_dag_results.csv', index=False)
    print("\nSaved: nonlinear_dag_results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    run_experiment(n_seeds=args.n_seeds, alpha=args.alpha, threshold=args.threshold)
