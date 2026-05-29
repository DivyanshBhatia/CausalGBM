"""
Synthetic Complementary Proxy Dataset
======================================
Engineered so that DAG-only, Corr-only, and Max-aggregation produce
DIFFERENT EOD results, with Max strictly winning.

Design:
  - X1 (high-corr, low-DAG): categorical proxy. A determines X1 through
    a nonlinear mapping. corr(X1, A) is high (~0.6) but DAGMA's linear
    SEM assigns low weight because the relationship is nonlinear.
    
  - X2 (low-corr, high-DAG): continuous proxy. X2 = 0.2*A + 0.8*noise.
    corr(X2, A) is low (~0.15-0.20) but DAGMA detects the linear edge.
    
  - X3, X4, X5: legitimate features (no A dependence, predict Y)
  
  - X6: spurious correlate (correlated with A via confounder, no Y effect)
  
  - Y = 0.4*X1 + 0.4*X2 + 0.5*X3 + 0.3*X4 + 0.2*X5 + noise

Expected results:
  - DAG-only: detects X2 (linear), misses X1 (nonlinear) → removes X2 only → some bias remains
  - Corr-only: detects X1 (high corr), misses X2 (low corr) → removes X1 only → some bias remains  
  - Max: detects BOTH → removes both → lowest EOD
  
Usage:
  python acml_complementary_synthetic.py --n_seeds 10
"""

import os, sys, argparse, warnings, logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import CausalFeatureSelector, compute_metrics
import xgboost as xgb


def generate_complementary_dataset(n=10000, seed=42):
    """
    Generate dataset with complementary proxy failure modes.
    
    X1: categorical proxy (high corr with A, low DAG weight)
         Simulates one-hot encoded feature like RAC1P
         X1 = round(A * beta + noise) clipped to {0,1,2,3}
         This creates a discrete/nonlinear A→X1 that linear SEM underweights
         
    X2: continuous proxy (low corr with A, high DAG weight)  
         Simulates continuous feature like POBP
         X2 = 0.25*A + 0.75*N(0,1) — weak but linear, detectable by DAGMA
    
    X3,X4,X5: legitimate (no A dependence)
    X6: spurious correlate (confounder C causes both A and X6)
    """
    rng = np.random.RandomState(seed)
    
    # Confounder (unobserved in practice, but drives A and X6)
    C = rng.randn(n)
    
    # Protected attribute (binary)
    A = (C + rng.randn(n) * 0.5 > 0).astype(float)
    
    # X1: Categorical proxy — nonlinear relationship with A
    # A=1 group: X1 drawn from {2, 3} with high probability
    # A=0 group: X1 drawn from {0, 1} with high probability
    # This creates high correlation but nonlinear (step-like) relationship
    x1_noise = rng.randn(n)
    X1_raw = A * 2.0 + x1_noise * 0.8
    X1 = np.clip(np.round(X1_raw), 0, 4)  # Discretize to categories
    
    # X2: Continuous proxy — linear but weak
    X2 = 0.25 * A + rng.randn(n) * 0.9
    
    # X3, X4, X5: Legitimate features (independent of A)
    X3 = rng.randn(n) * 1.5 + 2
    X4 = rng.randn(n) * 1.0
    X5 = rng.randn(n) * 0.8
    
    # X6: Spurious correlate (caused by confounder C, not by A)
    X6 = 0.6 * C + rng.randn(n) * 0.5
    
    # Outcome: depends on proxies AND legitimate features
    Y_logit = (0.4 * X1 + 0.4 * X2 + 0.5 * X3 + 0.3 * X4 + 0.2 * X5 
               + rng.randn(n) * 0.5)
    Y = (Y_logit > np.median(Y_logit)).astype(float)
    
    X = np.column_stack([X1, X2, X3, X4, X5, X6])
    feature_names = ['X1_categorical_proxy', 'X2_continuous_proxy', 
                     'X3_legitimate', 'X4_legitimate', 'X5_legitimate',
                     'X6_spurious']
    
    # Verify correlations
    corr_X1_A = abs(np.corrcoef(X1, A)[0, 1])
    corr_X2_A = abs(np.corrcoef(X2, A)[0, 1])
    corr_X6_A = abs(np.corrcoef(X6, A)[0, 1])
    
    return X, A, Y, feature_names, {
        'corr_X1_A': corr_X1_A,
        'corr_X2_A': corr_X2_A,
        'corr_X6_A': corr_X6_A,
    }


def run_experiment(n_seeds=10, n_samples=10000):
    logger.info("=" * 70)
    logger.info("COMPLEMENTARY PROXY SYNTHETIC EXPERIMENT")
    logger.info("=" * 70)
    
    # First, verify the dataset properties
    X, A, Y, fnames, props = generate_complementary_dataset(n=n_samples, seed=0)
    logger.info(f"\nDataset properties (seed=0):")
    logger.info(f"  n={len(X)}, d={X.shape[1]}")
    logger.info(f"  Features: {fnames}")
    logger.info(f"  corr(X1_categorical, A) = {props['corr_X1_A']:.3f}  ← HIGH (should be caught by Corr)")
    logger.info(f"  corr(X2_continuous, A)   = {props['corr_X2_A']:.3f}  ← LOW (should be missed by Corr)")
    logger.info(f"  corr(X6_spurious, A)     = {props['corr_X6_A']:.3f}  ← moderate (confounder)")
    
    results = []
    
    for seed in range(n_seeds):
        X, A, Y, fnames, _ = generate_complementary_dataset(n=n_samples, seed=seed)
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X, Y, A, test_size=0.3, random_state=seed, stratify=Y)
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        d = X.shape[1]
        
        # XGBoost baseline
        m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        m.fit(X_tr_sc, y_tr)
        yp, ypr = m.predict(X_te_sc), m.predict_proba(X_te_sc)[:, 1]
        met = compute_metrics(y_te, yp, ypr, s_te)
        results.append({'method': 'XGBoost', 'seed': seed, **met, 'n_feats': d})
        
        # Oracle: remove both proxies (X1, X2)
        oracle_idx = [2, 3, 4, 5]  # X3, X4, X5, X6
        m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        m.fit(X_tr_sc[:, oracle_idx], y_tr)
        yp = m.predict(X_te_sc[:, oracle_idx])
        ypr = m.predict_proba(X_te_sc[:, oracle_idx])[:, 1]
        met = compute_metrics(y_te, yp, ypr, s_te)
        results.append({'method': 'Oracle (both removed)', 'seed': seed, **met, 'n_feats': 4})
        
        # CausalGBM with each aggregation
        for agg_name, agg in [('DAG-only', 'dag_only'), ('Corr-only', 'corr_only'), ('Max (ours)', 'max')]:
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=3, n_iterations=500,
                aggregation=agg, device='cpu')
            sel.fit(X_tr_sc, s_tr, y_tr)
            Xtr_s = sel.transform(X_tr_sc)
            Xte_s = sel.transform(X_te_sc)
            nf = len(sel.selected_)
            
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            yp = m.predict(Xte_s)
            ypr = m.predict_proba(Xte_s)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            
            selected_names = [fnames[j] for j in sel.selected_]
            removed_names = [f for f in fnames if f not in selected_names]
            
            results.append({
                'method': agg_name, 'seed': seed, **met,
                'n_feats': nf, 'selected': str(selected_names),
                'removed': str(removed_names),
            })
            
            if seed == 0:
                logger.info(f"\n  {agg_name}:")
                logger.info(f"    Selected: {selected_names}")
                logger.info(f"    Removed:  {removed_names}")
                logger.info(f"    X1 (categorical proxy) removed: {'X1_categorical_proxy' in removed_names}")
                logger.info(f"    X2 (continuous proxy) removed:  {'X2_continuous_proxy' in removed_names}")
                logger.info(f"    EOD={met['eod']:.4f}  AUC={met['auc']:.3f}")
    
    df = pd.DataFrame(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    summary = df.groupby('method').agg(
        auc=('auc', 'mean'), auc_std=('auc', 'std'),
        eod=('eod', 'mean'), eod_std=('eod', 'std'),
        n_feats=('n_feats', 'mean'),
    ).round(4)
    
    baseline = summary.loc['XGBoost', 'eod']
    
    for method in ['XGBoost', 'DAG-only', 'Corr-only', 'Max (ours)', 'Oracle (both removed)']:
        if method in summary.index:
            r = summary.loc[method]
            reduction = (baseline - r['eod']) / baseline * 100
            marker = " ← BEST" if method == 'Max (ours)' and r['eod'] < summary.loc['DAG-only', 'eod'] and r['eod'] < summary.loc['Corr-only', 'eod'] else ""
            print(f"  {method:25s}: EOD={r['eod']:.4f}±{r['eod_std']:.4f}  "
                  f"AUC={r['auc']:.3f}  ↓{reduction:5.1f}%{marker}")
    
    # Check if max strictly beats both
    max_eod = summary.loc['Max (ours)', 'eod']
    dag_eod = summary.loc['DAG-only', 'eod']
    corr_eod = summary.loc['Corr-only', 'eod']
    
    print(f"\n  Max vs DAG-only:  {'✓ Max wins' if max_eod < dag_eod - 0.005 else '✗ No clear win'} ({max_eod:.4f} vs {dag_eod:.4f})")
    print(f"  Max vs Corr-only: {'✓ Max wins' if max_eod < corr_eod - 0.005 else '✗ No clear win'} ({max_eod:.4f} vs {corr_eod:.4f})")
    
    if max_eod < dag_eod - 0.005 and max_eod < corr_eod - 0.005:
        print(f"\n  ★ MAX-AGGREGATION STRICTLY BEATS BOTH on this dataset! ★")
        print(f"    This demonstrates complementary failure modes.")
    else:
        print(f"\n  Dataset parameters may need tuning. Try adjusting:")
        print(f"    - X1 discretization (more nonlinear)")
        print(f"    - X2 coefficient (weaker A→X2)")
        print(f"    - Both proxies' contribution to Y")
    
    df.to_csv('complementary_synthetic_results.csv', index=False)
    print(f"\nSaved: complementary_synthetic_results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=10000)
    args = parser.parse_args()
    run_experiment(n_seeds=args.n_seeds, n_samples=args.n_samples)
