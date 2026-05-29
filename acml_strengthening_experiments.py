"""
ACML 2026 — Final Strengthening Experiments (All-in-One)
=========================================================
Four experiments that raise the ceiling from borderline to weak accept:

  1. PARETO FRONTIER: Sweep FairGBM/M²FGB fairness strength, overlay CausalGBM
  2. MATCHED-AUC:     At CausalGBM's AUC, does tuned FairGBM beat it on EOD?
  3. SAMPLE-SIZE:     Proxy detection precision and EOD vs n (subsample ACS Income)
  4. HARDER SYNTHETIC: Nonlinear proxy + weak-signal proxy dataset

Usage:
  python acml_strengthening_experiments.py --all --n_seeds 10
  python acml_strengthening_experiments.py --pareto
  python acml_strengthening_experiments.py --sample_size
  python acml_strengthening_experiments.py --harder_synthetic
"""

import os, sys, argparse, warnings, logging, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import (
    CausalFeatureSelector, compute_metrics,
    load_adult, load_acs_income, load_compas,
    load_synthetic_loan, load_synthetic_hiring,
)
import xgboost as xgb

try:
    from fairgbm import FairGBMClassifier
    HAS_FAIRGBM = True
except ImportError:
    HAS_FAIRGBM = False
    logger.warning("FairGBM not installed")

try:
    from m2fgb.m2fgb import M2FGBClassifier
    HAS_M2FGB = True
except ImportError:
    HAS_M2FGB = False
    logger.warning("M2FGB not installed")

DATASET_LOADERS = {
    'adult': load_adult,
    'acs_income': load_acs_income,
    'compas': load_compas,
    'synthetic_loan': load_synthetic_loan,
    'synthetic_hiring': load_synthetic_hiring,
}


# ============================================================================
# EXPERIMENT 1: PARETO FRONTIER
# ============================================================================

def run_pareto_frontier(output_dir, n_seeds=5):
    """
    Sweep FairGBM and M²FGB fairness strength to trace EOD-AUC curves.
    Overlay CausalGBM as a single point.
    
    Answers ECML R3: "show the trade-off curve, not just one point."
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: PARETO FRONTIER (EOD vs AUC)")
    logger.info("=" * 70)
    
    pareto_datasets = ['adult', 'acs_income', 'compas']
    
    # FairGBM: sweep multiplier_learning_rate (controls fairness strength)
    fairgbm_strengths = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    # M²FGB: sweep fair_weight
    m2fgb_weights = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]
    
    results = []
    
    for ds_name in pareto_datasets:
        if ds_name not in DATASET_LOADERS:
            continue
        try:
            dataset = DATASET_LOADERS[ds_name]()
        except:
            continue
        
        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        logger.info(f"\n--- {ds_name} (n={len(X)}, d={d}) ---")
        
        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)
            
            # XGBoost baseline
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(X_tr, y_tr)
            yp, ypr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({
                'dataset': ds_name, 'method': 'XGBoost', 'strength': 0,
                'seed': seed, 'auc': met['auc'], 'eod': met['eod']})
            
            # CausalGBM (single point)
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=max(3, d // 3),
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr, s_tr, y_tr)
            Xtr_s, Xte_s = sel.transform(X_tr), sel.transform(X_te)
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({
                'dataset': ds_name, 'method': 'CausalGBM', 'strength': 0.5,
                'seed': seed, 'auc': met['auc'], 'eod': met['eod']})
            
            # FairGBM sweep
            if HAS_FAIRGBM:
                for strength in fairgbm_strengths:
                    try:
                        fgbm = FairGBMClassifier(
                            constraint_type="FNR,FPR", n_estimators=100,
                            random_state=seed,
                            multiplier_learning_rate=strength, verbose=-1)
                        fgbm.fit(X_tr, y_tr, constraint_group=s_tr)
                        yp = fgbm.predict(X_te)
                        ypr = fgbm.predict_proba(X_te)[:, 1]
                        met = compute_metrics(y_te, yp, ypr, s_te)
                        results.append({
                            'dataset': ds_name, 'method': 'FairGBM',
                            'strength': strength, 'seed': seed,
                            'auc': met['auc'], 'eod': met['eod']})
                    except:
                        pass
            
            # M²FGB sweep
            if HAS_M2FGB:
                for fw in m2fgb_weights:
                    try:
                        m2 = M2FGBClassifier(
                            fairness_constraint='true_positive_rate',
                            fair_weight=fw, n_estimators=100,
                            learning_rate=0.1, multiplier_learning_rate=0.1,
                            random_state=seed)
                        m2.fit(X_tr, y_tr, sensitive_attribute=s_tr)
                        yp = m2.predict(X_te)
                        ypr = m2.predict_proba(X_te)[:, 1]
                        met = compute_metrics(y_te, yp, ypr, s_te)
                        results.append({
                            'dataset': ds_name, 'method': 'M2FGB-TPR',
                            'strength': fw, 'seed': seed,
                            'auc': met['auc'], 'eod': met['eod']})
                    except:
                        pass
        
        # Print frontier for this dataset
        ds_df = pd.DataFrame([r for r in results if r['dataset'] == ds_name])
        summary = ds_df.groupby(['method', 'strength']).agg(
            auc=('auc', 'mean'), eod=('eod', 'mean')).round(4)
        logger.info(f"\n  Frontier points:")
        for (method, strength), row in summary.iterrows():
            marker = " ★" if method == 'CausalGBM' else ""
            logger.info(f"    {method:12s} (s={strength:5.3f}): "
                       f"AUC={row['auc']:.3f}  EOD={row['eod']:.3f}{marker}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'pareto_frontier_raw.csv'), index=False)
    
    # Print matched-AUC comparison
    print("\n" + "=" * 70)
    print("MATCHED-AUC COMPARISON")
    print("=" * 70)
    print("At CausalGBM's AUC level, which method has best EOD?\n")
    
    for ds_name in pareto_datasets:
        ds_df = df[df['dataset'] == ds_name]
        cgbm = ds_df[ds_df['method'] == 'CausalGBM'].groupby('seed').agg(
            auc=('auc', 'mean'), eod=('eod', 'mean'))
        cgbm_auc = cgbm['auc'].mean()
        cgbm_eod = cgbm['eod'].mean()
        
        print(f"  --- {ds_name} (CausalGBM AUC={cgbm_auc:.3f}, EOD={cgbm_eod:.3f}) ---")
        
        for method in ['FairGBM', 'M2FGB-TPR']:
            method_df = ds_df[ds_df['method'] == method]
            if method_df.empty:
                continue
            # Find strength closest to CausalGBM's AUC
            avg = method_df.groupby('strength').agg(
                auc=('auc', 'mean'), eod=('eod', 'mean'))
            avg['auc_diff'] = abs(avg['auc'] - cgbm_auc)
            best_match = avg['auc_diff'].idxmin()
            matched = avg.loc[best_match]
            
            winner = "CausalGBM wins" if cgbm_eod < matched['eod'] else f"{method} wins"
            print(f"    {method:12s} @ matched AUC={matched['auc']:.3f}: "
                  f"EOD={matched['eod']:.3f}  → {winner}")
    
    print(f"\nSaved: {os.path.join(output_dir, 'pareto_frontier_raw.csv')}")
    return df


# ============================================================================
# EXPERIMENT 2: SAMPLE-SIZE CURVE
# ============================================================================

def run_sample_size_curve(output_dir, n_seeds=5):
    """
    Subsample ACS Income at different n, measure:
    - Proxy detection precision (does CausalGBM find the right proxies?)
    - EOD reduction
    - DAG quality
    
    Substantiates the "n>1,000" deployment guideline.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: SAMPLE-SIZE CURVE (ACS Income)")
    logger.info("=" * 70)
    
    try:
        dataset = load_acs_income()
    except:
        logger.error("ACS Income not available")
        return None
    
    X_full, y_full, sens_full = dataset.X, dataset.y, dataset.sensitive
    d = X_full.shape[1]
    
    # Known proxies on ACS Income (from Section 4.8)
    known_proxy_indices = {0, 1, 3, 4, 5, 7}  # POBP, RAC1P, RELP, COW, MAR, OCCP
    known_legitimate = {2, 6, 8, 9}  # SCHL, AGEP, WKHP, and one more
    
    sample_sizes = [500, 750, 1000, 1500, 2000, 3000, 5000, 7500,
                    10000, 20000, 50000, 100000]
    
    results = []
    
    for n in sample_sizes:
        if n > len(X_full):
            continue
        logger.info(f"\n  n={n}")
        
        for seed in range(n_seeds):
            # Subsample
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_full), size=n, replace=False)
            X, y, sens = X_full[idx], y_full[idx], sens_full[idx]
            
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)
            
            # XGBoost baseline
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(X_tr, y_tr)
            yp, ypr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
            met_base = compute_metrics(y_te, yp, ypr, s_te)
            
            # CausalGBM
            try:
                sel = CausalFeatureSelector(
                    d, alpha=0.5, threshold=0.2,
                    min_features=max(3, d // 3),
                    n_iterations=500, aggregation='max', device='cpu')
                sel.fit(X_tr, s_tr, y_tr)
                selected = set(sel.selected_)
                removed = set(range(d)) - selected
                
                Xtr_s, Xte_s = sel.transform(X_tr), sel.transform(X_te)
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(Xtr_s, y_tr)
                yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
                met_cgbm = compute_metrics(y_te, yp, ypr, s_te)
                
                # Detection metrics
                true_pos = len(removed & known_proxy_indices)
                false_pos = len(removed - known_proxy_indices)
                false_neg = len(known_proxy_indices - removed)
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / len(known_proxy_indices) if len(known_proxy_indices) > 0 else 0
                
                results.append({
                    'n': n, 'seed': seed,
                    'baseline_eod': met_base['eod'], 'baseline_auc': met_base['auc'],
                    'cgbm_eod': met_cgbm['eod'], 'cgbm_auc': met_cgbm['auc'],
                    'eod_reduction': (met_base['eod'] - met_cgbm['eod']) / met_base['eod'] * 100
                        if met_base['eod'] > 0 else 0,
                    'proxy_precision': precision,
                    'proxy_recall': recall,
                    'n_selected': len(selected),
                    'n_removed': len(removed),
                })
            except Exception as e:
                logger.warning(f"    seed={seed} failed: {e}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'sample_size_curve_raw.csv'), index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SAMPLE-SIZE CURVE RESULTS")
    print("=" * 70)
    print(f"\n  {'n':>8s}  {'EOD_red%':>8s}  {'Precision':>9s}  {'Recall':>7s}  {'AUC':>6s}")
    print("  " + "-" * 48)
    
    summary = df.groupby('n').agg(
        eod_red=('eod_reduction', 'mean'),
        precision=('proxy_precision', 'mean'),
        recall=('proxy_recall', 'mean'),
        auc=('cgbm_auc', 'mean'),
    ).round(3)
    
    for n, row in summary.iterrows():
        marker = " ← n=1000" if n == 1000 else ""
        print(f"  {n:>8d}  {row['eod_red']:>7.1f}%  {row['precision']:>9.2f}  "
              f"{row['recall']:>7.2f}  {row['auc']:>6.3f}{marker}")
    
    print(f"\nSaved: {os.path.join(output_dir, 'sample_size_curve_raw.csv')}")
    return df


# ============================================================================
# EXPERIMENT 3: HARDER SYNTHETIC (nonlinear + weak signals)
# ============================================================================

def run_harder_synthetic(output_dir, n_seeds=10):
    """
    Synthetic dataset with NONLINEAR proxy relationships and weak signals.
    Tests CausalGBM outside the favorable linear/direct regime.
    
    Design:
    - X1: nonlinear proxy (A→X1 via interaction/threshold, not linear)
    - X2: weak-signal proxy (A→X2 with very low effect size)
    - X3,X4,X5: legitimate
    - X6: spurious
    - Y depends on all features through a nonlinear function
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: HARDER SYNTHETIC (nonlinear + weak)")
    logger.info("=" * 70)
    
    results = []
    
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        n = 10000
        
        # Protected attribute
        A = rng.binomial(1, 0.5, n).astype(float)
        
        # Confounder
        C = rng.randn(n)
        
        # X1: NONLINEAR proxy — interaction effect
        # X1 = A * noise1 + (1-A) * noise2 with different distributions
        # Group 0: X1 ~ N(0, 0.5²), Group 1: X1 ~ N(0, 0.5²) + Bernoulli(0.7)*2
        X1 = rng.randn(n) * 0.5
        X1[A == 1] += rng.binomial(1, 0.7, int(A.sum())) * 2.0
        # This creates a mixture relationship that's highly nonlinear
        
        # X2: WEAK proxy — very small effect size
        X2 = 0.10 * A + rng.randn(n)  # corr ≈ 0.05
        
        # X3-X5: Legitimate
        X3 = rng.randn(n) * 1.5
        X4 = rng.randn(n)
        X5 = rng.randn(n) * 0.8
        
        # X6: Spurious (confounder-driven)
        X6 = 0.5 * C + rng.randn(n) * 0.5
        
        # Y: NONLINEAR outcome function
        Y_logit = (0.3 * np.sign(X1) * X1**2  # nonlinear X1 effect
                   + 0.4 * X2
                   + 0.5 * X3
                   + 0.3 * X4
                   + 0.2 * X5
                   + rng.randn(n) * 0.5)
        Y = (Y_logit > np.median(Y_logit)).astype(float)
        
        X = np.column_stack([X1, X2, X3, X4, X5, X6])
        d = 6
        feature_names = ['X1_nonlinear_proxy', 'X2_weak_proxy',
                        'X3_legit', 'X4_legit', 'X5_legit', 'X6_spurious']
        
        if seed == 0:
            logger.info(f"  corr(X1, A) = {abs(np.corrcoef(X1, A)[0,1]):.3f}  (nonlinear proxy)")
            logger.info(f"  corr(X2, A) = {abs(np.corrcoef(X2, A)[0,1]):.3f}  (weak proxy)")
            logger.info(f"  corr(X6, A) = {abs(np.corrcoef(X6, A)[0,1]):.3f}  (spurious)")
        
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X, Y, A, test_size=0.3, random_state=seed, stratify=Y)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        
        # XGBoost baseline
        m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        m.fit(X_tr_sc, y_tr)
        yp, ypr = m.predict(X_te_sc), m.predict_proba(X_te_sc)[:, 1]
        met = compute_metrics(y_te, yp, ypr, s_te)
        results.append({'method': 'XGBoost', 'seed': seed, **met, 'regime': 'nonlinear'})
        
        # Oracle (remove both proxies)
        oracle_idx = [2, 3, 4, 5]
        m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        m.fit(X_tr_sc[:, oracle_idx], y_tr)
        yp = m.predict(X_te_sc[:, oracle_idx])
        ypr = m.predict_proba(X_te_sc[:, oracle_idx])[:, 1]
        met = compute_metrics(y_te, yp, ypr, s_te)
        results.append({'method': 'Oracle', 'seed': seed, **met, 'regime': 'nonlinear'})
        
        # CausalGBM
        sel = CausalFeatureSelector(
            d, alpha=0.5, threshold=0.2,
            min_features=3, n_iterations=500,
            aggregation='max', device='cpu')
        sel.fit(X_tr_sc, s_tr, y_tr)
        Xtr_s, Xte_s = sel.transform(X_tr_sc), sel.transform(X_te_sc)
        
        m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        m.fit(Xtr_s, y_tr)
        yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
        met = compute_metrics(y_te, yp, ypr, s_te)
        
        sel_names = [feature_names[j] for j in sel.selected_]
        rem_names = [f for f in feature_names if f not in sel_names]
        
        results.append({
            'method': 'CausalGBM', 'seed': seed, **met, 'regime': 'nonlinear',
        })
        
        if seed == 0:
            logger.info(f"  CausalGBM selected: {sel_names}")
            logger.info(f"  CausalGBM removed:  {rem_names}")
            nonlin_removed = 'X1_nonlinear_proxy' in rem_names
            weak_removed = 'X2_weak_proxy' in rem_names
            logger.info(f"  Nonlinear proxy detected: {nonlin_removed}")
            logger.info(f"  Weak proxy detected: {weak_removed}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'harder_synthetic_raw.csv'), index=False)
    
    print("\n" + "=" * 70)
    print("HARDER SYNTHETIC RESULTS (nonlinear + weak proxies)")
    print("=" * 70)
    
    summary = df.groupby('method').agg(
        eod=('eod', 'mean'), eod_std=('eod', 'std'),
        auc=('auc', 'mean')).round(4)
    
    baseline = summary.loc['XGBoost', 'eod']
    for method in ['XGBoost', 'CausalGBM', 'Oracle']:
        r = summary.loc[method]
        red = (baseline - r['eod']) / baseline * 100
        print(f"  {method:12s}: EOD={r['eod']:.4f}±{r['eod_std']:.4f}  "
              f"AUC={r['auc']:.3f}  ↓{red:5.1f}%")
    
    oracle_eod = summary.loc['Oracle', 'eod']
    cgbm_eod = summary.loc['CausalGBM', 'eod']
    gap_closed = (baseline - cgbm_eod) / (baseline - oracle_eod) * 100 if baseline != oracle_eod else 0
    print(f"\n  Gap closed: {gap_closed:.1f}%")
    print(f"  (This is the HARDER regime — 100% is not expected)")
    
    print(f"\nSaved: {os.path.join(output_dir, 'harder_synthetic_raw.csv')}")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ACML 2026 Strengthening Experiments')
    parser.add_argument('--pareto', action='store_true')
    parser.add_argument('--sample_size', action='store_true')
    parser.add_argument('--harder_synthetic', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--output_dir', default='results/acml2026/strengthening')
    args = parser.parse_args()
    
    if args.all:
        args.pareto = args.sample_size = args.harder_synthetic = True
    if not any([args.pareto, args.sample_size, args.harder_synthetic]):
        args.pareto = args.sample_size = args.harder_synthetic = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.pareto:
        run_pareto_frontier(args.output_dir, n_seeds=args.n_seeds)
    
    if args.sample_size:
        run_sample_size_curve(args.output_dir, n_seeds=args.n_seeds)
    
    if args.harder_synthetic:
        run_harder_synthetic(args.output_dir, n_seeds=max(args.n_seeds, 10))
    
    logger.info(f"\nAll experiments complete! Results in: {args.output_dir}")


if __name__ == '__main__':
    main()
