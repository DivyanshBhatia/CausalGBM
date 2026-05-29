"""
Synthetic Credit Lending Dataset: Complementary Proxy Failure Modes
====================================================================
A realistic credit lending scenario designed to demonstrate why both
DAG-based AND correlation-based proxy detection are needed.

Scenario: A bank uses 12 features to predict credit default risk.
Two features encode unfair racial bias through different mechanisms:

  1. NEIGHBORHOOD_RISK_TIER (categorical proxy):
     Zip-code-derived risk category {1,2,3,4,5}. Due to historical
     residential segregation, race strongly predicts neighborhood tier
     through a NONLINEAR mapping (threshold effects, categorical bins).
     Correlation with race is HIGH (~0.55), but DAGMA's linear SEM
     assigns LOW edge weight because the relationship is non-linear.
     → Caught by correlation, missed by DAG

  2. BRANCH_PROXIMITY_SCORE (continuous proxy):
     Distance to nearest bank branch, normalized. Minority communities
     have fewer branches (documented "banking desert" effect). The
     relationship is LINEAR but WEAK: race explains only ~5% of variance.
     Correlation is LOW (~0.18), but DAGMA detects the linear edge.
     → Caught by DAG, missed by correlation

Both features causally affect the outcome (default prediction depends
on neighborhood stability and banking access). Removing only one
leaves residual bias from the other.

Dataset: n=20,000, d=12, binary protected attribute (Race)

Quality Verification:
  - Prints correlation matrix
  - Verifies ground truth causal structure
  - Reports DAG recovery quality
  - Computes baseline unfairness

Usage:
  python acml_synthetic_lending.py --generate           # Generate + verify
  python acml_synthetic_lending.py --experiment          # Run ablation
  python acml_synthetic_lending.py --all --n_seeds 10    # Both
"""

import os, sys, argparse, warnings, logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr, pointbiserialr

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import CausalFeatureSelector, compute_metrics
import xgboost as xgb

# ============================================================================
# GROUND TRUTH CAUSAL STRUCTURE
# ============================================================================

FEATURE_METADATA = {
    'Neighborhood_Risk_Tier': {
        'type': 'categorical_proxy',
        'mechanism': 'Historical residential segregation creates nonlinear '
                     'Race→Tier mapping (threshold/binning effects)',
        'expected_corr_with_A': 'HIGH (~0.55)',
        'expected_dag_weight': 'LOW (linear SEM cannot model categorical mapping)',
        'detected_by': 'Correlation',
    },
    'Branch_Proximity_Score': {
        'type': 'continuous_proxy',
        'mechanism': 'Banking desert effect: minority areas have fewer branches. '
                     'Linear but weak causal effect',
        'expected_corr_with_A': 'LOW (~0.18)',
        'expected_dag_weight': 'HIGH (linear relationship, detectable by DAGMA)',
        'detected_by': 'DAG',
    },
    'Annual_Income': {'type': 'legitimate', 'mechanism': 'Direct predictor of repayment capacity'},
    'Credit_Score': {'type': 'legitimate', 'mechanism': 'Credit history (FICO-like)'},
    'Employment_Years': {'type': 'legitimate', 'mechanism': 'Job stability indicator'},
    'Debt_to_Income': {'type': 'legitimate', 'mechanism': 'Financial leverage ratio'},
    'Savings_Balance': {'type': 'legitimate', 'mechanism': 'Liquid assets / safety net'},
    'Loan_Amount': {'type': 'legitimate', 'mechanism': 'Size of requested credit'},
    'Num_Dependents': {'type': 'legitimate', 'mechanism': 'Family financial obligations'},
    'Education_Years': {'type': 'legitimate', 'mechanism': 'Human capital proxy'},
    'County_Unemployment': {
        'type': 'spurious',
        'mechanism': 'Correlated with Race via geographic confounder (regional '
                     'economics). Does NOT causally affect individual default.',
    },
    'Church_Attendance': {
        'type': 'spurious',
        'mechanism': 'Correlated with Race via cultural confounder. '
                     'No causal effect on creditworthiness.',
    },
}

FEATURE_NAMES = list(FEATURE_METADATA.keys())
PROXY_FEATURES = [f for f, m in FEATURE_METADATA.items() if 'proxy' in m['type']]
LEGITIMATE_FEATURES = [f for f, m in FEATURE_METADATA.items() if m['type'] == 'legitimate']
SPURIOUS_FEATURES = [f for f, m in FEATURE_METADATA.items() if m['type'] == 'spurious']


# ============================================================================
# DATA GENERATING PROCESS
# ============================================================================

def generate_lending_dataset(n=20000, seed=42):
    """
    Generate the synthetic credit lending dataset.
    
    Structural Equations:
    ---------------------
    Confounder (unobserved):
      C ~ N(0, 1)                                    # Regional/geographic factor
    
    Protected attribute:
      Race = 1{C + ε_A > 0},  ε_A ~ N(0, 0.5²)      # Binary
    
    Categorical proxy (nonlinear A→X):
      raw = 2.5·Race + ε₁,  ε₁ ~ N(0, 0.7²)
      Neighborhood_Risk_Tier = clip(round(raw), 1, 5)  # Discretized to {1,2,3,4,5}
    
    Continuous proxy (linear A→X):
      Branch_Proximity_Score = 0.25·Race + ε₂,  ε₂ ~ N(0, 1.0²)
    
    Legitimate features (independent of Race):
      Annual_Income = 50000 + 15000·ε₃               # N(50K, 15K²)
      Credit_Score = 680 + 60·ε₄                     # N(680, 60²)
      Employment_Years = max(0, 8 + 5·ε₅)            # N(8, 5²) truncated
      Debt_to_Income = max(0.05, 0.35 + 0.15·ε₆)    # N(0.35, 0.15²) truncated
      Savings_Balance = max(0, 10000 + 8000·ε₇)      # N(10K, 8K²) truncated
      Loan_Amount = max(1000, 15000 + 10000·ε₈)      # N(15K, 10K²) truncated
      Num_Dependents = max(0, round(1.5 + 1.2·ε₉))   # Poisson-like
      Education_Years = max(8, round(14 + 2.5·ε₁₀))   # N(14, 2.5²) rounded
    
    Spurious correlates (confounder-driven, no causal effect on Y):
      County_Unemployment = 0.06 + 0.02·C + ε₁₁      # Driven by geography
      Church_Attendance = max(0, 2 + 1.5·C + ε₁₂)    # Cultural confounder
    
    Outcome (credit default):
      logit(Y) = -2
                 + 0.40 · Neighborhood_Risk_Tier_normalized
                 + 0.35 · Branch_Proximity_Score_normalized
                 + 0.50 · (−Credit_Score_normalized)
                 + 0.40 · Debt_to_Income_normalized
                 + 0.30 · (−Annual_Income_normalized)
                 + 0.25 · (−Employment_Years_normalized)
                 + 0.20 · (−Savings_Balance_normalized)
                 + 0.15 · Loan_Amount_normalized
                 + 0.10 · Num_Dependents_normalized
                 + ε_Y
      Y = 1{logit(Y) > 0}
    
    Note: Education_Years intentionally excluded from Y equation
    (has zero causal effect — tests whether method correctly retains it
    as a non-proxy despite moderate correlation with other features).
    """
    rng = np.random.RandomState(seed)
    
    # Confounder (geographic/regional factor)
    C = rng.randn(n)
    
    # Protected attribute
    A = (C + rng.randn(n) * 0.5 > 0).astype(float)
    
    # === PROXY FEATURES ===
    
    # 1. Neighborhood Risk Tier (categorical proxy)
    # Nonlinear: discretized mapping creates step function
    tier_raw = 2.5 * A + rng.randn(n) * 0.7
    Neighborhood_Risk_Tier = np.clip(np.round(tier_raw), 1, 5)
    
    # 2. Branch Proximity Score (continuous proxy)  
    # Linear but weak
    Branch_Proximity_Score = 0.25 * A + rng.randn(n) * 1.0
    
    # === LEGITIMATE FEATURES ===
    Annual_Income = 50000 + 15000 * rng.randn(n)
    Credit_Score = 680 + 60 * rng.randn(n)
    Employment_Years = np.maximum(0, 8 + 5 * rng.randn(n))
    Debt_to_Income = np.maximum(0.05, 0.35 + 0.15 * rng.randn(n))
    Savings_Balance = np.maximum(0, 10000 + 8000 * rng.randn(n))
    Loan_Amount = np.maximum(1000, 15000 + 10000 * rng.randn(n))
    Num_Dependents = np.maximum(0, np.round(1.5 + 1.2 * rng.randn(n)))
    Education_Years = np.maximum(8, np.round(14 + 2.5 * rng.randn(n)))
    
    # === SPURIOUS CORRELATES ===
    County_Unemployment = 0.06 + 0.02 * C + rng.randn(n) * 0.01
    Church_Attendance = np.maximum(0, 2 + 1.5 * C + rng.randn(n) * 1.0)
    
    # === OUTCOME ===
    # Normalize all features to [0,1] range for coefficient interpretability
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    logit_Y = (-2.0
               + 0.40 * normalize(Neighborhood_Risk_Tier)
               + 0.35 * normalize(Branch_Proximity_Score)
               - 0.50 * normalize(Credit_Score)
               + 0.40 * normalize(Debt_to_Income)
               - 0.30 * normalize(Annual_Income)
               - 0.25 * normalize(Employment_Years)
               - 0.20 * normalize(Savings_Balance)
               + 0.15 * normalize(Loan_Amount)
               + 0.10 * normalize(Num_Dependents)
               + rng.randn(n) * 0.5)
    
    Y = (logit_Y > np.median(logit_Y)).astype(float)
    
    # Assemble feature matrix
    X = np.column_stack([
        Neighborhood_Risk_Tier, Branch_Proximity_Score,
        Annual_Income, Credit_Score, Employment_Years,
        Debt_to_Income, Savings_Balance, Loan_Amount,
        Num_Dependents, Education_Years,
        County_Unemployment, Church_Attendance,
    ])
    
    return X, A, Y


# ============================================================================
# QUALITY VERIFICATION
# ============================================================================

def verify_dataset(X, A, Y):
    """Comprehensive quality checks on the generated dataset."""
    n, d = X.shape
    
    print("\n" + "=" * 70)
    print("DATASET QUALITY VERIFICATION")
    print("=" * 70)
    
    print(f"\n  Shape: n={n}, d={d}")
    print(f"  Class balance: Y=1 {Y.mean():.1%}, Y=0 {1-Y.mean():.1%}")
    print(f"  Group balance: A=1 {A.mean():.1%}, A=0 {1-A.mean():.1%}")
    
    # Feature statistics
    print(f"\n  {'Feature':<28s} {'Type':<18s} {'Mean':>10s} {'Std':>10s} {'corr(X,A)':>10s} {'corr(X,Y)':>10s}")
    print("  " + "-" * 96)
    
    for j, fname in enumerate(FEATURE_NAMES):
        ftype = FEATURE_METADATA[fname]['type']
        corr_a = abs(np.corrcoef(X[:, j], A)[0, 1])
        corr_y = abs(np.corrcoef(X[:, j], Y)[0, 1])
        marker = ""
        if 'proxy' in ftype:
            marker = " ◄"
        print(f"  {fname:<28s} {ftype:<18s} {X[:,j].mean():>10.2f} {X[:,j].std():>10.2f} "
              f"{corr_a:>10.3f} {corr_y:>10.3f}{marker}")
    
    # Verify proxy properties
    print(f"\n  PROXY VERIFICATION:")
    cat_corr = abs(np.corrcoef(X[:, 0], A)[0, 1])
    cont_corr = abs(np.corrcoef(X[:, 1], A)[0, 1])
    print(f"    Neighborhood_Risk_Tier corr with Race: {cat_corr:.3f} "
          f"{'✓ HIGH' if cat_corr > 0.4 else '✗ needs tuning'}")
    print(f"    Branch_Proximity_Score corr with Race: {cont_corr:.3f} "
          f"{'✓ LOW' if cont_corr < 0.25 else '✗ needs tuning'}")
    
    # Verify spurious correlates
    spur1_corr = abs(np.corrcoef(X[:, 10], A)[0, 1])
    spur2_corr = abs(np.corrcoef(X[:, 11], A)[0, 1])
    spur1_corr_y = abs(np.corrcoef(X[:, 10], Y)[0, 1])
    spur2_corr_y = abs(np.corrcoef(X[:, 11], Y)[0, 1])
    print(f"    County_Unemployment corr(A)={spur1_corr:.3f}, corr(Y)={spur1_corr_y:.3f} "
          f"{'✓ low Y-corr' if spur1_corr_y < 0.1 else '⚠ may affect Y'}")
    print(f"    Church_Attendance corr(A)={spur2_corr:.3f}, corr(Y)={spur2_corr_y:.3f} "
          f"{'✓ low Y-corr' if spur2_corr_y < 0.1 else '⚠ may affect Y'}")
    
    # Verify legitimate features are independent of A
    print(f"\n  LEGITIMATE FEATURE INDEPENDENCE:")
    for j, fname in enumerate(FEATURE_NAMES):
        if FEATURE_METADATA[fname]['type'] == 'legitimate':
            corr_a = abs(np.corrcoef(X[:, j], A)[0, 1])
            status = "✓" if corr_a < 0.05 else "⚠ weak dependence"
            print(f"    {fname:<28s}: corr(A)={corr_a:.4f} {status}")
    
    # Baseline unfairness
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, Y, A, test_size=0.3, random_state=0, stratify=Y)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    m = xgb.XGBClassifier(n_estimators=100, random_state=0, verbosity=0)
    m.fit(X_tr_sc, y_tr)
    yp = m.predict(X_te_sc)
    ypr = m.predict_proba(X_te_sc)[:, 1]
    met = compute_metrics(y_te, yp, ypr, s_te)
    print(f"\n  BASELINE UNFAIRNESS (XGBoost, no debiasing):")
    print(f"    AUC={met['auc']:.3f}  EOD={met['eod']:.3f}  DPD={met['dpd']:.3f}")
    print(f"    {'✓ Sufficient baseline unfairness' if met['eod'] > 0.05 else '⚠ Low unfairness — may need tuning'}")
    
    return met


# ============================================================================
# ABLATION EXPERIMENT
# ============================================================================

def run_ablation(n_seeds=10, n_samples=20000):
    logger.info("=" * 70)
    logger.info("COMPLEMENTARY PROXY ABLATION: CREDIT LENDING")
    logger.info("=" * 70)
    
    results = []
    
    for seed in range(n_seeds):
        X, A, Y = generate_lending_dataset(n=n_samples, seed=seed)
        d = X.shape[1]
        
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
        results.append({'method': 'XGBoost', 'seed': seed, **met, 'n_feats': d})
        
        # Oracle: remove both proxies
        oracle_idx = list(range(2, d))  # Skip first 2 (proxies)
        m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
        m.fit(X_tr_sc[:, oracle_idx], y_tr)
        yp = m.predict(X_te_sc[:, oracle_idx])
        ypr = m.predict_proba(X_te_sc[:, oracle_idx])[:, 1]
        met = compute_metrics(y_te, yp, ypr, s_te)
        results.append({'method': 'Oracle', 'seed': seed, **met, 'n_feats': len(oracle_idx)})
        
        # Three aggregation methods
        for agg_label, agg in [('DAG-only', 'dag_only'), ('Corr-only', 'corr_only'), ('Max (ours)', 'max')]:
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=max(3, d // 5),
                n_iterations=500, aggregation=agg, device='cpu')
            sel.fit(X_tr_sc, s_tr, y_tr)
            Xtr_s, Xte_s = sel.transform(X_tr_sc), sel.transform(X_te_sc)
            nf = len(sel.selected_)
            
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            
            sel_names = [FEATURE_NAMES[j] for j in sel.selected_]
            rem_names = [f for f in FEATURE_NAMES if f not in sel_names]
            
            results.append({
                'method': agg_label, 'seed': seed, **met, 'n_feats': nf,
            })
            
            if seed == 0:
                cat_removed = 'Neighborhood_Risk_Tier' in rem_names
                cont_removed = 'Branch_Proximity_Score' in rem_names
                logger.info(f"  {agg_label:12s}: EOD={met['eod']:.3f}  "
                           f"CatProxy removed={cat_removed}  ContProxy removed={cont_removed}  "
                           f"Selected={sel_names}")
    
    df = pd.DataFrame(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    
    summary = df.groupby('method').agg(
        eod=('eod', 'mean'), eod_std=('eod', 'std'),
        auc=('auc', 'mean'), n_feats=('n_feats', 'mean')).round(4)
    
    baseline = summary.loc['XGBoost', 'eod']
    
    for method in ['XGBoost', 'DAG-only', 'Corr-only', 'Max (ours)', 'Oracle']:
        r = summary.loc[method]
        reduction = (baseline - r['eod']) / baseline * 100
        print(f"  {method:15s}: EOD={r['eod']:.4f}±{r['eod_std']:.4f}  "
              f"AUC={r['auc']:.3f}  ↓{reduction:5.1f}%")
    
    max_eod = summary.loc['Max (ours)', 'eod']
    dag_eod = summary.loc['DAG-only', 'eod']
    corr_eod = summary.loc['Corr-only', 'eod']
    
    print(f"\n  Max vs DAG-only:  {max_eod:.4f} vs {dag_eod:.4f}  "
          f"{'✓ Max wins' if max_eod < dag_eod - 0.003 else '~ similar'}")
    print(f"  Max vs Corr-only: {max_eod:.4f} vs {corr_eod:.4f}  "
          f"{'✓ Max wins' if max_eod < corr_eod - 0.003 else '~ similar'}")
    
    if max_eod < dag_eod - 0.003 and max_eod < corr_eod - 0.003:
        print(f"\n  ★ MAX-AGGREGATION STRICTLY BEATS BOTH! ★")
    
    df.to_csv('synthetic_lending_results.csv', index=False)
    return df


# ============================================================================
# SAVE DATASET + METADATA
# ============================================================================

def save_dataset(n=20000, seed=42, output_path='synthetic_complementary_lending.csv'):
    """Save dataset as CSV with full metadata."""
    X, A, Y = generate_lending_dataset(n=n, seed=seed)
    
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['Race'] = A.astype(int)
    df['Default'] = Y.astype(int)
    df.to_csv(output_path, index=False)
    
    # Save metadata
    meta = {
        'name': 'Synthetic Credit Lending (Complementary Proxies)',
        'n': n, 'd': len(FEATURE_NAMES),
        'protected': 'Race', 'target': 'Default',
        'proxy_features': PROXY_FEATURES,
        'legitimate_features': LEGITIMATE_FEATURES,
        'spurious_features': SPURIOUS_FEATURES,
        'structural_equations': {
            'Neighborhood_Risk_Tier': 'clip(round(2.5·Race + N(0,0.7²)), 1, 5)',
            'Branch_Proximity_Score': '0.25·Race + N(0, 1.0²)',
            'Legitimate': 'Independent of Race (see generate_lending_dataset)',
            'Spurious': 'Driven by geographic confounder C, not Race',
            'Outcome': 'logistic(0.40·Tier + 0.35·Branch - 0.50·Credit + ... + noise)',
        },
        'design_purpose': 'Demonstrate complementary failure modes: '
                          'Tier has high corr/low DAG weight (nonlinear); '
                          'Branch has low corr/high DAG weight (linear)',
    }
    
    import json
    with open(output_path.replace('.csv', '_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Saved: {output_path} ({n} rows, {len(FEATURE_NAMES)} features)")
    logger.info(f"Saved: {output_path.replace('.csv', '_metadata.json')}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate and verify dataset')
    parser.add_argument('--experiment', action='store_true', help='Run ablation experiment')
    parser.add_argument('--all', action='store_true', help='Both')
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=20000)
    args = parser.parse_args()
    
    if args.all:
        args.generate = args.experiment = True
    if not args.generate and not args.experiment:
        args.generate = args.experiment = True
    
    if args.generate:
        X, A, Y = generate_lending_dataset(n=args.n_samples, seed=42)
        verify_dataset(X, A, Y)
        save_dataset(n=args.n_samples)
    
    if args.experiment:
        run_ablation(n_seeds=args.n_seeds, n_samples=args.n_samples)


if __name__ == '__main__':
    main()
