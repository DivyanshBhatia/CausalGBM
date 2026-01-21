#!/usr/bin/env python3
"""
=============================================================================
CausalGBM Evaluation on Synthetic Indirect Proxy Dataset
=============================================================================

Run this script to evaluate CausalGBM's ability to detect indirect proxies.
Results can be shared for inclusion in Appendix L of the paper.

Usage:
    python evaluate_indirect_proxy.py

Requirements:
    pip install numpy pandas scikit-learn scipy

Input:
    - synthetic_indirect_proxy_loan.csv (in same directory or specify path)

Output:
    - Console output with all results
    - appendix_l_evaluation_results.csv (detailed results)
    - appendix_l_summary.txt (copy-paste ready summary)
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to dataset (modify if needed)
DATASET_PATH = 'synthetic_indirect_proxy_loan.csv'

# Ground truth (from data generation)
GROUND_TRUTH = {
    'direct_proxies': ['Zip_Code_Risk', 'School_Rating'],
    'indirect_proxies': ['Property_Value', 'Branch_Quality'],
    'legitimate': ['Annual_Income', 'Employment_Years', 'Credit_Score'],
    'spurious': ['Name_Pattern']
}

# CausalGBM parameters
ALPHA = 0.5      # Fairness-accuracy tradeoff
TAU = 0.2        # Score threshold
MIN_FEATURES = 3 # Minimum features to retain

# Experiment settings
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_ESTIMATORS = 100

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_eod(y_true, y_pred, sensitive):
    """Compute Equalized Odds Difference."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive)
    
    # TPR by group
    mask_1_0 = (y_true == 1) & (sensitive == 0)
    mask_1_1 = (y_true == 1) & (sensitive == 1)
    tpr_0 = y_pred[mask_1_0].mean() if mask_1_0.sum() > 0 else 0
    tpr_1 = y_pred[mask_1_1].mean() if mask_1_1.sum() > 0 else 0
    
    # FPR by group
    mask_0_0 = (y_true == 0) & (sensitive == 0)
    mask_0_1 = (y_true == 0) & (sensitive == 1)
    fpr_0 = y_pred[mask_0_0].mean() if mask_0_0.sum() > 0 else 0
    fpr_1 = y_pred[mask_0_1].mean() if mask_0_1.sum() > 0 else 0
    
    return max(abs(tpr_1 - tpr_0), abs(fpr_1 - fpr_0))

def compute_dpd(y_pred, sensitive):
    """Compute Demographic Parity Difference."""
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive)
    rate_0 = y_pred[sensitive == 0].mean()
    rate_1 = y_pred[sensitive == 1].mean()
    return abs(rate_1 - rate_0)

def get_feature_type(feature_name):
    """Get ground truth type for a feature."""
    for ftype, features in GROUND_TRUTH.items():
        if feature_name in features:
            return ftype.replace('_', ' ').title()
    return 'Unknown'

def train_evaluate(X_train, X_test, y_train, y_test, A_test, name):
    """Train GBM and compute all metrics."""
    model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS, 
        max_depth=4, 
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'Method': name,
        'AUC': roc_auc_score(y_test, y_prob),
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'EOD': compute_eod(y_test, y_pred, A_test),
        'DPD': compute_dpd(y_pred, A_test)
    }

# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    print("=" * 80)
    print("CAUSALGBM EVALUATION ON SYNTHETIC INDIRECT PROXY DATASET")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    if not os.path.exists(DATASET_PATH):
        print(f"\nERROR: Dataset not found at '{DATASET_PATH}'")
        print("Please ensure synthetic_indirect_proxy_loan.csv is in the current directory.")
        return
    
    df = pd.read_csv(DATASET_PATH)
    print(f"\nLoaded dataset: {len(df):,} samples")
    
    # Feature columns (all except Race and Loan_Approved)
    feature_cols = ['Zip_Code_Risk', 'School_Rating', 'Property_Value', 'Branch_Quality',
                    'Annual_Income', 'Employment_Years', 'Credit_Score', 'Name_Pattern']
    
    X = df[feature_cols].values.astype(np.float32)
    A = df['Race'].values.astype(np.float32)
    y = df['Loan_Approved'].values.astype(np.float32)
    
    # Split data
    X_train, X_test, A_train, A_test, y_train, y_test = train_test_split(
        X, A, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    n_train, d = X_train.shape
    print(f"Train: {n_train:,}, Test: {len(X_test):,}, Features: {d}")
    
    # -------------------------------------------------------------------------
    # Dataset Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    approval_rate = y.mean()
    minority_approval = y[A == 0].mean()
    majority_approval = y[A == 1].mean()
    
    print(f"\nOverall Approval Rate: {approval_rate:.1%}")
    print(f"Minority (Race=0) Approval: {minority_approval:.1%}")
    print(f"Majority (Race=1) Approval: {majority_approval:.1%}")
    print(f"Disparity: {(majority_approval - minority_approval)*100:.1f} percentage points")
    
    # -------------------------------------------------------------------------
    # Stage 1: Feature Weight Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 1: FEATURE WEIGHT ANALYSIS")
    print("=" * 80)
    
    # Standardize features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_std = (X_train - X_mean) / X_std
    X_test_std = (X_test - X_mean) / X_std
    
    # Compute correlations with protected attribute
    correlations = np.array([abs(pearsonr(X_train[:, j], A_train)[0]) for j in range(d)])
    
    # Estimate W_A->X (proxy for DAG edge weight)
    W_A_X = np.zeros(d)
    for j in range(d):
        reg = LinearRegression()
        reg.fit(A_train.reshape(-1, 1), X_train_std[:, j])
        W_A_X[j] = abs(reg.coef_[0])
    
    # Estimate W_X->Y (predictive importance)
    W_X_Y = np.zeros(d)
    for j in range(d):
        reg = LinearRegression()
        reg.fit(X_train_std[:, j].reshape(-1, 1), y_train)
        W_X_Y[j] = abs(reg.coef_[0])
    
    # Max aggregation
    W_prime = np.maximum(W_A_X, correlations)
    
    # Compute scores
    scores = W_X_Y - ALPHA * W_prime
    
    print(f"\nParameters: alpha={ALPHA}, tau={TAU}, min_features={MIN_FEATURES}")
    print(f"\n{'Feature':<20} {'|ρ|':>7} {'W_A→X':>8} {'W_X→Y':>8} {'W_prime':>8} {'Score':>8} {'Type':<18}")
    print("-" * 90)
    
    feature_analysis = []
    for j, feat in enumerate(feature_cols):
        ftype = get_feature_type(feat)
        print(f"{feat:<20} {correlations[j]:>7.3f} {W_A_X[j]:>8.3f} {W_X_Y[j]:>8.3f} {W_prime[j]:>8.3f} {scores[j]:>+8.3f} {ftype:<18}")
        feature_analysis.append({
            'Feature': feat,
            'Correlation': correlations[j],
            'W_A_X': W_A_X[j],
            'W_X_Y': W_X_Y[j],
            'W_prime': W_prime[j],
            'Score': scores[j],
            'Type': ftype
        })
    
    # -------------------------------------------------------------------------
    # Stage 2: Feature Selection
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: FEATURE SELECTION")
    print("=" * 80)
    
    # Select features
    above_threshold = np.where(scores >= TAU)[0]
    if len(above_threshold) >= MIN_FEATURES:
        causalgbm_idx = list(above_threshold)
    else:
        causalgbm_idx = list(np.argsort(scores)[::-1][:MIN_FEATURES])
    
    causalgbm_features = [feature_cols[j] for j in causalgbm_idx]
    causalgbm_removed = [f for f in feature_cols if f not in causalgbm_features]
    
    print(f"\nCausalGBM SELECTED ({len(causalgbm_features)}): {causalgbm_features}")
    print(f"CausalGBM REMOVED ({len(causalgbm_removed)}):  {causalgbm_removed}")
    
    # -------------------------------------------------------------------------
    # Stage 3: Detection Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 3: DETECTION ANALYSIS")
    print("=" * 80)
    
    removed_set = set(causalgbm_removed)
    selected_set = set(causalgbm_features)
    
    # Calculate detection rates
    direct_detected = sum(1 for f in GROUND_TRUTH['direct_proxies'] if f in removed_set)
    direct_total = len(GROUND_TRUTH['direct_proxies'])
    
    indirect_detected = sum(1 for f in GROUND_TRUTH['indirect_proxies'] if f in removed_set)
    indirect_total = len(GROUND_TRUTH['indirect_proxies'])
    
    legit_retained = sum(1 for f in GROUND_TRUTH['legitimate'] if f in selected_set)
    legit_total = len(GROUND_TRUTH['legitimate'])
    
    spurious_retained = sum(1 for f in GROUND_TRUTH['spurious'] if f in selected_set)
    spurious_total = len(GROUND_TRUTH['spurious'])
    
    print(f"\n{'Metric':<35} {'Result':>15} {'Rate':>10} {'Expected':>12}")
    print("-" * 75)
    print(f"{'Direct Proxy Detection':<35} {direct_detected}/{direct_total}{'':<10} {direct_detected/direct_total*100:>9.0f}% {'100%':>12}")
    print(f"{'Indirect Proxy Detection':<35} {indirect_detected}/{indirect_total}{'':<10} {indirect_detected/indirect_total*100:>9.0f}% {'0%':>12}")
    print(f"{'Legitimate Feature Retention':<35} {legit_retained}/{legit_total}{'':<10} {legit_retained/legit_total*100:>9.0f}% {'100%':>12}")
    print(f"{'Spurious Correlate Retention':<35} {spurious_retained}/{spurious_total}{'':<10} {spurious_retained/spurious_total*100:>9.0f}% {'100%':>12}")
    
    # -------------------------------------------------------------------------
    # Stage 4: Fairness Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 4: FAIRNESS EVALUATION")
    print("=" * 80)
    
    results = []
    
    # Helper to get feature indices
    def get_idx(features):
        return [feature_cols.index(f) for f in features]
    
    # 1. Baseline (all features)
    res = train_evaluate(X_train, X_test, y_train, y_test, A_test, "Baseline (all features)")
    results.append(res)
    eod_baseline = res['EOD']
    
    # 2. Direct Proxy Removal Only
    direct_keep = [f for f in feature_cols if f not in GROUND_TRUTH['direct_proxies']]
    idx = get_idx(direct_keep)
    res = train_evaluate(X_train[:, idx], X_test[:, idx], y_train, y_test, A_test, "Direct Proxy Removal")
    results.append(res)
    eod_direct = res['EOD']
    
    # 3. CausalGBM
    res = train_evaluate(X_train[:, causalgbm_idx], X_test[:, causalgbm_idx], y_train, y_test, A_test, "CausalGBM (ours)")
    results.append(res)
    eod_causalgbm = res['EOD']
    
    # 4. Oracle (remove all proxies)
    oracle_keep = GROUND_TRUTH['legitimate'] + GROUND_TRUTH['spurious']
    idx = get_idx(oracle_keep)
    res = train_evaluate(X_train[:, idx], X_test[:, idx], y_train, y_test, A_test, "Oracle (all proxies removed)")
    results.append(res)
    eod_oracle = res['EOD']
    
    # 5. Perfect (legitimate only)
    idx = get_idx(GROUND_TRUTH['legitimate'])
    res = train_evaluate(X_train[:, idx], X_test[:, idx], y_train, y_test, A_test, "Perfect (legitimate only)")
    results.append(res)
    
    # Print results
    print(f"\n{'Method':<30} {'AUC':>7} {'Acc':>7} {'F1':>7} {'EOD':>7} {'DPD':>7} {'ΔEOD':>8}")
    print("-" * 80)
    for r in results:
        delta = f"{(eod_baseline - r['EOD'])/eod_baseline*100:+.0f}%" if r['Method'] != "Baseline (all features)" else "---"
        print(f"{r['Method']:<30} {r['AUC']:>7.3f} {r['Accuracy']:>7.3f} {r['F1']:>7.3f} {r['EOD']:>7.3f} {r['DPD']:>7.3f} {delta:>8}")
    
    # -------------------------------------------------------------------------
    # Stage 5: Indirect Gap Calculation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 5: INDIRECT GAP ANALYSIS (Key Metric)")
    print("=" * 80)
    
    # Calculate indirect gap
    if eod_baseline != eod_oracle:
        indirect_gap = (eod_causalgbm - eod_oracle) / (eod_baseline - eod_oracle)
    else:
        indirect_gap = 0
    
    # Achievement rate
    possible_improvement = eod_baseline - eod_oracle
    achieved_improvement = eod_baseline - eod_causalgbm
    achievement_rate = achieved_improvement / possible_improvement if possible_improvement > 0 else 0
    
    print(f"""
INDIRECT GAP CALCULATION:
=========================
EOD_Baseline   = {eod_baseline:.3f}
EOD_CausalGBM  = {eod_causalgbm:.3f}
EOD_Oracle     = {eod_oracle:.3f}

Formula: (EOD_CausalGBM - EOD_Oracle) / (EOD_Baseline - EOD_Oracle)
       = ({eod_causalgbm:.3f} - {eod_oracle:.3f}) / ({eod_baseline:.3f} - {eod_oracle:.3f})
       = {eod_causalgbm - eod_oracle:.3f} / {eod_baseline - eod_oracle:.3f}
       = {indirect_gap:.1%}

INTERPRETATION:
- CausalGBM achieves {achievement_rate:.1%} of maximum possible EOD reduction
- {indirect_gap:.1%} of remaining unfairness is due to undetected indirect proxies
""")
    
    # Interpretation
    if indirect_gap < 0.30:
        interpretation = "FAVORABLE (Gap < 30%): Indirect proxy limitation doesn't matter in practice"
        recommendation = "Direct proxy removal is sufficient for most applications."
    elif indirect_gap < 0.60:
        interpretation = "MODERATE (Gap 30-60%): Indirect proxies contribute meaningfully to unfairness"
        recommendation = "Consider supplementing with domain knowledge for multi-hop proxy structures."
    else:
        interpretation = "SIGNIFICANT (Gap > 60%): Indirect proxies dominate unfairness"
        recommendation = "Path-specific fairness methods may be needed for this type of data."
    
    print(f"VERDICT: {interpretation}")
    print(f"RECOMMENDATION: {recommendation}")
    
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df['Delta_EOD_pct'] = [(eod_baseline - r['EOD'])/eod_baseline*100 for r in results]
    results_df.to_csv('appendix_l_evaluation_results.csv', index=False)
    print(f"\nSaved: appendix_l_evaluation_results.csv")
    
    # Save feature analysis
    feature_df = pd.DataFrame(feature_analysis)
    feature_df['Selected'] = [f in selected_set for f in feature_cols]
    feature_df.to_csv('appendix_l_feature_analysis.csv', index=False)
    print(f"Saved: appendix_l_feature_analysis.csv")
    
    # Save summary for paper
    summary = f"""
================================================================================
APPENDIX L: INDIRECT PROXY LIMITATION ANALYSIS - SUMMARY
================================================================================

DATASET: Synthetic Loan Approval with Indirect Proxies
- Samples: {len(df):,}
- Features: {d}
- Approval Rate: {approval_rate:.1%}
- Disparity: {(majority_approval - minority_approval)*100:.1f} percentage points

GROUND TRUTH:
- Direct Proxies: {GROUND_TRUTH['direct_proxies']}
- Indirect Proxies: {GROUND_TRUTH['indirect_proxies']}
- Legitimate: {GROUND_TRUTH['legitimate']}
- Spurious: {GROUND_TRUTH['spurious']}

CAUSALGBM DETECTION RESULTS:
- Direct Proxy Detection: {direct_detected}/{direct_total} ({direct_detected/direct_total*100:.0f}%)
- Indirect Proxy Detection: {indirect_detected}/{indirect_total} ({indirect_detected/indirect_total*100:.0f}%)
- Legitimate Retention: {legit_retained}/{legit_total} ({legit_retained/legit_total*100:.0f}%)
- Spurious Retention: {spurious_retained}/{spurious_total} ({spurious_retained/spurious_total*100:.0f}%)

FAIRNESS RESULTS (Table L3):
| Method                      | AUC   | EOD   | ΔEOD  |
|-----------------------------|-------|-------|-------|
| Baseline (all features)     | {results[0]['AUC']:.3f} | {results[0]['EOD']:.3f} |  ---  |
| Direct Proxy Removal        | {results[1]['AUC']:.3f} | {results[1]['EOD']:.3f} | {(eod_baseline-eod_direct)/eod_baseline*100:+.0f}%  |
| CausalGBM (ours)            | {results[2]['AUC']:.3f} | {results[2]['EOD']:.3f} | {(eod_baseline-eod_causalgbm)/eod_baseline*100:+.0f}%  |
| Oracle (all proxies)        | {results[3]['AUC']:.3f} | {results[3]['EOD']:.3f} | {(eod_baseline-eod_oracle)/eod_baseline*100:+.0f}%  |

INDIRECT GAP: {indirect_gap:.1%}

VERDICT: {interpretation}

RECOMMENDATION: {recommendation}

KEY FINDING FOR PAPER:
On synthetic data with low-correlation indirect proxies (|ρ|<0.1), CausalGBM 
achieves {(eod_baseline-eod_causalgbm)/eod_baseline*100:.0f}% EOD reduction vs Oracle's {(eod_baseline-eod_oracle)/eod_baseline*100:.0f}%. The {indirect_gap:.0%} Indirect Gap confirms 
that indirect proxies represent a {'significant' if indirect_gap > 0.6 else 'moderate' if indirect_gap > 0.3 else 'minor'} limitation when they have weak 
correlation with the protected attribute.
================================================================================
"""
    
    with open('appendix_l_summary.txt', 'w') as f:
        f.write(summary)
    print(f"Saved: appendix_l_summary.txt")
    
    print(summary)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nPlease share the following files:")
    print("  1. appendix_l_evaluation_results.csv")
    print("  2. appendix_l_feature_analysis.csv") 
    print("  3. appendix_l_summary.txt")

if __name__ == "__main__":
    main()
