"""
Indirect Proxy Detection v3: Beyond DAG Edge Propagation
==========================================================
The propagation approaches (total effect, taint, k-hop) all fail because
DAGMA doesn't learn intermediate edges. These approaches DON'T require
intermediate edges — they detect indirect proxies through different signals.

Approach 1: ITERATIVE DAGMA
  Remove identified direct proxies → re-learn DAG → indirect proxies
  may now appear as "direct" edges (mediator removed, A→indirect becomes visible)

Approach 2: CORRELATION CASCADE  
  After identifying direct proxies, check if remaining features correlate
  with known proxies. If corr(X_k, proxy_j) > threshold → indirect proxy.

Approach 3: DOWNSTREAM REGRESSION
  Regress each remaining feature on all direct proxies. If R² is high,
  the feature is substantially determined by proxies → indirect proxy.

Approach 4: COMBINED (Iterative + Cascade)
  Use both signals: re-learn DAG AND check correlation with known proxies.

Usage:
  python acml_indirect_proxy_v3.py --n_seeds 10
"""

import os, sys, warnings, logging, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from dagma.linear import DagmaLinear
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import compute_metrics

GROUND_TRUTH = {
    'direct_proxies': ['Zip_Code_Risk', 'School_Rating'],
    'indirect_proxies': ['Property_Value', 'Branch_Quality'],
    'legitimate': ['Annual_Income', 'Employment_Years', 'Credit_Score'],
    'spurious': ['Name_Pattern'],
}
ALL_PROXIES = GROUND_TRUTH['direct_proxies'] + GROUND_TRUTH['indirect_proxies']


# ============================================================================
# STAGE 1: Standard CausalGBM direct proxy detection
# ============================================================================

def detect_direct_proxies(X, A, y, feature_names, alpha=0.5, threshold=0.2):
    """Standard CausalGBM: learn DAG, identify direct proxies."""
    n, d = X.shape
    Z = np.column_stack([X, A.reshape(-1,1), y.reshape(-1,1)])
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)

    model = DagmaLinear(loss_type='l2')
    W = model.fit(Z, lambda1=0.1)

    idx_A, idx_Y = d, d+1
    W_A_X = np.abs(W[idx_A, :d])
    corr_A = np.array([abs(np.corrcoef(X[:,j], A)[0,1]) for j in range(d)])
    W_prime = np.maximum(W_A_X, corr_A)
    W_X_Y = np.abs(W[:d, idx_Y])
    scores = W_X_Y - alpha * W_prime

    # Identify direct proxies: features with high W_prime and low/negative score
    proxy_mask = W_prime > 0.1  # features with detectable A→X influence
    proxy_indices = np.where(proxy_mask)[0]
    proxy_names = [feature_names[j] for j in proxy_indices]

    return proxy_indices, proxy_names, W, scores, W_prime, corr_A


# ============================================================================
# APPROACH 1: ITERATIVE DAGMA
# ============================================================================

def iterative_dagma(X, A, y, feature_names, max_rounds=3, alpha=0.5, threshold=0.2):
    """
    Iterative proxy detection:
    1. Learn DAG, find direct proxies
    2. Remove them from X
    3. Re-learn DAG on remaining features
    4. Previously indirect proxies may now appear as direct
    5. Repeat until no new proxies found
    """
    n, d = X.shape
    all_proxy_indices = set()
    remaining_indices = list(range(d))
    remaining_names = list(feature_names)

    for round_num in range(max_rounds):
        if len(remaining_indices) <= 3:
            break

        X_remaining = X[:, remaining_indices]
        n_rem = len(remaining_indices)

        Z = np.column_stack([X_remaining, A.reshape(-1,1), y.reshape(-1,1)])
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)

        model = DagmaLinear(loss_type='l2')
        W = model.fit(Z, lambda1=0.1)

        idx_A = n_rem
        W_A_X = np.abs(W[idx_A, :n_rem])
        corr_A = np.array([abs(np.corrcoef(X_remaining[:,j], A)[0,1]) for j in range(n_rem)])
        W_prime = np.maximum(W_A_X, corr_A)

        # Find new proxies this round
        new_proxies = []
        for j in range(n_rem):
            if W_prime[j] > 0.1:
                orig_idx = remaining_indices[j]
                if orig_idx not in all_proxy_indices:
                    new_proxies.append(j)
                    all_proxy_indices.add(orig_idx)

        new_proxy_names = [remaining_names[j] for j in new_proxies]
        logger.info(f"  Round {round_num+1}: found {len(new_proxies)} new proxies: {new_proxy_names}")

        if len(new_proxies) == 0:
            break

        # Remove new proxies for next round
        remaining_indices = [remaining_indices[j] for j in range(n_rem) if j not in new_proxies]
        remaining_names = [feature_names[j] for j in remaining_indices]

    selected_indices = [j for j in range(d) if j not in all_proxy_indices]
    return selected_indices, all_proxy_indices


# ============================================================================
# APPROACH 2: CORRELATION CASCADE
# ============================================================================

def correlation_cascade(X, A, feature_names, direct_proxy_indices,
                        cascade_threshold=0.15):
    """
    After finding direct proxies, flag features correlated with them.
    If corr(X_k, direct_proxy_j) > threshold → indirect proxy.
    """
    d = X.shape[1]
    indirect_indices = set()

    for k in range(d):
        if k in direct_proxy_indices:
            continue
        for p_idx in direct_proxy_indices:
            corr = abs(np.corrcoef(X[:, k], X[:, p_idx])[0, 1])
            if corr > cascade_threshold:
                indirect_indices.add(k)
                break

    all_proxies = set(direct_proxy_indices) | indirect_indices
    selected = [j for j in range(d) if j not in all_proxies]

    return selected, indirect_indices


# ============================================================================
# APPROACH 3: DOWNSTREAM REGRESSION
# ============================================================================

def downstream_regression(X, A, feature_names, direct_proxy_indices,
                          r2_threshold=0.05):
    """
    Regress each remaining feature on all direct proxies.
    If R² > threshold → feature is substantially determined by proxies.
    """
    d = X.shape[1]
    proxy_features = X[:, list(direct_proxy_indices)]
    indirect_indices = set()

    for k in range(d):
        if k in direct_proxy_indices:
            continue
        target = X[:, k]
        reg = LinearRegression().fit(proxy_features, target)
        r2 = r2_score(target, reg.predict(proxy_features))
        if r2 > r2_threshold:
            indirect_indices.add(k)

    all_proxies = set(direct_proxy_indices) | indirect_indices
    selected = [j for j in range(d) if j not in all_proxies]

    return selected, indirect_indices, {
        feature_names[k]: r2_score(X[:,k], LinearRegression().fit(proxy_features, X[:,k]).predict(proxy_features))
        for k in range(d) if k not in direct_proxy_indices
    }


# ============================================================================
# APPROACH 4: COMBINED (Iterative + Cascade)
# ============================================================================

def combined_approach(X, A, y, feature_names, cascade_threshold=0.15,
                      r2_threshold=0.05, alpha=0.5):
    """Combine iterative DAGMA with correlation cascade."""
    d = X.shape[1]

    # Step 1: Get direct proxies
    direct_idx, direct_names, W, scores, W_prime, corr_A = detect_direct_proxies(
        X, A, y, feature_names, alpha=alpha
    )

    # Step 2: Correlation cascade from direct proxies
    _, cascade_indirect = correlation_cascade(
        X, A, feature_names, set(direct_idx), cascade_threshold
    )

    # Step 3: Downstream regression
    _, regression_indirect, r2_scores = downstream_regression(
        X, A, feature_names, set(direct_idx), r2_threshold
    )

    # Step 4: Union of all detected proxies
    all_proxies = set(direct_idx) | cascade_indirect | regression_indirect
    selected = [j for j in range(d) if j not in all_proxies]

    # Enforce min features
    min_features = max(3, d // 3)
    if len(selected) < min_features:
        # Add back least-proxy features
        proxy_list = list(all_proxies - set(direct_idx))
        proxy_scores_sorted = sorted(proxy_list, key=lambda j: max(
            max(abs(np.corrcoef(X[:,j], X[:,p])[0,1]) for p in direct_idx) if direct_idx.size > 0 else 0,
            abs(np.corrcoef(X[:,j], A)[0,1])
        ))
        while len(selected) < min_features and proxy_scores_sorted:
            selected.append(proxy_scores_sorted.pop(0))

    return selected, all_proxies, cascade_indirect, regression_indirect, r2_scores


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment(n_seeds=10, alpha=0.5, threshold=0.2):
    csv_path = 'synthetic_indirect_proxy_loan.csv'
    if not os.path.exists(csv_path):
        logger.error(f"Not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ['Race', 'Loan_Approved']]
    X_raw = df[feature_cols].values.astype(np.float64)
    y = df['Loan_Approved'].values.astype(np.float64)
    sens = df['Race'].values.astype(np.float64)
    d = len(feature_cols)
    min_features = max(3, d // 3)

    logger.info(f"Features: {feature_cols}")
    logger.info(f"d={d}, min_features={min_features}")

    methods = {
        'XGBoost (baseline)': None,
        'Oracle': None,
        'CausalGBM (direct only)': None,
        'Iterative DAGMA': None,
        'Correlation Cascade (τ=0.15)': None,
        'Correlation Cascade (τ=0.10)': None,
        'Downstream Regression (R²>0.05)': None,
        'Downstream Regression (R²>0.02)': None,
        'Combined (Iter+Cascade+Reg)': None,
    }

    results = []

    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X_raw, y, sens, test_size=0.3, random_state=seed, stratify=y
        )
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        def evaluate(selected_indices, method_name):
            if len(selected_indices) == 0:
                selected_indices = list(range(d))
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(X_tr_sc[:, selected_indices], y_tr)
            yp = m.predict(X_te_sc[:, selected_indices])
            ypr = m.predict_proba(X_te_sc[:, selected_indices])[:, 1]
            metrics = compute_metrics(y_te, yp, ypr, s_te)
            sel_names = [feature_cols[j] for j in selected_indices]
            rem_names = [f for f in feature_cols if f not in sel_names]
            return {
                'method': method_name, 'seed': seed,
                'auc': metrics['auc'], 'eod': metrics['eod'],
                'n_features': len(selected_indices),
                'selected': sel_names, 'removed': rem_names,
                'direct_detected': len([f for f in GROUND_TRUTH['direct_proxies'] if f in rem_names]),
                'indirect_detected': len([f for f in GROUND_TRUTH['indirect_proxies'] if f in rem_names]),
                'legit_kept': len([f for f in GROUND_TRUTH['legitimate'] if f in sel_names]),
                'spurious_removed': len([f for f in GROUND_TRUTH['spurious'] if f in rem_names]),
            }

        # Baseline
        results.append(evaluate(list(range(d)), 'XGBoost (baseline)'))

        # Oracle
        oracle_keep = [j for j,f in enumerate(feature_cols) if f not in ALL_PROXIES]
        results.append(evaluate(oracle_keep, 'Oracle'))

        # Stage 1: detect direct proxies
        direct_idx, direct_names, W, scores, W_prime, corr_A = detect_direct_proxies(
            X_tr_sc, s_tr, y_tr, feature_cols, alpha=alpha
        )

        # CausalGBM original (direct only, min_features applied)
        above = np.where(scores >= threshold)[0]
        if len(above) < min_features:
            above = np.argsort(scores)[::-1][:min_features]
        results.append(evaluate(list(above), 'CausalGBM (direct only)'))

        # Approach 1: Iterative DAGMA
        iter_selected, iter_proxies = iterative_dagma(
            X_tr_sc, s_tr, y_tr, feature_cols, max_rounds=3, alpha=alpha
        )
        if len(iter_selected) < min_features:
            iter_selected = np.argsort(scores)[::-1][:min_features].tolist()
        results.append(evaluate(iter_selected, 'Iterative DAGMA'))

        # Approach 2: Correlation Cascade
        for tau in [0.15, 0.10]:
            casc_selected, casc_indirect = correlation_cascade(
                X_tr_sc, s_tr, feature_cols, set(direct_idx), tau
            )
            if len(casc_selected) < min_features:
                casc_selected = np.argsort(scores)[::-1][:min_features].tolist()
            results.append(evaluate(casc_selected, f'Correlation Cascade (τ={tau})'))

        # Approach 3: Downstream Regression
        for r2t in [0.05, 0.02]:
            reg_selected, reg_indirect, r2_vals = downstream_regression(
                X_tr_sc, s_tr, feature_cols, set(direct_idx), r2t
            )
            if len(reg_selected) < min_features:
                reg_selected = np.argsort(scores)[::-1][:min_features].tolist()
            results.append(evaluate(reg_selected, f'Downstream Regression (R²>{r2t})'))

        # Approach 4: Combined
        comb_selected, comb_all, comb_casc, comb_reg, comb_r2 = combined_approach(
            X_tr_sc, s_tr, y_tr, feature_cols, cascade_threshold=0.15,
            r2_threshold=0.05, alpha=alpha
        )
        if len(comb_selected) < min_features:
            comb_selected = np.argsort(scores)[::-1][:min_features].tolist()
        results.append(evaluate(comb_selected, 'Combined (Iter+Cascade+Reg)'))

        # Debug for seed 0
        if seed == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"SEED 0 DEBUG")
            logger.info(f"{'='*60}")
            logger.info(f"Direct proxies found: {direct_names}")
            logger.info(f"\nCorrelation with direct proxies:")
            for k, fname in enumerate(feature_cols):
                if k not in direct_idx:
                    corrs = {feature_cols[p]: abs(np.corrcoef(X_tr_sc[:,k], X_tr_sc[:,p])[0,1])
                             for p in direct_idx}
                    marker = " ◄ INDIRECT" if fname in GROUND_TRUTH['indirect_proxies'] else \
                             " ◄ SPURIOUS" if fname in GROUND_TRUTH['spurious'] else ""
                    logger.info(f"  {fname:20s}: {corrs}{marker}")

            logger.info(f"\nR² from regressing on direct proxies:")
            proxy_X = X_tr_sc[:, list(direct_idx)]
            for k, fname in enumerate(feature_cols):
                if k not in direct_idx:
                    reg = LinearRegression().fit(proxy_X, X_tr_sc[:,k])
                    r2 = r2_score(X_tr_sc[:,k], reg.predict(proxy_X))
                    marker = " ◄ INDIRECT" if fname in GROUND_TRUTH['indirect_proxies'] else \
                             " ◄ SPURIOUS" if fname in GROUND_TRUTH['spurious'] else ""
                    logger.info(f"  {fname:20s}: R²={r2:.4f}{marker}")

    # Summary
    rdf = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    baseline_eod = rdf[rdf['method']=='XGBoost (baseline)']['eod'].mean()
    oracle_eod = rdf[rdf['method']=='Oracle']['eod'].mean()

    print(f"\nBaseline EOD: {baseline_eod:.4f}")
    print(f"Oracle EOD:   {oracle_eod:.4f}")
    print(f"Gap: {baseline_eod - oracle_eod:.4f}\n")

    summary = rdf.groupby('method').agg(
        auc=('auc', 'mean'), eod=('eod', 'mean'), eod_std=('eod', 'std'),
        n_feats=('n_features', 'mean'),
        direct=('direct_detected', 'mean'),
        indirect=('indirect_detected', 'mean'),
        legit=('legit_kept', 'mean'),
    ).round(4).sort_values('eod')

    for method, row in summary.iterrows():
        gap_closed = (baseline_eod - row['eod']) / (baseline_eod - oracle_eod) * 100 \
                     if baseline_eod != oracle_eod else 0
        print(f"  {method:40s}: EOD={row['eod']:.4f}±{row['eod_std']:.4f}  "
              f"AUC={row['auc']:.3f}  direct={row['direct']:.1f}/2  "
              f"indirect={row['indirect']:.1f}/2  legit={row['legit']:.1f}/3  "
              f"gap={gap_closed:5.1f}%")

    rdf.to_csv('indirect_proxy_v3_results.csv', index=False)
    print("\nSaved: indirect_proxy_v3_results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    run_experiment(n_seeds=args.n_seeds, alpha=args.alpha, threshold=args.threshold)
