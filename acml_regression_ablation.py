"""
DAGMA Joint SEM vs Independent Regressions
============================================
The decisive test: does DAGMA's O(d³) acyclicity-constrained joint SEM
outperform two cheap independent regressions?

Conditions (all use same scoring function c_j = p_j - α·q_j):
  1. CausalGBM (reference): p=W_{X→Y} from DAGMA, q=max(|W_{A→X}|,|corr|)
  2. OLS-based:   p=|β_j| from OLS(Y~X), q=max(|β_j| from OLS(X_j~A), |corr|)
  3. Lasso-based:  p=|β_j| from Lasso(Y~X), q=max(|β_j| from Lasso(X_j~A), |corr|)
  4. Ridge-based:  p=|β_j| from Ridge(Y~X), q=max(|β_j| from Ridge(X_j~A), |corr|)

All signals rank-normalised. Same α, τ, min-features, rollback, seeds, splits.

If OLS/Lasso match CausalGBM → DAG is decorative (joint SEM ≈ independent regressions)
If CausalGBM wins → acyclicity constraint provides value beyond cheap coefficients

Usage:
  python acml_regression_ablation.py --n_seeds 10
"""

import os, sys, argparse, warnings, logging, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from scipy.stats import rankdata

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import (
    CausalFeatureSelector, compute_metrics,
    load_adult, load_acs_income, load_compas,
    load_german, load_bank, load_taiwan_credit,
    load_online_shoppers,
    load_synthetic_loan, load_synthetic_hiring,
)
import xgboost as xgb

DATASETS = {
    'acs_income': load_acs_income,
    'adult': load_adult,
    'compas': load_compas,
    'german': load_german,
    'bank': load_bank,
    'taiwan': load_taiwan_credit,
    'online_shoppers': load_online_shoppers,
    'synthetic_loan': load_synthetic_loan,
    'synthetic_hiring': load_synthetic_hiring,
}


def rank_normalize(arr):
    ranks = rankdata(arr, method='average')
    return (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-10)


def get_regression_signals(X, A, Y, reg_type='ols'):
    """
    Compute proxy and predictive signals from independent regressions.
    
    Predictive: fit reg(Y ~ X), take |coefficients|
    Proxy: for each j, fit reg(X_j ~ A), take |coefficient|
    
    Returns (predictive_weights, proxy_weights) as arrays of shape (d,)
    """
    d = X.shape[1]
    
    # Predictive: Y ~ X (multivariate)
    if reg_type == 'ols':
        reg = LinearRegression()
    elif reg_type == 'lasso':
        reg = Lasso(alpha=0.01, max_iter=5000)
    elif reg_type == 'ridge':
        reg = Ridge(alpha=1.0)
    
    reg.fit(X, Y)
    predictive = np.abs(reg.coef_).flatten()
    
    # Proxy: X_j ~ A (univariate, per feature)
    proxy = np.zeros(d)
    for j in range(d):
        if reg_type == 'ols':
            r = LinearRegression()
        elif reg_type == 'lasso':
            r = Lasso(alpha=0.01, max_iter=5000)
        elif reg_type == 'ridge':
            r = Ridge(alpha=1.0)
        
        r.fit(A.reshape(-1, 1), X[:, j])
        proxy[j] = np.abs(r.coef_[0]) if hasattr(r.coef_, '__len__') else np.abs(r.coef_)
    
    return predictive, proxy


def select_features_from_signals(predictive, proxy, corr_with_A, 
                                  alpha=0.5, tau=0.2, min_features=3):
    """Apply CausalGBM's scoring with given signals (all rank-normalised)."""
    # Max-aggregate proxy with correlation (same as CausalGBM)
    proxy_max = np.maximum(rank_normalize(proxy), rank_normalize(np.abs(corr_with_A)))
    pred_norm = rank_normalize(predictive)
    
    scores = pred_norm - alpha * proxy_max
    selected = set(np.where(scores >= tau)[0])
    if len(selected) < min_features:
        selected = set(np.argsort(scores)[-min_features:])
    return selected, scores


def run_ablation(n_seeds=10, output_dir='results/acml2026/regression_ablation'):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for ds_name, loader in DATASETS.items():
        try:
            dataset = loader()
        except Exception as e:
            logger.warning(f"Skipping {ds_name}: {e}")
            continue

        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        min_feat = max(3, d // 3)
        logger.info(f"\n{'='*60}")
        logger.info(f"{ds_name} (n={len(X)}, d={d})")
        logger.info(f"{'='*60}")

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            s_tr = np.asarray(s_tr, dtype=float)
            y_tr = np.asarray(y_tr, dtype=float)
            s_te = np.asarray(s_te, dtype=float)
            y_te = np.asarray(y_te, dtype=float)

            # Correlation with A (shared across conditions)
            corr_A = np.array([abs(np.corrcoef(X_tr_sc[:, j], s_tr)[0, 1]) for j in range(d)])

            # ============================================================
            # CONDITION 1: CausalGBM (DAGMA joint SEM)
            # ============================================================
            t0 = time.time()
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=min_feat,
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr_sc, s_tr, y_tr)
            cgbm_selected = set(sel.selected_)
            cgbm_time = time.time() - t0

            Xtr_s = sel.transform(X_tr_sc)
            Xte_s = sel.transform(X_te_sc)
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)

            if met['auc'] < 0.60:
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(X_tr_sc, y_tr)
                yp, ypr = m.predict(X_te_sc), m.predict_proba(X_te_sc)[:, 1]
                met = compute_metrics(y_te, yp, ypr, s_te)
                cgbm_selected = set(range(d))

            results.append({
                'dataset': ds_name, 'condition': 'CausalGBM (DAGMA)',
                'seed': seed, **met,
                'n_selected': len(cgbm_selected),
                'runtime': cgbm_time,
            })

            # ============================================================
            # CONDITIONS 2-4: Independent regressions (OLS, Lasso, Ridge)
            # ============================================================
            for reg_name, reg_type in [('OLS', 'ols'), ('Lasso', 'lasso'), ('Ridge', 'ridge')]:
                t0 = time.time()
                pred_w, proxy_w = get_regression_signals(X_tr_sc, s_tr, y_tr, reg_type)
                reg_selected, _ = select_features_from_signals(
                    pred_w, proxy_w, corr_A,
                    alpha=0.5, tau=0.2, min_features=min_feat)
                reg_time = time.time() - t0

                Xtr_r = X_tr_sc[:, sorted(reg_selected)]
                Xte_r = X_te_sc[:, sorted(reg_selected)]
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(Xtr_r, y_tr)
                yp, ypr = m.predict(Xte_r), m.predict_proba(Xte_r)[:, 1]
                met_r = compute_metrics(y_te, yp, ypr, s_te)

                if met_r['auc'] < 0.60:
                    m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                    m.fit(X_tr_sc, y_tr)
                    yp, ypr = m.predict(X_te_sc), m.predict_proba(X_te_sc)[:, 1]
                    met_r = compute_metrics(y_te, yp, ypr, s_te)
                    reg_selected = set(range(d))

                jaccard = len(cgbm_selected & reg_selected) / len(cgbm_selected | reg_selected) if len(cgbm_selected | reg_selected) > 0 else 1.0

                results.append({
                    'dataset': ds_name, 'condition': f'{reg_name}-based',
                    'seed': seed, **met_r,
                    'n_selected': len(reg_selected),
                    'jaccard_vs_dagma': jaccard,
                    'runtime': reg_time,
                })

            if seed == 0:
                logger.info(f"  DAGMA:  EOD={results[-4]['eod']:.4f}  AUC={results[-4]['auc']:.3f}  "
                           f"K={results[-4]['n_selected']}  t={results[-4]['runtime']:.2f}s")
                for i, name in enumerate(['OLS', 'Lasso', 'Ridge']):
                    r = results[-(3-i)]
                    logger.info(f"  {name:6s}: EOD={r['eod']:.4f}  AUC={r['auc']:.3f}  "
                               f"K={r['n_selected']}  J={r.get('jaccard_vs_dagma', 0):.2f}  "
                               f"t={r['runtime']:.4f}s")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'regression_ablation_raw.csv'), index=False)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 90)
    print("DAGMA JOINT SEM vs INDEPENDENT REGRESSIONS")
    print("=" * 90)
    print(f"\n{'Dataset':<18s} {'Condition':<22s} {'EOD':>7s} {'AUC':>7s} {'K':>4s} {'Jaccard':>8s} {'Time':>8s}")
    print("-" * 90)

    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        for cond in ['CausalGBM (DAGMA)', 'OLS-based', 'Lasso-based', 'Ridge-based']:
            c_df = ds_df[ds_df['condition'] == cond]
            if c_df.empty:
                continue
            eod = c_df['eod'].mean()
            auc = c_df['auc'].mean()
            k = c_df['n_selected'].mean()
            rt = c_df['runtime'].mean()
            jac = c_df['jaccard_vs_dagma'].mean() if 'jaccard_vs_dagma' in c_df.columns and c_df['jaccard_vs_dagma'].notna().any() else float('nan')
            jac_str = f"{jac:.2f}" if not np.isnan(jac) else "ref"
            marker = " ★" if cond == 'CausalGBM (DAGMA)' else ""
            print(f"{ds_name:<18s} {cond:<22s} {eod:>7.4f} {auc:>7.3f} {k:>4.0f} {jac_str:>8s} {rt:>7.3f}s{marker}")
        print()

    # Decisive comparison
    print("\n" + "=" * 60)
    print("DOES DAGMA BEAT CHEAP REGRESSIONS?")
    print("=" * 60)
    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        dagma_eod = ds_df[ds_df['condition'] == 'CausalGBM (DAGMA)']['eod'].mean()
        best_reg_name = 'OLS'
        best_reg_eod = 999
        for reg in ['OLS-based', 'Lasso-based', 'Ridge-based']:
            reg_eod = ds_df[ds_df['condition'] == reg]['eod'].mean()
            if reg_eod < best_reg_eod:
                best_reg_eod = reg_eod
                best_reg_name = reg

        if dagma_eod < best_reg_eod - 0.005:
            print(f"  {ds_name}: DAGMA wins ({dagma_eod:.4f} vs {best_reg_name} {best_reg_eod:.4f})")
        elif best_reg_eod < dagma_eod - 0.005:
            print(f"  {ds_name}: {best_reg_name} wins ({best_reg_eod:.4f} vs DAGMA {dagma_eod:.4f})")
        else:
            jac = ds_df[ds_df['condition'] == best_reg_name]['jaccard_vs_dagma'].mean()
            print(f"  {ds_name}: Tie ({dagma_eod:.4f} vs {best_reg_eod:.4f}, J={jac:.2f})")

    # Timing comparison
    print("\n" + "=" * 60)
    print("RUNTIME: DAGMA vs REGRESSION")
    print("=" * 60)
    dagma_times = df[df['condition'] == 'CausalGBM (DAGMA)'].groupby('dataset')['runtime'].mean()
    ols_times = df[df['condition'] == 'OLS-based'].groupby('dataset')['runtime'].mean()
    for ds in dagma_times.index:
        if ds in ols_times.index:
            speedup = dagma_times[ds] / ols_times[ds] if ols_times[ds] > 0 else float('inf')
            print(f"  {ds:<18s}: DAGMA={dagma_times[ds]:.2f}s  OLS={ols_times[ds]:.4f}s  ({speedup:.0f}× slower)")

    print(f"\nSaved: {os.path.join(output_dir, 'regression_ablation_raw.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/regression_ablation')
    args = parser.parse_args()
    run_ablation(n_seeds=args.n_seeds, output_dir=args.output_dir)
