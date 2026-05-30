"""
Dedicated Acyclicity Ablation
==============================
The clean isolation: does the acyclicity constraint h(W)=0 matter,
holding joint estimation fixed?

Condition 1: DAGMA — minimize ||Z-ZW||² + λ||W||₁ subject to h(W)=0
Condition 2: Unconstrained — minimize ||Z-ZW||² + λ||W||₁ (NO acyclicity)
             = multivariate Ridge/Lasso regression of each variable on all others

Both produce the SAME weight matrix W over [X, A, Y], from which we
extract W_{A→X_j} (proxy) and W_{X_j→Y} (predictive). Same scoring
function, same α, τ, min-features, rollback, Stage-2 XGBoost.

If they match → acyclicity is decorative (joint estimation is what matters)
If DAGMA wins → acyclicity constraint provides genuine value

Usage:
  python acml_acyclicity_ablation.py --n_seeds 10
"""

import os, sys, argparse, warnings, logging, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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


def learn_unconstrained_weights(X, A, Y, alpha_ridge=1.0):
    """
    Unconstrained joint estimation: regress each variable on all others
    WITHOUT acyclicity constraint. Same structure as DAGMA's W matrix
    but cycles are allowed.
    
    Returns W matrix of shape (d+2, d+2) over [X₁,...,X_d, A, Y]
    """
    Z = np.column_stack([X, A.reshape(-1, 1), Y.reshape(-1, 1)])
    n, p = Z.shape
    W = np.zeros((p, p))
    
    for j in range(p):
        # Regress z_j on all other variables
        others = [i for i in range(p) if i != j]
        reg = Ridge(alpha=alpha_ridge)
        reg.fit(Z[:, others], Z[:, j])
        for k, idx in enumerate(others):
            W[idx, j] = reg.coef_[k]
    
    return W


def extract_signals_from_W(W, d):
    """
    Extract proxy and predictive signals from weight matrix W.
    W is (d+2, d+2) over [X₁,...,X_d, A, Y].
    
    A is at index d, Y is at index d+1.
    proxy_j = |W[d, j]|      (A → X_j edge)
    pred_j  = |W[j, d+1]|    (X_j → Y edge)
    """
    A_idx = d      # A is column d
    Y_idx = d + 1  # Y is column d+1
    
    proxy = np.abs(W[A_idx, :d])      # |W_{A→X_j}| for each feature
    predictive = np.abs(W[:d, Y_idx]) # |W_{X_j→Y}| for each feature
    
    return predictive, proxy


def select_features(predictive, proxy, corr_A, alpha=0.5, tau=0.2, min_features=3):
    """CausalGBM scoring with max-aggregation."""
    # Max-aggregate proxy with correlation (same as CausalGBM)
    proxy_max = np.maximum(np.abs(proxy), np.abs(corr_A))
    
    scores = predictive - alpha * proxy_max
    selected = set(np.where(scores >= tau)[0])
    if len(selected) < min_features:
        selected = set(np.argsort(scores)[-min_features:])
    return selected, scores


def run_ablation(n_seeds=10, output_dir='results/acml2026/acyclicity_ablation'):
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

            corr_A = np.array([abs(np.corrcoef(X_tr_sc[:, j], s_tr)[0, 1]) for j in range(d)])

            # ============================================================
            # CONDITION 1: DAGMA (with acyclicity)
            # ============================================================
            t0 = time.time()
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=min_feat,
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr_sc, s_tr, y_tr)
            dagma_selected = set(sel.selected_)
            dagma_time = time.time() - t0

            Xtr_s = sel.transform(X_tr_sc)
            Xte_s = sel.transform(X_te_sc)
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
            met_dagma = compute_metrics(y_te, yp, ypr, s_te)

            if met_dagma['auc'] < 0.60:
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(X_tr_sc, y_tr)
                yp, ypr = m.predict(X_te_sc), m.predict_proba(X_te_sc)[:, 1]
                met_dagma = compute_metrics(y_te, yp, ypr, s_te)
                dagma_selected = set(range(d))

            results.append({
                'dataset': ds_name, 'condition': 'DAGMA (acyclic)',
                'seed': seed, **met_dagma,
                'n_selected': len(dagma_selected),
                'runtime': dagma_time,
            })

            # ============================================================
            # CONDITION 2: Unconstrained (no acyclicity)
            # ============================================================
            t0 = time.time()
            W_unc = learn_unconstrained_weights(X_tr_sc, s_tr, y_tr, alpha_ridge=1.0)
            pred_unc, proxy_unc = extract_signals_from_W(W_unc, d)
            unc_selected, unc_scores = select_features(
                pred_unc, proxy_unc, corr_A,
                alpha=0.5, tau=0.2, min_features=min_feat)
            unc_time = time.time() - t0

            Xtr_u = X_tr_sc[:, sorted(unc_selected)]
            Xte_u = X_te_sc[:, sorted(unc_selected)]
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_u, y_tr)
            yp, ypr = m.predict(Xte_u), m.predict_proba(Xte_u)[:, 1]
            met_unc = compute_metrics(y_te, yp, ypr, s_te)

            if met_unc['auc'] < 0.60:
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(X_tr_sc, y_tr)
                yp, ypr = m.predict(X_te_sc), m.predict_proba(X_te_sc)[:, 1]
                met_unc = compute_metrics(y_te, yp, ypr, s_te)
                unc_selected = set(range(d))

            jaccard = len(dagma_selected & unc_selected) / len(dagma_selected | unc_selected) if len(dagma_selected | unc_selected) > 0 else 1.0

            results.append({
                'dataset': ds_name, 'condition': 'Unconstrained (no DAG)',
                'seed': seed, **met_unc,
                'n_selected': len(unc_selected),
                'jaccard_vs_dagma': jaccard,
                'runtime': unc_time,
            })

            if seed == 0:
                logger.info(f"  DAGMA:         EOD={met_dagma['eod']:.4f}  AUC={met_dagma['auc']:.3f}  "
                           f"K={len(dagma_selected)}  t={dagma_time:.2f}s")
                logger.info(f"  Unconstrained: EOD={met_unc['eod']:.4f}  AUC={met_unc['auc']:.3f}  "
                           f"K={len(unc_selected)}  J={jaccard:.2f}  t={unc_time:.4f}s")

                # Show weight comparison for first seed
                logger.info(f"  --- Weight comparison (seed 0) ---")
                for j in range(min(d, 5)):
                    dagma_proxy_j = sel.dag_weights_[d, j] if hasattr(sel, 'dag_weights_') else '?'
                    dagma_pred_j = sel.dag_weights_[j, d+1] if hasattr(sel, 'dag_weights_') else '?'
                    logger.info(f"    Feat {j}: DAGMA proxy={dagma_proxy_j}, pred={dagma_pred_j} | "
                               f"Unc proxy={proxy_unc[j]:.4f}, pred={pred_unc[j]:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'acyclicity_ablation_raw.csv'), index=False)

    # Summary
    print("\n" + "=" * 80)
    print("ACYCLICITY ABLATION: DAGMA vs UNCONSTRAINED JOINT ESTIMATION")
    print("=" * 80)
    print(f"\n{'Dataset':<18s} {'Condition':<28s} {'EOD':>7s} {'AUC':>7s} {'K':>4s} {'J':>6s} {'Time':>8s}")
    print("-" * 80)

    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        for cond in ['DAGMA (acyclic)', 'Unconstrained (no DAG)']:
            c_df = ds_df[ds_df['condition'] == cond]
            if c_df.empty:
                continue
            eod = c_df['eod'].mean()
            auc = c_df['auc'].mean()
            k = c_df['n_selected'].mean()
            rt = c_df['runtime'].mean()
            jac = c_df['jaccard_vs_dagma'].mean() if 'jaccard_vs_dagma' in c_df.columns and c_df['jaccard_vs_dagma'].notna().any() else float('nan')
            jac_str = f"{jac:.2f}" if not np.isnan(jac) else "ref"
            marker = " ★" if 'DAGMA' in cond else ""
            print(f"{ds_name:<18s} {cond:<28s} {eod:>7.4f} {auc:>7.3f} {k:>4.0f} {jac_str:>6s} {rt:>7.3f}s{marker}")
        print()

    # Decisive comparison
    print("=" * 60)
    print("DOES ACYCLICITY MATTER?")
    print("=" * 60)
    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        dagma_eod = ds_df[ds_df['condition'] == 'DAGMA (acyclic)']['eod'].mean()
        unc_eod = ds_df[ds_df['condition'] == 'Unconstrained (no DAG)']['eod'].mean()
        jac = ds_df[ds_df['condition'] == 'Unconstrained (no DAG)']['jaccard_vs_dagma'].mean()

        if dagma_eod < unc_eod - 0.005:
            print(f"  {ds_name}: DAGMA wins ({dagma_eod:.4f} vs {unc_eod:.4f}, J={jac:.2f})")
        elif unc_eod < dagma_eod - 0.005:
            print(f"  {ds_name}: Unconstrained wins ({unc_eod:.4f} vs {dagma_eod:.4f}, J={jac:.2f})")
        else:
            print(f"  {ds_name}: Tie ({dagma_eod:.4f} vs {unc_eod:.4f}, J={jac:.2f})")

    print(f"\nSaved: {os.path.join(output_dir, 'acyclicity_ablation_raw.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/acyclicity_ablation')
    args = parser.parse_args()
    run_ablation(n_seeds=args.n_seeds, output_dir=args.output_dir)
