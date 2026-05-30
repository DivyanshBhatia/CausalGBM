"""
Full DAG-Free Ablation
======================
Answers: "Does the O(d³) DAG step earn its keep over cheap statistics?"

Holds EVERYTHING fixed (α, τ, min-features, rollback, Stage-2 XGBoost, 
same 10 seeds, same standardisation) and swaps ONLY the signal vectors.

Conditions:
  1. CausalGBM (reference): proxy=max(|W_AX|,|corr|), predictive=W_XY
  2. No-DAG (corr/corr):    proxy=|corr(X,A)|,         predictive=|corr(X,Y)|
  3. No-DAG (corr/XGB):     proxy=|corr(X,A)|,         predictive=XGB feature importance

All signals are rank-normalised to [0,1] before scoring so τ=0.2 is 
comparable across conditions (reviewer trap avoided).

Reports per condition: EOD, AUC, Jaccard overlap vs CausalGBM, runtime.

Usage:
  python acml_dagfree_ablation.py --n_seeds 10
"""

import os, sys, argparse, warnings, logging, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    """Rank-normalise array to [0,1] so threshold τ is comparable across scales."""
    from scipy.stats import rankdata
    ranks = rankdata(arr, method='average')
    return (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-10)


def select_features(predictive_scores, proxy_scores, alpha=0.5, tau=0.2, 
                     min_features=3, d=None):
    """
    Apply CausalGBM's scoring function with given signal vectors.
    Both inputs should be rank-normalised to [0,1].
    Returns set of selected feature indices.
    """
    scores = predictive_scores - alpha * proxy_scores
    selected = set(np.where(scores >= tau)[0])
    if len(selected) < min_features:
        selected = set(np.argsort(scores)[-min_features:])
    return selected, scores


def run_ablation(n_seeds=10, output_dir='results/acml2026/dagfree_ablation'):
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

            # ============================================================
            # CONDITION 1: CausalGBM (reference)
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

            # Rollback check
            if met['auc'] < 0.60:
                m_full = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m_full.fit(X_tr_sc, y_tr)
                yp, ypr = m_full.predict(X_te_sc), m_full.predict_proba(X_te_sc)[:, 1]
                met = compute_metrics(y_te, yp, ypr, s_te)
                cgbm_selected = set(range(d))

            results.append({
                'dataset': ds_name, 'condition': 'CausalGBM',
                'seed': seed, **met,
                'n_selected': len(cgbm_selected),
                'selected': str(sorted(cgbm_selected)),
                'runtime': cgbm_time,
            })

            # ============================================================
            # CHEAP SIGNALS (shared across No-DAG conditions)
            # ============================================================
            t0 = time.time()
            
            # Proxy: |corr(X, A)|
            corr_proxy = np.array([
                abs(np.corrcoef(X_tr_sc[:, j], s_tr)[0, 1]) 
                for j in range(d)])
            
            # Predictive option 1: |corr(X, Y)|
            corr_pred = np.array([
                abs(np.corrcoef(X_tr_sc[:, j], y_tr)[0, 1]) 
                for j in range(d)])
            
            # Predictive option 2: XGBoost feature importance (gain)
            m_full = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m_full.fit(X_tr_sc, y_tr)
            xgb_gain = m_full.feature_importances_  # gain-based
            
            cheap_time = time.time() - t0

            # Rank-normalise ALL signals to [0,1]
            corr_proxy_norm = rank_normalize(corr_proxy)
            corr_pred_norm = rank_normalize(corr_pred)
            xgb_gain_norm = rank_normalize(xgb_gain)

            # ============================================================
            # CONDITION 2: No-DAG (corr/corr)
            # ============================================================
            selected_cc, scores_cc = select_features(
                corr_pred_norm, corr_proxy_norm, 
                alpha=0.5, tau=0.2, min_features=min_feat, d=d)
            
            Xtr_cc = X_tr_sc[:, sorted(selected_cc)]
            Xte_cc = X_te_sc[:, sorted(selected_cc)]
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_cc, y_tr)
            yp, ypr = m.predict(Xte_cc), m.predict_proba(Xte_cc)[:, 1]
            met_cc = compute_metrics(y_te, yp, ypr, s_te)

            # Rollback
            if met_cc['auc'] < 0.60:
                m_full2 = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m_full2.fit(X_tr_sc, y_tr)
                yp, ypr = m_full2.predict(X_te_sc), m_full2.predict_proba(X_te_sc)[:, 1]
                met_cc = compute_metrics(y_te, yp, ypr, s_te)
                selected_cc = set(range(d))

            jaccard_cc = len(cgbm_selected & selected_cc) / len(cgbm_selected | selected_cc) if len(cgbm_selected | selected_cc) > 0 else 1.0

            results.append({
                'dataset': ds_name, 'condition': 'No-DAG (corr/corr)',
                'seed': seed, **met_cc,
                'n_selected': len(selected_cc),
                'selected': str(sorted(selected_cc)),
                'jaccard_vs_cgbm': jaccard_cc,
                'runtime': cheap_time,
            })

            # ============================================================
            # CONDITION 3: No-DAG (corr/XGB-gain)
            # ============================================================
            selected_cx, scores_cx = select_features(
                xgb_gain_norm, corr_proxy_norm,
                alpha=0.5, tau=0.2, min_features=min_feat, d=d)
            
            Xtr_cx = X_tr_sc[:, sorted(selected_cx)]
            Xte_cx = X_te_sc[:, sorted(selected_cx)]
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_cx, y_tr)
            yp, ypr = m.predict(Xte_cx), m.predict_proba(Xte_cx)[:, 1]
            met_cx = compute_metrics(y_te, yp, ypr, s_te)

            # Rollback
            if met_cx['auc'] < 0.60:
                m_full3 = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m_full3.fit(X_tr_sc, y_tr)
                yp, ypr = m_full3.predict(X_te_sc), m_full3.predict_proba(X_te_sc)[:, 1]
                met_cx = compute_metrics(y_te, yp, ypr, s_te)
                selected_cx = set(range(d))

            jaccard_cx = len(cgbm_selected & selected_cx) / len(cgbm_selected | selected_cx) if len(cgbm_selected | selected_cx) > 0 else 1.0

            results.append({
                'dataset': ds_name, 'condition': 'No-DAG (corr/XGB)',
                'seed': seed, **met_cx,
                'n_selected': len(selected_cx),
                'selected': str(sorted(selected_cx)),
                'jaccard_vs_cgbm': jaccard_cx,
                'runtime': cheap_time,
            })

            if seed == 0:
                logger.info(f"  CausalGBM:       EOD={results[-3]['eod']:.4f}  AUC={results[-3]['auc']:.3f}  "
                           f"K={results[-3]['n_selected']}  t={cgbm_time:.1f}s")
                logger.info(f"  No-DAG (corr/corr): EOD={met_cc['eod']:.4f}  AUC={met_cc['auc']:.3f}  "
                           f"K={len(selected_cc)}  J={jaccard_cc:.2f}  t={cheap_time:.1f}s")
                logger.info(f"  No-DAG (corr/XGB):  EOD={met_cx['eod']:.4f}  AUC={met_cx['auc']:.3f}  "
                           f"K={len(selected_cx)}  J={jaccard_cx:.2f}  t={cheap_time:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'dagfree_ablation_raw.csv'), index=False)

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 90)
    print("DAG-FREE ABLATION RESULTS")
    print("=" * 90)
    print(f"\n{'Dataset':<18s} {'Condition':<22s} {'EOD':>7s} {'AUC':>7s} {'K':>4s} {'Jaccard':>8s} {'Time':>7s}")
    print("-" * 90)

    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        for cond in ['CausalGBM', 'No-DAG (corr/corr)', 'No-DAG (corr/XGB)']:
            c_df = ds_df[ds_df['condition'] == cond]
            if c_df.empty:
                continue
            eod = c_df['eod'].mean()
            auc = c_df['auc'].mean()
            k = c_df['n_selected'].mean()
            rt = c_df['runtime'].mean()
            jac = c_df['jaccard_vs_cgbm'].mean() if 'jaccard_vs_cgbm' in c_df.columns and c_df['jaccard_vs_cgbm'].notna().any() else float('nan')
            jac_str = f"{jac:.2f}" if not np.isnan(jac) else "ref"
            marker = " ★" if cond == 'CausalGBM' else ""
            print(f"{ds_name:<18s} {cond:<22s} {eod:>7.4f} {auc:>7.3f} {k:>4.0f} {jac_str:>8s} {rt:>6.1f}s{marker}")
        print()

    # Highlight where they differ
    print("\n" + "=" * 60)
    print("WHERE DOES THE DAG MATTER?")
    print("=" * 60)
    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        cgbm_eod = ds_df[ds_df['condition']=='CausalGBM']['eod'].mean()
        cc_eod = ds_df[ds_df['condition']=='No-DAG (corr/corr)']['eod'].mean()
        cx_eod = ds_df[ds_df['condition']=='No-DAG (corr/XGB)']['eod'].mean()
        best_nodag = min(cc_eod, cx_eod)
        
        if cgbm_eod < best_nodag - 0.005:
            print(f"  {ds_name}: CausalGBM wins (EOD {cgbm_eod:.4f} vs best No-DAG {best_nodag:.4f})")
        elif best_nodag < cgbm_eod - 0.005:
            print(f"  {ds_name}: No-DAG wins (EOD {best_nodag:.4f} vs CausalGBM {cgbm_eod:.4f})")
        else:
            jac = ds_df[ds_df['condition']=='No-DAG (corr/corr)']['jaccard_vs_cgbm'].mean()
            print(f"  {ds_name}: Tie (EOD diff < 0.005, Jaccard={jac:.2f})")

    print(f"\nSaved: {os.path.join(output_dir, 'dagfree_ablation_raw.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/dagfree_ablation')
    args = parser.parse_args()
    run_ablation(n_seeds=args.n_seeds, output_dir=args.output_dir)
