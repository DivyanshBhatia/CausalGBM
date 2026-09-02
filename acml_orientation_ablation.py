"""
Orientation Ablation
=====================
Proves edge directions don't matter by symmetrising the DAG:
  proxy = max(|W_{A→X}|, |W_{X→A}|)
  pred  = max(|W_{X→Y}|, |W_{Y→X}|)

If EOD/AUC and selected features are unchanged → causal framing
concession is cost-free and W1 is closed empirically.

No DAG refit needed — just a scoring change on existing W.

Usage: python acml_orientation_ablation.py --n_seeds 10
"""
import os, sys, argparse, warnings, logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

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

DATASETS = {
    'acs_income': load_acs_income,
    'adult': load_adult,
    'compas': load_compas,
    'bank': load_bank,
    'taiwan': load_taiwan_credit,
    'online_shoppers': load_online_shoppers,
    'synthetic_loan': load_synthetic_loan,
    'synthetic_hiring': load_synthetic_hiring,
}


def select_with_symmetrised_scores(sel, d, alpha=0.5, tau=0.2, min_features=3):
    """
    Re-score using symmetrised (orientation-destroyed) weights.
    W is (d+2, d+2) over [X_1..X_d, A, Y]. A=index d, Y=index d+1.
    
    Original:  proxy_j = |W[d, j]|,     pred_j = |W[j, d+1]|
    Symmetric: proxy_j = max(|W[d,j]|, |W[j,d]|)
               pred_j  = max(|W[j,d+1]|, |W[d+1,j]|)
    """
    if not hasattr(sel, 'W_'):
        return None, None
    
    W = sel.W_
    A_idx, Y_idx = d, d + 1
    
    # Original scores
    orig_proxy = np.abs(W[A_idx, :d])
    orig_pred = np.abs(W[:d, Y_idx])
    
    # Symmetrised scores
    sym_proxy = np.maximum(np.abs(W[A_idx, :d]), np.abs(W[:d, A_idx]))
    sym_pred = np.maximum(np.abs(W[:d, Y_idx]), np.abs(W[Y_idx, :d]))
    
    # Also get correlation for max-aggregation
    corr_A = np.abs(sel.corr_proxy_) if hasattr(sel, 'corr_proxy_') else orig_proxy
    
    # Apply max-aggregation with correlation (same as CausalGBM)
    sym_proxy_max = np.maximum(sym_proxy, corr_A)
    
    # Score and select
    scores = sym_pred - alpha * sym_proxy_max
    selected = set(np.where(scores >= tau)[0])
    if len(selected) < min_features:
        selected = set(np.argsort(scores)[-min_features:])
    
    return sorted(selected), scores


def run_ablation(n_seeds=10, output_dir='results/acml2026/orientation'):
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

            # Fit DAGMA once
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=min_feat,
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr_sc, s_tr, y_tr)
            orig_selected = set(sel.selected_)

            # --- Original CausalGBM ---
            Xtr_s = sel.transform(X_tr_sc)
            Xte_s = sel.transform(X_te_sc)
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            met_orig = compute_metrics(y_te, m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1], s_te)
            results.append({'dataset': ds_name, 'method': 'CausalGBM (directed)',
                           'seed': seed, 'n_feats': len(orig_selected), **met_orig})

            # --- Symmetrised (orientation destroyed) ---
            sym_selected, sym_scores = select_with_symmetrised_scores(
                sel, d, alpha=0.5, tau=0.2, min_features=min_feat)

            if sym_selected is not None:
                Xtr_sym = X_tr_sc[:, sym_selected]
                Xte_sym = X_te_sc[:, sym_selected]
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(Xtr_sym, y_tr)
                met_sym = compute_metrics(y_te, m.predict(Xte_sym), m.predict_proba(Xte_sym)[:, 1], s_te)

                jac = len(orig_selected & set(sym_selected)) / len(orig_selected | set(sym_selected)) if len(orig_selected | set(sym_selected)) > 0 else 1.0

                results.append({'dataset': ds_name, 'method': 'CausalGBM (symmetric)',
                               'seed': seed, 'n_feats': len(sym_selected),
                               'jaccard': jac, **met_sym})
            else:
                logger.warning(f"  {ds_name} seed {seed}: W_ not exposed, skipping symmetric")

            if seed == 0:
                logger.info(f"  Directed:  EOD={met_orig['eod']:.4f}  AUC={met_orig['auc']:.3f}  K={len(orig_selected)}")
                if sym_selected is not None:
                    logger.info(f"  Symmetric: EOD={met_sym['eod']:.4f}  AUC={met_sym['auc']:.3f}  K={len(sym_selected)}  J={jac:.2f}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'orientation_ablation.csv'), index=False)

    print("\n" + "=" * 70)
    print("ORIENTATION ABLATION: Directed vs Symmetrised DAG")
    print("=" * 70)
    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        d_eod = ds_df[ds_df['method'] == 'CausalGBM (directed)']['eod'].mean()
        s_df = ds_df[ds_df['method'] == 'CausalGBM (symmetric)']
        if s_df.empty:
            print(f"  {ds_name}: symmetric not available (W_ not exposed)")
            continue
        s_eod = s_df['eod'].mean()
        jac = s_df['jaccard'].mean()
        match = "IDENTICAL" if abs(d_eod - s_eod) < 0.001 and jac > 0.95 else (
            "SIMILAR" if abs(d_eod - s_eod) < 0.005 else "DIFFERENT")
        print(f"  {ds_name}: directed={d_eod:.4f}  symmetric={s_eod:.4f}  J={jac:.2f}  → {match}")

    print(f"\nSaved: {os.path.join(output_dir, 'orientation_ablation.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/orientation')
    args = parser.parse_args()
    run_ablation(n_seeds=args.n_seeds, output_dir=args.output_dir)
