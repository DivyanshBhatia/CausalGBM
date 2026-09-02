"""
Orientation Ablation — Table 2 Protocol
=========================================
Matches Table 2 exactly: full training set, no validation holdout,
min_features = max(3, d//3), same seeds.

Directed column = Table 2 (free consistency check).
Orientation-free column = same W, symmetrised scoring.

Usage: python acml_orientation_ablation.py --n_seeds 10
"""
import os, sys, argparse, warnings, logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import (
    CausalFeatureSelector, compute_metrics,
    load_adult, load_acs_income, load_compas,
    load_bank, load_taiwan_credit, load_online_shoppers,
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


def orientation_free_selection(adj, correlations, d, alpha=0.5, tau=0.2, min_features=3):
    """
    Apply orientation-free scoring to the SAME learned adjacency matrix.
    proxy_j = max(|W[A,j]|, |W[j,A]|)   (undirected skeleton)
    pred_j  = max(|W[j,Y]|, |W[Y,j]|)   (undirected skeleton)
    Both use max-aggregation with correlation for proxy signal.
    """
    A_idx, Y_idx = d, d + 1

    sym_proxy = np.maximum(np.abs(adj[A_idx, :d]), np.abs(adj[:d, A_idx]))
    sym_pred = np.maximum(np.abs(adj[:d, Y_idx]), np.abs(adj[Y_idx, :d]))

    # Max-aggregation with correlation
    sym_proxy_max = np.maximum(sym_proxy, correlations)

    scores = sym_pred - alpha * sym_proxy_max
    selected = set(np.where(scores >= tau)[0])
    if len(selected) < min_features:
        selected = set(np.argsort(scores)[-min_features:])

    return sorted(selected)


def run(n_seeds=10, output_dir='results/acml2026/orientation'):
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
        min_feat = max(3, d // 3)  # Table 2's rule
        logger.info(f"\n{'='*60}")
        logger.info(f"{ds_name} (n={len(X)}, d={d}, min_feat={min_feat})")
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

            # Fit DAGMA once on FULL training set (no validation holdout)
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=min_feat,
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr_sc, s_tr, y_tr)

            # ---- DIRECTED (should reproduce Table 2) ----
            dir_selected = sorted(sel.selected_)
            m_dir = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m_dir.fit(X_tr_sc[:, dir_selected], y_tr)
            met_dir = compute_metrics(
                y_te, m_dir.predict(X_te_sc[:, dir_selected]),
                m_dir.predict_proba(X_te_sc[:, dir_selected])[:, 1], s_te)
            results.append({'dataset': ds_name, 'method': 'Directed',
                           'seed': seed, 'n_feats': len(dir_selected), **met_dir})

            # ---- ORIENTATION-FREE (same W, symmetrised scoring) ----
            adj = sel.learned_adjacency_
            if adj is not None:
                sym_selected = orientation_free_selection(
                    adj, sel.correlations_, d,
                    alpha=0.5, tau=0.2, min_features=min_feat)

                m_sym = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m_sym.fit(X_tr_sc[:, sym_selected], y_tr)
                met_sym = compute_metrics(
                    y_te, m_sym.predict(X_te_sc[:, sym_selected]),
                    m_sym.predict_proba(X_te_sc[:, sym_selected])[:, 1], s_te)

                jac = len(set(dir_selected) & set(sym_selected)) / len(set(dir_selected) | set(sym_selected))

                results.append({'dataset': ds_name, 'method': 'Orientation-free',
                               'seed': seed, 'n_feats': len(sym_selected),
                               'jaccard': jac, **met_sym})

            if seed == 0:
                logger.info(f"  Directed:         EOD={met_dir['eod']:.4f}  AUC={met_dir['auc']:.3f}  K={len(dir_selected)}  feats={dir_selected}")
                if adj is not None:
                    logger.info(f"  Orientation-free: EOD={met_sym['eod']:.4f}  AUC={met_sym['auc']:.3f}  K={len(sym_selected)}  J={jac:.2f}  feats={sym_selected}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'orientation_ablation.csv'), index=False)

    print("\n" + "=" * 80)
    print("ORIENTATION ABLATION (Table 2 protocol)")
    print("Directed column should match Table 2 exactly")
    print("=" * 80)
    print(f"\n{'Dataset':<18s} {'Dir EOD':>8s} {'Free EOD':>9s} {'|D|':>6s} {'Dir AUC':>8s} {'Free AUC':>9s} {'J':>5s}")
    print("-" * 68)

    for ds in DATASETS:
        d_df = df[(df['dataset'] == ds) & (df['method'] == 'Directed')]
        s_df = df[(df['dataset'] == ds) & (df['method'] == 'Orientation-free')]
        if d_df.empty:
            continue
        d_eod, d_auc = d_df['eod'].mean(), d_df['auc'].mean()
        if s_df.empty:
            print(f"{ds:<18s} {d_eod:>8.4f}    N/A")
            continue
        s_eod, s_auc = s_df['eod'].mean(), s_df['auc'].mean()
        delta = abs(d_eod - s_eod)
        jac = s_df['jaccard'].mean()
        print(f"{ds:<18s} {d_eod:>8.4f} {s_eod:>9.4f} {delta:>6.3f} {d_auc:>8.3f} {s_auc:>9.3f} {jac:>5.2f}")

    print(f"\nSaved: {os.path.join(output_dir, 'orientation_ablation.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/orientation')
    args = parser.parse_args()
    run(n_seeds=args.n_seeds, output_dir=args.output_dir)
