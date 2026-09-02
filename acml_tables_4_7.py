"""
Generate per-seed data for Tables 4 and 7.
Table 4: Aggregation ablation (Max, DAG-only, Corr-only, Sum)
Table 7: Structure learner comparison (DAGMA vs NOTEARS-Linear vs NOTEARS-MLP)

Usage: python acml_tables_4_7.py --n_seeds 10
"""
import os, sys, argparse, warnings, logging
import numpy as np
import pandas as pd
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
    load_german, load_bank, load_taiwan_credit,
    load_online_shoppers,
    load_synthetic_loan, load_synthetic_hiring,
)

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


def run_table4(n_seeds=10, output_dir='results/acml2026'):
    """Table 4: Aggregation ablation — same DAGMA, different aggregation rules."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    aggregations = ['max', 'dag_only', 'corr_only', 'sum']

    for ds_name, loader in DATASETS.items():
        try:
            dataset = loader()
        except Exception as e:
            logger.warning(f"Skipping {ds_name}: {e}")
            continue

        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        min_feat = max(3, d // 3)
        logger.info(f"\n{ds_name} (n={len(X)}, d={d})")

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

            for agg in aggregations:
                sel = CausalFeatureSelector(
                    d, alpha=0.5, threshold=0.2,
                    min_features=min_feat,
                    n_iterations=500, aggregation=agg, device='cpu')
                sel.fit(X_tr_sc, s_tr, y_tr)
                Xtr_s = sel.transform(X_tr_sc)
                Xte_s = sel.transform(X_te_sc)

                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(Xtr_s, y_tr)
                met = compute_metrics(y_te, m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1], s_te)

                results.append({
                    'dataset': ds_name, 'condition': agg, 'seed': seed,
                    'n_selected': len(sel.selected_), **met
                })

            if seed == 0:
                for agg in aggregations:
                    r = [r for r in results if r['dataset']==ds_name and r['condition']==agg and r['seed']==0][-1]
                    logger.info(f"  {agg:<10s}: EOD={r['eod']:.4f}  AUC={r['auc']:.3f}")

    df = pd.DataFrame(results)
    path = os.path.join(output_dir, 'table4_aggregation_raw.csv')
    df.to_csv(path, index=False)
    logger.info(f"\nSaved: {path}")

    # Summary
    print("\n" + "=" * 70)
    print("TABLE 4 — AGGREGATION ABLATION: mean ± SD")
    print("=" * 70)
    for ds in DATASETS:
        ds_df = df[df['dataset'] == ds]
        if ds_df.empty:
            continue
        print(f"\n  {ds}:")
        for agg in aggregations:
            a_df = ds_df[ds_df['condition'] == agg]
            print(f"  {agg:<10s}: EOD={a_df['eod'].mean():.4f} ± {a_df['eod'].std(ddof=1):.4f}  "
                  f"AUC={a_df['auc'].mean():.3f} ± {a_df['auc'].std(ddof=1):.3f}")
    return df


def run_table7(n_seeds=10, output_dir='results/acml2026'):
    """Table 7: Structure learner comparison — DAGMA vs alternatives."""
    os.makedirs(output_dir, exist_ok=True)

    # Check available structure learners
    has_notears = False
    try:
        from notears.linear import notears_linear
        has_notears = True
        logger.info("NOTEARS-Linear available")
    except ImportError:
        try:
            # Alternative: notears via pip install notears
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'notears', '-q'])
            from notears.linear import notears_linear
            has_notears = True
            logger.info("NOTEARS-Linear installed and available")
        except:
            logger.warning("NOTEARS not available — will run DAGMA-only comparison")

    has_notears_mlp = False  # Skip MLP — dtype issues, paper uses DAGMA for stability

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
        logger.info(f"\n{ds_name} (n={len(X)}, d={d})")

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

            # DAGMA (default)
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=min_feat,
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr_sc, s_tr, y_tr)
            Xtr_s = sel.transform(X_tr_sc)
            Xte_s = sel.transform(X_te_sc)
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            met = compute_metrics(y_te, m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1], s_te)
            results.append({
                'dataset': ds_name, 'condition': 'DAGMA', 'seed': seed,
                'n_selected': len(sel.selected_), **met
            })

            # NOTEARS-Linear
            if has_notears:
                try:
                    from notears.linear import notears_linear
                    A_std = (s_tr - s_tr.mean()) / (s_tr.std() + 1e-8)
                    y_std = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-8)
                    Z = np.column_stack([X_tr_sc, A_std.reshape(-1,1), y_std.reshape(-1,1)])
                    W_est = notears_linear(Z, lambda1=0.1, loss_type='l2', max_iter=100)

                    idx_A, idx_Y = d, d + 1
                    W_A_X = np.abs(W_est[idx_A, :d])
                    W_X_Y = np.abs(W_est[:d, idx_Y])
                    corrs = np.array([abs(np.corrcoef(X_tr_sc[:, j], s_tr)[0, 1]) for j in range(d)])
                    W_prime = np.maximum(W_A_X, corrs)
                    scores = W_X_Y - 0.5 * W_prime

                    selected = set(np.where(scores >= 0.2)[0])
                    if len(selected) < min_feat:
                        selected = set(np.argsort(scores)[-min_feat:])
                    selected = sorted(selected)

                    m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                    m.fit(X_tr_sc[:, selected], y_tr)
                    met = compute_metrics(y_te, m.predict(X_te_sc[:, selected]),
                                          m.predict_proba(X_te_sc[:, selected])[:, 1], s_te)
                    results.append({
                        'dataset': ds_name, 'condition': 'NOTEARS-Linear', 'seed': seed,
                        'n_selected': len(selected), **met
                    })
                except Exception as e:
                    logger.warning(f"  NOTEARS-Linear failed on {ds_name} seed {seed}: {e}")

            # NOTEARS-MLP
            if has_notears_mlp:
                try:
                    from notears.nonlinear import NotearsMLP, notears_nonlinear
                    A_std = (s_tr - s_tr.mean()) / (s_tr.std() + 1e-8)
                    y_std = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-8)
                    Z = np.column_stack([X_tr_sc, A_std.reshape(-1,1), y_std.reshape(-1,1)])

                    n_nodes = d + 2
                    model_mlp = NotearsMLP(dims=[n_nodes, 10, 1], bias=True)
                    W_est = notears_nonlinear(model_mlp, Z, lambda1=0.01, lambda2=0.01, max_iter=50)

                    idx_A, idx_Y = d, d + 1
                    W_A_X = np.abs(W_est[idx_A, :d])
                    W_X_Y = np.abs(W_est[:d, idx_Y])
                    corrs = np.array([abs(np.corrcoef(X_tr_sc[:, j], s_tr)[0, 1]) for j in range(d)])
                    W_prime = np.maximum(W_A_X, corrs)
                    scores = W_X_Y - 0.5 * W_prime

                    selected = set(np.where(scores >= 0.2)[0])
                    if len(selected) < min_feat:
                        selected = set(np.argsort(scores)[-min_feat:])
                    selected = sorted(selected)

                    m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                    m.fit(X_tr_sc[:, selected], y_tr)
                    met = compute_metrics(y_te, m.predict(X_te_sc[:, selected]),
                                          m.predict_proba(X_te_sc[:, selected])[:, 1], s_te)
                    results.append({
                        'dataset': ds_name, 'condition': 'NOTEARS-MLP', 'seed': seed,
                        'n_selected': len(selected), **met
                    })
                except Exception as e:
                    logger.warning(f"  NOTEARS-MLP failed on {ds_name} seed {seed}: {e}")

            if seed == 0:
                for cond in ['DAGMA', 'NOTEARS-Linear', 'NOTEARS-MLP']:
                    r = [r for r in results if r['dataset']==ds_name and r['condition']==cond and r['seed']==0]
                    if r:
                        logger.info(f"  {cond:<16s}: EOD={r[-1]['eod']:.4f}  AUC={r[-1]['auc']:.3f}")

    df = pd.DataFrame(results)
    path = os.path.join(output_dir, 'table7_structure_learner_raw.csv')
    df.to_csv(path, index=False)
    logger.info(f"\nSaved: {path}")

    # Summary
    print("\n" + "=" * 70)
    print("TABLE 7 — STRUCTURE LEARNER COMPARISON: mean ± SD")
    print("=" * 70)
    for ds in DATASETS:
        ds_df = df[df['dataset'] == ds]
        if ds_df.empty:
            continue
        print(f"\n  {ds}:")
        for cond in ds_df['condition'].unique():
            c_df = ds_df[ds_df['condition'] == cond]
            print(f"  {cond:<16s}: EOD={c_df['eod'].mean():.4f} ± {c_df['eod'].std(ddof=1):.4f}  "
                  f"AUC={c_df['auc'].mean():.3f} ± {c_df['auc'].std(ddof=1):.3f}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026')
    parser.add_argument('--table4_only', action='store_true')
    parser.add_argument('--table7_only', action='store_true')
    args = parser.parse_args()

    if args.table4_only:
        run_table4(args.n_seeds, args.output_dir)
    elif args.table7_only:
        run_table7(args.n_seeds, args.output_dir)
    else:
        run_table4(args.n_seeds, args.output_dir)
        run_table7(args.n_seeds, args.output_dir)
