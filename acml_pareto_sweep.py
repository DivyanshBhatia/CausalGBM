"""
Pareto Frontier Sweep — All Three Methods
==========================================
Sweeps hyperparameters on SAME train/test splits:
  - CausalGBM: α ∈ {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0}
  - FairGBM:   multiplier_learning_rate ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}
  - M²FGB-TPR: fair_weight ∈ {0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}

Also includes single-point baselines: XGBoost, LightGBM, Corr-0.3, MutualInfo,
FairLearn-EO, CounterfactualFair, TabTransformer, FT-Transformer

Usage:
  python acml_pareto_sweep.py --n_seeds 5
  python acml_pareto_sweep.py --n_seeds 10 --device cuda
"""

import os, sys, argparse, warnings, logging, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import (
    CausalFeatureSelector, compute_metrics,
    load_adult, load_acs_income,
)
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

try:
    from fairgbm import FairGBMClassifier
    HAS_FAIRGBM = True
except:
    HAS_FAIRGBM = False
    logger.warning("FairGBM not installed")

try:
    from m2fgb.m2fgb import M2FGBClassifier
    HAS_M2FGB = True
except:
    HAS_M2FGB = False
    logger.warning("M2FGB not installed")


# Sweep ranges
ALPHA_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
FAIRGBM_LR = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
M2FGB_FW = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

DATASETS = {
    'adult': load_adult,
    'acs_income': load_acs_income,
}


def run_sweep(n_seeds=5, device='cpu', output_dir='results/acml2026/pareto_sweep'):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for ds_name, loader in DATASETS.items():
        try:
            dataset = loader()
        except Exception as e:
            logger.warning(f"Could not load {ds_name}: {e}")
            continue

        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        logger.info(f"\n{'='*60}")
        logger.info(f"{ds_name} (n={len(X)}, d={d})")
        logger.info(f"{'='*60}")

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)

            # ==========================================================
            # SINGLE-POINT BASELINES (same split)
            # ==========================================================

            # XGBoost
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(X_tr, y_tr)
            yp, ypr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({'dataset': ds_name, 'method': 'XGBoost', 'param': 0,
                           'seed': seed, 'auc': met['auc'], 'eod': met['eod']})

            # LightGBM
            if HAS_LGB:
                m = lgb.LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1)
                m.fit(X_tr, y_tr)
                yp, ypr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
                met = compute_metrics(y_te, yp, ypr, s_te)
                results.append({'dataset': ds_name, 'method': 'LightGBM', 'param': 0,
                               'seed': seed, 'auc': met['auc'], 'eod': met['eod']})

            # Corr-0.3 (remove features with |corr(X,A)| > 0.3)
            corrs = np.array([abs(np.corrcoef(X_tr[:, j], s_tr)[0, 1]) for j in range(d)])
            keep = np.where(corrs <= 0.3)[0]
            if len(keep) < 3:
                keep = np.argsort(corrs)[:3]
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(X_tr[:, keep], y_tr)
            yp, ypr = m.predict(X_te[:, keep]), m.predict_proba(X_te[:, keep])[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({'dataset': ds_name, 'method': 'Corr-0.3', 'param': 0.3,
                           'seed': seed, 'auc': met['auc'], 'eod': met['eod']})

            # MutualInfo (remove top-k by MI with A)
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(X_tr, s_tr, random_state=seed)
            mi_keep = np.argsort(mi)[:max(3, d // 2)]
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(X_tr[:, mi_keep], y_tr)
            yp, ypr = m.predict(X_te[:, mi_keep]), m.predict_proba(X_te[:, mi_keep])[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({'dataset': ds_name, 'method': 'MutualInfo', 'param': 0,
                           'seed': seed, 'auc': met['auc'], 'eod': met['eod']})

            # ==========================================================
            # CAUSALGBM α SWEEP
            # ==========================================================
            logger.info(f"  Seed {seed}: CausalGBM α sweep...")
            for alpha in ALPHA_VALUES:
                try:
                    if alpha == 0.0:
                        # α=0 means no fairness penalty → all features kept
                        m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                        m.fit(X_tr, y_tr)
                        yp, ypr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
                        met = compute_metrics(y_te, yp, ypr, s_te)
                        n_sel = d
                    else:
                        sel = CausalFeatureSelector(
                            d, alpha=alpha, threshold=0.2,
                            min_features=max(3, d // 3),
                            n_iterations=500, aggregation='max', device=device)
                        sel.fit(X_tr, s_tr, y_tr)
                        Xtr_s, Xte_s = sel.transform(X_tr), sel.transform(X_te)
                        n_sel = len(sel.selected_)
                        m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                        m.fit(Xtr_s, y_tr)
                        yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
                        met = compute_metrics(y_te, yp, ypr, s_te)

                    results.append({
                        'dataset': ds_name, 'method': 'CausalGBM',
                        'param': alpha, 'seed': seed,
                        'auc': met['auc'], 'eod': met['eod'],
                        'n_features': n_sel})
                except Exception as e:
                    logger.warning(f"    α={alpha} failed: {e}")

            # ==========================================================
            # FAIRGBM multiplier_learning_rate SWEEP
            # ==========================================================
            if HAS_FAIRGBM:
                logger.info(f"  Seed {seed}: FairGBM sweep...")
                for lr in FAIRGBM_LR:
                    try:
                        fgbm = FairGBMClassifier(
                            constraint_type="FNR,FPR", n_estimators=100,
                            random_state=seed,
                            multiplier_learning_rate=lr, verbose=-1)
                        fgbm.fit(X_tr, y_tr, constraint_group=s_tr)
                        yp = fgbm.predict(X_te)
                        ypr = fgbm.predict_proba(X_te)[:, 1]
                        met = compute_metrics(y_te, yp, ypr, s_te)
                        results.append({
                            'dataset': ds_name, 'method': 'FairGBM',
                            'param': lr, 'seed': seed,
                            'auc': met['auc'], 'eod': met['eod']})
                    except Exception as e:
                        logger.warning(f"    lr={lr} failed: {e}")

            # ==========================================================
            # M²FGB fair_weight SWEEP
            # ==========================================================
            if HAS_M2FGB:
                logger.info(f"  Seed {seed}: M²FGB sweep...")
                for fw in M2FGB_FW:
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
                            'param': fw, 'seed': seed,
                            'auc': met['auc'], 'eod': met['eod']})
                    except Exception as e:
                        logger.warning(f"    fw={fw} failed: {e}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'pareto_sweep_raw.csv'), index=False)

    # Print summary per dataset
    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue

        print(f"\n{'='*70}")
        print(f"  {ds_name}")
        print(f"{'='*70}")

        for method in ['XGBoost', 'LightGBM', 'Corr-0.3', 'MutualInfo',
                       'CausalGBM', 'FairGBM', 'M2FGB-TPR']:
            m_df = ds_df[ds_df['method'] == method]
            if m_df.empty:
                continue

            print(f"\n  {method}:")
            summary = m_df.groupby('param').agg(
                auc=('auc', 'mean'), eod=('eod', 'mean')).round(4)
            for param, row in summary.iterrows():
                n_feat = ''
                if 'n_features' in m_df.columns:
                    nf = m_df[m_df['param'] == param]['n_features'].mean()
                    if not np.isnan(nf):
                        n_feat = f'  n_feat={nf:.0f}'
                print(f"    param={param:6.3f}: AUC={row['auc']:.4f}  EOD={row['eod']:.4f}{n_feat}")

    print(f"\nSaved: {os.path.join(output_dir, 'pareto_sweep_raw.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='results/acml2026/pareto_sweep')
    args = parser.parse_args()
    run_sweep(n_seeds=args.n_seeds, device=args.device, output_dir=args.output_dir)
