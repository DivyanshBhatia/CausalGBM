"""
Complementarity: CausalGBM + M²FGB
====================================
Run M²FGB on CausalGBM's selected features, sweep γ.
If combined frontier dominates M²FGB-alone → "use both" is validated.

Usage: python acml_complementarity.py --n_seeds 10
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
    load_adult, load_acs_income,
)

try:
    from m2fgb.m2fgb import M2FGBClassifier
    HAS_M2FGB = True
except ImportError:
    HAS_M2FGB = False
    logger.warning("M2FGB not installed")

try:
    from fairgbm import FairGBMClassifier
    HAS_FAIRGBM = True
except ImportError:
    HAS_FAIRGBM = False

GAMMA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
LAMBDA_GRID = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]

DATASETS = {
    'acs_income': load_acs_income,
    'adult': load_adult,
}


def run_complementarity(n_seeds=10, output_dir='results/acml2026/complementarity'):
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

            # CausalGBM feature selection
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=min_feat,
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr_sc, s_tr, y_tr)
            cgbm_cols = sorted(sel.selected_)
            Xtr_cgbm = X_tr_sc[:, cgbm_cols]
            Xte_cgbm = X_te_sc[:, cgbm_cols]

            # --- M²FGB alone (full features, sweep γ) ---
            if HAS_M2FGB:
                for gam in GAMMA_GRID:
                    try:
                        if gam == 0.0:
                            mm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                            mm.fit(X_tr_sc, y_tr)
                        else:
                            mm = M2FGBClassifier(
                                fairness_constraint='true_positive_rate',
                                fair_weight=gam, n_estimators=100,
                                learning_rate=0.1, multiplier_learning_rate=0.1,
                                random_state=seed)
                            mm.fit(X_tr_sc, y_tr, sensitive_attribute=s_tr)
                        yp = mm.predict(X_te_sc)
                        ypr = mm.predict_proba(X_te_sc)[:, 1]
                        met = compute_metrics(y_te, yp, ypr, s_te)
                        results.append({'dataset': ds_name, 'method': 'M2FGB-alone',
                                       'gamma': gam, 'seed': seed, **met})
                    except:
                        pass

                # --- CausalGBM + M²FGB (CausalGBM features, sweep γ) ---
                for gam in GAMMA_GRID:
                    try:
                        if gam == 0.0:
                            mm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                            mm.fit(Xtr_cgbm, y_tr)
                        else:
                            mm = M2FGBClassifier(
                                fairness_constraint='true_positive_rate',
                                fair_weight=gam, n_estimators=100,
                                learning_rate=0.1, multiplier_learning_rate=0.1,
                                random_state=seed)
                            mm.fit(Xtr_cgbm, y_tr, sensitive_attribute=s_tr)
                        yp = mm.predict(Xte_cgbm)
                        ypr = mm.predict_proba(Xte_cgbm)[:, 1]
                        met = compute_metrics(y_te, yp, ypr, s_te)
                        results.append({'dataset': ds_name, 'method': 'CausalGBM+M2FGB',
                                       'gamma': gam, 'seed': seed, **met})
                    except:
                        pass

            # --- FairGBM alone + CausalGBM+FairGBM ---
            if HAS_FAIRGBM:
                for lam in LAMBDA_GRID:
                    try:
                        if lam == 0.0:
                            fm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                            fm.fit(X_tr_sc, y_tr)
                        else:
                            fm = FairGBMClassifier(
                                constraint_type="FNR,FPR", n_estimators=100,
                                random_state=seed, multiplier_learning_rate=lam, verbose=-1)
                            fm.fit(X_tr_sc, y_tr, constraint_group=s_tr)
                        yp = fm.predict(X_te_sc)
                        ypr = fm.predict_proba(X_te_sc)[:, 1]
                        met = compute_metrics(y_te, yp, ypr, s_te)
                        results.append({'dataset': ds_name, 'method': 'FairGBM-alone',
                                       'gamma': lam, 'seed': seed, **met})
                    except:
                        pass

                for lam in LAMBDA_GRID:
                    try:
                        if lam == 0.0:
                            fm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                            fm.fit(Xtr_cgbm, y_tr)
                        else:
                            fm = FairGBMClassifier(
                                constraint_type="FNR,FPR", n_estimators=100,
                                random_state=seed, multiplier_learning_rate=lam, verbose=-1)
                            fm.fit(Xtr_cgbm, y_tr, constraint_group=s_tr)
                        yp = fm.predict(Xte_cgbm)
                        ypr = fm.predict_proba(Xte_cgbm)[:, 1]
                        met = compute_metrics(y_te, yp, ypr, s_te)
                        results.append({'dataset': ds_name, 'method': 'CausalGBM+FairGBM',
                                       'gamma': lam, 'seed': seed, **met})
                    except:
                        pass

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'complementarity.csv'), index=False)

    # Summary: compare best Pareto points
    print("\n" + "=" * 70)
    print("COMPLEMENTARITY: Does CausalGBM + in-processing beat in-processing alone?")
    print("=" * 70)

    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        print(f"\n  {ds_name}:")
        print(f"  {'Method':<25s} {'Best EOD':>9s} {'@ AUC':>7s} {'γ/λ':>5s}")
        print("  " + "-" * 50)

        for method in ['M2FGB-alone', 'CausalGBM+M2FGB', 'FairGBM-alone', 'CausalGBM+FairGBM']:
            m_df = ds_df[ds_df['method'] == method]
            if m_df.empty:
                continue
            # Find best EOD across γ (mean over seeds)
            best_eod = float('inf')
            best_auc = 0
            best_g = 0
            for g in m_df['gamma'].unique():
                g_df = m_df[m_df['gamma'] == g]
                eod = g_df['eod'].mean()
                auc = g_df['auc'].mean()
                if eod < best_eod:
                    best_eod = eod
                    best_auc = auc
                    best_g = g
            marker = " ★" if 'CausalGBM+' in method else ""
            print(f"  {method:<25s} {best_eod:>9.4f} {best_auc:>7.3f} {best_g:>5.2f}{marker}")

    print(f"\nSaved: {os.path.join(output_dir, 'complementarity.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/complementarity')
    args = parser.parse_args()
    run_complementarity(n_seeds=args.n_seeds, output_dir=args.output_dir)
