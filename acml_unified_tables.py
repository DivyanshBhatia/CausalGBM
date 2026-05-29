"""
ACML 2026 — Unified Table Generation
======================================
Generates Tables 2, 3, and 4 from ONE set of experiment runs.
Ensures zero inconsistency: same (dataset, seed) → same DAGMA run →
same feature selection → same CausalGBM numbers everywhere.

Output: unified_results.csv + LaTeX-ready tables for copy-paste

Usage:
  python acml_unified_tables.py --n_seeds 10 --device cuda
"""

import os, sys, time, argparse, warnings, logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import ttest_rel

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import (
    CausalFeatureSelector, compute_metrics,
    load_adult, load_compas, load_german, load_acs_income,
    load_taiwan_credit, load_bank, load_online_shoppers,
    load_synthetic_loan, load_synthetic_hiring,
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

try:
    from m2fgb.m2fgb import M2FGBClassifier
    HAS_M2FGB = True
except:
    HAS_M2FGB = False

DATASET_LOADERS = {
    'adult': load_adult,
    'acs_income': load_acs_income,
    'compas': load_compas,
    'german': load_german,
    'taiwan_credit': load_taiwan_credit,
    'bank': load_bank,
    'online_shoppers': load_online_shoppers,
    'synthetic_loan': load_synthetic_loan,
    'synthetic_hiring': load_synthetic_hiring,
}

ABLATION_DATASETS = ['adult', 'acs_income', 'online_shoppers', 'taiwan_credit', 'bank',
                     'compas', 'german', 'synthetic_loan', 'synthetic_hiring']


def compute_extended_metrics(y_true, y_pred, y_prob, sensitive):
    base = compute_metrics(y_true, y_pred, y_prob, sensitive)
    groups = np.unique(sensitive)
    if len(groups) != 2:
        base['eq_opp'] = np.nan
        base['avg_odds'] = np.nan
        return base
    g0, g1 = groups
    m0, m1 = sensitive == g0, sensitive == g1
    tpr0 = y_pred[(m0) & (y_true == 1)].mean() if ((m0) & (y_true == 1)).sum() > 0 else 0
    tpr1 = y_pred[(m1) & (y_true == 1)].mean() if ((m1) & (y_true == 1)).sum() > 0 else 0
    fpr0 = y_pred[(m0) & (y_true == 0)].mean() if ((m0) & (y_true == 0)).sum() > 0 else 0
    fpr1 = y_pred[(m1) & (y_true == 0)].mean() if ((m1) & (y_true == 0)).sum() > 0 else 0
    base['eq_opp'] = abs(tpr0 - tpr1)
    base['avg_odds'] = (abs(fpr0 - fpr1) + abs(tpr0 - tpr1)) / 2
    return base


def run_unified(dataset_names, output_dir, n_seeds=10, device='cpu'):
    logger.info("Loading datasets...")
    datasets = {}
    for name in dataset_names:
        if name in DATASET_LOADERS:
            try:
                ds = DATASET_LOADERS[name]()
                datasets[name] = ds
                logger.info(f"  {name}: n={len(ds.X)}, d={ds.X.shape[1]}")
            except Exception as e:
                logger.warning(f"  {name}: {e}")

    all_results = []

    for ds_name, dataset in datasets.items():
        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        min_feat = max(3, d // 3)
        do_ablation = ds_name in ABLATION_DATASETS

        logger.info(f"\n{'='*60}")
        logger.info(f"{ds_name} (n={len(X)}, d={d})")
        logger.info(f"{'='*60}")

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)

            # ----------------------------------------------------------
            # XGBoost baseline (no feature selection)
            # ----------------------------------------------------------
            m_xgb = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m_xgb.fit(X_tr, y_tr)
            yp = m_xgb.predict(X_te)
            ypr = m_xgb.predict_proba(X_te)[:, 1]
            met = compute_extended_metrics(y_te, yp, ypr, s_te)
            all_results.append({
                'table': 'T2', 'dataset': ds_name, 'method': 'XGBoost',
                'seed': seed, **met, 'n_features': d})

            # ----------------------------------------------------------
            # CausalGBM with MAX aggregation (default) — used in T2, T3, T4
            # Learn DAG ONCE, reuse for all aggregation variants
            # ----------------------------------------------------------
            # We need to run all 3 aggregation variants from the SAME split
            aggregation_results = {}
            for agg in (['max', 'dag_only', 'corr_only'] if do_ablation else ['max']):
                sel = CausalFeatureSelector(
                    d, alpha=0.5, threshold=0.2, min_features=min_feat,
                    n_iterations=500, aggregation=agg, device=device)
                sel.fit(X_tr, s_tr, y_tr)
                Xtr_s = sel.transform(X_tr)
                Xte_s = sel.transform(X_te)
                nf = len(sel.selected_)

                m_cgbm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m_cgbm.fit(Xtr_s, y_tr)
                yp = m_cgbm.predict(Xte_s)
                ypr = m_cgbm.predict_proba(Xte_s)[:, 1]
                met = compute_extended_metrics(y_te, yp, ypr, s_te)

                aggregation_results[agg] = met

                # Table 4: ablation row
                if do_ablation:
                    all_results.append({
                        'table': 'T4', 'dataset': ds_name,
                        'method': f'CausalGBM-{agg}', 'seed': seed,
                        **met, 'n_features': nf})

                # Table 2: main result (max aggregation only)
                if agg == 'max':
                    all_results.append({
                        'table': 'T2', 'dataset': ds_name,
                        'method': 'CausalGBM', 'seed': seed,
                        **met, 'n_features': nf})

                    # Table 3: same CausalGBM value
                    all_results.append({
                        'table': 'T3', 'dataset': ds_name,
                        'method': 'CausalGBM', 'seed': seed,
                        **met, 'n_features': nf})

            # ----------------------------------------------------------
            # FairGBM (Table 3)
            # ----------------------------------------------------------
            if HAS_FAIRGBM:
                try:
                    fgbm = FairGBMClassifier(
                        constraint_type="FNR,FPR", n_estimators=100,
                        random_state=seed, multiplier_learning_rate=0.1, verbose=-1)
                    fgbm.fit(X_tr, y_tr, constraint_group=s_tr)
                    yp = fgbm.predict(X_te)
                    ypr = fgbm.predict_proba(X_te)[:, 1]
                    met = compute_extended_metrics(y_te, yp, ypr, s_te)
                    all_results.append({
                        'table': 'T3', 'dataset': ds_name,
                        'method': 'FairGBM', 'seed': seed,
                        **met, 'n_features': d})
                except Exception as e:
                    logger.warning(f"  FairGBM seed={seed}: {e}")

            # ----------------------------------------------------------
            # M2FGB-TPR (Table 3)
            # ----------------------------------------------------------
            if HAS_M2FGB:
                try:
                    m2 = M2FGBClassifier(
                        fairness_constraint='true_positive_rate',
                        fair_weight=0.5, n_estimators=100,
                        learning_rate=0.1, multiplier_learning_rate=0.1,
                        random_state=seed)
                    m2.fit(X_tr, y_tr, sensitive_attribute=s_tr)
                    yp = m2.predict(X_te)
                    ypr = m2.predict_proba(X_te)[:, 1]
                    met = compute_extended_metrics(y_te, yp, ypr, s_te)
                    all_results.append({
                        'table': 'T3', 'dataset': ds_name,
                        'method': 'M2FGB-TPR', 'seed': seed,
                        **met, 'n_features': d})
                except Exception as e:
                    logger.warning(f"  M2FGB seed={seed}: {e}")

        # Print progress
        ds_res = [r for r in all_results if r['dataset'] == ds_name]
        for method in ['XGBoost', 'CausalGBM', 'FairGBM', 'M2FGB-TPR']:
            m_res = [r for r in ds_res if r['method'] == method and r['table'] in ['T2','T3']]
            if m_res:
                avg_eod = np.mean([r['eod'] for r in m_res])
                avg_auc = np.mean([r['auc'] for r in m_res])
                logger.info(f"  {method:15s}: AUC={avg_auc:.3f}  EOD={avg_eod:.3f}")

    # Save everything
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'unified_results.csv'), index=False)

    # ================================================================
    # GENERATE TABLES
    # ================================================================

    print("\n" + "=" * 70)
    print("TABLE 1: Dataset characteristics (Base EOD from XGBoost)")
    print("=" * 70)
    t1 = df[(df['table'] == 'T2') & (df['method'] == 'XGBoost')].groupby('dataset')['eod'].mean().round(3)
    for ds in dataset_names:
        if ds in t1.index:
            print(f"  {ds:20s}: Base EOD = {t1[ds]}")

    print("\n" + "=" * 70)
    print("TABLE 2: CausalGBM vs XGBoost (main results)")
    print("=" * 70)
    t2 = df[df['table'] == 'T2'].groupby(['dataset', 'method']).agg(
        auc=('auc', 'mean'), wga=('wga', 'mean'),
        eod=('eod', 'mean'), dpd=('dpd', 'mean')).round(3)
    print(t2.to_string())

    # Significance
    print("\n  Significance:")
    for ds in dataset_names:
        xgb_eod = df[(df['table']=='T2') & (df['dataset']==ds) & (df['method']=='XGBoost')]['eod'].values
        cgbm_eod = df[(df['table']=='T2') & (df['dataset']==ds) & (df['method']=='CausalGBM')]['eod'].values
        if len(xgb_eod) >= 2 and len(cgbm_eod) >= 2:
            ml = min(len(xgb_eod), len(cgbm_eod))
            t, p = ttest_rel(xgb_eod[:ml], cgbm_eod[:ml])
            delta = np.mean(xgb_eod[:ml]) - np.mean(cgbm_eod[:ml])
            diffs = xgb_eod[:ml] - cgbm_eod[:ml]
            d_cohen = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {ds:20s}: ΔEOD={delta:.3f}  d={d_cohen:.2f}  p={p:.4f}  ({sig})")

    print("\n" + "=" * 70)
    print("TABLE 3: Fair GBM comparison")
    print("=" * 70)
    t3_methods = ['XGBoost', 'CausalGBM', 'FairGBM', 'M2FGB-TPR']
    t3 = df[(df['table'].isin(['T2','T3'])) & (df['method'].isin(t3_methods))].groupby(
        ['dataset', 'method']).agg(
        auc=('auc', 'mean'), eod=('eod', 'mean'),
        dpd=('dpd', 'mean'), wga=('wga', 'mean')).round(3)
    for ds in dataset_names:
        if ds in t3.index.get_level_values(0):
            print(f"\n  --- {ds} ---")
            print(t3.loc[ds].to_string())

    print("\n" + "=" * 70)
    print("TABLE 4: Ablation (aggregation method)")
    print("=" * 70)
    t4 = df[df['table'] == 'T4'].groupby(['dataset', 'method'])['eod'].mean().round(3)
    for ds in ABLATION_DATASETS:
        if ds in t4.index.get_level_values(0):
            row = t4.loc[ds]
            dag = row.get('CausalGBM-dag_only', np.nan)
            corr = row.get('CausalGBM-corr_only', np.nan)
            mx = row.get('CausalGBM-max', np.nan)
            print(f"  {ds:20s}: DAG={dag:.3f}  Corr={corr:.3f}  Max={mx:.3f}")

    # CONSISTENCY CHECK
    print("\n" + "=" * 70)
    print("CONSISTENCY CHECK: CausalGBM EOD across tables")
    print("=" * 70)
    for ds in dataset_names:
        t2_val = df[(df['table']=='T2') & (df['dataset']==ds) & (df['method']=='CausalGBM')]['eod'].mean()
        t3_val = df[(df['table']=='T3') & (df['dataset']==ds) & (df['method']=='CausalGBM')]['eod'].mean()
        t4_val = df[(df['table']=='T4') & (df['dataset']==ds) & (df['method']=='CausalGBM-max')]['eod'].mean()

        t2_s = f"{t2_val:.4f}" if not np.isnan(t2_val) else "N/A"
        t3_s = f"{t3_val:.4f}" if not np.isnan(t3_val) else "N/A"
        t4_s = f"{t4_val:.4f}" if not np.isnan(t4_val) else "N/A"

        match = "✓" if (np.isnan(t4_val) or abs(t2_val - t4_val) < 0.0001) else "✗ MISMATCH"
        print(f"  {ds:20s}: T2={t2_s}  T3={t3_s}  T4={t4_s}  {match}")

    print(f"\nAll results saved to: {os.path.join(output_dir, 'unified_results.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output_dir', default='results/acml2026/unified')
    parser.add_argument('--datasets', nargs='+',
                       default=['adult', 'acs_income', 'compas', 'german',
                                'taiwan_credit', 'bank', 'online_shoppers',
                                'synthetic_loan', 'synthetic_hiring'])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_unified(args.datasets, args.output_dir, args.n_seeds, args.device)
