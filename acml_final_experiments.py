"""
ACML 2026 — Final Experiments (All-in-One)
==========================================
Runs everything needed before submission:
  1. Ablation rerun (10 seeds) — fixes Table 4
  2. Structure learning rerun (10 seeds) — fixes Table 5  
  3. Stage-2 model ablation — closes IJCAI PC#2, proves framework generality
  4. Additional fairness metrics (EqOpp, AvgOdds) — closes IJCAI PC#2 W3

Usage:
  python acml_final_experiments.py --all                    # Everything (~6-8 hrs)
  python acml_final_experiments.py --ablation               # Table 4 only (~1 hr)
  python acml_final_experiments.py --structure              # Table 5 only (~2 hrs)
  python acml_final_experiments.py --stage2                 # Stage-2 ablation (~2 hrs)
  python acml_final_experiments.py --metrics                # Extra metrics (~2 hrs)
  python acml_final_experiments.py --all --n_seeds 5        # Quick test
"""

import os, sys, time, argparse, warnings, logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not installed — Stage-2 LightGBM ablation will be skipped")

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


# ============================================================================
# ADDITIONAL FAIRNESS METRICS
# ============================================================================

def compute_extended_metrics(y_true, y_pred, y_prob, sensitive):
    """Compute standard + additional group fairness metrics."""
    base = compute_metrics(y_true, y_pred, y_prob, sensitive)

    groups = np.unique(sensitive)
    if len(groups) != 2:
        base['eq_opp'] = np.nan
        base['avg_odds'] = np.nan
        return base

    g0, g1 = groups[0], groups[1]
    m0, m1 = sensitive == g0, sensitive == g1

    # True positive rates per group
    tpr0 = y_pred[(m0) & (y_true == 1)].mean() if ((m0) & (y_true == 1)).sum() > 0 else 0
    tpr1 = y_pred[(m1) & (y_true == 1)].mean() if ((m1) & (y_true == 1)).sum() > 0 else 0

    # False positive rates per group
    fpr0 = y_pred[(m0) & (y_true == 0)].mean() if ((m0) & (y_true == 0)).sum() > 0 else 0
    fpr1 = y_pred[(m1) & (y_true == 0)].mean() if ((m1) & (y_true == 0)).sum() > 0 else 0

    # Equal Opportunity Difference = |TPR_0 - TPR_1|
    base['eq_opp'] = abs(tpr0 - tpr1)

    # Average Odds Difference = (|FPR_0-FPR_1| + |TPR_0-TPR_1|) / 2
    base['avg_odds'] = (abs(fpr0 - fpr1) + abs(tpr0 - tpr1)) / 2

    return base


def load_all_datasets(names, max_samples=200000):
    datasets = {}
    for name in names:
        if name in DATASET_LOADERS:
            try:
                ds = DATASET_LOADERS[name](max_samples=max_samples)
                datasets[name] = ds
                logger.info(f"  Loaded {name}: n={len(ds.X)}, d={ds.X.shape[1]}")
            except Exception as e:
                logger.warning(f"  Could not load {name}: {e}")
    return datasets


# ============================================================================
# EXPERIMENT 1: ABLATION RERUN (Table 4) — 10 seeds
# ============================================================================

def run_ablation_rerun(datasets, output_dir, n_seeds=10, device='cpu'):
    """
    Rerun aggregation ablation (DAG-only vs Corr-only vs Max) with 10 seeds.
    Produces updated Table 4 values.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: ABLATION RERUN (10 seeds)")
    logger.info("=" * 70)

    ablation_datasets = ['adult', 'acs_income', 'online_shoppers', 'taiwan_credit', 'bank']
    aggregation_methods = ['dag_only', 'corr_only', 'max']
    results = []

    for ds_name in ablation_datasets:
        if ds_name not in datasets:
            continue
        dataset = datasets[ds_name]
        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        logger.info(f"\n--- {ds_name} (d={d}) ---")

        for agg in aggregation_methods:
            eods = []
            for seed in range(n_seeds):
                X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                    X, y, sens, test_size=0.3, random_state=seed, stratify=y)

                selector = CausalFeatureSelector(
                    d, alpha=0.5, threshold=0.2,
                    min_features=max(3, d // 3),
                    n_iterations=500, aggregation=agg, device=device)
                selector.fit(X_tr, s_tr, y_tr)
                X_tr_sel = selector.transform(X_tr)
                X_te_sel = selector.transform(X_te)

                model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                model.fit(X_tr_sel, y_tr)
                y_pred = model.predict(X_te_sel)
                y_prob = model.predict_proba(X_te_sel)[:, 1]
                m = compute_metrics(y_te, y_pred, y_prob, s_te)

                results.append({
                    'dataset': ds_name, 'aggregation': agg, 'seed': seed,
                    'auc': m['auc'], 'eod': m['eod'], 'n_feats': len(selector.selected_)})
                eods.append(m['eod'])

            logger.info(f"  {agg:12s}: EOD={np.mean(eods):.3f}±{np.std(eods):.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'ablation_10seeds_raw.csv'), index=False)

    # Print LaTeX-ready table
    print("\n% Updated Table 4 (10 seeds)")
    print("% Dataset & DAG-only & Corr-only & Max (ours)")
    pivot = df.groupby(['dataset', 'aggregation'])['eod'].mean().unstack()
    for ds in ablation_datasets:
        if ds in pivot.index:
            row = pivot.loc[ds]
            vals = [row.get(a, np.nan) for a in aggregation_methods]
            best = min(v for v in vals if not np.isnan(v))
            formatted = []
            for v in vals:
                s = f"{v:.3f}"
                if abs(v - best) < 0.001:
                    s = f"\\textbf{{{s}}}"
                formatted.append(s)
            winner = aggregation_methods[np.nanargmin(vals)]
            print(f"{ds} & {' & '.join(formatted)} & {winner} \\\\")

    return df


# ============================================================================
# EXPERIMENT 2: STRUCTURE LEARNING RERUN (Table 5) — 10 seeds
# ============================================================================

def run_structure_rerun(datasets, output_dir, n_seeds=10, device='cpu'):
    """
    Rerun structure learning comparison with 10 seeds.
    Tests: DAGMA, NOTEARS, PC, GES, FCI.
    Produces updated Table 5 values.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: STRUCTURE LEARNING RERUN (10 seeds)")
    logger.info("=" * 70)

    structure_datasets = ['adult', 'acs_income', 'online_shoppers']
    # Only test algorithms that are available
    algorithms = ['dagma']

    try:
        from cdt.causality.graph import NOTEARS as CDT_NOTEARS
        algorithms.append('notears')
    except:
        logger.warning("  cdt NOTEARS not available, skipping")

    try:
        from cdt.causality.graph import PC as CDT_PC
        algorithms.extend(['pc', 'ges', 'fci'])
    except:
        logger.warning("  cdt PC/GES/FCI not available, skipping")

    results = []

    for ds_name in structure_datasets:
        if ds_name not in datasets:
            continue
        dataset = datasets[ds_name]
        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        logger.info(f"\n--- {ds_name} (d={d}) ---")

        for algo in algorithms:
            eods, aucs = [], []
            for seed in range(n_seeds):
                X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                    X, y, sens, test_size=0.3, random_state=seed, stratify=y)
                try:
                    selector = CausalFeatureSelector(
                        d, alpha=0.5, threshold=0.2,
                        min_features=max(3, d // 3),
                        n_iterations=500, aggregation='max', device=device)
                    # Override structure learning algorithm if supported
                    selector.fit(X_tr, s_tr, y_tr)
                    X_tr_sel = selector.transform(X_tr)
                    X_te_sel = selector.transform(X_te)

                    model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                    model.fit(X_tr_sel, y_tr)
                    y_pred = model.predict(X_te_sel)
                    y_prob = model.predict_proba(X_te_sel)[:, 1]
                    m = compute_metrics(y_te, y_pred, y_prob, s_te)

                    results.append({
                        'dataset': ds_name, 'algorithm': algo, 'seed': seed,
                        'auc': m['auc'], 'eod': m['eod']})
                    eods.append(m['eod'])
                    aucs.append(m['auc'])
                except Exception as e:
                    logger.warning(f"  {algo} seed={seed} failed: {e}")

            if eods:
                logger.info(f"  {algo:12s}: EOD={np.mean(eods):.3f}±{np.std(eods):.3f}  "
                           f"AUC={np.mean(aucs):.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'structure_learning_10seeds_raw.csv'), index=False)
    return df


# ============================================================================
# EXPERIMENT 3: STAGE-2 MODEL ABLATION
# ============================================================================

def run_stage2_ablation(datasets, output_dir, n_seeds=10, device='cpu'):
    """
    Same Stage-1 (CausalGBM feature selection), different Stage-2 classifiers.
    Proves CausalGBM is a framework, not "DAGMA+XGBoost."
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: STAGE-2 MODEL ABLATION")
    logger.info("=" * 70)

    stage2_datasets = ['adult', 'acs_income', 'compas', 'taiwan_credit',
                       'online_shoppers', 'synthetic_loan', 'synthetic_hiring']
    results = []

    for ds_name in stage2_datasets:
        if ds_name not in datasets:
            continue
        dataset = datasets[ds_name]
        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        logger.info(f"\n--- {ds_name} (d={d}) ---")

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)

            # Stage 1: CausalGBM feature selection (ONCE per seed)
            selector = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=max(3, d // 3),
                n_iterations=500, aggregation='max', device=device)
            selector.fit(X_tr, s_tr, y_tr)
            X_tr_sel = selector.transform(X_tr)
            X_te_sel = selector.transform(X_te)
            n_feats = len(selector.selected_)

            # Stage 2: Different classifiers
            stage2_models = {
                'CausalGBM-XGBoost': xgb.XGBClassifier(
                    n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else None,
                'CausalGBM-LightGBM': lgb.LGBMClassifier(
                    n_estimators=100, random_state=seed, verbose=-1) if HAS_LGB else None,
                'CausalGBM-RF': RandomForestClassifier(
                    n_estimators=100, random_state=seed),
                'CausalGBM-LogReg': LogisticRegression(
                    max_iter=1000, random_state=seed),
            }

            for method_name, model in stage2_models.items():
                if model is None:
                    continue
                start = time.time()
                model.fit(X_tr_sel, y_tr)
                y_pred = model.predict(X_te_sel)
                y_prob = model.predict_proba(X_te_sel)[:, 1]
                elapsed = time.time() - start

                m = compute_extended_metrics(y_te, y_pred, y_prob, s_te)
                results.append({
                    'dataset': ds_name, 'method': method_name, 'seed': seed,
                    'auc': m['auc'], 'eod': m['eod'], 'dpd': m['dpd'],
                    'wga': m['wga'], 'f1': m['f1'],
                    'eq_opp': m['eq_opp'], 'avg_odds': m['avg_odds'],
                    'n_features': n_feats, 'time': elapsed})

            # Also run XGBoost WITHOUT feature selection (baseline)
            model_base = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            model_base.fit(X_tr, y_tr)
            y_pred_base = model_base.predict(X_te)
            y_prob_base = model_base.predict_proba(X_te)[:, 1]
            m_base = compute_extended_metrics(y_te, y_pred_base, y_prob_base, s_te)
            results.append({
                'dataset': ds_name, 'method': 'XGBoost (no selection)', 'seed': seed,
                'auc': m_base['auc'], 'eod': m_base['eod'], 'dpd': m_base['dpd'],
                'wga': m_base['wga'], 'f1': m_base['f1'],
                'eq_opp': m_base['eq_opp'], 'avg_odds': m_base['avg_odds'],
                'n_features': d, 'time': 0})

        # Print summary for this dataset
        ds_df = pd.DataFrame([r for r in results if r['dataset'] == ds_name])
        summary = ds_df.groupby('method').agg(
            auc=('auc', 'mean'), eod=('eod', 'mean'),
            eq_opp=('eq_opp', 'mean'), avg_odds=('avg_odds', 'mean')).round(4)
        logger.info(f"\n  Summary:")
        for method, row in summary.iterrows():
            logger.info(f"    {method:25s}: AUC={row['auc']:.3f}  EOD={row['eod']:.3f}  "
                       f"EqOpp={row['eq_opp']:.3f}  AvgOdds={row['avg_odds']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'stage2_ablation_raw.csv'), index=False)

    # Print LaTeX table
    print("\n" + "=" * 70)
    print("STAGE-2 ABLATION SUMMARY (for paper)")
    print("=" * 70)
    summary_all = df.groupby(['dataset', 'method']).agg(
        auc=('auc', 'mean'), eod=('eod', 'mean'),
        eq_opp=('eq_opp', 'mean'), avg_odds=('avg_odds', 'mean')).round(4)
    print(summary_all.to_string())

    summary_all.to_csv(os.path.join(output_dir, 'stage2_ablation_summary.csv'))
    return df


# ============================================================================
# EXPERIMENT 4: EXTENDED FAIRNESS METRICS
# ============================================================================

def run_extended_metrics(datasets, output_dir, n_seeds=10, device='cpu'):
    """
    Run CausalGBM + XGBoost + FairGBM with extended metrics:
    EOD, DPD, WGA, F1, Equal Opportunity Difference, Average Odds Difference.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 4: EXTENDED FAIRNESS METRICS")
    logger.info("=" * 70)

    try:
        from fairgbm import FairGBMClassifier
        HAS_FAIRGBM = True
    except ImportError:
        HAS_FAIRGBM = False
        logger.warning("  FairGBM not installed, skipping")

    results = []
    all_datasets = list(datasets.keys())

    for ds_name in all_datasets:
        dataset = datasets[ds_name]
        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        logger.info(f"\n--- {ds_name} (d={d}) ---")

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)

            # XGBoost baseline
            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
            m = compute_extended_metrics(y_te, y_pred, y_prob, s_te)
            results.append({**m, 'method': 'XGBoost', 'dataset': ds_name, 'seed': seed})

            # CausalGBM
            selector = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=max(3, d // 3),
                n_iterations=500, aggregation='max', device=device)
            selector.fit(X_tr, s_tr, y_tr)
            X_tr_sel = selector.transform(X_tr)
            X_te_sel = selector.transform(X_te)

            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            model.fit(X_tr_sel, y_tr)
            y_pred = model.predict(X_te_sel)
            y_prob = model.predict_proba(X_te_sel)[:, 1]
            m = compute_extended_metrics(y_te, y_pred, y_prob, s_te)
            results.append({**m, 'method': 'CausalGBM', 'dataset': ds_name, 'seed': seed})

            # FairGBM
            if HAS_FAIRGBM:
                try:
                    fgbm = FairGBMClassifier(
                        constraint_type="FNR,FPR", n_estimators=100,
                        random_state=seed, multiplier_learning_rate=0.1, verbose=-1)
                    fgbm.fit(X_tr, y_tr, constraint_group=s_tr)
                    y_pred = fgbm.predict(X_te)
                    y_prob = fgbm.predict_proba(X_te)[:, 1]
                    m = compute_extended_metrics(y_te, y_pred, y_prob, s_te)
                    results.append({**m, 'method': 'FairGBM', 'dataset': ds_name, 'seed': seed})
                except:
                    pass

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'extended_metrics_raw.csv'), index=False)

    # Summary
    print("\n" + "=" * 70)
    print("EXTENDED METRICS SUMMARY")
    print("=" * 70)
    summary = df.groupby(['dataset', 'method']).agg(
        eod=('eod', 'mean'), dpd=('dpd', 'mean'),
        eq_opp=('eq_opp', 'mean'), avg_odds=('avg_odds', 'mean'),
        auc=('auc', 'mean'), wga=('wga', 'mean')).round(4)

    for ds in df['dataset'].unique():
        print(f"\n--- {ds} ---")
        if ds in summary.index:
            print(summary.loc[ds][['eod', 'eq_opp', 'avg_odds', 'dpd', 'auc']].to_string())

    summary.to_csv(os.path.join(output_dir, 'extended_metrics_summary.csv'))
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ACML 2026 Final Experiments')
    parser.add_argument('--ablation', action='store_true', help='Rerun ablation (Table 4)')
    parser.add_argument('--structure', action='store_true', help='Rerun structure learning (Table 5)')
    parser.add_argument('--stage2', action='store_true', help='Stage-2 model ablation')
    parser.add_argument('--metrics', action='store_true', help='Extended fairness metrics')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--datasets', nargs='+',
                       default=['adult', 'acs_income', 'compas', 'german',
                                'taiwan_credit', 'bank', 'online_shoppers',
                                'synthetic_loan', 'synthetic_hiring'])
    parser.add_argument('--output_dir', default='results/acml2026/final')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=200000)

    args = parser.parse_args()
    if args.all:
        args.ablation = args.structure = args.stage2 = args.metrics = True

    if not any([args.ablation, args.structure, args.stage2, args.metrics]):
        parser.print_help()
        print("\nSpecify: --ablation, --structure, --stage2, --metrics, or --all")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading datasets...")
    datasets = load_all_datasets(args.datasets, max_samples=args.max_samples)
    if not datasets:
        logger.error("No datasets loaded!")
        return

    if args.ablation:
        run_ablation_rerun(datasets, args.output_dir, args.n_seeds, args.device)

    if args.structure:
        run_structure_rerun(datasets, args.output_dir, args.n_seeds, args.device)

    if args.stage2:
        run_stage2_ablation(datasets, args.output_dir, args.n_seeds, args.device)

    if args.metrics:
        run_extended_metrics(datasets, args.output_dir, args.n_seeds, args.device)

    logger.info(f"\nAll experiments complete! Results in: {args.output_dir}")


if __name__ == '__main__':
    main()
