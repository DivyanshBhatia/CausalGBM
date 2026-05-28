"""
ACML 2026 Supplementary Experiments
====================================
1. LightGBM as Stage 2 (proves framework generality)
2. Calibration metric (Expected Calibration Error)
3. Min-features ablation across ALL datasets (m=1,2,3,5,auto)

Usage:
  python acml_supplementary_experiments.py --all
  python acml_supplementary_experiments.py --stage2
  python acml_supplementary_experiments.py --calibration
  python acml_supplementary_experiments.py --min_features
"""

import os
import sys
import time
import argparse
import warnings
import logging
import numpy as np
import pandas as pd

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
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

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


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)


def load_all_datasets(dataset_names, max_samples=200000):
    datasets = {}
    for name in dataset_names:
        if name in DATASET_LOADERS:
            try:
                ds = DATASET_LOADERS[name](max_samples=max_samples)
                datasets[name] = ds
                logger.info(f"  Loaded {name}: n={len(ds.X)}, d={ds.X.shape[1]}")
            except Exception as e:
                logger.warning(f"  Could not load {name}: {e}")
    return datasets


# ============================================================================
# EXPERIMENT 1: LIGHTGBM AS STAGE 2
# ============================================================================

def run_stage2_comparison(datasets, output_dir, n_seeds=10, device='cpu'):
    """
    Compare CausalGBM with XGBoost vs LightGBM vs RandomForest as Stage 2.
    Proves CausalGBM is a general framework, not tied to XGBoost.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: STAGE 2 MODEL COMPARISON")
    logger.info("=" * 70)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    results = []

    for ds_name, dataset in datasets.items():
        logger.info(f"\n--- {ds_name} (d={dataset.X.shape[1]}) ---")
        X, y, sens = dataset.X, dataset.y, dataset.sensitive

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y
            )

            # Run CausalGBM feature selection ONCE
            selector = CausalFeatureSelector(
                X_tr.shape[1], alpha=0.5, threshold=0.2,
                min_features=max(3, X_tr.shape[1] // 3),
                n_iterations=500, aggregation='max', device=device
            )
            selector.fit(X_tr, s_tr, y_tr)
            X_tr_sel = selector.transform(X_tr)
            X_te_sel = selector.transform(X_te)
            n_feats = len(selector.selected_)

            # Test with different Stage 2 models
            stage2_models = {
                'CausalGBM-XGBoost': xgb.XGBClassifier(
                    n_estimators=100, random_state=seed, verbosity=0
                ) if HAS_XGB else GradientBoostingClassifier(
                    n_estimators=100, random_state=seed
                ),
                'CausalGBM-LightGBM': lgb.LGBMClassifier(
                    n_estimators=100, random_state=seed, verbose=-1
                ) if HAS_LGB else None,
                'CausalGBM-RF': RandomForestClassifier(
                    n_estimators=100, random_state=seed
                ),
            }

            for method_name, model in stage2_models.items():
                if model is None:
                    continue
                start = time.time()
                model.fit(X_tr_sel, y_tr)
                y_pred = model.predict(X_te_sel)
                y_prob = model.predict_proba(X_te_sel)[:, 1]
                elapsed = time.time() - start

                metrics = compute_metrics(y_te, y_pred, y_prob, s_te)
                results.append({
                    'method': method_name,
                    'dataset': ds_name,
                    'seed': seed,
                    'auc': metrics['auc'],
                    'eod': metrics['eod'],
                    'dpd': metrics['dpd'],
                    'wga': metrics['wga'],
                    'f1': metrics['f1'],
                    'n_features': n_feats,
                    'time': elapsed,
                })

            if seed == 0:
                for m in ['CausalGBM-XGBoost', 'CausalGBM-LightGBM', 'CausalGBM-RF']:
                    r = [x for x in results if x['method'] == m and x['dataset'] == ds_name and x['seed'] == 0]
                    if r:
                        logger.info(f"  {m:25s}: AUC={r[0]['auc']:.3f}  EOD={r[0]['eod']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'stage2_comparison_raw.csv'), index=False)

    # Summary
    summary = df.groupby(['dataset', 'method']).agg(
        auc=('auc', 'mean'), eod=('eod', 'mean'),
        dpd=('dpd', 'mean'), wga=('wga', 'mean'),
    ).round(4)

    print("\n" + "=" * 70)
    print("STAGE 2 MODEL COMPARISON SUMMARY")
    print("=" * 70)
    for ds in df['dataset'].unique():
        print(f"\n--- {ds} ---")
        print(summary.loc[ds].to_string())

    summary.to_csv(os.path.join(output_dir, 'stage2_comparison_summary.csv'))
    return df


# ============================================================================
# EXPERIMENT 2: CALIBRATION METRICS
# ============================================================================

def run_calibration_comparison(datasets, output_dir, n_seeds=10, device='cpu'):
    """
    Compare calibration (ECE) of CausalGBM vs XGBoost vs FairGBM.
    Shows proxy removal doesn't harm calibration.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: CALIBRATION (ECE) COMPARISON")
    logger.info("=" * 70)

    try:
        from fairgbm import FairGBMClassifier
        HAS_FAIRGBM = True
    except ImportError:
        HAS_FAIRGBM = False
        logger.warning("FairGBM not installed, skipping FairGBM calibration")

    results = []

    for ds_name, dataset in datasets.items():
        logger.info(f"\n--- {ds_name} ---")
        X, y, sens = dataset.X, dataset.y, dataset.sensitive

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y
            )

            # XGBoost baseline
            model_xgb = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else None
            if model_xgb:
                model_xgb.fit(X_tr, y_tr)
                y_prob_xgb = model_xgb.predict_proba(X_te)[:, 1]
                y_pred_xgb = model_xgb.predict(X_te)
                m = compute_metrics(y_te, y_pred_xgb, y_prob_xgb, s_te)
                results.append({
                    'method': 'XGBoost', 'dataset': ds_name, 'seed': seed,
                    'auc': m['auc'], 'eod': m['eod'],
                    'ece': expected_calibration_error(y_te, y_prob_xgb),
                })

            # CausalGBM
            selector = CausalFeatureSelector(
                X_tr.shape[1], alpha=0.5, threshold=0.2,
                min_features=max(3, X_tr.shape[1] // 3),
                n_iterations=500, aggregation='max', device=device
            )
            selector.fit(X_tr, s_tr, y_tr)
            X_tr_sel = selector.transform(X_tr)
            X_te_sel = selector.transform(X_te)

            model_cgbm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else None
            if model_cgbm:
                model_cgbm.fit(X_tr_sel, y_tr)
                y_prob_cgbm = model_cgbm.predict_proba(X_te_sel)[:, 1]
                y_pred_cgbm = model_cgbm.predict(X_te_sel)
                m = compute_metrics(y_te, y_pred_cgbm, y_prob_cgbm, s_te)
                results.append({
                    'method': 'CausalGBM', 'dataset': ds_name, 'seed': seed,
                    'auc': m['auc'], 'eod': m['eod'],
                    'ece': expected_calibration_error(y_te, y_prob_cgbm),
                })

            # FairGBM
            if HAS_FAIRGBM:
                try:
                    model_fgbm = FairGBMClassifier(
                        constraint_type="FNR,FPR", n_estimators=100,
                        random_state=seed, multiplier_learning_rate=0.1, verbose=-1
                    )
                    model_fgbm.fit(X_tr, y_tr, constraint_group=s_tr)
                    y_prob_fgbm = model_fgbm.predict_proba(X_te)[:, 1]
                    y_pred_fgbm = model_fgbm.predict(X_te)
                    m = compute_metrics(y_te, y_pred_fgbm, y_prob_fgbm, s_te)
                    results.append({
                        'method': 'FairGBM', 'dataset': ds_name, 'seed': seed,
                        'auc': m['auc'], 'eod': m['eod'],
                        'ece': expected_calibration_error(y_te, y_prob_fgbm),
                    })
                except Exception as e:
                    logger.warning(f"  FairGBM failed on {ds_name} seed={seed}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'calibration_raw.csv'), index=False)

    summary = df.groupby(['dataset', 'method']).agg(
        auc=('auc', 'mean'), eod=('eod', 'mean'),
        ece=('ece', 'mean'),
    ).round(4)

    print("\n" + "=" * 70)
    print("CALIBRATION (ECE) COMPARISON")
    print("=" * 70)
    for ds in df['dataset'].unique():
        print(f"\n--- {ds} ---")
        if ds in summary.index:
            print(summary.loc[ds].to_string())

    summary.to_csv(os.path.join(output_dir, 'calibration_summary.csv'))
    return df


# ============================================================================
# EXPERIMENT 3: MIN-FEATURES ABLATION (ALL DATASETS)
# ============================================================================

def run_min_features_ablation(datasets, output_dir, n_seeds=10, device='cpu'):
    """
    Vary min_features parameter across ALL datasets.
    Tests: m=1, m=2, m=3, m=5, m=auto (default = max(3, d//3))

    This disentangles:
    - Whether the m-floor is masking potential improvements (COMPAS: d=6, m=3)
    - Whether aggressive removal (m=1) helps or hurts
    - Optimal m per dataset
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: MIN-FEATURES ABLATION (ALL DATASETS)")
    logger.info("=" * 70)

    results = []

    for ds_name, dataset in datasets.items():
        d = dataset.X.shape[1]
        m_auto = max(3, d // 3)

        # Test different m values
        m_values = sorted(set([1, 2, 3, 5, m_auto, d]))  # include d (no removal possible)
        m_values = [m for m in m_values if m <= d]  # cap at d

        logger.info(f"\n--- {ds_name} (d={d}, m_auto={m_auto}) ---")
        logger.info(f"  Testing m ∈ {m_values}")

        X, y, sens = dataset.X, dataset.y, dataset.sensitive

        for m in m_values:
            for seed in range(n_seeds):
                X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                    X, y, sens, test_size=0.3, random_state=seed, stratify=y
                )

                try:
                    selector = CausalFeatureSelector(
                        X_tr.shape[1], alpha=0.5, threshold=0.2,
                        min_features=m,
                        n_iterations=500, aggregation='max', device=device
                    )
                    selector.fit(X_tr, s_tr, y_tr)
                    X_tr_sel = selector.transform(X_tr)
                    X_te_sel = selector.transform(X_te)
                    n_selected = len(selector.selected_)

                    model = xgb.XGBClassifier(
                        n_estimators=100, random_state=seed, verbosity=0
                    ) if HAS_XGB else None

                    if model:
                        model.fit(X_tr_sel, y_tr)
                        y_pred = model.predict(X_te_sel)
                        y_prob = model.predict_proba(X_te_sel)[:, 1]
                        metrics = compute_metrics(y_te, y_pred, y_prob, s_te)

                        results.append({
                            'dataset': ds_name,
                            'min_features': m,
                            'is_default': m == m_auto,
                            'seed': seed,
                            'n_selected': n_selected,
                            'd_total': d,
                            'auc': metrics['auc'],
                            'eod': metrics['eod'],
                            'dpd': metrics['dpd'],
                            'wga': metrics['wga'],
                            'f1': metrics['f1'],
                        })
                except Exception as e:
                    logger.warning(f"  m={m}, seed={seed} failed: {e}")

        # Print summary for this dataset
        ds_results = [r for r in results if r['dataset'] == ds_name]
        if ds_results:
            ds_df = pd.DataFrame(ds_results)
            summary = ds_df.groupby('min_features').agg(
                n_sel=('n_selected', 'mean'),
                auc=('auc', 'mean'),
                eod=('eod', 'mean'),
            ).round(4)
            logger.info(f"  Results:")
            for m_val, row in summary.iterrows():
                default_marker = " (default)" if m_val == m_auto else ""
                logger.info(
                    f"    m={m_val:2d}{default_marker:10s}: "
                    f"selected={row['n_sel']:.1f}/{d}, "
                    f"AUC={row['auc']:.3f}, EOD={row['eod']:.3f}"
                )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'min_features_ablation_raw.csv'), index=False)

    # Full summary
    summary = df.groupby(['dataset', 'min_features']).agg(
        n_selected=('n_selected', 'mean'),
        auc=('auc', 'mean'), auc_std=('auc', 'std'),
        eod=('eod', 'mean'), eod_std=('eod', 'std'),
        wga=('wga', 'mean'),
    ).round(4)

    print("\n" + "=" * 70)
    print("MIN-FEATURES ABLATION SUMMARY (ALL DATASETS)")
    print("=" * 70)
    for ds in df['dataset'].unique():
        print(f"\n--- {ds} (d={df[df['dataset']==ds]['d_total'].iloc[0]}) ---")
        if ds in summary.index:
            print(summary.loc[ds][['n_selected', 'auc', 'eod', 'wga']].to_string())

    summary.to_csv(os.path.join(output_dir, 'min_features_ablation_summary.csv'))

    # Find optimal m per dataset
    print("\n" + "=" * 70)
    print("OPTIMAL m PER DATASET")
    print("=" * 70)
    opt_rows = []
    for ds in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds]
        d = ds_df['d_total'].iloc[0]
        m_auto = max(3, d // 3)

        for m_val in ds_df['min_features'].unique():
            m_data = ds_df[ds_df['min_features'] == m_val]
            opt_rows.append({
                'dataset': ds, 'd': d, 'm': m_val,
                'default': '✓' if m_val == m_auto else '',
                'AUC': round(m_data['auc'].mean(), 3),
                'EOD': round(m_data['eod'].mean(), 3),
                'n_sel': round(m_data['n_selected'].mean(), 1),
            })

    opt_df = pd.DataFrame(opt_rows)

    # Best EOD per dataset
    for ds in df['dataset'].unique():
        ds_opt = opt_df[opt_df['dataset'] == ds]
        best = ds_opt.loc[ds_opt['EOD'].idxmin()]
        default = ds_opt[ds_opt['default'] == '✓'].iloc[0] if len(ds_opt[ds_opt['default'] == '✓']) > 0 else None
        if default is not None:
            diff = default['EOD'] - best['EOD']
            print(f"  {ds:20s}: best m={int(best['m'])} (EOD={best['EOD']:.3f}), "
                  f"default m={int(default['m'])} (EOD={default['EOD']:.3f}), "
                  f"gap={diff:.3f}")

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ACML 2026 Supplementary Experiments')
    parser.add_argument('--stage2', action='store_true', help='Stage 2 model comparison')
    parser.add_argument('--calibration', action='store_true', help='Calibration (ECE) comparison')
    parser.add_argument('--min_features', action='store_true', help='Min-features ablation')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--datasets', nargs='+',
                       default=['adult', 'acs_income', 'compas', 'german',
                                'taiwan_credit', 'bank', 'online_shoppers',
                                'synthetic_loan', 'synthetic_hiring'])
    parser.add_argument('--output_dir', default='results/acml2026/supplementary')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=200000)

    args = parser.parse_args()

    if args.all:
        args.stage2 = True
        args.calibration = True
        args.min_features = True

    if not (args.stage2 or args.calibration or args.min_features):
        parser.print_help()
        print("\nSpecify: --stage2, --calibration, --min_features, or --all")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading datasets...")
    datasets = load_all_datasets(args.datasets, max_samples=args.max_samples)

    if not datasets:
        logger.error("No datasets loaded!")
        return

    if args.stage2:
        run_stage2_comparison(datasets, args.output_dir,
                            n_seeds=args.n_seeds, device=args.device)

    if args.calibration:
        run_calibration_comparison(datasets, args.output_dir,
                                  n_seeds=args.n_seeds, device=args.device)

    if args.min_features:
        run_min_features_ablation(datasets, args.output_dir,
                                  n_seeds=args.n_seeds, device=args.device)

    logger.info("\nAll experiments complete! Results in: " + args.output_dir)


if __name__ == '__main__':
    main()
