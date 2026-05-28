"""
M²FGB Comparison Experiments for ACML 2026
==========================================
Compares CausalGBM against M²FGB (Pereira et al., FAccT 2025)
"Min-Max Gradient Boosting Framework for Subgroup Fairness"

Setup:
  pip install lightgbm
  pip install git+https://github.com/hiaac-finance/m2fgb.git

Usage:
  python acml_m2fgb_experiments.py --all
  python acml_m2fgb_experiments.py --datasets adult acs_income
  python acml_m2fgb_experiments.py --all --n_seeds 10 --device cuda
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

# Import from existing codebase
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import (
    CausalFeatureSelector, compute_metrics,
    load_adult, load_compas, load_german, load_acs_income,
    load_taiwan_credit, load_bank, load_online_shoppers,
    load_synthetic_loan, load_synthetic_hiring,
)
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

# M2FGB import
try:
    from m2fgb.m2fgb import M2FGBClassifier
    HAS_M2FGB = True
    logger.info("M2FGB loaded successfully")
except ImportError:
    HAS_M2FGB = False
    logger.warning("M2FGB not installed. Run: pip install git+https://github.com/hiaac-finance/m2fgb.git")


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


def run_single_experiment(dataset, method_name, seed, alpha=0.5, 
                          fair_weight=0.5, fairness_constraint='equalized_loss',
                          device='cpu'):
    """Run one method on one dataset with one seed."""
    X, y, sens = dataset.X, dataset.y, dataset.sensitive
    
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sens, test_size=0.3, random_state=seed, stratify=y
    )
    
    start = time.time()
    n_feats = X_tr.shape[1]
    
    try:
        if method_name == 'XGBoost':
            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else \
                    GradientBoostingClassifier(n_estimators=100, random_state=seed)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
        
        elif method_name == 'CausalGBM':
            selector = CausalFeatureSelector(
                X_tr.shape[1], alpha=alpha, threshold=0.2,
                min_features=max(3, X_tr.shape[1] // 3),
                n_iterations=500, aggregation='max', device=device
            )
            selector.fit(X_tr, s_tr, y_tr)
            X_tr_sel = selector.transform(X_tr)
            X_te_sel = selector.transform(X_te)
            n_feats = len(selector.selected_)
            
            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else \
                    GradientBoostingClassifier(n_estimators=100, random_state=seed)
            model.fit(X_tr_sel, y_tr)
            y_pred = model.predict(X_te_sel)
            y_prob = model.predict_proba(X_te_sel)[:, 1]
        
        elif method_name.startswith('M2FGB'):
            if not HAS_M2FGB:
                logger.error("M2FGB not installed!")
                return None
            
            model = M2FGBClassifier(
                fairness_constraint=fairness_constraint,
                fair_weight=fair_weight,
                n_estimators=100,
                learning_rate=0.1,
                multiplier_learning_rate=0.1,
                random_state=seed,
            )
            model.fit(X_tr, y_tr, sensitive_attribute=s_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
        
        elif method_name == 'LightGBM':
            import lightgbm as lgb
            model = lgb.LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
        
        else:
            logger.warning(f"Unknown method: {method_name}")
            return None
        
        elapsed = time.time() - start
        metrics = compute_metrics(y_te, y_pred, y_prob, s_te)
        
        return {
            'method': method_name,
            'dataset': dataset.name,
            'seed': seed,
            'auc': metrics['auc'],
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'eod': metrics['eod'],
            'dpd': metrics['dpd'],
            'wga': metrics['wga'],
            'n_features': n_feats,
            'time': elapsed,
            'fairness_constraint': fairness_constraint if 'M2FGB' in method_name else '',
            'fair_weight': fair_weight if 'M2FGB' in method_name else '',
        }
    
    except Exception as e:
        logger.error(f"{method_name} on {dataset.name} seed={seed} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_m2fgb_comparison(datasets, output_dir, n_seeds=10, device='cpu'):
    """
    Full comparison: XGBoost vs CausalGBM vs M²FGB variants.
    Tests M²FGB with multiple fairness constraints.
    """
    logger.info("=" * 70)
    logger.info("M²FGB COMPARISON EXPERIMENT")
    logger.info("=" * 70)
    
    if not HAS_M2FGB:
        logger.error("M2FGB not installed! Run: pip install git+https://github.com/hiaac-finance/m2fgb.git")
        return None
    
    # Methods to compare
    methods = [
        ('XGBoost', {}),
        ('LightGBM', {}),
        ('CausalGBM', {}),
        ('M2FGB-EqLoss', {'fairness_constraint': 'equalized_loss', 'fair_weight': 0.5}),
        ('M2FGB-TPR', {'fairness_constraint': 'true_positive_rate', 'fair_weight': 0.5}),
        ('M2FGB-PR', {'fairness_constraint': 'positive_rate', 'fair_weight': 0.5}),
    ]
    
    results = []
    seeds = list(range(n_seeds))
    
    for ds_name, dataset in datasets.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Dataset: {ds_name} (n={len(dataset.X)}, d={dataset.X.shape[1]})")
        logger.info(f"{'='*50}")
        
        for method_name, kwargs in methods:
            method_results = []
            
            for seed in seeds:
                r = run_single_experiment(
                    dataset, method_name, seed,
                    fairness_constraint=kwargs.get('fairness_constraint', 'equalized_loss'),
                    fair_weight=kwargs.get('fair_weight', 0.5),
                    device=device
                )
                if r:
                    results.append(r)
                    method_results.append(r)
            
            if method_results:
                avg_auc = np.mean([r['auc'] for r in method_results])
                avg_eod = np.mean([r['eod'] for r in method_results])
                avg_dpd = np.mean([r['dpd'] for r in method_results])
                avg_time = np.mean([r['time'] for r in method_results])
                logger.info(
                    f"  {method_name:15s}: AUC={avg_auc:.3f}  EOD={avg_eod:.3f}  "
                    f"DPD={avg_dpd:.3f}  Time={avg_time:.2f}s"
                )
    
    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'm2fgb_comparison_raw.csv'), index=False)
    
    # Summary table
    summary = df.groupby(['dataset', 'method']).agg(
        auc_mean=('auc', 'mean'), auc_std=('auc', 'std'),
        eod_mean=('eod', 'mean'), eod_std=('eod', 'std'),
        dpd_mean=('dpd', 'mean'), dpd_std=('dpd', 'std'),
        wga_mean=('wga', 'mean'),
        f1_mean=('f1', 'mean'),
        time_mean=('time', 'mean'),
    ).round(4)
    
    summary.to_csv(os.path.join(output_dir, 'm2fgb_comparison_summary.csv'))
    
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 70)
    
    # Pretty print for each dataset
    for ds_name in df['dataset'].unique():
        logger.info(f"\n--- {ds_name} ---")
        ds_summary = summary.loc[ds_name] if ds_name in summary.index.get_level_values(0) else None
        if ds_summary is not None:
            print(ds_summary[['auc_mean', 'eod_mean', 'dpd_mean', 'wga_mean', 'f1_mean', 'time_mean']].to_string())
    
    # LaTeX table for paper
    _generate_latex_table(df, output_dir)
    
    return df


def _generate_latex_table(df, output_dir):
    """Generate LaTeX table for direct inclusion in paper."""
    
    lines = []
    lines.append("% Auto-generated M2FGB comparison table")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison with fair gradient boosting methods. Bold = best EOD.}")
    lines.append("\\label{tab:m2fgb}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}llcccc@{}}")
    lines.append("\\toprule")
    lines.append("Dataset & Method & AUC$\\uparrow$ & EOD$\\downarrow$ & DPD$\\downarrow$ & Time \\\\")
    lines.append("\\midrule")
    
    for ds_name in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds_name]
        methods_order = ['XGBoost', 'LightGBM', 'CausalGBM', 'M2FGB-EqLoss', 'M2FGB-TPR', 'M2FGB-PR']
        
        best_eod = ds_df.groupby('method')['eod'].mean().min()
        
        first = True
        for method in methods_order:
            m_df = ds_df[ds_df['method'] == method]
            if m_df.empty:
                continue
            
            auc = m_df['auc'].mean()
            eod = m_df['eod'].mean()
            dpd = m_df['dpd'].mean()
            tm = m_df['time'].mean()
            
            eod_str = f"\\textbf{{{eod:.3f}}}" if abs(eod - best_eod) < 0.001 else f"{eod:.3f}"
            
            ds_label = ds_name if first else ""
            first = False
            
            lines.append(f"{ds_label} & {method} & {auc:.3f} & {eod_str} & {dpd:.3f} & {tm:.1f}s \\\\")
        
        lines.append("\\midrule")
    
    # Remove last midrule, replace with bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    latex_str = "\n".join(lines)
    
    with open(os.path.join(output_dir, 'm2fgb_latex_table.tex'), 'w') as f:
        f.write(latex_str)
    
    logger.info(f"\nLaTeX table saved to: {os.path.join(output_dir, 'm2fgb_latex_table.tex')}")
    print("\n" + latex_str)


def main():
    parser = argparse.ArgumentParser(description='M²FGB Comparison for ACML 2026')
    parser.add_argument('--all', action='store_true', help='Run on all datasets')
    parser.add_argument('--datasets', nargs='+',
                       default=['adult', 'acs_income', 'taiwan_credit', 'online_shoppers',
                                'compas', 'german', 'bank', 'synthetic_loan', 'synthetic_hiring'],
                       help='Datasets to use')
    parser.add_argument('--output_dir', default='results/acml2026/m2fgb', help='Output directory')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--n_seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--max_samples', type=int, default=200000, help='Max samples per dataset')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    datasets = {}
    for name in args.datasets:
        if name in DATASET_LOADERS:
            try:
                ds = DATASET_LOADERS[name](max_samples=args.max_samples)
                datasets[name] = ds
                logger.info(f"  Loaded {name}: n={len(ds.X)}, d={ds.X.shape[1]}")
            except Exception as e:
                logger.warning(f"  Could not load {name}: {e}")
    
    if not datasets:
        logger.error("No datasets loaded!")
        return
    
    # Run comparison
    run_m2fgb_comparison(datasets, args.output_dir, n_seeds=args.n_seeds, device=args.device)
    
    logger.info("\nDone! Results in: " + args.output_dir)


if __name__ == '__main__':
    main()
