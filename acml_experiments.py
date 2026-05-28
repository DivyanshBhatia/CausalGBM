"""
ACML 2026 Additional Experiments for CausalGBM
===============================================
Addresses ECML-PKDD reviewer feedback:
  1. Pareto-front analysis (R3): vary alpha, plot full tradeoff curve
  2. FairGBM + M2FGB baselines (R2, R3): add missing fair GBM comparisons  
  3. 10-seed rerun (R1): increase statistical power

Usage:
  python acml_experiments.py --pareto              # Pareto-front only
  python acml_experiments.py --fairgbm             # FairGBM comparison only
  python acml_experiments.py --rerun_seeds          # 10-seed rerun
  python acml_experiments.py --all                  # Everything
  python acml_experiments.py --pareto --datasets adult acs_income

Requirements:
  pip install fairgbm                              # For FairGBM baseline
  (Other deps same as existing requirements.txt)

Author: CausalGBM Team
"""

import os
import sys
import time
import argparse
import warnings
import logging
import traceback
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    DatasetBundle
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

# ============================================================================
# DATASET LOADERS
# ============================================================================

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

def load_datasets(dataset_names, max_samples=200000):
    """Load specified datasets."""
    datasets = {}
    for name in dataset_names:
        if name in DATASET_LOADERS:
            try:
                ds = DATASET_LOADERS[name](max_samples=max_samples)
                datasets[name] = ds
                logger.info(f"Loaded {name}: n={len(ds.X)}, d={ds.X.shape[1]}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
    return datasets


def run_single(dataset, method_name, seed, alpha=0.5, threshold=0.2,
               aggregation='max', device='cpu'):
    """Run a single experiment and return metrics dict."""
    X, y, sens = dataset.X, dataset.y, dataset.sensitive
    
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sens, test_size=0.3, random_state=seed, stratify=y
    )
    
    start = time.time()
    
    if method_name == 'XGBoost':
        model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else \
                GradientBoostingClassifier(n_estimators=100, random_state=seed)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        n_feats = X_tr.shape[1]
    
    elif method_name == 'CausalGBM':
        selector = CausalFeatureSelector(
            X_tr.shape[1], alpha=alpha, threshold=threshold,
            min_features=max(3, X_tr.shape[1] // 3),
            n_iterations=500, aggregation=aggregation, device=device
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
    
    elif method_name == 'FairGBM':
        try:
            from fairgbm import FairGBMClassifier
            model = FairGBMClassifier(
                constraint_type="FNR,FPR",
                n_estimators=100,
                random_state=seed,
                multiplier_learning_rate=0.1,
                verbose=-1
            )
            model.fit(X_tr, y_tr, constraint_group=s_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
            n_feats = X_tr.shape[1]
        except ImportError:
            logger.error("FairGBM not installed. Run: pip install fairgbm")
            return None
        except Exception as e:
            logger.error(f"FairGBM failed on {dataset.name}: {e}")
            return None
    
    else:
        logger.warning(f"Unknown method: {method_name}")
        return None
    
    elapsed = time.time() - start
    metrics = compute_metrics(y_te, y_pred, y_prob, s_te)
    
    return {
        'method': method_name,
        'dataset': dataset.name,
        'seed': seed,
        'alpha': alpha,
        'auc': metrics['auc'],
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'eod': metrics['eod'],
        'dpd': metrics['dpd'],
        'wga': metrics['wga'],
        'n_features': n_feats,
        'time': elapsed,
    }


# ============================================================================
# EXPERIMENT 1: PARETO-FRONT ANALYSIS
# ============================================================================

def run_pareto_front(datasets, output_dir, seeds=range(5), device='cpu'):
    """
    Vary alpha to produce Pareto frontier for CausalGBM.
    Plots AUC vs EOD with baselines as single points and CausalGBM as a curve.
    
    Addresses ECML R3: "lacks a Pareto-front analysis varying hyperparameters"
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: PARETO-FRONT ANALYSIS")
    logger.info("=" * 70)
    
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    for ds_name, dataset in datasets.items():
        logger.info(f"\n--- {ds_name} ---")
        
        # Run XGBoost baseline
        for seed in seeds:
            r = run_single(dataset, 'XGBoost', seed, device=device)
            if r: results.append(r)
        
        # Run CausalGBM at each alpha
        for alpha in alphas:
            for seed in seeds:
                r = run_single(dataset, 'CausalGBM', seed, alpha=alpha, device=device)
                if r:
                    r['alpha'] = alpha
                    results.append(r)
                    logger.info(f"  α={alpha:.2f}, seed={seed}: AUC={r['auc']:.3f}, EOD={r['eod']:.3f}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'pareto_front_results.csv'), index=False)
    
    # Generate Pareto frontier plots
    _plot_pareto(df, output_dir)
    
    return df


def _plot_pareto(df, output_dir):
    """Generate Pareto frontier scatter plots."""
    dataset_names = df['dataset'].unique()
    
    for ds_name in dataset_names:
        ds_df = df[df['dataset'] == ds_name]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # XGBoost baseline (single point)
        xgb_df = ds_df[ds_df['method'] == 'XGBoost']
        if not xgb_df.empty:
            ax.scatter(
                xgb_df['auc'].mean(), xgb_df['eod'].mean(),
                c='blue', s=120, marker='s', zorder=5, edgecolors='black',
                label=f"XGBoost (EOD={xgb_df['eod'].mean():.3f})"
            )
        
        # CausalGBM Pareto curve
        cgbm_df = ds_df[ds_df['method'] == 'CausalGBM']
        if not cgbm_df.empty:
            # Average across seeds for each alpha
            pareto = cgbm_df.groupby('alpha').agg(
                auc_mean=('auc', 'mean'),
                auc_std=('auc', 'std'),
                eod_mean=('eod', 'mean'),
                eod_std=('eod', 'std'),
            ).reset_index()
            
            # Sort by alpha for line
            pareto = pareto.sort_values('alpha')
            
            # Plot line
            ax.plot(pareto['auc_mean'], pareto['eod_mean'],
                    'r-o', markersize=8, linewidth=2, zorder=4,
                    label='CausalGBM (varying α)')
            
            # Error bars
            ax.errorbar(pareto['auc_mean'], pareto['eod_mean'],
                       xerr=pareto['auc_std'], yerr=pareto['eod_std'],
                       fmt='none', color='red', alpha=0.3, capsize=3)
            
            # Annotate alpha values
            for _, row in pareto.iterrows():
                alpha_val = row['alpha']
                label = f"α={alpha_val}"
                if alpha_val == 0.5:
                    label += " (default)"
                    ax.scatter(row['auc_mean'], row['eod_mean'],
                              c='red', s=200, marker='*', zorder=6, edgecolors='black')
                
                ax.annotate(label, 
                           (row['auc_mean'], row['eod_mean']),
                           textcoords="offset points", xytext=(8, 5),
                           fontsize=7, color='darkred')
        
        ax.set_xlabel('AUC (higher is better) →', fontsize=12)
        ax.set_ylabel('EOD (lower is fairer) ↓', fontsize=12)
        ax.set_title(f'Accuracy-Fairness Tradeoff: {ds_name}', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pareto_front_{ds_name}.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Pareto plot for {ds_name}")


# ============================================================================
# EXPERIMENT 2: FAIRGBM COMPARISON
# ============================================================================

def run_fairgbm_comparison(datasets, output_dir, seeds=range(10), device='cpu'):
    """
    Compare CausalGBM against FairGBM on all datasets.
    
    Addresses ECML R2, R3: "FairGBM is the most natural direct competitor"
    
    Install: pip install fairgbm
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: FAIRGBM COMPARISON")
    logger.info("=" * 70)
    
    try:
        from fairgbm import FairGBMClassifier
        logger.info("FairGBM installed successfully")
    except ImportError:
        logger.error("FairGBM not installed! Run: pip install fairgbm")
        logger.error("See: https://github.com/microsoft/fairgbm")
        return None
    
    methods = ['XGBoost', 'CausalGBM', 'FairGBM']
    results = []
    
    for ds_name, dataset in datasets.items():
        logger.info(f"\n--- {ds_name} ---")
        
        for method in methods:
            for seed in seeds:
                r = run_single(dataset, method, seed, device=device)
                if r:
                    results.append(r)
                    logger.info(f"  {method}, seed={seed}: AUC={r['auc']:.3f}, EOD={r['eod']:.3f}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'fairgbm_comparison_results.csv'), index=False)
    
    # Print summary table
    summary = df.groupby(['dataset', 'method']).agg(
        auc=('auc', 'mean'),
        eod=('eod', 'mean'),
        dpd=('dpd', 'mean'),
        f1=('f1', 'mean'),
        wga=('wga', 'mean'),
    ).round(3)
    
    logger.info("\n" + "=" * 70)
    logger.info("FAIRGBM COMPARISON SUMMARY")
    logger.info("=" * 70)
    print(summary.to_string())
    summary.to_csv(os.path.join(output_dir, 'fairgbm_comparison_summary.csv'))
    
    return df


# ============================================================================
# EXPERIMENT 3: 10-SEED RERUN
# ============================================================================

def run_10seed_rerun(datasets, output_dir, n_seeds=10, device='cpu'):
    """
    Rerun main experiments with 10 seeds for stronger statistical claims.
    
    Addresses ECML R1: "each experiment is repeated only five times"
    """
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT 3: {n_seeds}-SEED RERUN")
    logger.info("=" * 70)
    
    seeds = range(n_seeds)
    methods = ['XGBoost', 'CausalGBM']
    results = []
    
    for ds_name, dataset in datasets.items():
        logger.info(f"\n--- {ds_name} ---")
        
        for method in methods:
            for seed in seeds:
                r = run_single(dataset, method, seed, device=device)
                if r:
                    results.append(r)
        
        # Compute significance
        ds_results = [r for r in results if r['dataset'] == ds_name]
        xgb_eods = [r['eod'] for r in ds_results if r['method'] == 'XGBoost']
        cgbm_eods = [r['eod'] for r in ds_results if r['method'] == 'CausalGBM']
        
        if len(xgb_eods) >= 2 and len(cgbm_eods) >= 2:
            from scipy.stats import ttest_rel
            min_len = min(len(xgb_eods), len(cgbm_eods))
            t_stat, p_val = ttest_rel(xgb_eods[:min_len], cgbm_eods[:min_len])
            
            delta = np.mean(xgb_eods[:min_len]) - np.mean(cgbm_eods[:min_len])
            pooled_std = np.std(np.array(xgb_eods[:min_len]) - np.array(cgbm_eods[:min_len]))
            cohens_d = delta / pooled_std if pooled_std > 0 else 0
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            logger.info(f"  {ds_name}: ΔEOD={delta:.3f}, d={cohens_d:.2f}, p={p_val:.4f} ({sig})")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f'rerun_{n_seeds}seeds_results.csv'), index=False)
    
    # Summary with significance
    _print_significance_table(df, n_seeds, output_dir)
    
    return df


def _print_significance_table(df, n_seeds, output_dir):
    """Print formatted significance table."""
    from scipy.stats import ttest_rel
    
    rows = []
    for ds_name in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds_name]
        xgb = ds_df[ds_df['method'] == 'XGBoost']['eod'].values
        cgbm = ds_df[ds_df['method'] == 'CausalGBM']['eod'].values
        
        min_len = min(len(xgb), len(cgbm))
        if min_len < 2:
            continue
        
        xgb, cgbm = xgb[:min_len], cgbm[:min_len]
        delta_eod = np.mean(xgb) - np.mean(cgbm)
        diffs = xgb - cgbm
        cohens_d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
        t_stat, p_val = ttest_rel(xgb, cgbm)
        
        sig = "Yes***" if p_val < 0.001 else "Yes**" if p_val < 0.01 else \
              "Yes*" if p_val < 0.05 else "No"
        
        rows.append({
            'Dataset': ds_name,
            'ΔEOD': f"{delta_eod:.3f}",
            "Cohen's d": f"{cohens_d:.2f}",
            'p-value': f"{p_val:.4f}" if p_val >= 0.001 else "<0.001",
            'Significant?': sig,
            'Seeds': n_seeds,
        })
    
    sig_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(f"STATISTICAL SIGNIFICANCE ({n_seeds} seeds)")
    print("=" * 70)
    print(sig_df.to_string(index=False))
    sig_df.to_csv(os.path.join(output_dir, f'significance_{n_seeds}seeds.csv'), index=False)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ACML 2026 Additional Experiments')
    parser.add_argument('--pareto', action='store_true', help='Run Pareto-front analysis')
    parser.add_argument('--fairgbm', action='store_true', help='Run FairGBM comparison')
    parser.add_argument('--rerun_seeds', action='store_true', help='Rerun with 10 seeds')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--datasets', nargs='+', 
                       default=['adult', 'acs_income', 'taiwan_credit', 'online_shoppers',
                                'compas', 'german', 'bank', 'synthetic_loan', 'synthetic_hiring'],
                       help='Datasets to use')
    parser.add_argument('--output_dir', default='results/acml2026', help='Output directory')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--n_seeds', type=int, default=10, help='Number of seeds for rerun')
    parser.add_argument('--max_samples', type=int, default=200000, help='Max samples per dataset')
    
    args = parser.parse_args()
    
    if args.all:
        args.pareto = True
        args.fairgbm = True
        args.rerun_seeds = True
    
    if not (args.pareto or args.fairgbm or args.rerun_seeds):
        parser.print_help()
        print("\nPlease specify at least one experiment: --pareto, --fairgbm, --rerun_seeds, or --all")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    datasets = load_datasets(args.datasets, max_samples=args.max_samples)
    
    if not datasets:
        logger.error("No datasets loaded!")
        return
    
    # Run experiments
    if args.pareto:
        pareto_dir = os.path.join(args.output_dir, 'pareto')
        os.makedirs(pareto_dir, exist_ok=True)
        # Use only key datasets for Pareto analysis
        pareto_datasets = {k: v for k, v in datasets.items() 
                         if k in ['adult', 'acs_income', 'taiwan_credit', 'online_shoppers']}
        if pareto_datasets:
            run_pareto_front(pareto_datasets, pareto_dir, 
                           seeds=range(args.n_seeds), device=args.device)
    
    if args.fairgbm:
        fairgbm_dir = os.path.join(args.output_dir, 'fairgbm')
        os.makedirs(fairgbm_dir, exist_ok=True)
        run_fairgbm_comparison(datasets, fairgbm_dir,
                              seeds=range(args.n_seeds), device=args.device)
    
    if args.rerun_seeds:
        rerun_dir = os.path.join(args.output_dir, 'rerun')
        os.makedirs(rerun_dir, exist_ok=True)
        run_10seed_rerun(datasets, rerun_dir, 
                        n_seeds=args.n_seeds, device=args.device)
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
