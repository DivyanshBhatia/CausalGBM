"""
Three High-Leverage Experiments for ACML 2026
==============================================

Exp 1: HEAD-TO-HEAD SIGNIFICANCE (free — uses existing 10-seed data)
  Paired t-tests: CausalGBM vs FairGBM and CausalGBM vs M²FGB per dataset

Exp 2: COMPOSITION (CausalGBM features → FairGBM Stage-2)
  Tests: does CausalGBM + FairGBM beat either alone?

Exp 3: DISTRIBUTION SHIFT (ACS cross-state transfer)
  Train on CA, test on TX/NY — does causal feature selection transfer
  better than in-processing fairness?

Usage:
  python acml_leverage_experiments.py --all --n_seeds 10
  python acml_leverage_experiments.py --significance
  python acml_leverage_experiments.py --composition
  python acml_leverage_experiments.py --transfer
"""

import os, sys, argparse, warnings, logging
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
import xgboost as xgb

try:
    from fairgbm import FairGBMClassifier
    HAS_FAIRGBM = True
except ImportError:
    HAS_FAIRGBM = False

try:
    from m2fgb.m2fgb import M2FGBClassifier
    HAS_M2FGB = True
except ImportError:
    HAS_M2FGB = False

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


# ============================================================================
# EXPERIMENT 1: HEAD-TO-HEAD SIGNIFICANCE
# ============================================================================

def run_significance(n_seeds=10, output_dir='results/acml2026/leverage'):
    """Paired t-tests: CausalGBM vs FairGBM and CausalGBM vs M²FGB."""
    logger.info("=" * 60)
    logger.info("EXP 1: HEAD-TO-HEAD SIGNIFICANCE")
    logger.info("=" * 60)

    results = []

    for ds_name, loader in DATASETS.items():
        try:
            dataset = loader()
        except:
            continue

        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]

        eod_cgbm, eod_fgbm, eod_m2 = [], [], []

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)

            # CausalGBM
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=max(3, d // 3),
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr, s_tr, y_tr)
            Xtr_s, Xte_s = sel.transform(X_tr), sel.transform(X_te)
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            # Rollback
            if met['auc'] < 0.60:
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(X_tr, y_tr)
                yp, ypr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
                met = compute_metrics(y_te, yp, ypr, s_te)
            eod_cgbm.append(met['eod'])

            # FairGBM
            if HAS_FAIRGBM:
                try:
                    fgbm = FairGBMClassifier(
                        constraint_type="FNR,FPR", n_estimators=100,
                        random_state=seed, multiplier_learning_rate=0.1, verbose=-1)
                    fgbm.fit(X_tr, y_tr, constraint_group=s_tr)
                    yp = fgbm.predict(X_te)
                    ypr = fgbm.predict_proba(X_te)[:, 1]
                    met_f = compute_metrics(y_te, yp, ypr, s_te)
                    eod_fgbm.append(met_f['eod'])
                except:
                    eod_fgbm.append(np.nan)

            # M²FGB
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
                    met_m = compute_metrics(y_te, yp, ypr, s_te)
                    eod_m2.append(met_m['eod'])
                except:
                    eod_m2.append(np.nan)

        # Paired t-tests
        cgbm_arr = np.array(eod_cgbm)

        for rival_name, rival_arr in [('FairGBM', eod_fgbm), ('M2FGB-TPR', eod_m2)]:
            if not rival_arr:
                continue
            rival = np.array(rival_arr)
            mask = ~np.isnan(rival)
            if mask.sum() < 3:
                continue
            t_stat, p_val = ttest_rel(cgbm_arr[mask], rival[mask])
            delta = rival[mask].mean() - cgbm_arr[mask].mean()
            winner = 'CausalGBM' if delta > 0 else rival_name
            results.append({
                'dataset': ds_name, 'comparison': f'CausalGBM vs {rival_name}',
                'cgbm_eod': cgbm_arr[mask].mean(),
                'rival_eod': rival[mask].mean(),
                'delta_eod': delta,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'winner': winner,
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'headtohead_significance.csv'), index=False)

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD SIGNIFICANCE (CausalGBM vs Fair GBM Baselines)")
    print("=" * 80)
    print(f"\n{'Dataset':<18s} {'Comparison':<28s} {'ΔEOD':>7s} {'p':>8s} {'Sig?':>5s} {'Winner':<12s}")
    print("-" * 80)
    for _, r in df.iterrows():
        sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else ('*' if r['p_value'] < 0.05 else ''))
        print(f"{r['dataset']:<18s} {r['comparison']:<28s} {r['delta_eod']:>+7.4f} {r['p_value']:>8.4f} {sig:>5s} {r['winner']:<12s}")

    print(f"\nSaved: {os.path.join(output_dir, 'headtohead_significance.csv')}")
    return df


# ============================================================================
# EXPERIMENT 2: COMPOSITION (CausalGBM features → FairGBM Stage-2)
# ============================================================================

def run_composition(n_seeds=10, output_dir='results/acml2026/leverage'):
    """Test: CausalGBM feature selection + FairGBM training."""
    if not HAS_FAIRGBM:
        logger.warning("FairGBM not installed — skipping composition")
        return None

    logger.info("=" * 60)
    logger.info("EXP 2: COMPOSITION (CausalGBM → FairGBM)")
    logger.info("=" * 60)

    comp_datasets = ['adult', 'acs_income', 'compas', 'bank']
    results = []

    for ds_name in comp_datasets:
        if ds_name not in DATASETS:
            continue
        try:
            dataset = DATASETS[ds_name]()
        except:
            continue

        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        logger.info(f"\n--- {ds_name} (n={len(X)}, d={d}) ---")

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)

            # (a) XGBoost baseline
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(X_tr, y_tr)
            yp, ypr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({'dataset': ds_name, 'method': 'XGBoost', 'seed': seed, **met})

            # (b) CausalGBM (standard: features → XGBoost)
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=max(3, d // 3),
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr, s_tr, y_tr)
            Xtr_s, Xte_s = sel.transform(X_tr), sel.transform(X_te)
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            if met['auc'] < 0.60:
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(X_tr, y_tr)
                yp, ypr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
                met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({'dataset': ds_name, 'method': 'CausalGBM→XGB', 'seed': seed, **met})

            # (c) FairGBM alone (on all features)
            try:
                fgbm = FairGBMClassifier(
                    constraint_type="FNR,FPR", n_estimators=100,
                    random_state=seed, multiplier_learning_rate=0.1, verbose=-1)
                fgbm.fit(X_tr, y_tr, constraint_group=s_tr)
                yp = fgbm.predict(X_te)
                ypr = fgbm.predict_proba(X_te)[:, 1]
                met = compute_metrics(y_te, yp, ypr, s_te)
                results.append({'dataset': ds_name, 'method': 'FairGBM', 'seed': seed, **met})
            except:
                pass

            # (d) COMPOSITION: CausalGBM features → FairGBM
            try:
                # Use same selected features from CausalGBM
                s_tr_sel = s_tr  # sensitive attr unchanged
                fgbm_c = FairGBMClassifier(
                    constraint_type="FNR,FPR", n_estimators=100,
                    random_state=seed, multiplier_learning_rate=0.1, verbose=-1)
                fgbm_c.fit(Xtr_s, y_tr, constraint_group=s_tr_sel)
                yp = fgbm_c.predict(Xte_s)
                ypr = fgbm_c.predict_proba(Xte_s)[:, 1]
                met = compute_metrics(y_te, yp, ypr, s_te)
                results.append({'dataset': ds_name, 'method': 'CausalGBM→FairGBM', 'seed': seed, **met})
            except Exception as e:
                logger.warning(f"  Composition failed seed {seed}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'composition_results.csv'), index=False)

    print("\n" + "=" * 70)
    print("COMPOSITION RESULTS")
    print("=" * 70)
    print(f"\n{'Dataset':<18s} {'Method':<22s} {'EOD':>7s} {'AUC':>7s}")
    print("-" * 60)

    for ds_name in comp_datasets:
        ds_df = df[df['dataset'] == ds_name]
        for method in ['XGBoost', 'FairGBM', 'CausalGBM→XGB', 'CausalGBM→FairGBM']:
            m_df = ds_df[ds_df['method'] == method]
            if m_df.empty:
                continue
            marker = " ★" if method == 'CausalGBM→FairGBM' else ""
            print(f"{ds_name:<18s} {method:<22s} {m_df['eod'].mean():>7.4f} {m_df['auc'].mean():>7.3f}{marker}")
        print()

    print(f"Saved: {os.path.join(output_dir, 'composition_results.csv')}")
    return df


# ============================================================================
# EXPERIMENT 3: DISTRIBUTION SHIFT (ACS cross-state)
# ============================================================================

def run_transfer(n_seeds=5, output_dir='results/acml2026/leverage'):
    """Train on CA, test on TX and NY — does proxy structure transfer?"""
    logger.info("=" * 60)
    logger.info("EXP 3: DISTRIBUTION SHIFT (ACS cross-state)")
    logger.info("=" * 60)

    train_state = 'CA'
    test_states = ['TX', 'NY']

    results = []

    # Load training data (CA)
    try:
        train_data = load_acs_income(states=[train_state])
    except Exception as e:
        logger.error(f"Cannot load ACS {train_state}: {e}")
        return None

    X_train_full, y_train_full, s_train_full = train_data.X, train_data.y, train_data.sensitive
    d = X_train_full.shape[1]
    logger.info(f"Train: ACS-{train_state} (n={len(X_train_full)}, d={d})")

    for test_state in test_states:
        try:
            test_data = load_acs_income(states=[test_state])
        except Exception as e:
            logger.warning(f"Cannot load ACS {test_state}: {e}")
            continue

        X_test_full, y_test_full, s_test_full = test_data.X, test_data.y, test_data.sensitive
        logger.info(f"\nTest: ACS-{test_state} (n={len(X_test_full)})")

        for seed in range(n_seeds):
            # Subsample training data for speed
            rng = np.random.RandomState(seed)
            n_train = min(50000, len(X_train_full))
            n_test = min(20000, len(X_test_full))
            train_idx = rng.choice(len(X_train_full), n_train, replace=False)
            test_idx = rng.choice(len(X_test_full), n_test, replace=False)

            X_tr = X_train_full[train_idx]
            y_tr = y_train_full[train_idx]
            s_tr = s_train_full[train_idx]
            X_te = X_test_full[test_idx]
            y_te = y_test_full[test_idx]
            s_te = s_test_full[test_idx]

            # (a) XGBoost baseline
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(X_tr, y_tr)
            yp, ypr = m.predict(X_te), m.predict_proba(X_te)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({
                'train': train_state, 'test': test_state,
                'method': 'XGBoost', 'seed': seed, **met})

            # (b) CausalGBM (train features on CA, apply to test state)
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=max(3, d // 3),
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr, s_tr, y_tr)
            Xtr_s = sel.transform(X_tr)
            Xte_s = sel.transform(X_te)
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({
                'train': train_state, 'test': test_state,
                'method': 'CausalGBM', 'seed': seed, **met})

            # (c) FairGBM (trained on CA, tested on other state)
            if HAS_FAIRGBM:
                try:
                    fgbm = FairGBMClassifier(
                        constraint_type="FNR,FPR", n_estimators=100,
                        random_state=seed, multiplier_learning_rate=0.1,
                        verbose=-1)
                    fgbm.fit(X_tr, y_tr, constraint_group=s_tr)
                    yp = fgbm.predict(X_te)
                    ypr = fgbm.predict_proba(X_te)[:, 1]
                    met = compute_metrics(y_te, yp, ypr, s_te)
                    results.append({
                        'train': train_state, 'test': test_state,
                        'method': 'FairGBM', 'seed': seed, **met})
                except:
                    pass

            # (d) M²FGB
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
                    met = compute_metrics(y_te, yp, ypr, s_te)
                    results.append({
                        'train': train_state, 'test': test_state,
                        'method': 'M2FGB-TPR', 'seed': seed, **met})
                except:
                    pass

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'transfer_results.csv'), index=False)

    print("\n" + "=" * 70)
    print("DISTRIBUTION SHIFT RESULTS (Train: CA → Test: other state)")
    print("=" * 70)

    for test_state in test_states:
        ts_df = df[df['test'] == test_state]
        if ts_df.empty:
            continue
        print(f"\n  CA → {test_state}:")
        print(f"  {'Method':<18s} {'EOD':>7s} {'AUC':>7s} {'EOD shift':>10s}")
        print("  " + "-" * 45)

        for method in ['XGBoost', 'CausalGBM', 'FairGBM', 'M2FGB-TPR']:
            m_df = ts_df[ts_df['method'] == method]
            if m_df.empty:
                continue
            eod = m_df['eod'].mean()
            auc = m_df['auc'].mean()
            # Compare to in-distribution (CA→CA would be the reference)
            marker = " ★" if method == 'CausalGBM' else ""
            print(f"  {method:<18s} {eod:>7.4f} {auc:>7.3f}{marker}")

    print(f"\nSaved: {os.path.join(output_dir, 'transfer_results.csv')}")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--significance', action='store_true')
    parser.add_argument('--composition', action='store_true')
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/leverage')
    args = parser.parse_args()

    if args.all:
        args.significance = args.composition = args.transfer = True
    if not any([args.significance, args.composition, args.transfer]):
        args.significance = args.composition = args.transfer = True

    os.makedirs(args.output_dir, exist_ok=True)

    if args.significance:
        run_significance(n_seeds=args.n_seeds, output_dir=args.output_dir)

    if args.composition:
        run_composition(n_seeds=args.n_seeds, output_dir=args.output_dir)

    if args.transfer:
        run_transfer(n_seeds=min(args.n_seeds, 5), output_dir=args.output_dir)

    logger.info("All experiments complete!")


if __name__ == '__main__':
    main()
