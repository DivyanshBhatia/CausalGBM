"""
Complementarity: Pareto Frontier Comparison
=============================================
Compares Pareto-efficient frontiers (best EOD at each AUC level):
  - M²FGB alone (full features)
  - CausalGBM + M²FGB (CausalGBM's features)
  - FairGBM alone
  - CausalGBM + FairGBM

Filters out collapsed models (AUC < 0.60).
Reports: does the combined frontier dominate at matched AUC?

Usage: python acml_complementarity_pareto.py --n_seeds 10
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

try:
    from fairgbm import FairGBMClassifier
    HAS_FAIRGBM = True
except ImportError:
    HAS_FAIRGBM = False

GAMMA_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
LAMBDA_GRID = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

DATASETS = {
    'acs_income': load_acs_income,
    'adult': load_adult,
}

MIN_AUC = 0.60  # Filter out collapsed models


def pareto_frontier(points):
    """Extract Pareto-efficient points (maximize AUC, minimize EOD).
    Returns sorted by AUC ascending."""
    # Filter collapsed
    points = [(a, e) for a, e in points if a >= MIN_AUC]
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: -p[0])  # AUC descending
    frontier = []
    best_eod = float('inf')
    for auc, eod in sorted_pts:
        if eod < best_eod:
            frontier.append((auc, eod))
            best_eod = eod
    return sorted(frontier, key=lambda p: p[0])  # AUC ascending


def interpolate_eod(frontier, target_auc):
    """Interpolate EOD at target_auc on Pareto frontier."""
    if not frontier:
        return None
    aucs = [p[0] for p in frontier]
    eods = [p[1] for p in frontier]
    if target_auc < min(aucs) or target_auc > max(aucs):
        return None
    for i in range(len(aucs) - 1):
        if aucs[i] <= target_auc <= aucs[i + 1]:
            t = (target_auc - aucs[i]) / (aucs[i + 1] - aucs[i] + 1e-10)
            return eods[i] + t * (eods[i + 1] - eods[i])
    return eods[-1]


def run(n_seeds=10, output_dir='results/acml2026/complementarity_pareto'):
    os.makedirs(output_dir, exist_ok=True)
    all_points = []  # (dataset, method, seed, gamma, auc, eod)

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

            def eval_model(model, Xtr, Xte):
                model.fit(Xtr, y_tr) if not hasattr(model, 'constraint_type') else None
                yp = model.predict(Xte)
                ypr = model.predict_proba(Xte)[:, 1]
                return compute_metrics(y_te, yp, ypr, s_te)

            # M²FGB sweeps
            if HAS_M2FGB:
                for gam in GAMMA_GRID:
                    for method, Xtr, Xte in [('M2FGB-alone', X_tr_sc, X_te_sc),
                                              ('CGBM+M2FGB', Xtr_cgbm, Xte_cgbm)]:
                        try:
                            if gam == 0.0:
                                mm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                                mm.fit(Xtr, y_tr)
                            else:
                                mm = M2FGBClassifier(
                                    fairness_constraint='true_positive_rate',
                                    fair_weight=gam, n_estimators=100,
                                    learning_rate=0.1, multiplier_learning_rate=0.1,
                                    random_state=seed)
                                mm.fit(Xtr, y_tr, sensitive_attribute=s_tr)
                            met = compute_metrics(y_te, mm.predict(Xte),
                                                  mm.predict_proba(Xte)[:, 1], s_te)
                            all_points.append({
                                'dataset': ds_name, 'method': method,
                                'seed': seed, 'gamma': gam, **met})
                        except:
                            pass

            # FairGBM sweeps
            if HAS_FAIRGBM:
                for lam in LAMBDA_GRID:
                    for method, Xtr, Xte in [('FairGBM-alone', X_tr_sc, X_te_sc),
                                              ('CGBM+FairGBM', Xtr_cgbm, Xte_cgbm)]:
                        try:
                            if lam == 0.0:
                                fm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                                fm.fit(Xtr, y_tr)
                            else:
                                fm = FairGBMClassifier(
                                    constraint_type="FNR,FPR", n_estimators=100,
                                    random_state=seed, multiplier_learning_rate=lam, verbose=-1)
                                fm.fit(Xtr, y_tr, constraint_group=s_tr)
                            met = compute_metrics(y_te, fm.predict(Xte),
                                                  fm.predict_proba(Xte)[:, 1], s_te)
                            all_points.append({
                                'dataset': ds_name, 'method': method,
                                'seed': seed, 'gamma': lam, **met})
                        except:
                            pass

    import pandas as pd
    df = pd.DataFrame(all_points)
    df.to_csv(os.path.join(output_dir, 'complementarity_pareto.csv'), index=False)

    # Pareto comparison
    print("\n" + "=" * 80)
    print("PARETO FRONTIER COMPARISON (collapsed models filtered, AUC ≥ 0.60)")
    print("=" * 80)

    for ds_name in DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        print(f"\n  {ds_name}:")

        # Build per-method Pareto frontiers (mean over seeds per γ)
        frontiers = {}
        for method in ['M2FGB-alone', 'CGBM+M2FGB', 'FairGBM-alone', 'CGBM+FairGBM']:
            m_df = ds_df[ds_df['method'] == method]
            if m_df.empty:
                continue
            points = []
            for g in m_df['gamma'].unique():
                g_df = m_df[m_df['gamma'] == g]
                auc_mean = g_df['auc'].mean()
                eod_mean = g_df['eod'].mean()
                points.append((auc_mean, eod_mean))
            frontiers[method] = pareto_frontier(points)

        # Print frontiers
        for method, frontier in frontiers.items():
            print(f"\n    {method} Pareto frontier ({len(frontier)} points):")
            print(f"    {'AUC':>7s} {'EOD':>7s}")
            for auc, eod in frontier:
                print(f"    {auc:>7.3f} {eod:>7.4f}")

        # Compare at matched AUC levels
        print(f"\n    --- Matched-AUC comparison ---")
        if 'M2FGB-alone' in frontiers and 'CGBM+M2FGB' in frontiers:
            # Sample AUC points from the overlap region
            alone_aucs = [p[0] for p in frontiers['M2FGB-alone']]
            combo_aucs = [p[0] for p in frontiers['CGBM+M2FGB']]
            if alone_aucs and combo_aucs:
                lo = max(min(alone_aucs), min(combo_aucs))
                hi = min(max(alone_aucs), max(combo_aucs))
                if lo < hi:
                    test_aucs = np.linspace(lo, hi, 5)
                    combo_wins = 0
                    print(f"    {'AUC':>7s} {'M2FGB-alone':>12s} {'CGBM+M2FGB':>12s} {'Winner':>10s}")
                    for ta in test_aucs:
                        e_alone = interpolate_eod(frontiers['M2FGB-alone'], ta)
                        e_combo = interpolate_eod(frontiers['CGBM+M2FGB'], ta)
                        if e_alone is not None and e_combo is not None:
                            winner = 'COMBO' if e_combo < e_alone - 0.002 else (
                                'ALONE' if e_alone < e_combo - 0.002 else 'TIE')
                            if winner == 'COMBO':
                                combo_wins += 1
                            print(f"    {ta:>7.3f} {e_alone:>12.4f} {e_combo:>12.4f} {winner:>10s}")
                    print(f"    → CGBM+M2FGB dominates at {combo_wins}/{len(test_aucs)} AUC levels")
                else:
                    print("    No overlapping AUC range")

        if 'FairGBM-alone' in frontiers and 'CGBM+FairGBM' in frontiers:
            alone_aucs = [p[0] for p in frontiers['FairGBM-alone']]
            combo_aucs = [p[0] for p in frontiers['CGBM+FairGBM']]
            if alone_aucs and combo_aucs:
                lo = max(min(alone_aucs), min(combo_aucs))
                hi = min(max(alone_aucs), max(combo_aucs))
                if lo < hi:
                    test_aucs = np.linspace(lo, hi, 5)
                    combo_wins = 0
                    print(f"\n    {'AUC':>7s} {'FairGBM-alone':>14s} {'CGBM+FairGBM':>14s} {'Winner':>10s}")
                    for ta in test_aucs:
                        e_alone = interpolate_eod(frontiers['FairGBM-alone'], ta)
                        e_combo = interpolate_eod(frontiers['CGBM+FairGBM'], ta)
                        if e_alone is not None and e_combo is not None:
                            winner = 'COMBO' if e_combo < e_alone - 0.002 else (
                                'ALONE' if e_alone < e_combo - 0.002 else 'TIE')
                            if winner == 'COMBO':
                                combo_wins += 1
                            print(f"    {ta:>7.3f} {e_alone:>14.4f} {e_combo:>14.4f} {winner:>10s}")
                    print(f"    → CGBM+FairGBM dominates at {combo_wins}/{len(test_aucs)} AUC levels")

    print(f"\nSaved: {os.path.join(output_dir, 'complementarity_pareto.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/complementarity_pareto')
    args = parser.parse_args()
    run(n_seeds=args.n_seeds, output_dir=args.output_dir)
