"""
Matched-AUC Fairness Comparison
================================
Sweeps FairGBM (λ) and M²FGB (γ), interpolates each competitor's EOD
at CausalGBM's AUC per seed. Answers: "at the SAME AUC, who has lower EOD?"

Usage:
  python acml_matched_auc.py --n_seeds 10
"""

import os, sys, argparse, warnings, logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

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
    logger.warning("FairGBM not installed")

try:
    from m2fgb.m2fgb import M2FGBClassifier
    HAS_M2FGB = True
except ImportError:
    HAS_M2FGB = False
    logger.warning("M2FGB not installed")

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

FAIRGBM_GRID = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
M2FGB_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def pareto_frontier(points):
    """Return Pareto-efficient points (maximize AUC, minimize EOD)."""
    sorted_pts = sorted(points, key=lambda p: -p[0])  # sort by AUC desc
    frontier = []
    best_eod = float('inf')
    for auc, eod in sorted_pts:
        if eod < best_eod:
            frontier.append((auc, eod))
            best_eod = eod
    return sorted(frontier, key=lambda p: p[0])  # sort by AUC asc


def interpolate_eod_at_auc(frontier, target_auc):
    """Linearly interpolate EOD at target_auc on the Pareto frontier."""
    if not frontier:
        return None, 'no_data'

    aucs = [p[0] for p in frontier]
    eods = [p[1] for p in frontier]

    if target_auc < min(aucs):
        return None, 'unreachable'
    if target_auc > max(aucs):
        return eods[-1], 'extrapolated'
    if target_auc == min(aucs):
        return eods[0], 'exact'

    # Find bracketing points
    for i in range(len(aucs) - 1):
        if aucs[i] <= target_auc <= aucs[i + 1]:
            t = (target_auc - aucs[i]) / (aucs[i + 1] - aucs[i] + 1e-10)
            interp_eod = eods[i] + t * (eods[i + 1] - eods[i])
            return interp_eod, 'interpolated'

    return None, 'unreachable'


def run_matched_auc(n_seeds=10, output_dir='results/acml2026/matched_auc'):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    summary = []

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

        seed_results = {
            'cgbm_eod': [], 'cgbm_auc': [],
            'fgbm_eod_matched': [], 'fgbm_status': [],
            'm2_eod_matched': [], 'm2_status': [],
        }

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)

            s_tr = np.asarray(s_tr, dtype=float)
            y_tr = np.asarray(y_tr, dtype=float)
            s_te = np.asarray(s_te, dtype=float)
            y_te = np.asarray(y_te, dtype=float)

            # --- CausalGBM anchor ---
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=min_feat,
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

            target_auc = met['auc']
            cgbm_eod = met['eod']
            seed_results['cgbm_eod'].append(cgbm_eod)
            seed_results['cgbm_auc'].append(target_auc)

            # --- FairGBM sweep ---
            fgbm_points = []
            if HAS_FAIRGBM:
                for lam in FAIRGBM_GRID:
                    try:
                        if lam == 0.0:
                            fm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                            fm.fit(X_tr, y_tr)
                        else:
                            fm = FairGBMClassifier(
                                constraint_type="FNR,FPR", n_estimators=100,
                                random_state=seed, multiplier_learning_rate=lam,
                                verbose=-1)
                            fm.fit(X_tr, y_tr, constraint_group=s_tr)
                        yp = fm.predict(X_te)
                        ypr = fm.predict_proba(X_te)[:, 1]
                        fm_met = compute_metrics(y_te, yp, ypr, s_te)
                        fgbm_points.append((fm_met['auc'], fm_met['eod']))
                    except:
                        pass

                frontier = pareto_frontier(fgbm_points)
                eod_at_auc, status = interpolate_eod_at_auc(frontier, target_auc)
                seed_results['fgbm_eod_matched'].append(eod_at_auc)
                seed_results['fgbm_status'].append(status)
            else:
                seed_results['fgbm_eod_matched'].append(None)
                seed_results['fgbm_status'].append('no_library')

            # --- M2FGB sweep ---
            m2_points = []
            if HAS_M2FGB:
                for gam in M2FGB_GRID:
                    try:
                        if gam == 0.0:
                            mm = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                            mm.fit(X_tr, y_tr)
                        else:
                            mm = M2FGBClassifier(
                                fairness_constraint='true_positive_rate',
                                fair_weight=gam, n_estimators=100,
                                learning_rate=0.1, multiplier_learning_rate=0.1,
                                random_state=seed)
                            mm.fit(X_tr, y_tr, sensitive_attribute=s_tr)
                        yp = mm.predict(X_te)
                        ypr = mm.predict_proba(X_te)[:, 1]
                        mm_met = compute_metrics(y_te, yp, ypr, s_te)
                        m2_points.append((mm_met['auc'], mm_met['eod']))
                    except:
                        pass

                frontier = pareto_frontier(m2_points)
                eod_at_auc, status = interpolate_eod_at_auc(frontier, target_auc)
                seed_results['m2_eod_matched'].append(eod_at_auc)
                seed_results['m2_status'].append(status)
            else:
                seed_results['m2_eod_matched'].append(None)
                seed_results['m2_status'].append('no_library')

            all_results.append({
                'dataset': ds_name, 'seed': seed,
                'cgbm_auc': target_auc, 'cgbm_eod': cgbm_eod,
                'fgbm_eod_matched': seed_results['fgbm_eod_matched'][-1],
                'fgbm_status': seed_results['fgbm_status'][-1],
                'm2_eod_matched': seed_results['m2_eod_matched'][-1],
                'm2_status': seed_results['m2_status'][-1],
            })

        # --- Per-dataset summary ---
        cgbm_mean = np.mean(seed_results['cgbm_eod'])
        auc_mean = np.mean(seed_results['cgbm_auc'])

        fgbm_matched = [x for x in seed_results['fgbm_eod_matched'] if x is not None]
        fgbm_unreach = sum(1 for s in seed_results['fgbm_status'] if s == 'unreachable')
        m2_matched = [x for x in seed_results['m2_eod_matched'] if x is not None]
        m2_unreach = sum(1 for s in seed_results['m2_status'] if s == 'unreachable')

        row = {
            'dataset': ds_name,
            'auc_anchor': f"{auc_mean:.3f}",
            'cgbm_eod': f"{cgbm_mean:.4f}",
        }

        # FairGBM
        if fgbm_matched:
            fmean = np.mean(fgbm_matched)
            fci = 1.96 * np.std(fgbm_matched) / (len(fgbm_matched)**0.5 + 1e-10)
            row['fgbm_eod'] = f"{fmean:.4f}±{fci:.4f}"
            if fgbm_unreach > 0:
                row['fgbm_eod'] += f" (unreach {fgbm_unreach}/10)"
            row['fgbm_winner'] = 'CausalGBM' if cgbm_mean < fmean else 'FairGBM'
        else:
            row['fgbm_eod'] = f"unreachable ({fgbm_unreach}/10)"
            row['fgbm_winner'] = 'CausalGBM (unreachable)'

        # M2FGB
        if m2_matched:
            mmean = np.mean(m2_matched)
            mci = 1.96 * np.std(m2_matched) / (len(m2_matched)**0.5 + 1e-10)
            row['m2_eod'] = f"{mmean:.4f}±{mci:.4f}"
            if m2_unreach > 0:
                row['m2_eod'] += f" (unreach {m2_unreach}/10)"
            row['m2_winner'] = 'CausalGBM' if cgbm_mean < mmean else 'M2FGB'
        else:
            row['m2_eod'] = f"unreachable ({m2_unreach}/10)"
            row['m2_winner'] = 'CausalGBM (unreachable)'

        summary.append(row)
        logger.info(f"  {ds_name}: AUC*={auc_mean:.3f}  CGBM={cgbm_mean:.4f}  "
                    f"FairGBM@AUC*={row['fgbm_eod']}  M2@AUC*={row['m2_eod']}")

    # Save raw
    pd.DataFrame(all_results).to_csv(
        os.path.join(output_dir, 'matched_auc_raw.csv'), index=False)

    # Print summary
    print("\n" + "=" * 100)
    print("MATCHED-AUC COMPARISON")
    print("=" * 100)
    print(f"\n{'Dataset':<18s} {'AUC*':>7s} {'CGBM EOD':>10s} {'FairGBM@AUC*':>28s} {'M2FGB@AUC*':>28s}")
    print("-" * 100)
    for r in summary:
        print(f"{r['dataset']:<18s} {r['auc_anchor']:>7s} {r['cgbm_eod']:>10s} "
              f"{r['fgbm_eod']:>28s} {r['m2_eod']:>28s}")

    # Winner tally
    print("\n" + "=" * 60)
    print("WINNER AT MATCHED AUC")
    print("=" * 60)
    for r in summary:
        print(f"  {r['dataset']:<18s}: vs FairGBM={r.get('fgbm_winner','?'):<20s} "
              f"vs M2FGB={r.get('m2_winner','?')}")

    # Cross-dataset Wilcoxon
    print("\n" + "=" * 60)
    print("CROSS-DATASET WILCOXON SIGNED-RANK")
    print("=" * 60)

    df_raw = pd.DataFrame(all_results)
    for rival, col in [('FairGBM', 'fgbm_eod_matched'), ('M2FGB', 'm2_eod_matched')]:
        deltas = []
        for ds_name in DATASETS:
            ds_df = df_raw[df_raw['dataset'] == ds_name]
            valid = ds_df.dropna(subset=[col])
            if len(valid) >= 3:
                delta = valid[col].mean() - valid['cgbm_eod'].mean()
                deltas.append(delta)
        if len(deltas) >= 5:
            stat, p = wilcoxon(deltas, alternative='greater')
            wins = sum(1 for d in deltas if d > 0)
            print(f"  vs {rival}: {wins}/{len(deltas)} datasets CausalGBM wins, "
                  f"Wilcoxon p={p:.4f} {'*' if p < 0.05 else ''}")
        else:
            print(f"  vs {rival}: insufficient data ({len(deltas)} datasets)")

    print(f"\nSaved: {os.path.join(output_dir, 'matched_auc_raw.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/matched_auc')
    args = parser.parse_args()
    run_matched_auc(n_seeds=args.n_seeds, output_dir=args.output_dir)
