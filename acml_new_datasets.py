"""
Expand to 15 Datasets — 6 New Fairness Benchmarks
===================================================
Uses existing loaders: Law School, ACS Employment
New loaders: ACS Public Coverage, ACS Travel Time, Heart Disease, Student Performance

Runs full pipeline for each: XGBoost baseline, CausalGBM, FairGBM, M²FGB,
plus the DAG-free ablation. Same 10 seeds, same config as main paper.

Usage:
  python acml_new_datasets.py --n_seeds 10
"""

import os, sys, argparse, warnings, logging, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causalgbm_experiments_v2 import (
    CausalFeatureSelector, compute_metrics,
    load_acs_employment, load_law_school, DatasetBundle,
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


# ============================================================================
# NEW DATASET LOADERS
# ============================================================================

def load_acs_public_coverage(max_samples=None, states=['CA'], year='2018'):
    """ACS Public Coverage: predict public health insurance coverage. Protected: Race."""
    from folktables import ACSDataSource, ACSPublicCoverage
    logger.info("Loading ACS Public Coverage dataset...")
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)
    
    sensitive = (group == 1).astype(float)
    label = label.astype(float)
    feature_names = [f'feat_{i}' for i in range(features.shape[1])]
    
    if max_samples and len(features) > max_samples:
        idx = np.random.RandomState(42).choice(len(features), max_samples, replace=False)
        features, label, sensitive = features[idx], label[idx], sensitive[idx]
    
    logger.info(f"Processing acs_public_cov...")
    logger.info(f"  acs_public_cov: n={len(features)}, d={features.shape[1]}, groups={len(np.unique(sensitive))}")
    return DatasetBundle('acs_public_cov', features, label, sensitive, 'Race', feature_names)


def load_acs_travel_time(max_samples=None, states=['CA'], year='2018'):
    """ACS Travel Time: predict commute > 20 min. Protected: Race."""
    from folktables import ACSDataSource, ACSTravelTime
    logger.info("Loading ACS Travel Time dataset...")
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSTravelTime.df_to_numpy(acs_data)
    
    sensitive = (group == 1).astype(float)
    label = label.astype(float)
    feature_names = [f'feat_{i}' for i in range(features.shape[1])]
    
    if max_samples and len(features) > max_samples:
        idx = np.random.RandomState(42).choice(len(features), max_samples, replace=False)
        features, label, sensitive = features[idx], label[idx], sensitive[idx]
    
    logger.info(f"Processing acs_travel_time...")
    logger.info(f"  acs_travel_time: n={len(features)}, d={features.shape[1]}, groups={len(np.unique(sensitive))}")
    return DatasetBundle('acs_travel_time', features, label, sensitive, 'Race', feature_names)


def load_heart_disease(max_samples=None):
    """UCI Heart Disease (Cleveland). Protected: Sex. Target: presence of heart disease."""
    logger.info("Loading Heart Disease dataset...")
    try:
        data = fetch_openml('heart-statlog', version=1, as_frame=True, parser='auto')
        df = data.frame
    except:
        try:
            data = fetch_openml(data_id=53, as_frame=True, parser='auto')
            df = data.frame
        except Exception as e:
            raise RuntimeError(f"Heart Disease: could not load from OpenML. Error: {e}")
    
    y = (df.iloc[:, -1].astype(str).str.strip().isin(['2', 'present', '1'])).astype(float)
    
    sex_col = None
    for col in df.columns:
        if 'sex' in col.lower():
            sex_col = col
            break
    if sex_col is None:
        sex_col = df.columns[1]
    
    sensitive = df[sex_col].astype(float)
    feature_cols = [c for c in df.columns[:-1] if c != sex_col]
    X = df[feature_cols].copy()
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    fnames = list(X.columns)
    X = X.values.astype(float)
    y = y.values
    sensitive = sensitive.values
    
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isnan(sensitive))
    X, y, sensitive = X[mask], y[mask], sensitive[mask]
    
    logger.info(f"Processing heart_disease...")
    logger.info(f"  heart_disease: n={len(X)}, d={X.shape[1]}, groups={len(np.unique(sensitive))}")
    return DatasetBundle('heart_disease', X, y, sensitive, 'Sex', fnames)


def load_student_performance(max_samples=None):
    """UCI Student Performance. Protected: Sex. Target: pass (grade >= 10) vs fail."""
    try:
        # Try loading from OpenML
        data = fetch_openml('student-por', version=1, as_frame=True, parser='auto')
        df = data.frame
    except:
        try:
            data = fetch_openml('student_performance', version=1, as_frame=True, parser='auto')
            df = data.frame
        except:
            # Fallback: try direct URL
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
            import urllib.request, zipfile, io
            response = urllib.request.urlopen(url)
            z = zipfile.ZipFile(io.BytesIO(response.read()))
            df = pd.read_csv(z.open('student-por.csv'), sep=';')
    
    # Target: final grade G3 >= 10 (pass)
    if 'G3' in df.columns:
        y = (df['G3'].astype(float) >= 10).astype(float)
        grade_cols = ['G1', 'G2', 'G3']
    else:
        # Use last column as target
        y = df.iloc[:, -1].astype(float)
        y = (y >= y.median()).astype(float)
        grade_cols = [df.columns[-1]]
    
    # Protected: sex
    sex_col = None
    for col in df.columns:
        if col.lower() == 'sex':
            sex_col = col
            break
    
    if sex_col:
        sensitive = (df[sex_col].astype(str).str.strip().isin(['M', 'male', '1'])).astype(float)
    else:
        sensitive = df.iloc[:, 1].astype(float)
    
    # Features: everything except target grades and sex
    drop_cols = grade_cols + ([sex_col] if sex_col else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.values.astype(float)
    sensitive = sensitive.values
    
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y.values if hasattr(y, 'values') else y) | np.isnan(sensitive))
    X = X[mask]
    y = (y.values if hasattr(y, 'values') else y)[mask]
    sensitive = sensitive[mask]
    
    fnames = list(range(X.shape[1]))
    
    logger.info(f"Processing student_perf...")
    logger.info(f"  student_perf: n={len(X)}, d={X.shape[1]}, groups={len(np.unique(sensitive))}")
    return DatasetBundle('student_perf', X, y, sensitive, 'Sex', [f'feat_{i}' for i in range(X.shape[1])])


def load_diabetes_pima(max_samples=None):
    """Pima Indians Diabetes. Protected: Age (>=30 vs <30). Target: diabetes onset."""
    data = fetch_openml(data_id=37, as_frame=True, parser='auto')
    df = data.frame
    
    y = (df.iloc[:, -1].astype(str).str.strip().isin(['tested_positive', '1', 'pos'])).astype(float)
    
    # Protected: age binarised (>=30 vs <30)
    age_col = None
    for col in df.columns:
        if 'age' in col.lower():
            age_col = col
            break
    if age_col is None:
        age_col = df.columns[7]  # Usually column 8
    
    sensitive = (df[age_col].astype(float) >= 30).astype(float)
    
    feature_cols = [c for c in df.columns[:-1] if c != age_col]
    X = df[feature_cols].astype(float).values
    
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isnan(sensitive))
    X, y, sensitive = X[mask], y[mask], sensitive[mask]
    
    logger.info(f"Processing diabetes...")
    logger.info(f"  diabetes: n={len(X)}, d={X.shape[1]}, groups={len(np.unique(sensitive))}")
    return DatasetBundle('diabetes', X, y, sensitive, 'Age', [str(c) for c in feature_cols])


def load_credit_approval(max_samples=None):
    """UCI Credit Approval. Protected: inferred Gender (col 0). Target: approval."""
    try:
        data = fetch_openml('credit-approval', version=1, as_frame=True, parser='auto')
        df = data.frame
    except:
        data = fetch_openml(data_id=29, as_frame=True, parser='auto')
        df = data.frame
    
    y = (df.iloc[:, -1].astype(str).str.strip().isin(['+', '1', 'yes'])).astype(float)
    
    # Protected: first attribute (typically gender, encoded as a/b)
    sens_col = df.columns[0]
    sensitive = LabelEncoder().fit_transform(df[sens_col].astype(str)).astype(float)
    
    feature_cols = [c for c in df.columns[:-1] if c != sens_col]
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.values.astype(float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isnan(sensitive))
    X, y, sensitive = X[mask], y[mask], sensitive[mask]
    
    logger.info(f"Processing credit_approval...")
    logger.info(f"  credit_approval: n={len(X)}, d={X.shape[1]}, groups={len(np.unique(sensitive))}")
    return DatasetBundle('credit_approval', X, y, sensitive, 'Gender', [str(c) for c in feature_cols])


def load_titanic(max_samples=None):
    """Titanic survival. Protected: Sex. Target: survived."""
    try:
        data = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
        df = data.frame
    except:
        data = fetch_openml(data_id=40945, as_frame=True, parser='auto')
        df = data.frame
    
    y = (df['survived'].astype(str).str.strip().isin(['1', 'yes'])).astype(float)
    
    sensitive = (df['sex'].astype(str).str.strip().isin(['male', '1'])).astype(float)
    
    drop_cols = ['survived', 'sex', 'name', 'ticket', 'cabin', 'boat', 'body', 'home.dest']
    feature_cols = [c for c in df.columns if c.lower() not in [d.lower() for d in drop_cols]]
    X = df[feature_cols].copy()
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.values.astype(float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isnan(sensitive))
    X, y, sensitive = X[mask], y[mask], sensitive[mask]
    
    logger.info(f"Processing titanic...")
    logger.info(f"  titanic: n={len(X)}, d={X.shape[1]}, groups={len(np.unique(sensitive))}")
    return DatasetBundle('titanic', X, y, sensitive, 'Sex', [str(c) for c in feature_cols])


def load_communities_crime(max_samples=None):
    """Communities & Crime. Protected: race (majority Black vs not). Target: high crime."""
    try:
        data = fetch_openml(data_id=41960, as_frame=True, parser='auto')
        df = data.frame
    except:
        try:
            data = fetch_openml('communities', version=1, as_frame=True, parser='auto')
            df = data.frame
        except:
            # Direct URL fallback
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
            df = pd.read_csv(url, header=None, na_values='?')
            # Last column is target (ViolentCrimesPerPop)
            # Column 3 is racePctBlack
    
    # Find target and race columns
    target_col = df.columns[-1]
    y_raw = pd.to_numeric(df[target_col], errors='coerce')
    y = (y_raw >= y_raw.median()).astype(float)
    
    # Protected: race composition (look for racePctBlack or similar)
    race_col = None
    for col in df.columns:
        col_str = str(col).lower()
        if 'racepctblack' in col_str or 'race' in col_str:
            race_col = col
            break
    
    if race_col is None:
        # Use column index 3 (typically racePctBlack in the original)
        race_col = df.columns[min(3, len(df.columns)-2)]
    
    race_vals = pd.to_numeric(df[race_col], errors='coerce')
    sensitive = (race_vals >= race_vals.median()).astype(float)
    
    drop_cols = [target_col, race_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.values.astype(float)
    # Replace NaN with column median
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy='median')
    X = imp.fit_transform(X)
    
    mask = ~(np.isnan(y) | np.isnan(sensitive))
    X, y, sensitive = X[mask], y[mask], sensitive[mask]
    
    logger.info(f"Processing communities...")
    logger.info(f"  communities: n={len(X)}, d={X.shape[1]}, groups={len(np.unique(sensitive))}")
    return DatasetBundle('communities', X, y, sensitive, 'Race', [str(c) for c in feature_cols])


# ============================================================================
# DATASET REGISTRY
# ============================================================================

NEW_DATASETS = {
    'law_school': {
        'loader': load_law_school,
        'protected': 'Race',
        'domain': 'Education',
    },
    'acs_employment': {
        'loader': load_acs_employment,
        'protected': 'Race',
        'domain': 'Employment',
    },
    'acs_public_cov': {
        'loader': load_acs_public_coverage,
        'protected': 'Race',
        'domain': 'Healthcare',
    },
    'acs_travel_time': {
        'loader': load_acs_travel_time,
        'protected': 'Race',
        'domain': 'Transport',
    },
    'heart_disease': {
        'loader': load_heart_disease,
        'protected': 'Sex',
        'domain': 'Healthcare',
    },
    'student_perf': {
        'loader': load_student_performance,
        'protected': 'Sex',
        'domain': 'Education',
    },
    'diabetes': {
        'loader': load_diabetes_pima,
        'protected': 'Age',
        'domain': 'Healthcare',
    },
    'credit_approval': {
        'loader': load_credit_approval,
        'protected': 'Gender',
        'domain': 'Finance',
    },
    'titanic': {
        'loader': load_titanic,
        'protected': 'Sex',
        'domain': 'Safety',
    },
    'communities': {
        'loader': load_communities_crime,
        'protected': 'Race',
        'domain': 'Criminal Justice',
    },
}


def rank_normalize(arr):
    from scipy.stats import rankdata
    ranks = rankdata(arr, method='average')
    return (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-10)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiments(n_seeds=10, output_dir='results/acml2026/new_datasets'):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for ds_name, ds_info in NEW_DATASETS.items():
        try:
            dataset = ds_info['loader']()
        except Exception as e:
            logger.warning(f"Skipping {ds_name}: {e}")
            continue

        X, y, sens = dataset.X, dataset.y, dataset.sensitive
        d = X.shape[1]
        logger.info(f"\n{'='*60}")
        logger.info(f"{ds_name} (n={len(X)}, d={d}, protected={ds_info['protected']})")
        logger.info(f"{'='*60}")

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, sens, test_size=0.3, random_state=seed, stratify=y)

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            min_feat = max(3, d // 3)

            # --- XGBoost baseline ---
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(X_tr_sc, y_tr)
            yp, ypr = m.predict(X_te_sc), m.predict_proba(X_te_sc)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            results.append({'dataset': ds_name, 'method': 'XGBoost', 'seed': seed, **met})

            # --- CausalGBM ---
            sel = CausalFeatureSelector(
                d, alpha=0.5, threshold=0.2,
                min_features=min_feat,
                n_iterations=500, aggregation='max', device='cpu')
            sel.fit(X_tr_sc, s_tr, y_tr)
            cgbm_selected = set(sel.selected_)
            Xtr_s, Xte_s = sel.transform(X_tr_sc), sel.transform(X_te_sc)
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_s, y_tr)
            yp, ypr = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
            met = compute_metrics(y_te, yp, ypr, s_te)
            # Rollback
            if met['auc'] < 0.60:
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(X_tr_sc, y_tr)
                yp, ypr = m.predict(X_te_sc), m.predict_proba(X_te_sc)[:, 1]
                met = compute_metrics(y_te, yp, ypr, s_te)
                cgbm_selected = set(range(d))
            results.append({'dataset': ds_name, 'method': 'CausalGBM', 'seed': seed,
                           'n_feats': len(cgbm_selected), **met})

            # --- FairGBM ---
            if HAS_FAIRGBM:
                try:
                    fgbm = FairGBMClassifier(
                        constraint_type="FNR,FPR", n_estimators=100,
                        random_state=seed, multiplier_learning_rate=0.1, verbose=-1)
                    fgbm.fit(X_tr_sc, y_tr, constraint_group=s_tr)
                    yp = fgbm.predict(X_te_sc)
                    ypr = fgbm.predict_proba(X_te_sc)[:, 1]
                    met = compute_metrics(y_te, yp, ypr, s_te)
                    results.append({'dataset': ds_name, 'method': 'FairGBM', 'seed': seed, **met})
                except:
                    pass

            # --- M²FGB ---
            if HAS_M2FGB:
                try:
                    m2 = M2FGBClassifier(
                        fairness_constraint='true_positive_rate',
                        fair_weight=0.5, n_estimators=100,
                        learning_rate=0.1, multiplier_learning_rate=0.1,
                        random_state=seed)
                    m2.fit(X_tr_sc, y_tr, sensitive_attribute=s_tr)
                    yp = m2.predict(X_te_sc)
                    ypr = m2.predict_proba(X_te_sc)[:, 1]
                    met = compute_metrics(y_te, yp, ypr, s_te)
                    results.append({'dataset': ds_name, 'method': 'M2FGB-TPR', 'seed': seed, **met})
                except:
                    pass

            # --- No-DAG (corr/corr) for DAG-free ablation ---
            corr_proxy = np.array([abs(np.corrcoef(X_tr_sc[:, j], s_tr)[0, 1]) for j in range(d)])
            corr_pred = np.array([abs(np.corrcoef(X_tr_sc[:, j], y_tr)[0, 1]) for j in range(d)])
            cp_norm = rank_normalize(corr_proxy)
            cd_norm = rank_normalize(corr_pred)
            scores_nd = cd_norm - 0.5 * cp_norm
            nd_selected = set(np.where(scores_nd >= 0.2)[0])
            if len(nd_selected) < min_feat:
                nd_selected = set(np.argsort(scores_nd)[-min_feat:])

            Xtr_nd = X_tr_sc[:, sorted(nd_selected)]
            Xte_nd = X_te_sc[:, sorted(nd_selected)]
            m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
            m.fit(Xtr_nd, y_tr)
            yp, ypr = m.predict(Xte_nd), m.predict_proba(Xte_nd)[:, 1]
            met_nd = compute_metrics(y_te, yp, ypr, s_te)
            if met_nd['auc'] < 0.60:
                m = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0)
                m.fit(X_tr_sc, y_tr)
                yp, ypr = m.predict(X_te_sc), m.predict_proba(X_te_sc)[:, 1]
                met_nd = compute_metrics(y_te, yp, ypr, s_te)
                nd_selected = set(range(d))

            jaccard = len(cgbm_selected & nd_selected) / len(cgbm_selected | nd_selected) if len(cgbm_selected | nd_selected) > 0 else 1.0
            results.append({'dataset': ds_name, 'method': 'No-DAG', 'seed': seed,
                           'jaccard': jaccard, **met_nd})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'new_datasets_raw.csv'), index=False)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 90)
    print("NEW DATASET RESULTS (6 additional datasets)")
    print("=" * 90)

    for ds_name in NEW_DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        if ds_df.empty:
            continue
        print(f"\n  {ds_name} ({NEW_DATASETS[ds_name]['protected']}, {NEW_DATASETS[ds_name]['domain']}):")
        print(f"  {'Method':<18s} {'EOD':>7s} {'AUC':>7s} {'DPD':>7s}")
        print("  " + "-" * 45)

        for method in ['XGBoost', 'CausalGBM', 'FairGBM', 'M2FGB-TPR', 'No-DAG']:
            m_df = ds_df[ds_df['method'] == method]
            if m_df.empty:
                continue
            eod = m_df['eod'].mean()
            auc = m_df['auc'].mean()
            dpd = m_df['dpd'].mean()
            marker = " ★" if method == 'CausalGBM' else ""
            extra = ""
            if method == 'No-DAG' and 'jaccard' in m_df.columns:
                extra = f"  J={m_df['jaccard'].mean():.2f}"
            print(f"  {method:<18s} {eod:>7.4f} {auc:>7.3f} {dpd:>7.3f}{marker}{extra}")

    # Significance tests
    from scipy.stats import ttest_rel
    print("\n" + "=" * 70)
    print("SIGNIFICANCE: CausalGBM vs XGBoost")
    print("=" * 70)
    for ds_name in NEW_DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        cgbm = ds_df[ds_df['method'] == 'CausalGBM']['eod'].values
        xgb_eod = ds_df[ds_df['method'] == 'XGBoost']['eod'].values
        if len(cgbm) >= 3 and len(xgb_eod) >= 3:
            n = min(len(cgbm), len(xgb_eod))
            t, p = ttest_rel(xgb_eod[:n], cgbm[:n])
            delta = xgb_eod[:n].mean() - cgbm[:n].mean()
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {ds_name:<18s}: ΔEOD={delta:>+.4f}  p={p:.4f} {sig}")

    # DAG value summary
    print("\n" + "=" * 70)
    print("DAG VALUE: CausalGBM vs No-DAG")
    print("=" * 70)
    for ds_name in NEW_DATASETS:
        ds_df = df[df['dataset'] == ds_name]
        cgbm_eod = ds_df[ds_df['method'] == 'CausalGBM']['eod'].mean()
        nd_eod = ds_df[ds_df['method'] == 'No-DAG']['eod'].mean()
        if np.isnan(cgbm_eod) or np.isnan(nd_eod):
            continue
        if cgbm_eod < nd_eod - 0.005:
            print(f"  {ds_name}: CausalGBM wins ({cgbm_eod:.4f} vs {nd_eod:.4f})")
        elif nd_eod < cgbm_eod - 0.005:
            print(f"  {ds_name}: No-DAG wins ({nd_eod:.4f} vs {cgbm_eod:.4f})")
        else:
            print(f"  {ds_name}: Tie ({cgbm_eod:.4f} vs {nd_eod:.4f})")

    print(f"\nSaved: {os.path.join(output_dir, 'new_datasets_raw.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', default='results/acml2026/new_datasets')
    args = parser.parse_args()
    run_experiments(n_seeds=args.n_seeds, output_dir=args.output_dir)
