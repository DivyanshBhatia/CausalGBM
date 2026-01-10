#!/usr/bin/env python3
"""
Ablation study for minimum features parameter (m)
Reviewer request: Vary m ∈ {2, 3, 5, 7, ⌊d/4⌋, ⌊d/3⌋, ⌊d/2⌋}
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import argparse
import os

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

def compute_causal_importance(X, y, groups):
    """Compute causal importance for each feature."""
    n_features = X.shape[1]
    causal_importance = []
    
    for j in range(n_features):
        # Correlation with protected
        corr_g = abs(np.corrcoef(X[:, j], groups)[0, 1])
        
        # Partial correlation (residualize on groups)
        reg_x = LinearRegression().fit(groups.reshape(-1, 1), X[:, j])
        x_resid = X[:, j] - reg_x.predict(groups.reshape(-1, 1))
        
        reg_y = LinearRegression().fit(groups.reshape(-1, 1), y)
        y_resid = y - reg_y.predict(groups.reshape(-1, 1))
        
        partial_corr = abs(np.corrcoef(x_resid, y_resid)[0, 1])
        causal_imp = partial_corr * (1 - corr_g)
        causal_importance.append(causal_imp)
    
    return np.array(causal_importance)

def select_features(causal_importance, threshold=0.2, min_features=3):
    """Select features based on causal importance."""
    # Normalize
    causal_importance = causal_importance / causal_importance.max()
    
    # Select above threshold
    selected_idx = np.where(causal_importance >= threshold)[0]
    
    # Ensure minimum features
    if len(selected_idx) < min_features:
        selected_idx = np.argsort(causal_importance)[-min_features:]
    
    return selected_idx

def compute_eod(y_true, y_pred, groups):
    """Compute Equalized Odds Difference."""
    tpr_diff = 0
    fpr_diff = 0
    
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0
    
    tprs, fprs = [], []
    for g in unique_groups:
        mask = groups == g
        pos_mask = (groups == g) & (y_true == 1)
        neg_mask = (groups == g) & (y_true == 0)
        
        if pos_mask.sum() > 0:
            tprs.append(y_pred[pos_mask].mean())
        if neg_mask.sum() > 0:
            fprs.append(y_pred[neg_mask].mean())
    
    tpr_diff = max(tprs) - min(tprs) if len(tprs) >= 2 else 0
    fpr_diff = max(fprs) - min(fprs) if len(fprs) >= 2 else 0
    
    return max(tpr_diff, fpr_diff)

def compute_wga(y_true, y_pred, groups):
    """Compute Worst-Group Accuracy."""
    accs = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() > 0:
            accs.append(accuracy_score(y_true[mask], y_pred[mask]))
    return min(accs) if accs else 0

def run_min_features_ablation(X, y, groups, feature_names, dataset_name, seeds=[42, 43, 44, 45, 46]):
    """Run ablation study varying minimum features."""
    
    n_features = X.shape[1]
    
    # Different values of m to test
    m_values = [2, 3, 5, 7]
    m_values.extend([max(1, n_features // 4), max(1, n_features // 3), max(1, n_features // 2)])
    m_values = sorted(set(m_values))  # Remove duplicates and sort
    
    results = []
    
    for seed in seeds:
        np.random.seed(seed)
        
        # Split data
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y
        )
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train, g_test = groups[train_idx], groups[test_idx]
        
        # Compute causal importance on training data
        causal_imp = compute_causal_importance(X_train, y_train, g_train)
        
        for m in m_values:
            # Select features
            selected_idx = select_features(causal_imp, threshold=0.2, min_features=m)
            n_selected = len(selected_idx)
            
            X_train_sel = X_train[:, selected_idx]
            X_test_sel = X_test[:, selected_idx]
            
            # Train model
            if HAS_XGB:
                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4,
                    use_label_encoder=False, eval_metric='logloss',
                    random_state=seed, verbosity=0
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, random_state=seed
                )
            
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)
            y_prob = model.predict_proba(X_test_sel)[:, 1]
            
            # Compute metrics
            wga = compute_wga(y_test, y_pred, g_test)
            eod = compute_eod(y_test, y_pred, g_test)
            auc = roc_auc_score(y_test, y_prob)
            
            results.append({
                'dataset': dataset_name,
                'min_features': m,
                'n_selected': n_selected,
                'seed': seed,
                'wga': wga,
                'eod': eod,
                'auc': auc
            })
            
            print(f"  m={m}, n_sel={n_selected}, seed={seed}: WGA={wga:.3f}, EOD={eod:.3f}, AUC={auc:.3f}")
    
    return pd.DataFrame(results)

def load_adult():
    """Load Adult dataset."""
    from sklearn.datasets import fetch_openml
    
    print("Loading Adult dataset...")
    data = fetch_openml('adult', version=2, as_frame=True)
    df = data.frame
    
    # Prepare features
    df = df.dropna()
    
    # Protected attribute: sex
    df['sex_binary'] = (df['sex'] == 'Male').astype(int)
    
    # Target
    df['income_binary'] = (df['income'] == '>50K').astype(int)
    
    # Features (numeric only for simplicity)
    feature_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                    'capital-loss', 'hours-per-week']
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['income_binary'].values.astype(np.float32)
    groups = df['sex_binary'].values.astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, groups, feature_cols

def load_synthetic_loan(data_path='synthetic_loan_data.csv'):
    """Load synthetic loan dataset."""
    print("Loading Synthetic Loan dataset...")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Generate if not exists
        np.random.seed(42)
        n = 10000
        
        gender = np.random.binomial(1, 0.5, n)
        income = np.random.normal(50000, 15000, n)
        credit_score = np.random.normal(700, 50, n)
        employment_years = np.clip(np.random.poisson(5, n), 0, 30)
        
        works_in_tech = np.random.binomial(1, 0.15 + 0.60 * gender, n)
        has_stem_degree = np.random.binomial(1, 0.10 + 0.55 * gender, n)
        plays_golf = np.random.binomial(1, 0.05 + 0.50 * gender, n)
        
        favorite_color_blue = np.random.binomial(1, 0.3, n)
        birth_month = np.random.randint(1, 13, n)
        
        income_norm = (income - 50000) / 15000
        credit_norm = (credit_score - 700) / 50
        emp_norm = (employment_years - 5) / 3
        
        logit = (0.6 * income_norm + 0.8 * credit_norm + 0.4 * emp_norm +
                 1.2 * works_in_tech + 1.0 * has_stem_degree + 0.8 * plays_golf +
                 0.3 * np.random.randn(n))
        prob = 1 / (1 + np.exp(-logit))
        loan_approved = (np.random.rand(n) < prob).astype(int)
        
        df = pd.DataFrame({
            'income': income, 'credit_score': credit_score, 
            'employment_years': employment_years,
            'works_in_tech': works_in_tech, 'has_stem_degree': has_stem_degree,
            'plays_golf': plays_golf, 'favorite_color_blue': favorite_color_blue,
            'birth_month': birth_month, 'gender': gender, 'loan_approved': loan_approved
        })
    
    feature_cols = ['income', 'credit_score', 'employment_years',
                    'works_in_tech', 'has_stem_degree', 'plays_golf',
                    'favorite_color_blue', 'birth_month']
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['loan_approved'].values.astype(np.float32)
    groups = df['gender'].values.astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, groups, feature_cols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='synthetic_loan',
                       choices=['adult', 'synthetic_loan'])
    parser.add_argument('--data_path', type=str, default='synthetic_loan_data.csv')
    parser.add_argument('--output', type=str, default='min_features_ablation.csv')
    args = parser.parse_args()
    
    print("="*60)
    print("MINIMUM FEATURES ABLATION STUDY")
    print("="*60)
    
    if args.dataset == 'adult':
        X, y, groups, feature_names = load_adult()
    else:
        X, y, groups, feature_names = load_synthetic_loan(args.data_path)
    
    print(f"\nDataset: {args.dataset}")
    print(f"n={len(y)}, d={X.shape[1]}")
    print(f"Features: {feature_names}")
    
    results = run_min_features_ablation(X, y, groups, feature_names, args.dataset)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    summary = results.groupby('min_features').agg({
        'n_selected': 'mean',
        'wga': ['mean', 'std'],
        'eod': ['mean', 'std'],
        'auc': ['mean', 'std']
    }).round(4)
    
    print(f"\n{'m':<6} {'n_sel':>8} {'WGA':>16} {'EOD':>16} {'AUC':>16}")
    print("-"*70)
    
    for m in summary.index:
        n_sel = summary.loc[m, ('n_selected', 'mean')]
        wga = f"{summary.loc[m, ('wga', 'mean')]:.3f}±{summary.loc[m, ('wga', 'std')]:.3f}"
        eod = f"{summary.loc[m, ('eod', 'mean')]:.3f}±{summary.loc[m, ('eod', 'std')]:.3f}"
        auc = f"{summary.loc[m, ('auc', 'mean')]:.3f}±{summary.loc[m, ('auc', 'std')]:.3f}"
        print(f"{m:<6} {n_sel:>8.0f} {wga:>16} {eod:>16} {auc:>16}")
    
    # Save
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
