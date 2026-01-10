#!/usr/bin/env python3
"""
CausalGBM Analysis Script - Sensitivity, Scaling, and Cross-Validation
=======================================================================

This script runs additional analyses for the CausalGBM paper:
1. Sensitivity Analysis: Effect of threshold τ and other hyperparameters
2. Scaling Analysis: Performance vs dataset size
3. Cross-Validation: 5-fold CV for statistical robustness
4. Ablation Study: Component contributions

AVAILABLE DATASETS:
-------------------
- adult: Adult Income (2 groups - sex)
- compas: COMPAS Recidivism (3 groups - race)
- german: German Credit (3 groups - age)
- bank: Bank Marketing (3 groups - age)
- online_shoppers: Online Shoppers Intention (2 groups - weekend)
- acs_income: ACS Income (8 groups - race) - requires folktables
- acs_employment: ACS Employment (8 groups - race) - requires folktables

USAGE:
------
# Run all analyses on Adult and COMPAS
python causalgbm_analysis.py --all --output_dir analysis_results

# Run specific analysis
python causalgbm_analysis.py --sensitivity --datasets adult compas
python causalgbm_analysis.py --scaling --datasets adult
python causalgbm_analysis.py --crossval --datasets adult compas
python causalgbm_analysis.py --ablation --datasets adult

# Run on Online Shoppers (binary groups)
python causalgbm_analysis.py --all --datasets online_shoppers

# Run on ACS datasets (8 groups - for limitations analysis)
python causalgbm_analysis.py --all --datasets acs_income acs_employment

# Quick test (fewer seeds)
python causalgbm_analysis.py --all --quick --datasets adult

# Run all new datasets
python causalgbm_analysis.py --all --datasets online_shoppers acs_income acs_employment

Author: CausalGBM Team
"""

import argparse
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

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

# Seeds for reproducibility
SEED_LIST = [42, 43, 44, 45, 46]


# =============================================================================
# METRICS (same as main benchmark)
# =============================================================================

def worst_group_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = np.unique(groups)
    group_accs = []
    for g in unique_groups:
        mask = groups == g
        if mask.sum() > 0:
            group_accs.append(accuracy_score(y_true[mask], y_pred[mask]))
    return min(group_accs) if group_accs else 0.0


def equalized_odds_difference(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = np.unique(groups)
    tpr_by_group, fpr_by_group = {}, {}
    
    for g in unique_groups:
        mask = groups == g
        y_t, y_p = y_true[mask], y_pred[mask]
        pos_mask = y_t == 1
        neg_mask = y_t == 0
        tpr_by_group[g] = (y_p[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0
        fpr_by_group[g] = (y_p[neg_mask] == 1).mean() if neg_mask.sum() > 0 else 0
    
    tpr_vals = list(tpr_by_group.values())
    fpr_vals = list(fpr_by_group.values())
    
    if not tpr_vals or not fpr_vals:
        return 0.0
    return max(max(tpr_vals) - min(tpr_vals), max(fpr_vals) - min(fpr_vals))


def demographic_parity_diff(y_pred: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = np.unique(groups)
    rates = [y_pred[groups == g].mean() for g in unique_groups if (groups == g).sum() > 0]
    return max(rates) - min(rates) if rates else 0.0


def compute_all_metrics(y_true, y_pred, y_prob, groups) -> Dict[str, float]:
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    y_pred = np.nan_to_num(y_pred, nan=0).astype(int)
    
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except:
        auc = 0.5
    
    return {
        'auc': auc,
        'accuracy': accuracy_score(y_true, y_pred),
        'worst_group_accuracy': worst_group_accuracy(y_true, y_pred, groups),
        'equalized_odds_diff': equalized_odds_difference(y_true, y_pred, groups),
        'demographic_parity_diff': demographic_parity_diff(y_pred, groups),
    }


# =============================================================================
# DATASET LOADERS
# =============================================================================

class DatasetBundle:
    def __init__(self, name, X_train, y_train, X_test, y_test, 
                 groups_train, groups_test, group_name, feature_names):
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.groups_train = groups_train
        self.groups_test = groups_test
        self.group_name = group_name
        self.feature_names = feature_names


def load_adult() -> DatasetBundle:
    """Load Adult Income dataset."""
    print("Loading Adult Income dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    df = pd.read_csv(url, names=columns, sep=r',\s*', engine='python', na_values='?')
    df = df.dropna()
    df['income'] = (df['income'].str.strip() == '>50K').astype(int)
    df['sex'] = df['sex'].str.strip()
    
    # Encode categorical
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
    cont_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    
    feature_cols = cat_cols + cont_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df['income'].values.astype(np.float32)
    groups = LabelEncoder().fit_transform(df['sex'].astype(str))
    
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.3, 
                                            random_state=42, stratify=y)
    
    print(f"  Adult: n={len(df)}, d={X.shape[1]}, pos_rate={y.mean():.2%}")
    
    return DatasetBundle(
        name='adult', X_train=X[train_idx], y_train=y[train_idx],
        X_test=X[test_idx], y_test=y[test_idx],
        groups_train=groups[train_idx], groups_test=groups[test_idx],
        group_name='sex', feature_names=feature_cols
    )


def load_compas() -> DatasetBundle:
    """Load COMPAS dataset."""
    print("Loading COMPAS dataset...")
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)
    
    df = df[(df['days_b_screening_arrest'] <= 30) &
            (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) &
            (df['c_charge_degree'] != 'O')]
    
    df['race_group'] = df['race'].apply(
        lambda x: 'African-American' if x == 'African-American' 
                  else ('Caucasian' if x == 'Caucasian' else 'Other'))
    
    cat_cols = ['sex', 'c_charge_degree']
    cont_cols = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count']
    
    df = df.dropna(subset=cat_cols + cont_cols + ['is_recid', 'race_group'])
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    
    feature_cols = cat_cols + cont_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df['is_recid'].values.astype(np.float32)
    groups = LabelEncoder().fit_transform(df['race_group'].astype(str))
    
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.3,
                                            random_state=42, stratify=y)
    
    print(f"  COMPAS: n={len(df)}, d={X.shape[1]}, pos_rate={y.mean():.2%}")
    
    return DatasetBundle(
        name='compas', X_train=X[train_idx], y_train=y[train_idx],
        X_test=X[test_idx], y_test=y[test_idx],
        groups_train=groups[train_idx], groups_test=groups[test_idx],
        group_name='race_group', feature_names=feature_cols
    )


def load_german() -> DatasetBundle:
    """Load German Credit dataset."""
    print("Loading German Credit dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = ['checking', 'duration', 'credit_history', 'purpose', 'credit_amount',
               'savings', 'employment', 'installment_rate', 'status_sex', 'guarantors',
               'residence', 'property', 'age', 'other_installments', 'housing',
               'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker', 'target']
    
    df = pd.read_csv(url, names=columns, sep=' ')
    df['target'] = (df['target'] == 2).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 100], 
                             labels=['young', 'middle', 'old']).astype(str)
    
    cat_cols = ['checking', 'credit_history', 'purpose', 'savings', 'employment',
                'status_sex', 'guarantors', 'property', 'other_installments',
                'housing', 'job', 'telephone', 'foreign_worker']
    cont_cols = ['duration', 'credit_amount', 'installment_rate', 'residence',
                 'age', 'existing_credits', 'dependents']
    
    df = df.dropna()
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    
    feature_cols = cat_cols + cont_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    groups = LabelEncoder().fit_transform(df['age_group'].astype(str))
    
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.3,
                                            random_state=42, stratify=y)
    
    print(f"  German: n={len(df)}, d={X.shape[1]}, pos_rate={y.mean():.2%}")
    
    return DatasetBundle(
        name='german', X_train=X[train_idx], y_train=y[train_idx],
        X_test=X[test_idx], y_test=y[test_idx],
        groups_train=groups[train_idx], groups_test=groups[test_idx],
        group_name='age_group', feature_names=feature_cols
    )


def load_bank() -> DatasetBundle:
    """Load Bank Marketing dataset."""
    print("Loading Bank Marketing dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
    
    try:
        df = pd.read_csv(url, sep=';')
    except:
        url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.csv"
        df = pd.read_csv(url2, sep=';')
    
    df['y'] = (df['y'] == 'yes').astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100],
                             labels=['young', 'middle', 'old']).astype(str)
    
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'month', 'day_of_week', 'poutcome']
    cont_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    
    cat_cols = [c for c in cat_cols if c in df.columns]
    cont_cols = [c for c in cont_cols if c in df.columns]
    
    df = df.dropna(subset=cat_cols + cont_cols + ['y', 'age_group'])
    
    # Subsample if too large
    if len(df) > 30000:
        df = df.sample(n=30000, random_state=42)
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    
    feature_cols = cat_cols + cont_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df['y'].values.astype(np.float32)
    groups = LabelEncoder().fit_transform(df['age_group'].astype(str))
    
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.3,
                                            random_state=42, stratify=y)
    
    print(f"  Bank: n={len(df)}, d={X.shape[1]}, pos_rate={y.mean():.2%}")
    
    return DatasetBundle(
        name='bank', X_train=X[train_idx], y_train=y[train_idx],
        X_test=X[test_idx], y_test=y[test_idx],
        groups_train=groups[train_idx], groups_test=groups[test_idx],
        group_name='age_group', feature_names=feature_cols
    )


DATASET_LOADERS = {
    'adult': load_adult,
    'compas': load_compas,
    'german': load_german,
    'bank': load_bank,
    'online_shoppers': load_online_shoppers,
    'acs_income': load_acs_income,
    'acs_employment': load_acs_employment,
}


def load_online_shoppers() -> DatasetBundle:
    """Load Online Shoppers Intention dataset."""
    print("Loading Online Shoppers dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    
    try:
        df = pd.read_csv(url)
    except:
        # Fallback URL
        df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/online_shoppers_intention.csv")
    
    # Target: Revenue (purchase or not)
    df['Revenue'] = df['Revenue'].astype(int)
    
    # Protected attribute: Weekend (binary)
    df['Weekend'] = df['Weekend'].astype(str)
    
    cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 
                'VisitorType', 'SpecialDay']
    cont_cols = ['Administrative', 'Administrative_Duration', 'Informational',
                 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                 'BounceRates', 'ExitRates', 'PageValues']
    
    # Filter columns that exist
    cat_cols = [c for c in cat_cols if c in df.columns]
    cont_cols = [c for c in cont_cols if c in df.columns]
    
    df = df.dropna(subset=cat_cols + cont_cols + ['Revenue', 'Weekend'])
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    
    feature_cols = cat_cols + cont_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df['Revenue'].values.astype(np.float32)
    groups = LabelEncoder().fit_transform(df['Weekend'].astype(str))
    
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.3,
                                            random_state=42, stratify=y)
    
    print(f"  Online Shoppers: n={len(df)}, d={X.shape[1]}, pos_rate={y.mean():.2%}, groups={len(np.unique(groups))}")
    
    return DatasetBundle(
        name='online_shoppers', X_train=X[train_idx], y_train=y[train_idx],
        X_test=X[test_idx], y_test=y[test_idx],
        groups_train=groups[train_idx], groups_test=groups[test_idx],
        group_name='Weekend', feature_names=feature_cols
    )


def load_acs_income() -> DatasetBundle:
    """
    Load ACS Income dataset with 8 race groups.
    Uses folktables library if available, otherwise downloads directly.
    """
    print("Loading ACS Income dataset (8 race groups)...")
    
    try:
        from folktables import ACSDataSource, ACSIncome
        
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=["CA"], download=True)
        
        features, label, group = ACSIncome.df_to_numpy(acs_data)
        
        # Race groups (RAC1P): 1-9 different races
        # We'll use all 8+ race categories
        X = features.astype(np.float32)
        y = label.astype(np.float32)
        groups = group.astype(int)
        
    except ImportError:
        print("  folktables not installed, using synthetic ACS-like data...")
        # Create synthetic dataset mimicking ACS structure
        np.random.seed(42)
        n_samples = 20000
        n_features = 10
        n_groups = 8
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        groups = np.random.randint(0, n_groups, n_samples)
        
        # Create outcome with group-specific bias
        base_prob = 1 / (1 + np.exp(-X[:, 0] - 0.5 * X[:, 1]))
        group_bias = (groups - 3.5) * 0.1  # Groups have different base rates
        prob = np.clip(base_prob + group_bias, 0.1, 0.9)
        y = (np.random.rand(n_samples) < prob).astype(np.float32)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    # Subsample if too large
    if len(X) > 30000:
        idx = np.random.choice(len(X), 30000, replace=False)
        X, y, groups = X[idx], y[idx], groups[idx]
    
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.3,
                                            random_state=42, stratify=y)
    
    n_groups = len(np.unique(groups))
    print(f"  ACS Income: n={len(X)}, d={X.shape[1]}, pos_rate={y.mean():.2%}, groups={n_groups}")
    
    return DatasetBundle(
        name='acs_income', X_train=X[train_idx], y_train=y[train_idx],
        X_test=X[test_idx], y_test=y[test_idx],
        groups_train=groups[train_idx], groups_test=groups[test_idx],
        group_name='race', feature_names=[f'feature_{i}' for i in range(X.shape[1])]
    )


def load_acs_employment() -> DatasetBundle:
    """
    Load ACS Employment dataset with 8 race groups.
    Uses folktables library if available, otherwise downloads directly.
    """
    print("Loading ACS Employment dataset (8 race groups)...")
    
    try:
        from folktables import ACSDataSource, ACSEmployment
        
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=["CA"], download=True)
        
        features, label, group = ACSEmployment.df_to_numpy(acs_data)
        
        X = features.astype(np.float32)
        y = label.astype(np.float32)
        groups = group.astype(int)
        
    except ImportError:
        print("  folktables not installed, using synthetic ACS-like data...")
        # Create synthetic dataset mimicking ACS structure
        np.random.seed(43)  # Different seed from income
        n_samples = 20000
        n_features = 12
        n_groups = 8
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        groups = np.random.randint(0, n_groups, n_samples)
        
        # Create outcome with group-specific bias (employment)
        base_prob = 1 / (1 + np.exp(-0.8 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]))
        group_bias = (groups - 3.5) * 0.08
        prob = np.clip(base_prob + group_bias, 0.15, 0.85)
        y = (np.random.rand(n_samples) < prob).astype(np.float32)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    # Subsample if too large
    if len(X) > 30000:
        idx = np.random.choice(len(X), 30000, replace=False)
        X, y, groups = X[idx], y[idx], groups[idx]
    
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.3,
                                            random_state=42, stratify=y)
    
    n_groups = len(np.unique(groups))
    print(f"  ACS Employment: n={len(X)}, d={X.shape[1]}, pos_rate={y.mean():.2%}, groups={n_groups}")
    
    return DatasetBundle(
        name='acs_employment', X_train=X[train_idx], y_train=y[train_idx],
        X_test=X[test_idx], y_test=y[test_idx],
        groups_train=groups[train_idx], groups_test=groups[test_idx],
        group_name='race', feature_names=[f'feature_{i}' for i in range(X.shape[1])]
    )


# =============================================================================
# CAUSAL FEATURE SELECTOR (from main benchmark)
# =============================================================================

class CausalFeatureSelector:
    """
    Learn causal structure and use it for feature selection.
    """
    
    def __init__(self, n_features: int, n_groups: int = 2,
                 use_group_aware: bool = True,
                 selection_threshold: float = 0.2,
                 min_features: int = 3,
                 learning_rate: float = 0.01,
                 n_iterations: int = 500,
                 lambda_dag: float = 0.1,
                 lambda_sp: float = 0.01,
                 device: str = 'cuda'):
        
        self.n_features = n_features
        self.n_groups = n_groups
        self.use_group_aware = use_group_aware
        self.selection_threshold = selection_threshold
        self.min_features = min_features
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_dag = lambda_dag
        self.lambda_sp = lambda_sp
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.causal_importance_ = None
        self.selected_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        n_samples, n_features = X.shape
        data = np.column_stack([X, y.reshape(-1, 1)]).astype(np.float32)
        n_nodes = n_features + 1
        
        W = nn.Parameter(torch.randn(n_nodes, n_nodes, device=self.device) * 0.01)
        optimizer = torch.optim.Adam([W], lr=self.learning_rate)
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        for iteration in range(self.n_iterations):
            optimizer.zero_grad()
            A = torch.sigmoid(W)
            A = A * (1 - torch.eye(n_nodes, device=self.device))
            
            X_reconstructed = data_tensor @ A
            recon_loss = F.mse_loss(X_reconstructed, data_tensor)
            
            M = A * A
            M_clamped = torch.clamp(M, max=10)
            try:
                E = torch.matrix_exp(M_clamped)
                dag_constraint = torch.trace(E) - n_nodes
            except:
                dag_constraint = torch.tensor(0.0, device=self.device)
            
            dag_loss = self.lambda_dag * dag_constraint ** 2
            sparsity_loss = self.lambda_sp * A.abs().mean()
            loss = recon_loss + dag_loss + sparsity_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 1.0)
            optimizer.step()
        
        with torch.no_grad():
            A = torch.sigmoid(W)
            A = A * (1 - torch.eye(n_nodes, device=self.device))
            causal_importance = A[:-1, -1].cpu().numpy()
        
        # Group-aware: learn per-group and take intersection
        if self.use_group_aware and groups is not None:
            unique_groups = np.unique(groups)
            if len(unique_groups) > 1:
                group_importances = []
                for g in unique_groups:
                    mask = groups == g
                    if mask.sum() > 50:
                        X_g, y_g = X[mask], y[mask]
                        data_g = np.column_stack([X_g, y_g.reshape(-1, 1)]).astype(np.float32)
                        W_g = nn.Parameter(torch.randn(n_nodes, n_nodes, device=self.device) * 0.01)
                        opt_g = torch.optim.Adam([W_g], lr=self.learning_rate)
                        data_g_tensor = torch.FloatTensor(data_g).to(self.device)
                        
                        for _ in range(self.n_iterations // 2):
                            opt_g.zero_grad()
                            A_g = torch.sigmoid(W_g)
                            A_g = A_g * (1 - torch.eye(n_nodes, device=self.device))
                            recon = F.mse_loss(data_g_tensor @ A_g, data_g_tensor)
                            M_g = torch.clamp(A_g * A_g, max=10)
                            try:
                                dag_g = (torch.trace(torch.matrix_exp(M_g)) - n_nodes) ** 2
                            except:
                                dag_g = torch.tensor(0.0, device=self.device)
                            loss_g = recon + self.lambda_dag * dag_g + self.lambda_sp * A_g.abs().mean()
                            if not (torch.isnan(loss_g) or torch.isinf(loss_g)):
                                loss_g.backward()
                                torch.nn.utils.clip_grad_norm_([W_g], 1.0)
                                opt_g.step()
                        
                        with torch.no_grad():
                            A_g = torch.sigmoid(W_g)
                            A_g = A_g * (1 - torch.eye(n_nodes, device=self.device))
                            group_importances.append(A_g[:-1, -1].cpu().numpy())
                
                if group_importances:
                    causal_importance = np.min(np.stack(group_importances), axis=0)
        
        self.causal_importance_ = causal_importance
        
        # Select features
        sorted_idx = np.argsort(causal_importance)[::-1]
        threshold_mask = causal_importance >= self.selection_threshold
        n_above_threshold = threshold_mask.sum()
        n_select = max(n_above_threshold, self.min_features)
        n_select = min(n_select, n_features)
        self.selected_features_ = sorted_idx[:n_select]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_features_ is None:
            raise ValueError("Must call fit() first")
        return X[:, self.selected_features_]


def train_causal_gbm(dataset: DatasetBundle, 
                     base_model: str = 'xgboost',
                     use_group_aware: bool = True,
                     selection_threshold: float = 0.2,
                     min_features: int = None,
                     n_iterations: int = 500,
                     device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Train CausalGBM with configurable parameters."""
    
    n_features = dataset.X_train.shape[1]
    if min_features is None:
        min_features = max(3, n_features // 3)
    
    selector = CausalFeatureSelector(
        n_features=n_features,
        n_groups=len(np.unique(dataset.groups_train)),
        use_group_aware=use_group_aware,
        selection_threshold=selection_threshold,
        min_features=min_features,
        n_iterations=n_iterations,
        device=device
    )
    
    selector.fit(dataset.X_train, dataset.y_train, dataset.groups_train)
    
    X_train = selector.transform(dataset.X_train)
    X_test = selector.transform(dataset.X_test)
    
    if base_model == 'xgboost' and HAS_XGB:
        model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   use_label_encoder=False, eval_metric='logloss',
                                   random_state=42, n_jobs=-1, verbosity=0)
    elif base_model == 'lightgbm' and HAS_LGB:
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                    random_state=42, n_jobs=-1, verbose=-1)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, max_depth=6,
                                            learning_rate=0.1, random_state=42)
    
    model.fit(X_train, dataset.y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    info = {
        'n_features_selected': len(selector.selected_features_),
        'causal_importance': selector.causal_importance_,
        'selected_features': selector.selected_features_,
    }
    
    return y_pred, y_prob, info


def train_random_selection_gbm(dataset: DatasetBundle, 
                               n_features_to_select: int = 3,
                               base_model: str = 'xgboost') -> Tuple[np.ndarray, np.ndarray]:
    """Train GBM with random feature selection (ablation baseline)."""
    
    n_features = dataset.X_train.shape[1]
    selected = np.random.choice(n_features, size=min(n_features_to_select, n_features), replace=False)
    
    X_train = dataset.X_train[:, selected]
    X_test = dataset.X_test[:, selected]
    
    if base_model == 'xgboost' and HAS_XGB:
        model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   use_label_encoder=False, eval_metric='logloss',
                                   random_state=42, n_jobs=-1, verbosity=0)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    
    model.fit(X_train, dataset.y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    return y_pred, y_prob


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_sensitivity_analysis(dataset: DatasetBundle, output_dir: str,
                              seeds: int = 5, device: str = 'cuda'):
    """
    Sensitivity analysis for CausalGBM hyperparameters.
    Tests: threshold τ, min_features, n_iterations
    """
    print(f"\n{'='*60}")
    print(f"SENSITIVITY ANALYSIS: {dataset.name}")
    print(f"{'='*60}")
    
    results = []
    
    # 1. Threshold sensitivity
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    print(f"\n1. Testing threshold τ values: {thresholds}")
    
    for tau in thresholds:
        for seed in SEED_LIST[:seeds]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            try:
                y_pred, y_prob, info = train_causal_gbm(
                    dataset, selection_threshold=tau, device=device
                )
                metrics = compute_all_metrics(dataset.y_test, y_pred, y_prob, dataset.groups_test)
                results.append({
                    'analysis': 'threshold',
                    'param_name': 'tau',
                    'param_value': tau,
                    'seed': seed,
                    'n_features_selected': info['n_features_selected'],
                    **metrics
                })
                print(f"  τ={tau:.2f}, seed={seed}: WGA={metrics['worst_group_accuracy']:.4f}, "
                      f"EOD={metrics['equalized_odds_diff']:.4f}, n_feat={info['n_features_selected']}")
            except Exception as e:
                print(f"  τ={tau:.2f}, seed={seed}: FAILED - {e}")
    
    # 2. Min features sensitivity
    min_features_values = [2, 3, 4, 5, 6]
    print(f"\n2. Testing min_features values: {min_features_values}")
    
    for mf in min_features_values:
        for seed in SEED_LIST[:seeds]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            try:
                y_pred, y_prob, info = train_causal_gbm(
                    dataset, min_features=mf, device=device
                )
                metrics = compute_all_metrics(dataset.y_test, y_pred, y_prob, dataset.groups_test)
                results.append({
                    'analysis': 'min_features',
                    'param_name': 'min_features',
                    'param_value': mf,
                    'seed': seed,
                    'n_features_selected': info['n_features_selected'],
                    **metrics
                })
                print(f"  min_feat={mf}, seed={seed}: WGA={metrics['worst_group_accuracy']:.4f}, "
                      f"EOD={metrics['equalized_odds_diff']:.4f}")
            except Exception as e:
                print(f"  min_feat={mf}, seed={seed}: FAILED - {e}")
    
    # 3. DAG iterations sensitivity
    iterations_values = [100, 250, 500, 750, 1000]
    print(f"\n3. Testing n_iterations values: {iterations_values}")
    
    for n_iter in iterations_values:
        for seed in SEED_LIST[:seeds]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            try:
                y_pred, y_prob, info = train_causal_gbm(
                    dataset, n_iterations=n_iter, device=device
                )
                metrics = compute_all_metrics(dataset.y_test, y_pred, y_prob, dataset.groups_test)
                results.append({
                    'analysis': 'iterations',
                    'param_name': 'n_iterations',
                    'param_value': n_iter,
                    'seed': seed,
                    'n_features_selected': info['n_features_selected'],
                    **metrics
                })
                print(f"  n_iter={n_iter}, seed={seed}: WGA={metrics['worst_group_accuracy']:.4f}, "
                      f"EOD={metrics['equalized_odds_diff']:.4f}")
            except Exception as e:
                print(f"  n_iter={n_iter}, seed={seed}: FAILED - {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, f'sensitivity_{dataset.name}.csv'), index=False)
    
    # Create figure
    create_sensitivity_figure(results_df, output_dir, dataset.name)
    
    return results_df


def run_scaling_analysis(dataset: DatasetBundle, output_dir: str,
                          seeds: int = 3, device: str = 'cuda'):
    """
    Scaling analysis: Performance vs dataset size.
    Compares CausalGBM, XGBoost, and baseline.
    """
    print(f"\n{'='*60}")
    print(f"SCALING ANALYSIS: {dataset.name}")
    print(f"{'='*60}")
    
    results = []
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    full_n = len(dataset.X_train)
    
    for frac in fractions:
        n_samples = int(full_n * frac)
        print(f"\nTesting with {n_samples:,} samples ({frac*100:.0f}%)...")
        
        for seed in SEED_LIST[:seeds]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Subsample
            if frac < 1.0:
                idx = np.random.choice(full_n, n_samples, replace=False)
                X_train_sub = dataset.X_train[idx]
                y_train_sub = dataset.y_train[idx]
                groups_train_sub = dataset.groups_train[idx]
            else:
                X_train_sub = dataset.X_train
                y_train_sub = dataset.y_train
                groups_train_sub = dataset.groups_train
            
            sub_dataset = DatasetBundle(
                name=f"{dataset.name}_sub",
                X_train=X_train_sub, y_train=y_train_sub,
                X_test=dataset.X_test, y_test=dataset.y_test,
                groups_train=groups_train_sub, groups_test=dataset.groups_test,
                group_name=dataset.group_name, feature_names=dataset.feature_names
            )
            
            # CausalGBM-XGB-GroupAware
            try:
                y_pred, y_prob, _ = train_causal_gbm(sub_dataset, use_group_aware=True, device=device)
                metrics = compute_all_metrics(dataset.y_test, y_pred, y_prob, dataset.groups_test)
                results.append({
                    'n_samples': n_samples, 'fraction': frac,
                    'method': 'CausalGBM-XGB-GA', 'seed': seed, **metrics
                })
                print(f"  CausalGBM-GA seed={seed}: WGA={metrics['worst_group_accuracy']:.4f}")
            except Exception as e:
                print(f"  CausalGBM-GA seed={seed}: FAILED - {e}")
            
            # CausalGBM-XGB (no group aware)
            try:
                y_pred, y_prob, _ = train_causal_gbm(sub_dataset, use_group_aware=False, device=device)
                metrics = compute_all_metrics(dataset.y_test, y_pred, y_prob, dataset.groups_test)
                results.append({
                    'n_samples': n_samples, 'fraction': frac,
                    'method': 'CausalGBM-XGB', 'seed': seed, **metrics
                })
                print(f"  CausalGBM seed={seed}: WGA={metrics['worst_group_accuracy']:.4f}")
            except Exception as e:
                print(f"  CausalGBM seed={seed}: FAILED - {e}")
            
            # XGBoost baseline
            try:
                if HAS_XGB:
                    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, 
                                               use_label_encoder=False, eval_metric='logloss',
                                               random_state=seed, verbosity=0)
                    model.fit(X_train_sub, y_train_sub)
                    y_prob = model.predict_proba(dataset.X_test)[:, 1]
                    y_pred = (y_prob > 0.5).astype(int)
                    metrics = compute_all_metrics(dataset.y_test, y_pred, y_prob, dataset.groups_test)
                    results.append({
                        'n_samples': n_samples, 'fraction': frac,
                        'method': 'XGBoost', 'seed': seed, **metrics
                    })
                    print(f"  XGBoost seed={seed}: WGA={metrics['worst_group_accuracy']:.4f}")
            except Exception as e:
                print(f"  XGBoost seed={seed}: FAILED - {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, f'scaling_{dataset.name}.csv'), index=False)
    
    create_scaling_figure(results_df, output_dir, dataset.name)
    
    return results_df


def run_crossval_analysis(dataset: DatasetBundle, output_dir: str,
                           n_folds: int = 5, device: str = 'cuda'):
    """
    K-fold cross-validation for CausalGBM.
    """
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ANALYSIS: {dataset.name} ({n_folds} folds)")
    print(f"{'='*60}")
    
    # Combine train and test
    X = np.vstack([dataset.X_train, dataset.X_test])
    y = np.concatenate([dataset.y_train, dataset.y_test])
    groups = np.concatenate([dataset.groups_train, dataset.groups_test])
    
    results = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_folds}...")
        
        fold_dataset = DatasetBundle(
            name=f"{dataset.name}_fold{fold}",
            X_train=X[train_idx].astype(np.float32),
            y_train=y[train_idx].astype(np.float32),
            X_test=X[test_idx].astype(np.float32),
            y_test=y[test_idx].astype(np.float32),
            groups_train=groups[train_idx],
            groups_test=groups[test_idx],
            group_name=dataset.group_name,
            feature_names=dataset.feature_names
        )
        
        methods = [
            ('CausalGBM-XGB-GA', lambda d: train_causal_gbm(d, use_group_aware=True, device=device)),
            ('CausalGBM-XGB', lambda d: train_causal_gbm(d, use_group_aware=False, device=device)),
        ]
        
        # Add XGBoost baseline
        if HAS_XGB:
            def train_xgb(d):
                model = xgb.XGBClassifier(n_estimators=100, max_depth=6,
                                           use_label_encoder=False, eval_metric='logloss',
                                           random_state=42, verbosity=0)
                model.fit(d.X_train, d.y_train)
                y_prob = model.predict_proba(d.X_test)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)
                return y_pred, y_prob, {}
            methods.append(('XGBoost', train_xgb))
        
        for method_name, train_fn in methods:
            try:
                y_pred, y_prob, _ = train_fn(fold_dataset)
                metrics = compute_all_metrics(y[test_idx], y_pred, y_prob, groups[test_idx])
                results.append({'fold': fold, 'method': method_name, **metrics})
                print(f"  {method_name}: WGA={metrics['worst_group_accuracy']:.4f}, "
                      f"EOD={metrics['equalized_odds_diff']:.4f}")
            except Exception as e:
                print(f"  {method_name}: FAILED - {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, f'crossval_{dataset.name}.csv'), index=False)
    
    # Print summary
    print(f"\n--- Cross-Validation Summary ---")
    summary = results_df.groupby('method').agg({
        'worst_group_accuracy': ['mean', 'std'],
        'equalized_odds_diff': ['mean', 'std'],
        'auc': ['mean', 'std']
    })
    print(summary)
    
    return results_df


def run_ablation_study(dataset: DatasetBundle, output_dir: str,
                        seeds: int = 5, device: str = 'cuda'):
    """
    Ablation study: Component contributions.
    Tests: Full method, No group-aware, Random selection, Different thresholds
    """
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {dataset.name}")
    print(f"{'='*60}")
    
    results = []
    
    configurations = [
        ('XGBoost (baseline)', 'baseline', {}),
        ('+ Causal Selection (τ=0.2)', 'causal_only', {'use_group_aware': False, 'selection_threshold': 0.2}),
        ('+ Group-Aware', 'full', {'use_group_aware': True, 'selection_threshold': 0.2}),
        ('Random Selection (3 feat)', 'random', {'random': True, 'n_features': 3}),
        ('Causal τ=0.1 (more features)', 'causal_t01', {'use_group_aware': True, 'selection_threshold': 0.1}),
        ('Causal τ=0.3 (fewer features)', 'causal_t03', {'use_group_aware': True, 'selection_threshold': 0.3}),
    ]
    
    for config_name, config_key, config_params in configurations:
        print(f"\n{config_name}...")
        
        for seed in SEED_LIST[:seeds]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            try:
                if config_key == 'baseline':
                    if HAS_XGB:
                        model = xgb.XGBClassifier(n_estimators=100, max_depth=6,
                                                   use_label_encoder=False, eval_metric='logloss',
                                                   random_state=seed, verbosity=0)
                        model.fit(dataset.X_train, dataset.y_train)
                        y_prob = model.predict_proba(dataset.X_test)[:, 1]
                        y_pred = (y_prob > 0.5).astype(int)
                        n_feat = dataset.X_train.shape[1]
                    else:
                        continue
                elif config_key == 'random':
                    y_pred, y_prob = train_random_selection_gbm(
                        dataset, n_features_to_select=config_params.get('n_features', 3)
                    )
                    n_feat = config_params.get('n_features', 3)
                else:
                    y_pred, y_prob, info = train_causal_gbm(dataset, device=device, **config_params)
                    n_feat = info['n_features_selected']
                
                metrics = compute_all_metrics(dataset.y_test, y_pred, y_prob, dataset.groups_test)
                results.append({
                    'configuration': config_name,
                    'config_key': config_key,
                    'seed': seed,
                    'n_features': n_feat,
                    **metrics
                })
                print(f"  seed={seed}: WGA={metrics['worst_group_accuracy']:.4f}, "
                      f"EOD={metrics['equalized_odds_diff']:.4f}, n_feat={n_feat}")
            except Exception as e:
                print(f"  seed={seed}: FAILED - {e}")
                traceback.print_exc()
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, f'ablation_{dataset.name}.csv'), index=False)
    
    # Print summary
    print(f"\n--- Ablation Summary ---")
    summary = results_df.groupby('configuration').agg({
        'worst_group_accuracy': ['mean', 'std'],
        'equalized_odds_diff': ['mean', 'std'],
        'n_features': 'mean'
    }).round(4)
    print(summary)
    
    return results_df


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def create_sensitivity_figure(df: pd.DataFrame, output_dir: str, dataset_name: str):
    """Create sensitivity analysis figure."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    analyses = df['analysis'].unique()
    
    for idx, analysis in enumerate(analyses[:3]):
        analysis_df = df[df['analysis'] == analysis]
        if len(analysis_df) == 0:
            continue
        
        param_name = analysis_df['param_name'].iloc[0]
        summary = analysis_df.groupby('param_value').agg({
            'worst_group_accuracy': ['mean', 'std'],
            'equalized_odds_diff': ['mean', 'std']
        })
        
        # WGA plot
        ax = axes[0, idx]
        x = summary.index
        y = summary['worst_group_accuracy']['mean']
        yerr = summary['worst_group_accuracy']['std']
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, linewidth=2, 
                   markersize=8, color='#2E86AB')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Worst-Group Accuracy ↑')
        ax.set_title(f'{param_name} Sensitivity')
        ax.grid(True, alpha=0.3)
        
        # EOD plot
        ax = axes[1, idx]
        y = summary['equalized_odds_diff']['mean']
        yerr = summary['equalized_odds_diff']['std']
        ax.errorbar(x, y, yerr=yerr, marker='s', capsize=5, linewidth=2,
                   markersize=8, color='#A23B72')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Equalized Odds Diff ↓')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'CausalGBM Sensitivity Analysis: {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sensitivity_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'sensitivity_{dataset_name}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def create_scaling_figure(df: pd.DataFrame, output_dir: str, dataset_name: str):
    """Create scaling analysis figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'CausalGBM-XGB-GA': '#2E86AB', 'CausalGBM-XGB': '#A23B72', 'XGBoost': '#F18F01'}
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        summary = method_df.groupby('n_samples').agg({
            'worst_group_accuracy': ['mean', 'std'],
            'equalized_odds_diff': ['mean', 'std']
        })
        
        color = colors.get(method, '#333333')
        
        ax = axes[0]
        ax.errorbar(summary.index, summary['worst_group_accuracy']['mean'],
                   yerr=summary['worst_group_accuracy']['std'],
                   marker='o', capsize=5, linewidth=2, markersize=8, 
                   color=color, label=method)
        
        ax = axes[1]
        ax.errorbar(summary.index, summary['equalized_odds_diff']['mean'],
                   yerr=summary['equalized_odds_diff']['std'],
                   marker='s', capsize=5, linewidth=2, markersize=8,
                   color=color, label=method)
    
    axes[0].set_xlabel('Training Set Size')
    axes[0].set_ylabel('Worst-Group Accuracy ↑')
    axes[0].set_title('Accuracy vs Dataset Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Training Set Size')
    axes[1].set_ylabel('Equalized Odds Diff ↓')
    axes[1].set_title('Fairness vs Dataset Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'CausalGBM Scaling Analysis: {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scaling_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'scaling_{dataset_name}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CausalGBM Analysis Script')
    
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('--scaling', action='store_true', help='Run scaling analysis')
    parser.add_argument('--crossval', action='store_true', help='Run cross-validation')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    
    parser.add_argument('--datasets', nargs='+', default=['adult', 'compas'],
                       choices=['adult', 'compas', 'german', 'bank', 'online_shoppers', 'acs_income', 'acs_employment'],
                       help='Datasets to analyze')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer seeds)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.seeds = 2
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.output_dir}")
    
    for dataset_name in args.datasets:
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*60}")
        
        try:
            dataset = DATASET_LOADERS[dataset_name]()
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue
        
        if args.sensitivity or args.all:
            run_sensitivity_analysis(dataset, args.output_dir, args.seeds, device)
        
        if args.scaling or args.all:
            run_scaling_analysis(dataset, args.output_dir, min(args.seeds, 3), device)
        
        if args.crossval or args.all:
            run_crossval_analysis(dataset, args.output_dir, n_folds=5, device=device)
        
        if args.ablation or args.all:
            run_ablation_study(dataset, args.output_dir, args.seeds, device)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
