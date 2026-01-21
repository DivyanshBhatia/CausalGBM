#!/usr/bin/env python3
"""
CausalGBM: Comprehensive Experimental Framework v2
===================================================

Updates in v2:
- Support for synthetic datasets (loan, hiring) with known ground truth DAGs
- DAG recovery metrics: SHD, Precision, Recall, F1
- Ground truth causal structure comparison

Author: CausalGBM Team
"""

import argparse
import os
import warnings
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import traceback

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif

# Optional imports
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

try:
    from folktables import ACSDataSource, ACSIncome, ACSEmployment
    HAS_FOLKTABLES = True
except ImportError:
    HAS_FOLKTABLES = False

try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass 
class Config:
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    n_folds: int = 5
    n_cv_repeats: int = 3
    alpha: float = 0.5
    threshold: float = 0.2
    lambda_dag: float = 0.1
    lambda_sp: float = 0.01
    n_iterations: int = 500
    batch_size: int = 1024
    max_samples: int = 50000
    test_size: float = 0.3
    device: str = 'cuda'
    output_dir: str = 'results'

CONFIG = Config()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    sensitive: np.ndarray
    sensitive_name: str
    feature_names: List[str]
    X_train: np.ndarray = None
    X_test: np.ndarray = None
    y_train: np.ndarray = None
    y_test: np.ndarray = None
    sens_train: np.ndarray = None
    sens_test: np.ndarray = None
    # NEW: Ground truth DAG for synthetic datasets
    ground_truth_dag: np.ndarray = None
    causal_features: List[str] = None  # Features with causal effect on Y
    spurious_features: List[str] = None  # Features correlated with A but not causally affecting Y
    
    @property
    def n_features(self): return self.X.shape[1]
    
    def split(self, test_size=0.3, seed=42):
        self.X_train, self.X_test, self.y_train, self.y_test, self.sens_train, self.sens_test = \
            train_test_split(self.X, self.y, self.sensitive, test_size=test_size, 
                           random_state=seed, stratify=self.y)
        return self


# =============================================================================
# DAG RECOVERY METRICS
# =============================================================================

def compute_dag_metrics(learned_adj: np.ndarray, true_adj: np.ndarray, threshold: float = 0.1):
    """
    Compute DAG recovery metrics.
    
    Args:
        learned_adj: Learned adjacency matrix (continuous weights)
        true_adj: Ground truth adjacency matrix (binary)
        threshold: Threshold to binarize learned adjacency
    
    Returns:
        dict with SHD, precision, recall, F1
    """
    # Binarize learned adjacency
    learned_binary = (np.abs(learned_adj) > threshold).astype(int)
    true_binary = (np.abs(true_adj) > 0).astype(int)
    
    # Flatten for comparison (excluding diagonal)
    n = learned_binary.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    learned_flat = learned_binary[mask]
    true_flat = true_binary[mask]
    
    # True positives, false positives, false negatives
    tp = np.sum((learned_flat == 1) & (true_flat == 1))
    fp = np.sum((learned_flat == 1) & (true_flat == 0))
    fn = np.sum((learned_flat == 0) & (true_flat == 1))
    tn = np.sum((learned_flat == 0) & (true_flat == 0))
    
    # Structural Hamming Distance
    shd = fp + fn  # Missing edges + extra edges
    
    # Also count reversed edges if we care about direction
    # For undirected comparison:
    learned_undir = np.maximum(learned_binary, learned_binary.T)
    true_undir = np.maximum(true_binary, true_binary.T)
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Normalized SHD (by number of possible edges)
    n_possible_edges = n * (n - 1)
    shd_normalized = shd / n_possible_edges if n_possible_edges > 0 else 0.0
    
    return {
        'shd': shd,
        'shd_normalized': shd_normalized,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }


def compute_feature_selection_metrics(selected_features: np.ndarray, 
                                       causal_indices: List[int],
                                       spurious_indices: List[int],
                                       n_total_features: int):
    """
    Compute feature selection accuracy metrics.
    
    Args:
        selected_features: Indices of selected features
        causal_indices: Indices of truly causal features
        spurious_indices: Indices of spurious features
        n_total_features: Total number of features
    
    Returns:
        dict with feature selection metrics
    """
    selected_set = set(selected_features)
    causal_set = set(causal_indices)
    spurious_set = set(spurious_indices)
    
    # Causal feature recovery
    causal_selected = selected_set & causal_set
    causal_precision = len(causal_selected) / len(selected_set) if len(selected_set) > 0 else 0
    causal_recall = len(causal_selected) / len(causal_set) if len(causal_set) > 0 else 0
    causal_f1 = 2 * causal_precision * causal_recall / (causal_precision + causal_recall) \
                if (causal_precision + causal_recall) > 0 else 0
    
    # Spurious feature rejection
    spurious_selected = selected_set & spurious_set
    spurious_rejection_rate = 1 - (len(spurious_selected) / len(spurious_set)) \
                              if len(spurious_set) > 0 else 1.0
    
    return {
        'causal_precision': causal_precision,
        'causal_recall': causal_recall,
        'causal_f1': causal_f1,
        'spurious_rejection_rate': spurious_rejection_rate,
        'n_causal_selected': len(causal_selected),
        'n_spurious_selected': len(spurious_selected),
        'n_total_selected': len(selected_set)
    }


# =============================================================================
# FAIRNESS METRICS
# =============================================================================

def compute_metrics(y_true, y_pred, y_prob, sensitive):
    """Compute all fairness and performance metrics."""
    metrics = {}
    
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['auc'] = 0.5
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    unique_groups = np.unique(sensitive)
    tpr_by_g, fpr_by_g, acc_by_g, rate_by_g = {}, {}, {}, {}
    
    for g in unique_groups:
        mask = sensitive == g
        if mask.sum() == 0: continue
        
        y_t, y_p = y_true[mask], y_pred[mask]
        acc_by_g[g] = accuracy_score(y_t, y_p)
        rate_by_g[g] = y_p.mean()
        
        pos, neg = y_t == 1, y_t == 0
        tpr_by_g[g] = (y_p[pos] == 1).mean() if pos.sum() > 0 else 0
        fpr_by_g[g] = (y_p[neg] == 1).mean() if neg.sum() > 0 else 0
    
    metrics['wga'] = min(acc_by_g.values()) if acc_by_g else 0
    
    if len(rate_by_g) > 1:
        rates = list(rate_by_g.values())
        metrics['dpd'] = max(rates) - min(rates)
    else:
        metrics['dpd'] = 0
    
    if len(tpr_by_g) > 1:
        tprs, fprs = list(tpr_by_g.values()), list(fpr_by_g.values())
        metrics['eod'] = max(max(tprs) - min(tprs), max(fprs) - min(fprs))
    else:
        metrics['eod'] = 0
    
    return metrics


# =============================================================================
# DATASET LOADERS
# =============================================================================

def preprocess(df, name, target, sensitive, cat_cols, cont_cols, max_samples=None, seed=42):
    """Unified preprocessing."""
    logger.info(f"Processing {name}...")
    
    df = df.dropna(subset=[target, sensitive])
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
    
    existing_cat = [c for c in cat_cols if c in df.columns]
    existing_cont = [c for c in cont_cols if c in df.columns]
    
    for col in existing_cont:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    for col in existing_cat:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    if existing_cont:
        df[existing_cont] = StandardScaler().fit_transform(df[existing_cont])
    
    y = df[target].values.astype(np.float32)
    if y.max() > 1: y = (y > y.median()).astype(np.float32)
    
    sens = LabelEncoder().fit_transform(df[sensitive].astype(str))
    X = df[existing_cat + existing_cont].values.astype(np.float32)
    
    logger.info(f"  {name}: n={len(X)}, d={X.shape[1]}, groups={len(np.unique(sens))}")
    
    return DatasetBundle(name, X, y, sens, sensitive, existing_cat + existing_cont)


# =============================================================================
# SYNTHETIC DATASET LOADERS WITH GROUND TRUTH DAGs
# =============================================================================

def load_synthetic_loan(filepath: str = None, max_samples: int = None):
    """
    Load synthetic loan dataset with known ground truth causal structure.
    
    Ground Truth Causal Structure for Loan Approval:
    - Causal features (directly affect loan_approved): income, credit_score, employment_years
    - Spurious features (correlated with gender but don't cause approval): 
      works_in_tech, has_stem_degree, plays_golf, favorite_color_blue, birth_month
    - Sensitive attribute: gender
    - Target: loan_approved
    
    True DAG:
    income -> loan_approved
    credit_score -> loan_approved  
    employment_years -> loan_approved
    gender -> works_in_tech (gender affects tech employment)
    gender -> has_stem_degree (gender affects STEM degree)
    gender -> plays_golf (spurious correlation)
    """
    logger.info("Loading Synthetic Loan dataset...")
    
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        # Try default paths
        default_paths = [
            'synthetic_loan_data.csv',
            '/mnt/user-data/uploads/synthetic_loan_data.csv',
            'data/synthetic_loan_data.csv'
        ]
        df = None
        for path in default_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info(f"  Loaded from {path}")
                break
        
        if df is None:
            raise FileNotFoundError(
                "synthetic_loan_data.csv not found. Please provide the filepath or "
                "place the file in the current directory."
            )
    
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    # Define feature columns
    feature_cols = ['income', 'credit_score', 'employment_years', 'works_in_tech', 
                    'has_stem_degree', 'plays_golf', 'favorite_color_blue', 'birth_month']
    
    # Ensure all columns exist
    missing_cols = [c for c in feature_cols + ['gender', 'loan_approved'] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in synthetic_loan_data.csv: {missing_cols}")
    
    # Causal vs Spurious features
    causal_features = ['income', 'credit_score', 'employment_years']
    spurious_features = ['works_in_tech', 'has_stem_degree', 'plays_golf', 
                         'favorite_color_blue', 'birth_month']
    
    # Continuous vs Categorical
    cont_cols = ['income', 'credit_score', 'employment_years']
    cat_cols = ['works_in_tech', 'has_stem_degree', 'plays_golf', 
                'favorite_color_blue', 'birth_month']
    
    # Preprocess
    for col in cont_cols:
        df[col] = StandardScaler().fit_transform(df[[col]])
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['loan_approved'].values.astype(np.float32)
    sensitive = df['gender'].values.astype(np.float32)
    
    # Create ground truth DAG
    # Node order: [features..., gender (A), loan_approved (Y)]
    # features: 0-7, gender: 8, loan_approved: 9
    n_nodes = len(feature_cols) + 2  # features + A + Y
    ground_truth_dag = np.zeros((n_nodes, n_nodes))
    
    # Feature indices
    feat_idx = {f: i for i, f in enumerate(feature_cols)}
    idx_A = len(feature_cols)  # gender index
    idx_Y = len(feature_cols) + 1  # loan_approved index
    
    # Causal edges: causal features -> Y
    for feat in causal_features:
        ground_truth_dag[feat_idx[feat], idx_Y] = 1.0
    
    # Gender -> spurious features (confounding)
    for feat in ['works_in_tech', 'has_stem_degree', 'plays_golf']:
        if feat in feat_idx:
            ground_truth_dag[idx_A, feat_idx[feat]] = 1.0
    
    # Get indices for causal and spurious features
    causal_indices = [feat_idx[f] for f in causal_features]
    spurious_indices = [feat_idx[f] for f in spurious_features if f in feat_idx]
    
    logger.info(f"  Synthetic Loan: n={len(X)}, d={X.shape[1]}")
    logger.info(f"  Causal features: {causal_features}")
    logger.info(f"  Spurious features: {spurious_features}")
    
    dataset = DatasetBundle(
        name='synthetic_loan',
        X=X,
        y=y,
        sensitive=sensitive,
        sensitive_name='gender',
        feature_names=feature_cols,
        ground_truth_dag=ground_truth_dag,
        causal_features=causal_features,
        spurious_features=spurious_features
    )
    
    # Store indices for later use
    dataset.causal_indices = causal_indices
    dataset.spurious_indices = spurious_indices
    
    return dataset


def load_synthetic_hiring(filepath: str = None, max_samples: int = None):
    """
    Load synthetic hiring dataset with known ground truth causal structure.
    
    Ground Truth Causal Structure for Hiring:
    - Causal features (directly affect hired): years_experience, coding_score, 
      education_level, portfolio_quality
    - Spurious features (correlated with race but don't cause hiring):
      ivy_league, unpaid_internships, golf_club_member, lacrosse_player, 
      birth_month, zodiac_fire_sign
    - Sensitive attribute: race
    - Target: hired
    
    True DAG:
    years_experience -> hired
    coding_score -> hired
    education_level -> hired
    portfolio_quality -> hired
    race -> ivy_league (socioeconomic correlation)
    race -> unpaid_internships (socioeconomic - ability to take unpaid work)
    race -> golf_club_member (socioeconomic/cultural)
    race -> lacrosse_player (socioeconomic/cultural)
    """
    logger.info("Loading Synthetic Hiring dataset...")
    
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        # Try default paths
        default_paths = [
            'synthetic_hiring_data.csv',
            '/mnt/user-data/uploads/synthetic_hiring_data.csv',
            'data/synthetic_hiring_data.csv'
        ]
        df = None
        for path in default_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info(f"  Loaded from {path}")
                break
        
        if df is None:
            raise FileNotFoundError(
                "synthetic_hiring_data.csv not found. Please provide the filepath or "
                "place the file in the current directory."
            )
    
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    # Define feature columns
    feature_cols = ['years_experience', 'coding_score', 'education_level', 'portfolio_quality',
                    'ivy_league', 'unpaid_internships', 'golf_club_member', 'lacrosse_player',
                    'birth_month', 'zodiac_fire_sign']
    
    # Ensure all columns exist
    missing_cols = [c for c in feature_cols + ['race', 'hired'] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in synthetic_hiring_data.csv: {missing_cols}")
    
    # Causal vs Spurious features
    causal_features = ['years_experience', 'coding_score', 'education_level', 'portfolio_quality']
    spurious_features = ['ivy_league', 'unpaid_internships', 'golf_club_member', 
                         'lacrosse_player', 'birth_month', 'zodiac_fire_sign']
    
    # Continuous vs Categorical
    cont_cols = ['years_experience', 'coding_score', 'portfolio_quality']
    cat_cols = ['education_level', 'ivy_league', 'unpaid_internships', 'golf_club_member',
                'lacrosse_player', 'birth_month', 'zodiac_fire_sign']
    
    # Preprocess
    for col in cont_cols:
        if col in df.columns:
            df[col] = StandardScaler().fit_transform(df[[col]])
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['hired'].values.astype(np.float32)
    sensitive = df['race'].values.astype(np.float32)
    
    # Create ground truth DAG
    n_nodes = len(feature_cols) + 2  # features + A + Y
    ground_truth_dag = np.zeros((n_nodes, n_nodes))
    
    # Feature indices
    feat_idx = {f: i for i, f in enumerate(feature_cols)}
    idx_A = len(feature_cols)  # race index
    idx_Y = len(feature_cols) + 1  # hired index
    
    # Causal edges: causal features -> Y
    for feat in causal_features:
        ground_truth_dag[feat_idx[feat], idx_Y] = 1.0
    
    # Race -> spurious features (confounding through socioeconomic factors)
    for feat in ['ivy_league', 'unpaid_internships', 'golf_club_member', 'lacrosse_player']:
        if feat in feat_idx:
            ground_truth_dag[idx_A, feat_idx[feat]] = 1.0
    
    # Get indices for causal and spurious features
    causal_indices = [feat_idx[f] for f in causal_features]
    spurious_indices = [feat_idx[f] for f in spurious_features if f in feat_idx]
    
    logger.info(f"  Synthetic Hiring: n={len(X)}, d={X.shape[1]}")
    logger.info(f"  Causal features: {causal_features}")
    logger.info(f"  Spurious features: {spurious_features}")
    
    dataset = DatasetBundle(
        name='synthetic_hiring',
        X=X,
        y=y,
        sensitive=sensitive,
        sensitive_name='race',
        feature_names=feature_cols,
        ground_truth_dag=ground_truth_dag,
        causal_features=causal_features,
        spurious_features=spurious_features
    )
    
    # Store indices for later use
    dataset.causal_indices = causal_indices
    dataset.spurious_indices = spurious_indices
    
    return dataset


# =============================================================================
# STANDARD DATASET LOADERS
# =============================================================================

def load_adult(max_samples=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(url, names=cols, sep=r',\s*', engine='python', na_values='?')
    df['income'] = (df['income'].str.strip() == '>50K').astype(int)
    df['sex'] = df['sex'].str.strip()
    return preprocess(df, 'adult', 'income', 'sex',
                     ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
                     ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
                     max_samples)


def load_compas(max_samples=None):
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)
    df = df[(df['days_b_screening_arrest'] >= -30) & (df['days_b_screening_arrest'] <= 30) &
            (df['is_recid'] != -1) & (df['c_charge_degree'] != 'O')]
    df['race_binary'] = df['race'].apply(lambda x: 'AA' if x == 'African-American' else 'Other')
    return preprocess(df, 'compas', 'two_year_recid', 'race_binary',
                     ['sex', 'c_charge_degree'], ['age', 'juv_fel_count', 'juv_misd_count', 'priors_count'],
                     max_samples)


def load_german(max_samples=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    cols = ['status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings',
            'employment', 'installment_rate', 'personal_status_sex', 'other_debtors', 'residence',
            'property', 'age', 'other_installments', 'housing', 'existing_credits', 'job',
            'dependents', 'telephone', 'foreign_worker', 'credit']
    df = pd.read_csv(url, names=cols, sep=' ')
    df['credit'] = (df['credit'] == 1).astype(int)
    df['sex'] = df['personal_status_sex'].apply(lambda x: 'male' if x in ['A91', 'A93', 'A94'] else 'female')
    return preprocess(df, 'german', 'credit', 'sex',
                     ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors',
                      'property', 'other_installments', 'housing', 'job', 'telephone', 'foreign_worker'],
                     ['duration', 'credit_amount', 'installment_rate', 'residence', 'age', 
                      'existing_credits', 'dependents'], max_samples)


def load_bank(max_samples=None):
    import urllib.request, zipfile, io
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    with urllib.request.urlopen(url) as resp:
        with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
            with z.open('bank-additional/bank-additional-full.csv') as f:
                df = pd.read_csv(f, sep=';')
    df['y'] = (df['y'] == 'yes').astype(int)
    df['age_group'] = df['age'].apply(lambda x: 'adult' if x >= 25 else 'young')
    return preprocess(df, 'bank', 'y', 'age_group',
                     ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                      'month', 'day_of_week', 'poutcome'],
                     ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                      'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], max_samples)


def load_online_shoppers(max_samples=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    df = pd.read_csv(url)
    df['Revenue'] = df['Revenue'].astype(int)
    df['Weekend'] = df['Weekend'].apply(lambda x: 'weekend' if x else 'weekday')
    return preprocess(df, 'online_shoppers', 'Revenue', 'Weekend',
                     ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType'],
                     ['Administrative', 'Administrative_Duration', 'Informational',
                      'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                      'BounceRates', 'ExitRates', 'PageValues'], max_samples)


def load_acs_income(max_samples=None, states=['CA'], year='2018'):
    if not HAS_FOLKTABLES: 
        raise ImportError("folktables required")
    
    logger.info(f"Loading ACS Income dataset...")
    years_to_try = [str(year), '2019', '2017', '2021', '2020']
    acs = None
    
    for yr in years_to_try:
        try:
            data_source = ACSDataSource(survey_year=yr, horizon='1-Year', survey='person')
            acs = data_source.get_data(states=states, download=True)
            break
        except:
            continue
    
    if acs is None:
        raise RuntimeError("Could not load ACS Income data")
    
    features, labels, _ = ACSIncome.df_to_numpy(acs)
    feature_names = list(ACSIncome.features)
    
    df = pd.DataFrame(features, columns=feature_names)
    df['income'] = labels.astype(int)
    df['race'] = (df['RAC1P'] == 1).apply(lambda x: 'White' if x else 'NonWhite')
    
    cat_features = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P', 'SEX']
    cat_cols = [f for f in cat_features if f in feature_names]
    cont_cols = [f for f in feature_names if f not in cat_cols]
    
    return preprocess(df, 'acs_income', 'income', 'race', cat_cols, cont_cols, max_samples)


def load_acs_employment(max_samples=None, states=['CA'], year='2018'):
    if not HAS_FOLKTABLES: 
        raise ImportError("folktables required")
    
    logger.info(f"Loading ACS Employment dataset...")
    years_to_try = [str(year), '2019', '2017', '2021', '2020']
    acs = None
    
    for yr in years_to_try:
        try:
            data_source = ACSDataSource(survey_year=yr, horizon='1-Year', survey='person')
            acs = data_source.get_data(states=states, download=True)
            break
        except:
            continue
    
    if acs is None:
        raise RuntimeError("Could not load ACS Employment data")
    
    features, labels, _ = ACSEmployment.df_to_numpy(acs)
    feature_names = list(ACSEmployment.features)
    
    df = pd.DataFrame(features, columns=feature_names)
    df['employed'] = labels.astype(int)
    df['race'] = (df['RAC1P'] == 1).apply(lambda x: 'White' if x else 'NonWhite')
    
    cat_features = ['MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS',
                    'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC']
    cat_cols = [f for f in cat_features if f in feature_names]
    cont_cols = [f for f in feature_names if f not in cat_cols]
    
    return preprocess(df, 'acs_employment', 'employed', 'race', cat_cols, cont_cols, max_samples)


def load_law_school(max_samples=None):
    logger.info("Loading Law School dataset...")
    
    urls = [
        "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/law_school_clean.csv",
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/lawschool.csv",
    ]
    
    df = None
    for url in urls:
        try:
            df = pd.read_csv(url)
            break
        except:
            continue
    
    if df is None:
        # Create synthetic
        np.random.seed(42)
        n_samples = 20000
        lsat = np.random.normal(35, 8, n_samples).clip(10, 48)
        ugpa = np.random.normal(3.2, 0.5, n_samples).clip(1.0, 4.0)
        race = np.random.choice(['White', 'NonWhite'], n_samples, p=[0.75, 0.25])
        sex = np.random.choice([1, 2], n_samples, p=[0.52, 0.48])
        pass_prob = 0.3 + 0.4 * (lsat - 10) / 38 + 0.2 * (ugpa - 1) / 3
        passed = (np.random.random(n_samples) < pass_prob.clip(0.1, 0.95)).astype(int)
        df = pd.DataFrame({'lsat': lsat.astype(int), 'ugpa': ugpa.round(2), 
                          'sex': sex, 'race': race, 'passed': passed})
    
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    target_col = None
    for col in ['pass_bar', 'bar', 'passed', 'target']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    df['passed'] = (df[target_col] > 0).astype(int) if df[target_col].dtype != 'object' else \
                   df[target_col].astype(str).str.lower().isin(['yes', '1', 'true', 'pass']).astype(int)
    
    race_col = 'race' if 'race' in df.columns else None
    if race_col:
        df['race_bin'] = df[race_col].astype(str).str.lower().apply(
            lambda x: 'White' if 'white' in x else 'NonWhite')
    else:
        np.random.seed(42)
        df['race_bin'] = np.random.choice(['White', 'NonWhite'], size=len(df), p=[0.7, 0.3])
    
    feature_cols = [c for c in df.columns if c not in {'passed', target_col, 'race_bin', race_col}]
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object' or df[c].nunique() < 10]
    cont_cols = [c for c in feature_cols if c not in cat_cols]
    
    return preprocess(df, 'law_school', 'passed', 'race_bin', cat_cols, cont_cols, max_samples)


def load_taiwan_credit(max_samples=None):
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        df = pd.read_excel(url, header=1)
        target = [c for c in df.columns if 'default' in c.lower()][0]
        df['sex_name'] = df['SEX'].apply(lambda x: 'male' if x == 1 else 'female')
        return preprocess(df, 'taiwan_credit', target, 'sex_name',
                         ['SEX', 'EDUCATION', 'MARRIAGE'],
                         ['LIMIT_BAL', 'AGE'] + [f'PAY_{i}' for i in range(7)] + 
                         [f'BILL_AMT{i}' for i in range(1,7)] + [f'PAY_AMT{i}' for i in range(1,7)],
                         max_samples)
    except Exception as e:
        logger.warning(f"Could not load Taiwan Credit: {e}")
        return None


LOADERS = {
    'adult': load_adult, 'compas': load_compas, 'german': load_german,
    'bank': load_bank, 'online_shoppers': load_online_shoppers,
    'acs_income': load_acs_income, 'acs_employment': load_acs_employment,
    'law_school': load_law_school, 'taiwan_credit': load_taiwan_credit,
    'synthetic_loan': load_synthetic_loan, 'synthetic_hiring': load_synthetic_hiring,
}


# =============================================================================
# CAUSAL FEATURE SELECTOR (OUR METHOD) - ENHANCED WITH DAG EXTRACTION
# =============================================================================

class CausalFeatureSelector:
    """CausalGBM feature selection with DAG learning and extraction."""
    
    def __init__(self, n_features, alpha=0.5, threshold=0.2, min_features=3,
                 n_iterations=500, lambda_dag=0.1, lambda_sp=0.01,
                 aggregation='max', device='cuda'):
        self.n_features = n_features
        self.alpha = alpha
        self.threshold = threshold
        self.min_features = min_features
        self.n_iterations = n_iterations
        self.lambda_dag = lambda_dag
        self.lambda_sp = lambda_sp
        self.aggregation = aggregation
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.selected_ = None
        self.W_dag_ = None
        self.correlations_ = None
        self.causal_importance_ = None
        self.converged_ = False
        self.dag_loss_ = None
        self.learned_adjacency_ = None  # Full learned DAG adjacency matrix
    
    def fit(self, X, A, y):
        n, d = X.shape
        
        # Encode A if needed
        if not np.issubdtype(A.dtype, np.number):
            A = LabelEncoder().fit_transform(A).astype(np.float32)
        else:
            A = A.astype(np.float32)
        
        # Correlations
        self.correlations_ = np.array([
            abs(pearsonr(X[:, j], A)[0]) if np.std(X[:, j]) > 1e-6 else 0 for j in range(d)
        ])
        
        # Standardize for DAG learning
        A_std = (A - A.mean()) / (A.std() + 1e-8)
        y_std = (y - y.mean()) / (y.std() + 1e-8)
        Z = np.column_stack([X, A_std.reshape(-1,1), y_std.reshape(-1,1)]).astype(np.float32)
        
        n_nodes = d + 2
        idx_A, idx_Y = d, d + 1
        
        # DAG learning
        W = nn.Parameter(torch.randn(n_nodes, n_nodes, device=self.device) * 0.01)
        opt = torch.optim.Adam([W], lr=0.01)
        Z_t = torch.FloatTensor(Z).to(self.device)
        
        for _ in range(self.n_iterations):
            opt.zero_grad()
            A_mat = torch.sigmoid(W) * (1 - torch.eye(n_nodes, device=self.device))
            
            recon = F.mse_loss(Z_t @ A_mat, Z_t)
            
            # DAGMA constraint
            s = 1.0
            M = s * torch.eye(n_nodes, device=self.device) - A_mat * A_mat + 1e-6 * torch.eye(n_nodes, device=self.device)
            try:
                sign, logdet = torch.linalg.slogdet(M)
                dag_c = -logdet + n_nodes * np.log(s) if sign > 0 else torch.trace(torch.matrix_exp(A_mat * A_mat)) - n_nodes
            except:
                dag_c = torch.trace(torch.matrix_exp(A_mat * A_mat)) - n_nodes
            
            loss = recon + self.lambda_dag * dag_c ** 2 + self.lambda_sp * A_mat.abs().mean()
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_([W], 1.0)
                opt.step()
        
        with torch.no_grad():
            A_mat = torch.sigmoid(W) * (1 - torch.eye(n_nodes, device=self.device))
            adj = A_mat.cpu().numpy()
            self.dag_loss_ = dag_c.item() if not torch.isnan(dag_c) else float('inf')
            self.learned_adjacency_ = adj.copy()  # Store full learned adjacency
        
        # Extract weights
        W_X_Y = adj[:d, idx_Y]
        W_A_X = adj[idx_A, :d]
        self.W_dag_ = W_A_X
        self.W_X_Y_ = W_X_Y  # Store X->Y weights
        
        # Aggregation
        if self.aggregation == 'max':
            W_prime = np.maximum(W_A_X, self.correlations_)
        elif self.aggregation == 'dag_only':
            W_prime = W_A_X
        elif self.aggregation == 'corr_only':
            W_prime = self.correlations_
        elif self.aggregation == 'sum':
            W_prime = W_A_X + self.correlations_
        else:
            W_prime = np.maximum(W_A_X, self.correlations_)
        
        # Causal importance
        self.causal_importance_ = W_X_Y - self.alpha * W_prime
        
        # Selection
        above = np.where(self.causal_importance_ >= self.threshold)[0]
        if len(above) >= self.min_features:
            self.selected_ = above
        else:
            self.selected_ = np.argsort(self.causal_importance_)[::-1][:self.min_features]
        
        return self
    
    def transform(self, X):
        return X[:, self.selected_]
    
    def get_dag_adjacency(self):
        """Return the learned adjacency matrix."""
        return self.learned_adjacency_


# =============================================================================
# DEEP LEARNING MODELS
# =============================================================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class TabTransformer(nn.Module):
    def __init__(self, input_dim, n_heads=4, n_layers=2, d_model=64, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        x = self.transformer(x)
        return self.head(x.squeeze(1))


class FTTransformer(nn.Module):
    def __init__(self, input_dim, n_heads=4, n_layers=2, d_token=32, dropout=0.1):
        super().__init__()
        self.tokenizers = nn.ModuleList([nn.Linear(1, d_token) for _ in range(input_dim)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(d_token, n_heads, dim_feedforward=d_token*4,
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_token, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        tokens = [tok(x[:, i:i+1]) for i, tok in enumerate(self.tokenizers)]
        tokens = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        return self.head(out[:, 0])


class SAINT(nn.Module):
    def __init__(self, input_dim, n_heads=4, n_layers=2, d_model=64, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*2, 
                                       dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        for layer in self.self_attn_layers:
            x = layer(x)
        return self.head(x.squeeze(1))


def train_nn_model(model, X_train, y_train, X_test, device='cuda', epochs=100, batch_size=256, lr=1e-3):
    device = device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for _ in range(epochs):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        logits = model(X_test_t).squeeze().cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
    
    return preds, probs


# =============================================================================
# FAIRNESS METHODS
# =============================================================================

class GroupDRO:
    def __init__(self, base_model, n_groups, eta=0.1, n_epochs=50):
        self.base_model = base_model
        self.n_groups = n_groups
        self.eta = eta
        self.n_epochs = n_epochs
        self.group_weights = np.ones(n_groups) / n_groups
    
    def fit(self, X, y, groups):
        for _ in range(self.n_epochs):
            sample_weights = np.array([self.group_weights[g] for g in groups])
            sample_weights /= sample_weights.sum()
            self.base_model.fit(X, y, sample_weight=sample_weights)
            
            y_pred_prob = self.base_model.predict_proba(X)[:, 1]
            losses_per_group = []
            for g in range(self.n_groups):
                mask = groups == g
                if mask.sum() > 0:
                    group_loss = -np.mean(y[mask] * np.log(y_pred_prob[mask] + 1e-8) + 
                                          (1 - y[mask]) * np.log(1 - y_pred_prob[mask] + 1e-8))
                    losses_per_group.append(group_loss)
                else:
                    losses_per_group.append(0)
            
            losses_per_group = np.array(losses_per_group)
            self.group_weights = self.group_weights * np.exp(self.eta * losses_per_group)
            self.group_weights /= self.group_weights.sum()
        return self
    
    def predict(self, X):
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)


class CounterfactualFairClassifier:
    def __init__(self, base_model):
        self.base_model = base_model
        self.residualizer = None
    
    def fit(self, X, y, sensitive):
        self.residualizer = LinearRegression()
        self.residualizer.fit(sensitive.reshape(-1, 1), X)
        X_resid = X - self.residualizer.predict(sensitive.reshape(-1, 1))
        self.base_model.fit(X_resid, y)
        return self
    
    def predict(self, X, sensitive):
        X_resid = X - self.residualizer.predict(sensitive.reshape(-1, 1))
        return self.base_model.predict(X_resid)
    
    def predict_proba(self, X, sensitive):
        X_resid = X - self.residualizer.predict(sensitive.reshape(-1, 1))
        return self.base_model.predict_proba(X_resid)


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_method(method_name, dataset, seed, device='cuda', return_selector=False, **kwargs):
    """Run a single method on a dataset."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset.split(seed=seed)
    X_tr, X_te = dataset.X_train, dataset.X_test
    y_tr, y_te = dataset.y_train, dataset.y_test
    sens_tr, sens_te = dataset.sens_train, dataset.sens_test
    
    start = time.time()
    n_feats = X_tr.shape[1]
    selector = None
    
    try:
        # Standard ML methods
        if method_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
        
        elif method_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
        
        elif method_name == 'XGBoost' and HAS_XGB:
            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=seed, 
                                       eval_metric='logloss', verbosity=0)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
        
        elif method_name == 'LightGBM' and HAS_LGB:
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=seed, verbosity=-1)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
        
        # Deep learning
        elif method_name == 'MLP':
            model = MLP(X_tr.shape[1])
            y_pred, y_prob = train_nn_model(model, X_tr, y_tr, X_te, device)
        
        elif method_name == 'TabTransformer':
            model = TabTransformer(X_tr.shape[1])
            y_pred, y_prob = train_nn_model(model, X_tr, y_tr, X_te, device)
        
        elif method_name == 'FT-Transformer':
            model = FTTransformer(X_tr.shape[1])
            y_pred, y_prob = train_nn_model(model, X_tr, y_tr, X_te, device)
        
        elif method_name == 'SAINT':
            model = SAINT(X_tr.shape[1])
            y_pred, y_prob = train_nn_model(model, X_tr, y_tr, X_te, device)
        
        # Fairness methods
        elif method_name == 'GroupDRO':
            base = LogisticRegression(max_iter=500, random_state=seed)
            model = GroupDRO(base, n_groups=len(np.unique(sens_tr)))
            model.fit(X_tr, y_tr, sens_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
        
        elif method_name == 'CounterfactualFair':
            base = LogisticRegression(max_iter=500, random_state=seed)
            model = CounterfactualFairClassifier(base)
            model.fit(X_tr, y_tr, sens_tr)
            y_pred = model.predict(X_te, sens_te)
            y_prob = model.predict_proba(X_te, sens_te)[:, 1]
        
        elif method_name == 'FairLearn-EO' and HAS_FAIRLEARN:
            base = LogisticRegression(max_iter=500, random_state=seed)
            model = ExponentiatedGradient(base, constraints=EqualizedOdds())
            model.fit(X_tr, y_tr, sensitive_features=sens_tr)
            y_pred = model.predict(X_te)
            y_prob = model._pmf_predict(X_te)[:, 1] if hasattr(model, '_pmf_predict') else y_pred.astype(float)
        
        elif method_name == 'FairLearn-DP' and HAS_FAIRLEARN:
            base = LogisticRegression(max_iter=500, random_state=seed)
            model = ExponentiatedGradient(base, constraints=DemographicParity())
            model.fit(X_tr, y_tr, sensitive_features=sens_tr)
            y_pred = model.predict(X_te)
            y_prob = model._pmf_predict(X_te)[:, 1] if hasattr(model, '_pmf_predict') else y_pred.astype(float)
        
        # Feature selection methods
        elif method_name.startswith('Corr-'):
            thresh = float(method_name.split('-')[1])
            corrs = np.array([abs(pearsonr(X_tr[:, j], sens_tr)[0]) for j in range(X_tr.shape[1])])
            selected = np.where(corrs < thresh)[0]
            if len(selected) < 3: selected = np.argsort(corrs)[:max(3, X_tr.shape[1]//3)]
            n_feats = len(selected)
            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else \
                    GradientBoostingClassifier(n_estimators=100, random_state=seed)
            model.fit(X_tr[:, selected], y_tr)
            y_pred = model.predict(X_te[:, selected])
            y_prob = model.predict_proba(X_te[:, selected])[:, 1]
        
        elif method_name == 'MutualInfo':
            mi = mutual_info_classif(X_tr, sens_tr, random_state=seed)
            selected = np.argsort(mi)[:max(3, X_tr.shape[1]//3)]
            n_feats = len(selected)
            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else \
                    GradientBoostingClassifier(n_estimators=100, random_state=seed)
            model.fit(X_tr[:, selected], y_tr)
            y_pred = model.predict(X_te[:, selected])
            y_prob = model.predict_proba(X_te[:, selected])[:, 1]
        
        elif method_name == 'RF-Importance':
            rf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            imp = rf.feature_importances_
            selected = np.argsort(imp)[::-1][:max(3, X_tr.shape[1]//3)]
            n_feats = len(selected)
            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else \
                    GradientBoostingClassifier(n_estimators=100, random_state=seed)
            model.fit(X_tr[:, selected], y_tr)
            y_pred = model.predict(X_te[:, selected])
            y_prob = model.predict_proba(X_te[:, selected])[:, 1]
        
        elif method_name == 'RandomSelection':
            np.random.seed(seed)
            selected = np.random.choice(X_tr.shape[1], max(3, X_tr.shape[1]//3), replace=False)
            n_feats = len(selected)
            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else \
                    GradientBoostingClassifier(n_estimators=100, random_state=seed)
            model.fit(X_tr[:, selected], y_tr)
            y_pred = model.predict(X_te[:, selected])
            y_prob = model.predict_proba(X_te[:, selected])[:, 1]
        
        # CausalGBM variants
        elif method_name.startswith('CausalGBM'):
            agg = kwargs.get('aggregation', 'max')
            if '-' in method_name:
                agg = method_name.split('-')[1]
            
            selector = CausalFeatureSelector(
                X_tr.shape[1], 
                alpha=kwargs.get('alpha', 0.5),
                threshold=kwargs.get('threshold', 0.2),
                min_features=max(3, X_tr.shape[1]//3),
                n_iterations=kwargs.get('n_iterations', 500),
                lambda_dag=kwargs.get('lambda_dag', 0.1),
                lambda_sp=kwargs.get('lambda_sp', 0.01),
                aggregation=agg,
                device=device
            )
            selector.fit(X_tr, sens_tr, y_tr)
            X_tr_sel = selector.transform(X_tr)
            X_te_sel = selector.transform(X_te)
            n_feats = len(selector.selected_)
            
            model = xgb.XGBClassifier(n_estimators=100, random_state=seed, verbosity=0) if HAS_XGB else \
                    GradientBoostingClassifier(n_estimators=100, random_state=seed)
            model.fit(X_tr_sel, y_tr)
            y_pred = model.predict(X_te_sel)
            y_prob = model.predict_proba(X_te_sel)[:, 1]
        
        else:
            logger.warning(f"Unknown method: {method_name}")
            return None
        
        elapsed = time.time() - start
        metrics = compute_metrics(y_te, y_pred, y_prob, sens_te)
        
        result = {
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
            **kwargs
        }
        
        if return_selector and selector is not None:
            return result, selector
        return result
    
    except Exception as e:
        logger.error(f"{method_name} on {dataset.name} failed: {e}")
        traceback.print_exc()
        return None


def run_main_comparison(datasets, methods, seeds, output_dir, device='cuda'):
    """Run main comparison across all methods and datasets."""
    logger.info("="*70)
    logger.info("MAIN COMPARISON")
    logger.info("="*70)
    
    results = []
    
    for ds_name, dataset in datasets.items():
        logger.info(f"\nDataset: {ds_name}")
        for method in methods:
            for seed in seeds:
                result = run_method(method, dataset, seed, device)
                if result:
                    results.append(result)
                    logger.info(f"  {method} seed={seed}: AUC={result['auc']:.3f}, EOD={result['eod']:.3f}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'main_comparison_results.csv'), index=False)
    return df


def run_dag_recovery_analysis(datasets, seeds, output_dir, device='cuda'):
    """
    Run DAG recovery analysis on synthetic datasets with known ground truth.
    
    Computes:
    - SHD (Structural Hamming Distance)
    - Precision, Recall, F1 for edge recovery
    - Feature selection accuracy (causal vs spurious)
    """
    logger.info("="*70)
    logger.info("DAG RECOVERY ANALYSIS")
    logger.info("="*70)
    
    dag_results = []
    feature_selection_results = []
    
    # Only run on synthetic datasets with ground truth
    synthetic_datasets = {k: v for k, v in datasets.items() 
                         if v.ground_truth_dag is not None}
    
    if not synthetic_datasets:
        logger.warning("No synthetic datasets with ground truth DAG found!")
        return None, None
    
    aggregations = ['max', 'dag_only', 'corr_only']
    
    for ds_name, dataset in synthetic_datasets.items():
        logger.info(f"\nDataset: {ds_name}")
        logger.info(f"  Ground truth DAG shape: {dataset.ground_truth_dag.shape}")
        logger.info(f"  Causal features: {dataset.causal_features}")
        logger.info(f"  Spurious features: {dataset.spurious_features}")
        
        for seed in seeds:
            for agg in aggregations:
                method_name = f'CausalGBM-{agg}' if agg != 'max' else 'CausalGBM'
                
                result, selector = run_method(
                    method_name, dataset, seed, device, 
                    return_selector=True, aggregation=agg
                )
                
                if result is None or selector is None:
                    continue
                
                # Get learned adjacency matrix
                learned_adj = selector.get_dag_adjacency()
                
                if learned_adj is None:
                    continue
                
                # Compute DAG recovery metrics
                dag_metrics = compute_dag_metrics(
                    learned_adj, 
                    dataset.ground_truth_dag,
                    threshold=0.1
                )
                
                dag_result = {
                    'dataset': ds_name,
                    'method': method_name,
                    'aggregation': agg,
                    'seed': seed,
                    **dag_metrics,
                    'dag_loss': selector.dag_loss_,
                    # Also include prediction metrics
                    'auc': result['auc'],
                    'eod': result['eod'],
                    'dpd': result['dpd']
                }
                dag_results.append(dag_result)
                
                # Compute feature selection metrics
                if hasattr(dataset, 'causal_indices') and hasattr(dataset, 'spurious_indices'):
                    fs_metrics = compute_feature_selection_metrics(
                        selector.selected_,
                        dataset.causal_indices,
                        dataset.spurious_indices,
                        dataset.n_features
                    )
                    
                    fs_result = {
                        'dataset': ds_name,
                        'method': method_name,
                        'aggregation': agg,
                        'seed': seed,
                        **fs_metrics,
                        'selected_features': list(selector.selected_),
                        'feature_names_selected': [dataset.feature_names[i] for i in selector.selected_],
                        # Also include prediction metrics
                        'auc': result['auc'],
                        'eod': result['eod'],
                        'dpd': result['dpd']
                    }
                    feature_selection_results.append(fs_result)
                
                logger.info(f"  {method_name} seed={seed}: "
                           f"SHD={dag_metrics['shd']}, "
                           f"F1={dag_metrics['f1']:.3f}, "
                           f"EOD={result['eod']:.3f}")
    
    # Save results
    if dag_results:
        dag_df = pd.DataFrame(dag_results)
        dag_df.to_csv(os.path.join(output_dir, 'dag_recovery_results.csv'), index=False)
        logger.info(f"\nDAG recovery results saved to {output_dir}/dag_recovery_results.csv")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("DAG RECOVERY SUMMARY")
        logger.info("="*70)
        
        summary = dag_df.groupby(['dataset', 'method']).agg({
            'shd': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'eod': ['mean', 'std']
        }).round(4)
        logger.info(f"\n{summary}")
    
    if feature_selection_results:
        fs_df = pd.DataFrame(feature_selection_results)
        fs_df.to_csv(os.path.join(output_dir, 'feature_selection_results.csv'), index=False)
        logger.info(f"\nFeature selection results saved to {output_dir}/feature_selection_results.csv")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("FEATURE SELECTION SUMMARY")
        logger.info("="*70)
        
        fs_summary = fs_df.groupby(['dataset', 'method']).agg({
            'causal_precision': ['mean', 'std'],
            'causal_recall': ['mean', 'std'],
            'causal_f1': ['mean', 'std'],
            'spurious_rejection_rate': ['mean', 'std']
        }).round(4)
        logger.info(f"\n{fs_summary}")
    
    return dag_df if dag_results else None, fs_df if feature_selection_results else None


def run_ablation_analysis(datasets, seeds, output_dir, device='cuda'):
    """Run ablation on DAG vs correlation components."""
    logger.info("="*70)
    logger.info("ABLATION ANALYSIS")
    logger.info("="*70)
    
    results = []
    aggregations = ['max', 'dag_only', 'corr_only', 'sum']
    
    for ds_name, dataset in datasets.items():
        logger.info(f"\nDataset: {ds_name}")
        
        for seed in seeds:
            for agg in aggregations:
                result = run_method(f'CausalGBM-{agg}', dataset, seed, device, aggregation=agg)
                if result:
                    result['aggregation'] = agg
                    results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)
    return df


def run_sensitivity_analysis(datasets, seeds, output_dir, device='cuda'):
    """Run sensitivity analysis on CausalGBM hyperparameters."""
    logger.info("="*70)
    logger.info("SENSITIVITY ANALYSIS")
    logger.info("="*70)
    
    results = []
    
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    threshold_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    lambda_dag_values = [0.01, 0.1, 1.0]
    lambda_sp_values = [0.001, 0.01, 0.1]
    iteration_values = [200, 500, 1000]
    
    for ds_name, dataset in datasets.items():
        logger.info(f"\nDataset: {ds_name}")
        
        for seed in seeds:
            for alpha in alpha_values:
                result = run_method('CausalGBM', dataset, seed, device, alpha=alpha)
                if result:
                    result['hyperparameter'] = 'alpha'
                    result['hp_value'] = alpha
                    results.append(result)
            
            for thresh in threshold_values:
                result = run_method('CausalGBM', dataset, seed, device, threshold=thresh)
                if result:
                    result['hyperparameter'] = 'threshold'
                    result['hp_value'] = thresh
                    results.append(result)
            
            for lam in lambda_dag_values:
                result = run_method('CausalGBM', dataset, seed, device, lambda_dag=lam)
                if result:
                    result['hyperparameter'] = 'lambda_dag'
                    result['hp_value'] = lam
                    results.append(result)
            
            for lam in lambda_sp_values:
                result = run_method('CausalGBM', dataset, seed, device, lambda_sp=lam)
                if result:
                    result['hyperparameter'] = 'lambda_sp'
                    result['hp_value'] = lam
                    results.append(result)
            
            for iters in iteration_values:
                result = run_method('CausalGBM', dataset, seed, device, n_iterations=iters)
                if result:
                    result['hyperparameter'] = 'n_iterations'
                    result['hp_value'] = iters
                    results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'sensitivity_analysis_results.csv'), index=False)
    return df


def generate_summary(output_dir):
    """Generate summary of all experiments."""
    logger.info("="*70)
    logger.info("GENERATING SUMMARY")
    logger.info("="*70)
    
    lines = ["="*70, "EXPERIMENT SUMMARY", f"Generated: {datetime.now()}", "="*70]
    
    # Main comparison summary
    main_path = os.path.join(output_dir, 'main_comparison_results.csv')
    if os.path.exists(main_path):
        df = pd.read_csv(main_path)
        lines.append("\n\n=== MAIN COMPARISON ===")
        summary = df.groupby(['dataset', 'method']).agg({
            'auc': ['mean', 'std'],
            'eod': ['mean', 'std'],
            'dpd': ['mean', 'std'],
            'wga': ['mean', 'std']
        }).round(4)
        lines.append(str(summary))
    
    # DAG recovery summary
    dag_path = os.path.join(output_dir, 'dag_recovery_results.csv')
    if os.path.exists(dag_path):
        df = pd.read_csv(dag_path)
        lines.append("\n\n=== DAG RECOVERY ===")
        summary = df.groupby(['dataset', 'method']).agg({
            'shd': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1': ['mean', 'std']
        }).round(4)
        lines.append(str(summary))
    
    # Feature selection summary
    fs_path = os.path.join(output_dir, 'feature_selection_results.csv')
    if os.path.exists(fs_path):
        df = pd.read_csv(fs_path)
        lines.append("\n\n=== FEATURE SELECTION (Causal vs Spurious) ===")
        summary = df.groupby(['dataset', 'method']).agg({
            'causal_precision': ['mean', 'std'],
            'causal_recall': ['mean', 'std'],
            'causal_f1': ['mean', 'std'],
            'spurious_rejection_rate': ['mean', 'std']
        }).round(4)
        lines.append(str(summary))
    
    # Ablation summary
    abl_path = os.path.join(output_dir, 'ablation_results.csv')
    if os.path.exists(abl_path):
        df = pd.read_csv(abl_path)
        lines.append("\n\n=== ABLATION (DAG vs Correlation) ===")
        summary = df.groupby(['dataset', 'aggregation'])['eod'].agg(['mean', 'std']).round(4)
        lines.append(str(summary))
    
    # Save summary
    summary_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Summary saved to {summary_path}")
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CausalGBM Comprehensive Experiments v2')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--main_comparison', action='store_true')
    parser.add_argument('--dag_recovery', action='store_true', help='Run DAG recovery analysis on synthetic data')
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    
    parser.add_argument('--datasets', nargs='+', 
                        default=['adult', 'compas', 'german', 'bank', 'online_shoppers',
                                'acs_income', 'acs_employment', 'law_school', 'taiwan_credit',
                                'synthetic_loan', 'synthetic_hiring'],
                        help='Datasets to use')
    parser.add_argument('--synthetic_loan_path', type=str, default=None,
                        help='Path to synthetic_loan_data.csv')
    parser.add_argument('--synthetic_hiring_path', type=str, default=None,
                        help='Path to synthetic_hiring_data.csv')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--max_samples', type=int, default=200000)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    seeds = [42, 43] if args.quick else [42, 43, 44, 45, 46]
    
    logger.info("="*70)
    logger.info("CausalGBM Comprehensive Experiments v2")
    logger.info("="*70)
    logger.info(f"Device: {device}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Started: {datetime.now()}")
    
    # Update loaders with custom paths
    if args.synthetic_loan_path:
        LOADERS['synthetic_loan'] = lambda max_samples=None: load_synthetic_loan(
            args.synthetic_loan_path, max_samples)
    if args.synthetic_hiring_path:
        LOADERS['synthetic_hiring'] = lambda max_samples=None: load_synthetic_hiring(
            args.synthetic_hiring_path, max_samples)
    
    # Load datasets
    datasets = {}
    for name in args.datasets:
        if name in LOADERS:
            try:
                ds = LOADERS[name](max_samples=args.max_samples)
                if ds is not None:
                    datasets[name] = ds
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
    
    logger.info(f"Loaded {len(datasets)} datasets")
    
    # Define methods for main comparison
    all_methods = [
        'LogisticRegression', 'RandomForest', 
        'XGBoost', 'LightGBM',
        'MLP', 'TabTransformer', 'FT-Transformer', 'SAINT',
        'GroupDRO', 'CounterfactualFair',
        'CausalGBM', 'CausalGBM-corr_only', 'CausalGBM-dag_only',
        'Corr-0.3', 'MutualInfo', 'RF-Importance', 'RandomSelection'
    ]
    
    if HAS_FAIRLEARN:
        all_methods.extend(['FairLearn-EO', 'FairLearn-DP'])
    
    # Filter methods based on availability
    methods = []
    for m in all_methods:
        if m == 'XGBoost' and not HAS_XGB: continue
        if m == 'LightGBM' and not HAS_LGB: continue
        if m.startswith('FairLearn') and not HAS_FAIRLEARN: continue
        methods.append(m)
    
    # Run experiments
    if args.all or args.main_comparison:
        run_main_comparison(datasets, methods, seeds, args.output_dir, device)
    
    if args.all or args.dag_recovery:
        run_dag_recovery_analysis(datasets, seeds, args.output_dir, device)
    
    if args.all or args.sensitivity:
        run_sensitivity_analysis(datasets, seeds, args.output_dir, device)
    
    if args.all or args.ablation:
        run_ablation_analysis(datasets, seeds, args.output_dir, device)
    
    # Generate summary
    generate_summary(args.output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("COMPLETED")
    logger.info("="*70)
    logger.info(f"Results in: {args.output_dir}")
    logger.info(f"Finished: {datetime.now()}")


if __name__ == '__main__':
    main()
