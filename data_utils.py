"""
Data Preprocessing and Utilities for Tabular Models
=====================================================

This module provides:
1. Generic data preprocessing for tabular data
2. Dataset classes for PyTorch
3. Feature encoding utilities
4. Train/validation/test splitting

Compatible with both TabTransformer and CausalTab models.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union
import warnings


# =============================================================================
# TABULAR DATASET
# =============================================================================

class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data with categorical and continuous features.
    
    Handles:
    - Categorical features → Integer indices
    - Continuous features → Normalized float tensors
    - Target variable → Float tensor
    """
    def __init__(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        continuous_cols: List[str],
        target_col: str,
        cat_encoders: Optional[Dict[str, LabelEncoder]] = None,
        cont_scaler: Optional[StandardScaler] = None,
        is_train: bool = True
    ):
        """
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            continuous_cols: List of continuous column names
            target_col: Target column name
            cat_encoders: Pre-fitted label encoders (for val/test sets)
            cont_scaler: Pre-fitted scaler (for val/test sets)
            is_train: Whether this is training data (fit encoders if True)
        """
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.target_col = target_col
        
        # Make a copy to avoid modifying original
        self.df = df.copy()
        
        # Initialize or use provided encoders
        if cat_encoders is None:
            self.cat_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                # Handle unseen values by adding a placeholder
                self.df[col] = self.df[col].astype(str).fillna('__MISSING__')
                le.fit(self.df[col])
                self.cat_encoders[col] = le
        else:
            self.cat_encoders = cat_encoders
        
        # Initialize or use provided scaler
        if cont_scaler is None and len(continuous_cols) > 0:
            self.cont_scaler = StandardScaler()
            if is_train:
                cont_data = self.df[continuous_cols].fillna(0).values
                self.cont_scaler.fit(cont_data)
        else:
            self.cont_scaler = cont_scaler
        
        # Encode categorical features
        self.cat_data = np.zeros((len(self.df), len(categorical_cols)), dtype=np.int64)
        for i, col in enumerate(categorical_cols):
            self.df[col] = self.df[col].astype(str).fillna('__MISSING__')
            # Handle unseen categories
            known_classes = set(self.cat_encoders[col].classes_)
            self.df[col] = self.df[col].apply(
                lambda x: x if x in known_classes else '__UNKNOWN__'
            )
            # Add unknown to encoder if needed
            if '__UNKNOWN__' not in self.cat_encoders[col].classes_:
                self.cat_encoders[col].classes_ = np.append(
                    self.cat_encoders[col].classes_, '__UNKNOWN__'
                )
            self.cat_data[:, i] = self.cat_encoders[col].transform(self.df[col])
        
        # Scale continuous features
        if len(continuous_cols) > 0:
            cont_data = self.df[continuous_cols].fillna(0).values
            self.cont_data = self.cont_scaler.transform(cont_data).astype(np.float32)
        else:
            self.cont_data = np.zeros((len(self.df), 0), dtype=np.float32)
        
        # Target
        self.targets = self.df[target_col].values.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            cat_features: Categorical features as integer indices
            cont_features: Continuous features as float tensor
            target: Target value
        """
        return (
            torch.tensor(self.cat_data[idx], dtype=torch.long),
            torch.tensor(self.cont_data[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )
    
    def get_category_counts(self) -> List[int]:
        """Get number of unique values per categorical feature."""
        return [len(self.cat_encoders[col].classes_) for col in self.categorical_cols]
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names organized by type."""
        return {
            'categorical': self.categorical_cols,
            'continuous': self.continuous_cols,
            'target': self.target_col
        }


# =============================================================================
# DATA PREPROCESSOR
# =============================================================================

class TabularPreprocessor:
    """
    Preprocessing pipeline for tabular data.
    
    Features:
    - Automatic feature type detection
    - Handling missing values
    - Encoding categorical variables
    - Scaling continuous variables
    - Train/val/test splitting
    """
    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        target_col: str = 'target',
        id_cols: Optional[List[str]] = None,
        date_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.15,
        random_state: int = 42,
        scaling_method: str = 'standard'  # 'standard' or 'minmax'
    ):
        """
        Args:
            categorical_cols: List of categorical columns (auto-detect if None)
            continuous_cols: List of continuous columns (auto-detect if None)
            target_col: Name of target column
            id_cols: Columns to exclude (IDs, etc.)
            date_cols: Date columns to process specially
            test_size: Fraction for test set
            val_size: Fraction of training data for validation
            random_state: Random seed for splitting
            scaling_method: Method for scaling continuous features
        """
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.target_col = target_col
        self.id_cols = id_cols or []
        self.date_cols = date_cols or []
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaling_method = scaling_method
        
        # Will be set during fit
        self.cat_encoders = None
        self.cont_scaler = None
        self.feature_info = None
    
    def _detect_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Auto-detect categorical and continuous columns."""
        exclude_cols = set(self.id_cols + self.date_cols + [self.target_col])
        
        categorical = []
        continuous = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            # Check if categorical
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical.append(col)
            elif df[col].nunique() < 20 and df[col].dtype in ['int64', 'int32']:
                # Low cardinality integers treated as categorical
                categorical.append(col)
            else:
                continuous.append(col)
        
        return categorical, continuous
    
    def _extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from date columns."""
        df = df.copy()
        
        for col in self.date_cols:
            if col not in df.columns:
                continue
            
            try:
                # Try to parse dates
                dates = pd.to_datetime(df[col], errors='coerce')
                
                # Extract features
                df[f'{col}_year'] = dates.dt.year.fillna(0).astype(int)
                df[f'{col}_month'] = dates.dt.month.fillna(0).astype(int)
                df[f'{col}_day'] = dates.dt.day.fillna(0).astype(int)
                df[f'{col}_dayofweek'] = dates.dt.dayofweek.fillna(0).astype(int)
                
                # Add to continuous cols
                if self.continuous_cols is not None:
                    self.continuous_cols.extend([
                        f'{col}_year', f'{col}_month', 
                        f'{col}_day', f'{col}_dayofweek'
                    ])
            except Exception as e:
                warnings.warn(f"Could not parse date column {col}: {e}")
        
        return df
    
    def fit_transform(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit preprocessor and transform data into train/val/test splits.
        
        Args:
            df: Input DataFrame
        
        Returns:
            train_df, val_df, test_df: Split DataFrames
        """
        # Extract date features if specified
        if self.date_cols:
            df = self._extract_date_features(df)
        
        # Auto-detect feature types if not specified
        if self.categorical_cols is None or self.continuous_cols is None:
            auto_cat, auto_cont = self._detect_feature_types(df)
            self.categorical_cols = self.categorical_cols or auto_cat
            self.continuous_cols = self.continuous_cols or auto_cont
        
        print(f"Categorical features ({len(self.categorical_cols)}): {self.categorical_cols}")
        print(f"Continuous features ({len(self.continuous_cols)}): {self.continuous_cols}")
        
        # Split data
        train_val_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state,
            stratify=df[self.target_col] if df[self.target_col].nunique() <= 10 else None
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=self.val_size, random_state=self.random_state,
            stratify=train_val_df[self.target_col] if train_val_df[self.target_col].nunique() <= 10 else None
        )
        
        # Fit encoders on training data
        self.cat_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            train_df[col] = train_df[col].astype(str).fillna('__MISSING__')
            le.fit(train_df[col])
            self.cat_encoders[col] = le
        
        # Fit scaler on training data
        if len(self.continuous_cols) > 0:
            if self.scaling_method == 'standard':
                self.cont_scaler = StandardScaler()
            else:
                self.cont_scaler = MinMaxScaler()
            
            cont_data = train_df[self.continuous_cols].fillna(0).values
            self.cont_scaler.fit(cont_data)
        else:
            self.cont_scaler = None
        
        # Store feature info
        self.feature_info = {
            'categorical_cols': self.categorical_cols,
            'continuous_cols': self.continuous_cols,
            'target_col': self.target_col,
            'category_counts': [len(self.cat_encoders[col].classes_) for col in self.categorical_cols],
            'num_continuous': len(self.continuous_cols)
        }
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def create_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[TabularDataset, TabularDataset, TabularDataset]:
        """
        Create PyTorch datasets from DataFrames.
        
        Args:
            train_df, val_df, test_df: Split DataFrames
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        train_dataset = TabularDataset(
            train_df,
            self.categorical_cols,
            self.continuous_cols,
            self.target_col,
            cat_encoders=self.cat_encoders,
            cont_scaler=self.cont_scaler,
            is_train=True
        )
        
        val_dataset = TabularDataset(
            val_df,
            self.categorical_cols,
            self.continuous_cols,
            self.target_col,
            cat_encoders=self.cat_encoders,
            cont_scaler=self.cont_scaler,
            is_train=False
        )
        
        test_dataset = TabularDataset(
            test_df,
            self.categorical_cols,
            self.continuous_cols,
            self.target_col,
            cat_encoders=self.cat_encoders,
            cont_scaler=self.cont_scaler,
            is_train=False
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: TabularDataset,
        val_dataset: TabularDataset,
        test_dataset: TabularDataset,
        batch_size: int = 256,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders from datasets."""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_customs_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess customs fraud detection data."""
    df = pd.read_csv(filepath)
    
    # Define column types for customs data
    # Based on the schema provided
    return df


def create_synthetic_shift(
    df: pd.DataFrame,
    shift_col: str,
    shift_type: str = 'covariate',
    shift_strength: float = 0.5
) -> pd.DataFrame:
    """
    Create synthetic distribution shift for robustness testing.
    
    Args:
        df: Input DataFrame
        shift_col: Column to apply shift to
        shift_type: 'covariate' (feature shift) or 'label' (target shift)
        shift_strength: How much to shift (0-1)
    
    Returns:
        DataFrame with synthetic shift
    """
    df_shifted = df.copy()
    
    if shift_type == 'covariate':
        if df[shift_col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Shift numerical feature
            std = df[shift_col].std()
            df_shifted[shift_col] = df[shift_col] + shift_strength * std
        else:
            # Shift categorical by changing distribution
            values = df[shift_col].unique()
            n_to_change = int(len(df) * shift_strength)
            indices = np.random.choice(len(df), n_to_change, replace=False)
            df_shifted.loc[df_shifted.index[indices], shift_col] = np.random.choice(
                values, n_to_change
            )
    
    return df_shifted


def compute_class_weights(targets: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced classification."""
    classes, counts = np.unique(targets, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    return torch.FloatTensor(weights)


def get_feature_importance_from_attention(
    attention_weights: List[torch.Tensor],
    feature_names: List[str],
    aggregation: str = 'mean'
) -> Dict[str, float]:
    """
    Extract feature importance from attention weights.
    
    Args:
        attention_weights: List of attention tensors from Transformer layers
        feature_names: Names of features
        aggregation: How to aggregate across layers ('mean', 'last', 'max')
    
    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Stack attention weights from all layers
    # Shape: (num_layers, batch, heads, seq, seq)
    if aggregation == 'last':
        attn = attention_weights[-1]
    elif aggregation == 'max':
        attn = torch.stack(attention_weights).max(dim=0)[0]
    else:  # mean
        attn = torch.stack(attention_weights).mean(dim=0)
    
    # Average across batch and heads
    # Shape: (seq, seq)
    attn = attn.mean(dim=(0, 1))
    
    # Sum attention received by each feature
    importance = attn.sum(dim=0).cpu().numpy()
    
    # Normalize
    importance = importance / importance.sum()
    
    return {name: float(imp) for name, imp in zip(feature_names, importance)}
