"""
Main Experiment Script: TabTransformer vs CausalTab
====================================================

This script runs a complete comparison between TabTransformer and CausalTab
on the Customs Fraud Detection dataset.

Features:
- Data preprocessing and feature engineering
- Training both models with identical hyperparameters
- Comprehensive evaluation and comparison
- Causal graph analysis for CausalTab
- Distribution shift robustness testing (optional)

Usage:
    python run_experiment.py --data_path path/to/customs_data.csv
"""

import argparse
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from models import TabTransformer, CausalTab, create_model
from data_utils import TabularPreprocessor, TabularDataset, compute_class_weights
from training import TabularTrainer, ModelEvaluator, visualize_causal_graph, analyze_feature_importance


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Experiment configuration."""
    
    # Data settings
    DATA_PATH = 'customs_data.csv'
    TARGET_COL = 'illicit'
    
    # Columns to exclude from features
    ID_COLS = ['sgd.id']
    DATE_COLS = ['sgd.date']
    
    # Define feature types explicitly for customs data
    CATEGORICAL_COLS = [
        'importer.id', 'declarant.id', 'country', 
        'office.id', 'tariff.code'
    ]
    CONTINUOUS_COLS = [
        'quantity', 'gross.weight', 'fob.value', 
        'cif.value', 'total.taxes'
    ]
    
    # Model hyperparameters
    EMBEDDING_DIM = 32
    NUM_LAYERS = 6
    NUM_HEADS = 8
    DIM_HEAD = 16
    MLP_HIDDEN_MULTS = (4, 2)
    DROPOUT = 0.1
    COLUMN_EMBED_DIM = 8
    
    # CausalTab specific
    CAUSAL_THRESHOLD = 0.3
    DAG_LOSS_WEIGHT = 1.0
    SPARSITY_LOSS_WEIGHT = 0.1
    COUNTERFACTUAL_LOSS_WEIGHT = 0.0  # Set to 0 for faster training
    
    # Training settings
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Data splits
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output
    OUTPUT_DIR = 'results'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TabTransformer vs CausalTab Experiment')
    
    parser.add_argument('--data_path', type=str, default=Config.DATA_PATH,
                       help='Path to the CSV data file')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR,
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=Config.DEVICE,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--skip_causaltab', action='store_true',
                       help='Skip CausalTab training (for faster testing)')
    
    return parser.parse_args()


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(data_path: str, config: Config):
    """
    Load and preprocess the customs data.
    
    Returns:
        preprocessor: Fitted preprocessor
        train_loader, val_loader, test_loader: Data loaders
        feature_info: Dictionary with feature information
    """
    print("\n" + "="*60)
    print("DATA LOADING AND PREPROCESSING")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Display basic statistics
    print(f"\nTarget distribution:")
    print(df[config.TARGET_COL].value_counts())
    print(f"\nClass imbalance ratio: {df[config.TARGET_COL].mean():.2%} positive")
    
    # Handle high cardinality features
    # For importer.id and declarant.id, keep only top N frequent values
    for col in ['importer.id', 'declarant.id']:
        if col in df.columns:
            top_values = df[col].value_counts().head(100).index.tolist()
            df[col] = df[col].apply(lambda x: x if x in top_values else 'OTHER')
    
    # For tariff.code, use first 4 digits (chapter level)
    if 'tariff.code' in df.columns:
        df['tariff.code'] = df['tariff.code'].astype(str).str[:4]
    
    # Create preprocessor
    preprocessor = TabularPreprocessor(
        categorical_cols=config.CATEGORICAL_COLS,
        continuous_cols=config.CONTINUOUS_COLS,
        target_col=config.TARGET_COL,
        id_cols=config.ID_COLS,
        date_cols=config.DATE_COLS,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # Preprocess and split
    train_df, val_df, test_df = preprocessor.fit_transform(df)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = preprocessor.create_datasets(
        train_df, val_df, test_df
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.BATCH_SIZE
    )
    
    # Get feature info
    feature_info = {
        'categories': train_dataset.get_category_counts(),
        'num_continuous': len(config.CONTINUOUS_COLS),
        'feature_names': {
            'categorical': config.CATEGORICAL_COLS,
            'continuous': config.CONTINUOUS_COLS
        }
    }
    
    print(f"\nFeature info:")
    print(f"  Categorical features: {len(feature_info['categories'])}")
    print(f"  Category counts: {feature_info['categories']}")
    print(f"  Continuous features: {feature_info['num_continuous']}")
    
    return preprocessor, train_loader, val_loader, test_loader, feature_info


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_models(feature_info: dict, config: Config):
    """
    Create TabTransformer and CausalTab models.
    
    Returns:
        models: Dictionary of model instances
    """
    print("\n" + "="*60)
    print("MODEL CREATION")
    print("="*60)
    
    common_params = {
        'categories': feature_info['categories'],
        'num_continuous': feature_info['num_continuous'],
        'dim': config.EMBEDDING_DIM,
        'depth': config.NUM_LAYERS,
        'heads': config.NUM_HEADS,
        'dim_head': config.DIM_HEAD,
        'mlp_hidden_mults': config.MLP_HIDDEN_MULTS,
        'num_classes': 1,
        'attn_dropout': config.DROPOUT,
        'ff_dropout': config.DROPOUT,
        'column_embed_dim': config.COLUMN_EMBED_DIM
    }
    
    # TabTransformer
    print("\nCreating TabTransformer...")
    tabtransformer = TabTransformer(**common_params)
    
    # Count parameters
    tab_params = sum(p.numel() for p in tabtransformer.parameters())
    print(f"  Parameters: {tab_params:,}")
    
    # CausalTab
    print("\nCreating CausalTab...")
    causaltab_params = common_params.copy()
    causaltab_params.update({
        'causal_threshold': config.CAUSAL_THRESHOLD,
        'dag_loss_weight': config.DAG_LOSS_WEIGHT,
        'sparsity_loss_weight': config.SPARSITY_LOSS_WEIGHT,
        'counterfactual_loss_weight': config.COUNTERFACTUAL_LOSS_WEIGHT,
        'use_causal_mask': True
    })
    causaltab = CausalTab(**causaltab_params)
    
    causal_params = sum(p.numel() for p in causaltab.parameters())
    print(f"  Parameters: {causal_params:,}")
    
    models = {
        'TabTransformer': tabtransformer,
        'CausalTab': causaltab
    }
    
    return models


# =============================================================================
# TRAINING
# =============================================================================

def train_models(
    models: dict,
    train_loader,
    val_loader,
    config: Config,
    skip_causaltab: bool = False
):
    """
    Train all models.
    
    Returns:
        trained_models: Dictionary of trained models
        histories: Dictionary of training histories
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    trained_models = {}
    histories = {}
    
    for name, model in models.items():
        if skip_causaltab and name == 'CausalTab':
            print(f"\nSkipping {name} training...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Create trainer
        trainer = TabularTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            model_name=name
        )
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.NUM_EPOCHS,
            verbose=True
        )
        
        trained_models[name] = trainer.model
        histories[name] = history
    
    return trained_models, histories


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_models(
    models: dict,
    test_loader,
    feature_info: dict,
    output_dir: str,
    config: Config
):
    """
    Evaluate and compare models.
    
    Returns:
        results_df: DataFrame with comparison results
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Create evaluator
    evaluator = ModelEvaluator(device=config.DEVICE)
    
    # Compare models
    results_df = evaluator.compare_models(models, test_loader)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string())
    
    # Print detailed reports
    for name in models.keys():
        print(f"\n{'='*60}")
        print(f"Classification Report: {name}")
        print(f"{'='*60}")
        print(evaluator.get_classification_report(name))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'comparison_results.csv'))
    print(f"\nResults saved to {output_dir}/comparison_results.csv")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    evaluator.plot_comparison(
        save_path=os.path.join(output_dir, 'comparison_plots.png')
    )
    
    # Analyze CausalTab's causal graph if available
    if 'CausalTab' in models:
        print("\n" + "="*60)
        print("CAUSAL GRAPH ANALYSIS")
        print("="*60)
        
        try:
            visualize_causal_graph(
                models['CausalTab'],
                feature_info['feature_names']['categorical'],
                threshold=config.CAUSAL_THRESHOLD,
                save_path=os.path.join(output_dir, 'causal_graph.png')
            )
        except Exception as e:
            print(f"Could not visualize causal graph: {e}")
        
        # Feature importance comparison
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        for name, model in models.items():
            try:
                importance = analyze_feature_importance(
                    model, test_loader,
                    feature_info['feature_names']['categorical'],
                    device=config.DEVICE
                )
                print(f"\n{name} Feature Importance:")
                for feat, imp in list(importance.items())[:10]:
                    print(f"  {feat}: {imp:.4f}")
            except Exception as e:
                print(f"Could not analyze feature importance for {name}: {e}")
    
    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main experiment function."""
    args = parse_args()
    config = Config()
    
    # Override config with command line args
    config.DATA_PATH = args.data_path
    config.OUTPUT_DIR = args.output_dir
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.DEVICE = args.device
    
    print("\n" + "="*60)
    print("TABTRANSFORMER vs CAUSALTAB EXPERIMENT")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data path: {config.DATA_PATH}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    
    # Load and preprocess data
    preprocessor, train_loader, val_loader, test_loader, feature_info = \
        load_and_preprocess_data(config.DATA_PATH, config)
    
    # Create models
    models = create_models(feature_info, config)
    
    # Train models
    trained_models, histories = train_models(
        models, train_loader, val_loader, config,
        skip_causaltab=args.skip_causaltab
    )
    
    # Evaluate models
    results_df = evaluate_models(
        trained_models, test_loader, feature_info, 
        config.OUTPUT_DIR, config
    )
    
    # Save training histories
    for name, history in histories.items():
        history_df = pd.DataFrame(history)
        history_df.to_csv(
            os.path.join(config.OUTPUT_DIR, f'{name}_history.csv'),
            index=False
        )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print(f"\nResults saved to: {config.OUTPUT_DIR}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if len(results_df) >= 2:
        best_model = results_df['auc'].idxmax()
        print(f"\nBest performing model: {best_model}")
        print(f"  AUC: {results_df.loc[best_model, 'auc']:.4f}")
        print(f"  F1:  {results_df.loc[best_model, 'f1']:.4f}")
        
        # Compare TabTransformer vs CausalTab
        if 'TabTransformer' in results_df.index and 'CausalTab' in results_df.index:
            tab_auc = results_df.loc['TabTransformer', 'auc']
            causal_auc = results_df.loc['CausalTab', 'auc']
            diff = causal_auc - tab_auc
            
            print(f"\nCausalTab vs TabTransformer:")
            print(f"  AUC difference: {diff:+.4f}")
            if diff > 0:
                print(f"  CausalTab outperforms TabTransformer by {diff*100:.2f}%")
            else:
                print(f"  TabTransformer outperforms CausalTab by {-diff*100:.2f}%")
    
    return results_df


if __name__ == '__main__':
    main()
