"""
Quick Demo: TabTransformer vs CausalTab
========================================

This script provides a quick demonstration of both models
using a small sample of synthetic customs data.

Run with: python demo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TabTransformer, CausalTab
from data_utils import TabularPreprocessor
from training import TabularTrainer, ModelEvaluator


def create_sample_customs_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Create synthetic customs fraud detection data.
    
    This mimics the structure of real customs data with:
    - Categorical: importer, declarant, country, office, tariff code
    - Numerical: quantity, weight, values, taxes
    - Target: illicit (binary)
    """
    np.random.seed(42)
    
    # Generate categorical features
    n_importers = 200
    n_declarants = 100
    n_countries = 50
    n_offices = 10
    n_tariffs = 100
    
    data = {
        'sgd.id': [f'SGD{i}' for i in range(n_samples)],
        'sgd.date': pd.date_range('2020-01-01', periods=n_samples, freq='H').strftime('%y-%m-%d'),
        'importer.id': [f'IMP{np.random.randint(0, n_importers)}' for _ in range(n_samples)],
        'declarant.id': [f'DEC{np.random.randint(0, n_declarants)}' for _ in range(n_samples)],
        'country': [f'CNTRY{np.random.randint(0, n_countries)}' for _ in range(n_samples)],
        'office.id': [f'OFFICE{np.random.randint(0, n_offices)}' for _ in range(n_samples)],
        'tariff.code': [f'{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
    }
    
    # Generate numerical features
    data['quantity'] = np.random.exponential(100, n_samples).astype(int)
    data['gross.weight'] = np.random.exponential(1000, n_samples)
    data['fob.value'] = np.random.exponential(10000, n_samples)
    data['cif.value'] = data['fob.value'] * (1 + np.random.uniform(0.05, 0.3, n_samples))
    data['total.taxes'] = data['cif.value'] * np.random.uniform(0.05, 0.2, n_samples)
    
    df = pd.DataFrame(data)
    
    # Generate target with some realistic patterns
    # Higher fraud probability for:
    # - Certain countries
    # - Low declared values relative to weight
    # - Certain tariff codes
    
    fraud_prob = np.zeros(n_samples)
    
    # Country effect
    risky_countries = [f'CNTRY{i}' for i in range(5)]  # First 5 countries are risky
    fraud_prob += df['country'].isin(risky_countries).astype(float) * 0.15
    
    # Value/weight ratio effect (under-declaration)
    value_per_kg = df['fob.value'] / (df['gross.weight'] + 1)
    fraud_prob += (value_per_kg < np.percentile(value_per_kg, 20)).astype(float) * 0.2
    
    # Certain tariff codes
    risky_tariffs = [f'{1000 + i}' for i in range(10)]
    fraud_prob += df['tariff.code'].isin(risky_tariffs).astype(float) * 0.1
    
    # Add some noise
    fraud_prob += np.random.uniform(0, 0.1, n_samples)
    
    # Base fraud rate around 5-10%
    fraud_prob = np.clip(fraud_prob, 0.02, 0.5)
    
    # Generate binary labels
    df['illicit'] = (np.random.random(n_samples) < fraud_prob).astype(int)
    
    # Revenue (only for illicit cases)
    df['revenue'] = df['illicit'] * df['total.taxes'] * np.random.uniform(0.1, 0.5, n_samples)
    
    print(f"Generated {n_samples} samples")
    print(f"Fraud rate: {df['illicit'].mean():.2%}")
    
    return df


def run_demo():
    """Run a quick demonstration."""
    
    print("\n" + "="*70)
    print("  TABTRANSFORMER vs CAUSALTAB - QUICK DEMO")
    print("="*70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create output directory
    os.makedirs('demo_results', exist_ok=True)
    
    # Generate sample data
    print("\n" + "-"*50)
    print("STEP 1: Generating Sample Data")
    print("-"*50)
    
    df = create_sample_customs_data(n_samples=5000)
    
    # Save sample data
    df.to_csv('demo_customs_data.csv', index=False)
    print(f"Sample data saved to: demo_customs_data.csv")
    
    # Preprocess
    print("\n" + "-"*50)
    print("STEP 2: Preprocessing Data")
    print("-"*50)
    
    preprocessor = TabularPreprocessor(
        categorical_cols=['importer.id', 'declarant.id', 'country', 'office.id', 'tariff.code'],
        continuous_cols=['quantity', 'gross.weight', 'fob.value', 'cif.value', 'total.taxes'],
        target_col='illicit',
        id_cols=['sgd.id'],
        date_cols=['sgd.date'],
        test_size=0.2,
        val_size=0.15
    )
    
    train_df, val_df, test_df = preprocessor.fit_transform(df)
    train_dataset, val_dataset, test_dataset = preprocessor.create_datasets(train_df, val_df, test_df)
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=128
    )
    
    # Get feature info
    categories = train_dataset.get_category_counts()
    num_continuous = len(preprocessor.continuous_cols)
    feature_names = preprocessor.categorical_cols
    
    print(f"\nCategory counts: {categories}")
    print(f"Continuous features: {num_continuous}")
    
    # Create models
    print("\n" + "-"*50)
    print("STEP 3: Creating Models")
    print("-"*50)
    
    # Smaller models for quick demo
    model_config = {
        'categories': categories,
        'num_continuous': num_continuous,
        'dim': 32,
        'depth': 4,
        'heads': 4,
        'dim_head': 8,
        'mlp_hidden_mults': (2, 1),
        'num_classes': 1,
        'attn_dropout': 0.1,
        'ff_dropout': 0.1
    }
    
    # TabTransformer
    tabtransformer = TabTransformer(**model_config)
    tab_params = sum(p.numel() for p in tabtransformer.parameters())
    print(f"TabTransformer parameters: {tab_params:,}")
    
    # CausalTab
    causaltab_config = model_config.copy()
    causaltab_config.update({
        'causal_threshold': 0.3,
        'dag_loss_weight': 1.0,
        'sparsity_loss_weight': 0.1,
        'counterfactual_loss_weight': 0.0  # Disabled for speed
    })
    causaltab = CausalTab(**causaltab_config)
    causal_params = sum(p.numel() for p in causaltab.parameters())
    print(f"CausalTab parameters: {causal_params:,}")
    
    # Train TabTransformer
    print("\n" + "-"*50)
    print("STEP 4: Training TabTransformer")
    print("-"*50)
    
    tab_optimizer = optim.AdamW(tabtransformer.parameters(), lr=1e-3, weight_decay=1e-5)
    tab_trainer = TabularTrainer(
        model=tabtransformer,
        optimizer=tab_optimizer,
        device=device,
        early_stopping_patience=5,
        model_name='TabTransformer'
    )
    
    tab_history = tab_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        verbose=True
    )
    
    # Train CausalTab
    print("\n" + "-"*50)
    print("STEP 5: Training CausalTab")
    print("-"*50)
    
    causal_optimizer = optim.AdamW(causaltab.parameters(), lr=1e-3, weight_decay=1e-5)
    causal_trainer = TabularTrainer(
        model=causaltab,
        optimizer=causal_optimizer,
        device=device,
        early_stopping_patience=5,
        model_name='CausalTab'
    )
    
    causal_history = causal_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        verbose=True
    )
    
    # Evaluate
    print("\n" + "-"*50)
    print("STEP 6: Evaluation")
    print("-"*50)
    
    evaluator = ModelEvaluator(device=device)
    
    models = {
        'TabTransformer': tab_trainer.model,
        'CausalTab': causal_trainer.model
    }
    
    results = evaluator.compare_models(models, test_loader)
    
    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    print(results.to_string())
    
    # Print classification reports
    for name in models.keys():
        print(f"\n{'-'*50}")
        print(f"Classification Report: {name}")
        print(f"{'-'*50}")
        print(evaluator.get_classification_report(name))
    
    # Save results
    results.to_csv('demo_results/comparison_results.csv')
    print(f"\nResults saved to: demo_results/comparison_results.csv")
    
    # Plot comparison
    try:
        evaluator.plot_comparison(save_path='demo_results/comparison_plots.png')
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Analyze CausalTab's learned causal structure
    print("\n" + "-"*50)
    print("STEP 7: Causal Structure Analysis")
    print("-"*50)
    
    try:
        adj_matrix = causal_trainer.model.get_causal_graph(hard=False).detach().cpu().numpy()
        print("\nLearned Causal Adjacency Matrix:")
        print(f"Shape: {adj_matrix.shape}")
        print(f"Sparsity: {(adj_matrix < 0.3).mean():.2%}")
        
        # Show top causal relationships
        print("\nTop learned causal relationships (feature j → feature i):")
        n_features = min(len(feature_names), adj_matrix.shape[0])
        
        edges = []
        for i in range(n_features):
            for j in range(n_features):
                if i != j and adj_matrix[i, j] > 0.3:
                    edges.append((feature_names[j], feature_names[i], adj_matrix[i, j]))
        
        edges.sort(key=lambda x: -x[2])
        for src, dst, weight in edges[:10]:
            print(f"  {src} → {dst}: {weight:.3f}")
        
        if len(edges) == 0:
            print("  No strong causal edges detected (all weights < 0.3)")
            print("  This may indicate independent features or need for more training")
        
    except Exception as e:
        print(f"Could not analyze causal structure: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    tab_auc = results.loc['TabTransformer', 'auc']
    causal_auc = results.loc['CausalTab', 'auc']
    diff = causal_auc - tab_auc
    
    print(f"\nTabTransformer AUC: {tab_auc:.4f}")
    print(f"CausalTab AUC:      {causal_auc:.4f}")
    print(f"Difference:         {diff:+.4f}")
    
    if diff > 0:
        print(f"\n✓ CausalTab outperforms TabTransformer by {diff*100:.2f}%")
    elif diff < 0:
        print(f"\n✓ TabTransformer outperforms CausalTab by {-diff*100:.2f}%")
    else:
        print(f"\n✓ Both models perform equally")
    
    print("\n" + "="*70)
    print("  DEMO COMPLETED")
    print("="*70)
    
    return results


if __name__ == '__main__':
    results = run_demo()
