"""
Training and Evaluation Module
===============================

This module provides:
1. Training loops for TabTransformer and CausalTab
2. Comprehensive evaluation metrics
3. Model comparison utilities
4. Visualization functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional, Callable
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.mode == 'max':
            improved = self.best_score is None or score > self.best_score + self.min_delta
        else:
            improved = self.best_score is None or score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_best_model(self, model: nn.Module):
        """Restore the best model weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# =============================================================================
# TRAINER CLASS
# =============================================================================

class TabularTrainer:
    """
    Unified trainer for TabTransformer and CausalTab models.
    
    Features:
    - Automatic handling of auxiliary losses (for CausalTab)
    - Early stopping
    - Learning rate scheduling
    - Comprehensive logging
    - Model checkpointing
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        criterion: nn.Module = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        early_stopping_patience: int = 15,
        model_name: str = 'model'
    ):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        
        # Default optimizer
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5
        )
        
        # Default criterion (binary cross entropy with logits)
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        
        # Scheduler
        self.scheduler = scheduler
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max'  # Maximize AUC
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'learning_rate': []
        }
        
        # Check if model is CausalTab (has auxiliary losses)
        self.is_causaltab = hasattr(model, 'compute_total_loss')
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        aux_losses = {'dag_loss': 0.0, 'sparsity_loss': 0.0, 'cf_loss': 0.0}
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (x_cat, x_cont, targets) in enumerate(pbar):
            x_cat = x_cat.to(self.device)
            x_cont = x_cont.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(x_cat, x_cont)
            
            # Compute loss
            if self.is_causaltab:
                loss, loss_dict = self.model.compute_total_loss(
                    predictions, targets, x_cat, x_cont, self.criterion
                )
                aux_losses['dag_loss'] += loss_dict['dag_loss'].item()
                aux_losses['sparsity_loss'] += loss_dict['sparsity_loss'].item()
            else:
                loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(predictions).detach().cpu())
            all_targets.append(targets.cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.5
        
        avg_loss = total_loss / len(train_loader)
        
        metrics = {
            'loss': avg_loss,
            'auc': auc,
        }
        
        if self.is_causaltab:
            metrics['dag_loss'] = aux_losses['dag_loss'] / len(train_loader)
            metrics['sparsity_loss'] = aux_losses['sparsity_loss'] / len(train_loader)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self, 
        data_loader: DataLoader,
        desc: str = 'Eval'
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for x_cat, x_cont, targets in tqdm(data_loader, desc=desc):
            x_cat = x_cat.to(self.device)
            x_cont = x_cont.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)
            
            predictions = self.model(x_cat, x_cont)
            loss = self.criterion(predictions, targets)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(predictions).cpu())
            all_targets.append(targets.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.5
        
        # Binary predictions for other metrics
        binary_preds = (all_preds > 0.5).astype(int)
        
        metrics = {
            'loss': total_loss / len(data_loader),
            'auc': auc,
            'accuracy': accuracy_score(all_targets, binary_preds),
            'precision': precision_score(all_targets, binary_preds, zero_division=0),
            'recall': recall_score(all_targets, binary_preds, zero_division=0),
            'f1': f1_score(all_targets, binary_preds, zero_division=0),
            'average_precision': average_precision_score(all_targets, all_preds)
        }
        
        return metrics, all_preds, all_targets
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            verbose: Whether to print progress
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model type: {'CausalTab' if self.is_causaltab else 'TabTransformer'}")
        print(f"{'='*60}\n")
        
        best_val_auc = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics, _, _ = self.evaluate(val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            epoch_time = time.time() - start_time
            if verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")
                
                if self.is_causaltab:
                    print(f"  CausalTab - DAG Loss: {train_metrics.get('dag_loss', 0):.4f}, "
                          f"Sparsity: {train_metrics.get('sparsity_loss', 0):.4f}")
                
                print(f"  LR: {current_lr:.2e}")
            
            # Check for best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                if verbose:
                    print(f"  ★ New best validation AUC: {best_val_auc:.4f}")
            
            # Early stopping
            if self.early_stopping(val_metrics['auc'], self.model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Restore best model
        self.early_stopping.restore_best_model(self.model)
        print(f"\nRestored best model with validation AUC: {self.early_stopping.best_score:.4f}")
        
        return self.history


# =============================================================================
# EVALUATION AND COMPARISON
# =============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison.
    
    Features:
    - Multiple metrics computation
    - Statistical significance testing
    - Robustness analysis
    - Visualization
    """
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        model_name: str,
        criterion: nn.Module = None
    ) -> Dict[str, float]:
        """
        Evaluate a single model comprehensively.
        
        Returns dictionary with all metrics.
        """
        model.eval()
        model = model.to(self.device)
        
        criterion = criterion or nn.BCEWithLogitsLoss()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        for x_cat, x_cont, targets in tqdm(data_loader, desc=f'Evaluating {model_name}'):
            x_cat = x_cat.to(self.device)
            x_cont = x_cont.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)
            
            predictions = model(x_cat, x_cont)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(predictions).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        binary_preds = (all_preds > 0.5).astype(int)
        
        # Compute all metrics
        metrics = {
            'loss': total_loss / len(data_loader),
            'auc': roc_auc_score(all_targets, all_preds),
            'accuracy': accuracy_score(all_targets, binary_preds),
            'precision': precision_score(all_targets, binary_preds, zero_division=0),
            'recall': recall_score(all_targets, binary_preds, zero_division=0),
            'f1': f1_score(all_targets, binary_preds, zero_division=0),
            'average_precision': average_precision_score(all_targets, all_preds)
        }
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, binary_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': all_preds,
            'targets': all_targets
        }
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data_loader: DataLoader
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances
            data_loader: Test data loader
        
        Returns:
            DataFrame with comparison results
        """
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating: {name}")
            print(f"{'='*50}")
            
            metrics = self.evaluate_model(model, data_loader, name)
            results[name] = metrics
        
        # Create comparison DataFrame
        df = pd.DataFrame(results).T
        
        # Sort by AUC
        df = df.sort_values('auc', ascending=False)
        
        return df
    
    def get_classification_report(self, model_name: str) -> str:
        """Get detailed classification report for a model."""
        if model_name not in self.results:
            return "Model not evaluated yet."
        
        targets = self.results[model_name]['targets']
        preds = (self.results[model_name]['predictions'] > 0.5).astype(int)
        
        return classification_report(targets, preds, target_names=['Legitimate', 'Illicit'])
    
    def plot_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Create visualization comparing models.
        """
        if len(self.results) < 1:
            print("No models evaluated yet.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Metrics comparison bar chart
        ax1 = axes[0, 0]
        metrics_to_plot = ['auc', 'accuracy', 'precision', 'recall', 'f1']
        model_names = list(self.results.keys())
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        for i, name in enumerate(model_names):
            values = [self.results[name]['metrics'][m] for m in metrics_to_plot]
            offset = width * (i - len(model_names)/2 + 0.5)
            ax1.bar(x + offset, values, width, label=name)
        
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in metrics_to_plot])
        ax1.legend()
        ax1.set_ylim([0, 1])
        
        # 2. ROC Curves
        ax2 = axes[0, 1]
        from sklearn.metrics import roc_curve
        
        for name in model_names:
            preds = self.results[name]['predictions']
            targets = self.results[name]['targets']
            fpr, tpr, _ = roc_curve(targets, preds)
            auc = self.results[name]['metrics']['auc']
            ax2.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        
        # 3. Precision-Recall Curves
        ax3 = axes[1, 0]
        
        for name in model_names:
            preds = self.results[name]['predictions']
            targets = self.results[name]['targets']
            precision, recall, _ = precision_recall_curve(targets, preds)
            ap = self.results[name]['metrics']['average_precision']
            ax3.plot(recall, precision, label=f'{name} (AP={ap:.3f})')
        
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves')
        ax3.legend()
        
        # 4. Prediction distribution
        ax4 = axes[1, 1]
        
        for i, name in enumerate(model_names):
            preds = self.results[name]['predictions'].flatten()
            ax4.hist(preds, bins=50, alpha=0.5, label=name, density=True)
        
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Prediction Distribution')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig


# =============================================================================
# CAUSAL ANALYSIS UTILITIES
# =============================================================================

def visualize_causal_graph(
    model,
    feature_names: List[str],
    threshold: float = 0.3,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Visualize the learned causal graph from CausalTab.
    
    Args:
        model: CausalTab model
        feature_names: Names of features
        threshold: Threshold for edge visualization
        save_path: Path to save figure
        figsize: Figure size
    """
    if not hasattr(model, 'get_causal_graph'):
        print("Model does not have a causal graph (not CausalTab)")
        return
    
    # Get adjacency matrix
    adj = model.get_causal_graph(hard=False).detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Heatmap of adjacency matrix
    ax1 = axes[0]
    n_features = min(len(feature_names), adj.shape[0])
    
    sns.heatmap(
        adj[:n_features, :n_features],
        xticklabels=feature_names[:n_features],
        yticklabels=feature_names[:n_features],
        cmap='YlOrRd',
        ax=ax1,
        vmin=0,
        vmax=1
    )
    ax1.set_title('Learned Causal Adjacency Matrix\n(Row i, Col j: j → i)')
    
    # 2. Thresholded graph visualization
    ax2 = axes[1]
    adj_binary = (adj[:n_features, :n_features] > threshold).astype(int)
    
    try:
        import networkx as nx
        
        G = nx.DiGraph()
        G.add_nodes_from(range(n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if adj_binary[i, j] > 0:
                    G.add_edge(j, i, weight=adj[i, j])
        
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='lightblue', 
                              node_size=500, alpha=0.9)
        
        # Draw edges with weights
        edges = G.edges(data=True)
        weights = [e[2]['weight'] * 2 for e in edges]
        nx.draw_networkx_edges(G, pos, ax=ax2, width=weights, 
                              alpha=0.7, edge_color='red',
                              arrows=True, arrowsize=15)
        
        # Labels
        labels = {i: feature_names[i][:10] for i in range(n_features)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax2, font_size=8)
        
        ax2.set_title(f'Causal Graph (threshold={threshold})')
        ax2.axis('off')
        
    except ImportError:
        ax2.text(0.5, 0.5, 'Install networkx for graph visualization',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def analyze_feature_importance(
    model: nn.Module,
    data_loader: DataLoader,
    feature_names: List[str],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_batches: int = 10
) -> Dict[str, float]:
    """
    Analyze feature importance using attention weights.
    
    Args:
        model: Trained model
        data_loader: Data loader
        feature_names: Names of categorical features
        device: Device to use
        num_batches: Number of batches to analyze
    
    Returns:
        Dictionary of feature importance scores
    """
    model.eval()
    model = model.to(device)
    
    all_importance = []
    
    with torch.no_grad():
        for i, (x_cat, x_cont, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            
            # Get predictions with attention weights
            _, attention_weights = model(x_cat, x_cont, return_attention=True)
            
            # Average attention across layers, heads, and batch
            # Last layer attention
            attn = attention_weights[-1].mean(dim=(0, 1))  # (seq, seq)
            
            # Importance = how much each feature is attended to
            importance = attn.sum(dim=0).cpu().numpy()
            all_importance.append(importance)
    
    # Average across batches
    avg_importance = np.mean(all_importance, axis=0)
    avg_importance = avg_importance / avg_importance.sum()
    
    # Create dictionary
    n_features = min(len(feature_names), len(avg_importance))
    importance_dict = {
        feature_names[i]: float(avg_importance[i]) 
        for i in range(n_features)
    }
    
    # Sort by importance
    importance_dict = dict(sorted(
        importance_dict.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    return importance_dict
