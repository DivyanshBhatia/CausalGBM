"""
TabTransformer and CausalTab Implementation
============================================

This module implements:
1. TabTransformer - Based on the original paper "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
2. CausalTab - A novel causal-aware extension with causal discovery and attention masking

Author: Research Implementation
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from einops import rearrange


# =============================================================================
# UTILITY MODULES
# =============================================================================

class GEGLU(nn.Module):
    """Gated Linear Unit activation as used in the original TabTransformer."""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Position-wise Feed Forward Network with GEGLU activation."""
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention as described in 'Attention Is All You Need'.
    
    Supports optional causal attention mask for CausalTab.
    """
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, causal_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            causal_mask: Optional mask tensor of shape (seq_len, seq_len) or (batch, seq_len, seq_len)
                        Values should be 0 for positions to attend, -inf for positions to mask
        """
        b, n, _ = x.shape
        h = self.heads
        
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply causal mask if provided
        if causal_mask is not None:
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)
            elif causal_mask.dim() == 3:
                causal_mask = causal_mask.unsqueeze(1)  # (b, 1, n, n)
            dots = dots + causal_mask
        
        # Softmax and apply to values
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape and project out
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm architecture."""
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_mult, ff_dropout)
    
    def forward(self, x, causal_mask: Optional[torch.Tensor] = None):
        # Pre-norm attention
        attn_out, attn_weights = self.attn(self.norm1(x), causal_mask)
        x = x + attn_out
        
        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        
        return x, attn_weights


# =============================================================================
# COLUMN EMBEDDING (as per original paper)
# =============================================================================

class ColumnEmbedding(nn.Module):
    """
    Column Embedding as described in the TabTransformer paper.
    
    For each categorical feature i with value j:
        e_φi(j) = [c_φi, w_φij]
    
    Where:
        - c_φi: Unique column identifier (shared across all values in column i)
        - w_φij: Feature-value specific embedding
    """
    def __init__(
        self, 
        num_categories: List[int],  # Number of unique values per categorical feature
        dim: int,
        column_embed_dim: int = 8  # Dimension of the column identifier (ℓ in paper)
    ):
        super().__init__()
        self.num_features = len(num_categories)
        self.dim = dim
        self.column_embed_dim = column_embed_dim
        self.value_embed_dim = dim - column_embed_dim
        
        # Column identifiers - one per categorical feature
        self.column_embeddings = nn.Parameter(
            torch.randn(self.num_features, column_embed_dim)
        )
        
        # Value embeddings - separate embedding table per categorical feature
        self.value_embeddings = nn.ModuleList([
            nn.Embedding(num_cat + 1, self.value_embed_dim)  # +1 for missing values
            for num_cat in num_categories
        ])
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.column_embeddings, std=0.02)
        for emb in self.value_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
    
    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: Categorical features of shape (batch, num_cat_features)
                   Values should be integer indices
        
        Returns:
            Embeddings of shape (batch, num_cat_features, dim)
        """
        batch_size = x_cat.shape[0]
        embeddings = []
        
        for i in range(self.num_features):
            # Get column identifier (same for all samples)
            col_emb = self.column_embeddings[i].unsqueeze(0).expand(batch_size, -1)
            
            # Get value embedding
            val_emb = self.value_embeddings[i](x_cat[:, i])
            
            # Concatenate [column_id, value_embedding]
            emb = torch.cat([col_emb, val_emb], dim=-1)
            embeddings.append(emb)
        
        return torch.stack(embeddings, dim=1)  # (batch, num_features, dim)


# =============================================================================
# NUMERICAL FEATURE PROCESSING
# =============================================================================

class NumericalEmbedding(nn.Module):
    """
    Embedding layer for numerical features.
    
    Options:
        - 'linear': Simple linear projection (as in original TabTransformer)
        - 'ple': Piecewise Linear Encoding (from "On Embeddings for Numerical Features")
    """
    def __init__(
        self, 
        num_numerical: int, 
        dim: int,
        method: str = 'linear'
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.dim = dim
        self.method = method
        
        if method == 'linear':
            # Simple linear projection per feature
            self.embeddings = nn.ModuleList([
                nn.Linear(1, dim) for _ in range(num_numerical)
            ])
        else:
            raise ValueError(f"Unknown numerical embedding method: {method}")
    
    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_num: Numerical features of shape (batch, num_numerical)
        
        Returns:
            Embeddings of shape (batch, num_numerical, dim)
        """
        embeddings = []
        for i in range(self.num_numerical):
            emb = self.embeddings[i](x_num[:, i:i+1])
            embeddings.append(emb)
        
        return torch.stack(embeddings, dim=1)


# =============================================================================
# TABTRANSFORMER (Original Paper Implementation)
# =============================================================================

class TabTransformer(nn.Module):
    """
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings
    
    Architecture:
        1. Column Embedding for categorical features
        2. Stack of N Transformer layers (applied only to categorical embeddings)
        3. Concatenation with continuous features
        4. MLP for prediction
    
    Reference: https://arxiv.org/abs/2012.06678
    """
    def __init__(
        self,
        categories: List[int],          # Number of unique values per categorical feature
        num_continuous: int,             # Number of continuous features
        dim: int = 32,                   # Embedding dimension
        depth: int = 6,                  # Number of Transformer layers
        heads: int = 8,                  # Number of attention heads
        dim_head: int = 16,              # Dimension per attention head
        mlp_hidden_mults: Tuple[int, ...] = (4, 2),  # MLP hidden layer multipliers
        mlp_act: nn.Module = nn.ReLU(),  # MLP activation
        num_classes: int = 1,            # Output dimension (1 for binary classification)
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        column_embed_dim: int = 8,       # Column identifier dimension (ℓ in paper)
        use_numerical_embedding: bool = False  # Whether to embed numerical features
    ):
        super().__init__()
        
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.dim = dim
        
        # Column embedding for categorical features
        self.column_embedding = ColumnEmbedding(
            num_categories=categories,
            dim=dim,
            column_embed_dim=column_embed_dim
        )
        
        # Optional numerical embedding
        self.use_numerical_embedding = use_numerical_embedding
        if use_numerical_embedding and num_continuous > 0:
            self.numerical_embedding = NumericalEmbedding(num_continuous, dim)
        
        # Transformer layers (for categorical embeddings)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Continuous feature normalization
        if num_continuous > 0:
            self.cont_norm = nn.LayerNorm(num_continuous)
        
        # Calculate MLP input dimension
        if use_numerical_embedding and num_continuous > 0:
            mlp_input_dim = dim * (self.num_categories + num_continuous)
        else:
            mlp_input_dim = dim * self.num_categories + num_continuous
        
        # MLP layers
        mlp_layers = []
        dims = [mlp_input_dim] + [mlp_input_dim * m for m in mlp_hidden_mults]
        
        for i in range(len(dims) - 1):
            mlp_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                mlp_act,
                nn.Dropout(ff_dropout)
            ])
        
        mlp_layers.append(nn.Linear(dims[-1], num_classes))
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Store attention weights for interpretability
        self.attention_weights = None
    
    def forward(
        self, 
        x_cat: torch.Tensor, 
        x_cont: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x_cat: Categorical features (batch, num_cat_features) - integer indices
            x_cont: Continuous features (batch, num_continuous) - float values
            return_attention: Whether to return attention weights
        
        Returns:
            Predictions of shape (batch, num_classes)
        """
        # Embed categorical features
        x = self.column_embedding(x_cat)  # (batch, num_cat, dim)
        
        # Pass through Transformer layers
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn = block(x)
            attention_weights.append(attn)
        
        # Apply final norm
        x = self.norm(x)
        
        # Flatten categorical embeddings
        x = x.flatten(start_dim=1)  # (batch, num_cat * dim)
        
        # Process continuous features
        if self.num_continuous > 0 and x_cont is not None:
            if self.use_numerical_embedding:
                cont_emb = self.numerical_embedding(x_cont)
                cont_emb = cont_emb.flatten(start_dim=1)
                x = torch.cat([x, cont_emb], dim=1)
            else:
                x_cont = self.cont_norm(x_cont)
                x = torch.cat([x, x_cont], dim=1)
        
        # MLP prediction
        out = self.mlp(x)
        
        if return_attention:
            self.attention_weights = attention_weights
            return out, attention_weights
        
        return out
    
    def get_embeddings(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Get contextual embeddings for categorical features (for interpretability)."""
        x = self.column_embedding(x_cat)
        for block in self.transformer_blocks:
            x, _ = block(x)
        return self.norm(x)


# =============================================================================
# CAUSAL DISCOVERY MODULE
# =============================================================================

class CausalDiscoveryModule(nn.Module):
    """
    Differentiable Causal Discovery Module.
    
    Learns a DAG (Directed Acyclic Graph) structure among features using:
    1. Learnable adjacency matrix with continuous relaxation
    2. DAG constraint via NOTEARS algorithm
    3. Sparsity regularization
    
    Reference: NOTEARS - "DAGs with NO TEARS: Continuous Optimization for Structure Learning"
    """
    def __init__(
        self, 
        num_features: int,
        hidden_dim: int = 64,
        threshold: float = 0.3  # Threshold for edge existence
    ):
        super().__init__()
        self.num_features = num_features
        self.threshold = threshold
        
        # Learnable adjacency matrix parameters
        # We learn W such that W[i,j] represents the causal effect of feature j on feature i
        self.W = nn.Parameter(torch.zeros(num_features, num_features))
        nn.init.uniform_(self.W, -0.1, 0.1)
        
        # Mask diagonal (no self-loops)
        self.register_buffer('diag_mask', 1 - torch.eye(num_features))
    
    def get_adjacency_matrix(self, hard: bool = False) -> torch.Tensor:
        """
        Get the learned adjacency matrix.
        
        Args:
            hard: If True, apply threshold to get binary matrix
        
        Returns:
            Adjacency matrix of shape (num_features, num_features)
        """
        # Apply sigmoid to get values in [0, 1]
        A = torch.sigmoid(self.W) * self.diag_mask
        
        if hard:
            A = (A > self.threshold).float()
        
        return A
    
    def dag_loss(self) -> torch.Tensor:
        """
        Compute DAG constraint loss using NOTEARS formulation.
        
        The constraint is: h(A) = tr(e^A) - d = 0 for DAG
        """
        A = self.get_adjacency_matrix(hard=False)
        d = self.num_features
        
        # Matrix exponential trace
        # h(A) = tr(e^(A ⊙ A)) - d
        A_sq = A * A
        M = torch.matrix_exp(A_sq)
        h = torch.trace(M) - d
        
        return h * h  # Squared constraint
    
    def sparsity_loss(self) -> torch.Tensor:
        """L1 regularization for sparsity."""
        A = self.get_adjacency_matrix(hard=False)
        return torch.sum(torch.abs(A))
    
    def forward(self) -> torch.Tensor:
        """Returns the soft adjacency matrix."""
        return self.get_adjacency_matrix(hard=False)


# =============================================================================
# CAUSAL ATTENTION MASK
# =============================================================================

class CausalAttentionMask(nn.Module):
    """
    Creates attention masks based on learned causal structure.
    
    The mask ensures that:
    1. Features can only attend to their causal ancestors
    2. Non-causal correlations are blocked
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
    
    def forward(
        self, 
        adjacency_matrix: torch.Tensor,
        include_self: bool = True
    ) -> torch.Tensor:
        """
        Create attention mask from adjacency matrix.
        
        Args:
            adjacency_matrix: DAG adjacency matrix (num_features, num_features)
                             A[i,j] = 1 means j causes i
            include_self: Whether to allow self-attention
        
        Returns:
            Attention mask where 0 = attend, -inf = block
        """
        # Compute transitive closure to get all ancestors
        # If j causes i directly or indirectly, feature i can attend to j
        A = adjacency_matrix
        
        # Add self-connections if needed
        if include_self:
            A = A + torch.eye(self.num_features, device=A.device)
        
        # Compute transitive closure using matrix powers
        # (I + A)^n converges to transitive closure for DAGs
        closure = A.clone()
        for _ in range(self.num_features):
            closure = torch.clamp(closure @ A + closure, 0, 1)
        
        # Convert to attention mask: 0 for allowed, -inf for blocked
        mask = torch.where(
            closure > 0.5,
            torch.zeros_like(closure),
            torch.full_like(closure, float('-inf'))
        )
        
        return mask


# =============================================================================
# COUNTERFACTUAL REGULARIZATION
# =============================================================================

class CounterfactualRegularizer(nn.Module):
    """
    Counterfactual Regularization Loss.
    
    Encourages the model to rely on causally relevant features by:
    1. Generating counterfactual samples by intervening on non-causal features
    2. Penalizing prediction changes when only non-causal features change
    """
    def __init__(self, num_features: int, intervention_strength: float = 0.5):
        super().__init__()
        self.num_features = num_features
        self.intervention_strength = intervention_strength
    
    def generate_counterfactual(
        self,
        x: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        target_feature_idx: int
    ) -> torch.Tensor:
        """
        Generate counterfactual by intervening on non-causal features.
        
        Args:
            x: Original input (batch, num_features)
            adjacency_matrix: Causal DAG
            target_feature_idx: Index of feature to analyze
        
        Returns:
            Counterfactual samples
        """
        batch_size = x.shape[0]
        
        # Find non-causal features (not ancestors of target)
        A = adjacency_matrix
        
        # Features that are NOT ancestors of the target
        # can be intervened without affecting the target through causal paths
        ancestors = A[target_feature_idx, :]  # Which features cause target
        non_causal = (ancestors < 0.5)  # Features that don't cause target
        
        # Create counterfactual by perturbing non-causal features
        x_cf = x.clone()
        noise = torch.randn_like(x) * self.intervention_strength
        
        # Only perturb non-causal features
        mask = non_causal.unsqueeze(0).expand(batch_size, -1).float()
        x_cf = x_cf + noise * mask
        
        return x_cf
    
    def forward(
        self,
        model: nn.Module,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute counterfactual regularization loss.
        
        Penalizes the model if predictions change significantly when
        only non-causal features are modified.
        """
        if x_cont is None or x_cont.shape[1] == 0:
            return torch.tensor(0.0, device=x_cat.device)
        
        total_loss = 0.0
        num_cont = x_cont.shape[1]
        num_cat = x_cat.shape[1]
        
        # Generate counterfactuals for continuous features
        for i in range(num_cont):
            # Get adjacency for this continuous feature
            # (Continuous features come after categorical in the adjacency matrix)
            feat_idx = num_cat + i
            
            if feat_idx < adjacency_matrix.shape[0]:
                x_cont_cf = self.generate_counterfactual(
                    x_cont, 
                    adjacency_matrix[num_cat:, num_cat:] if num_cat < adjacency_matrix.shape[0] else adjacency_matrix,
                    i
                )
                
                # Get predictions for counterfactual
                with torch.no_grad():
                    pred_cf = model(x_cat, x_cont_cf)
                
                # Loss: predictions should be similar if only non-causal features changed
                # This is weighted by how non-causal the intervened features are
                loss = F.mse_loss(predictions, pred_cf)
                total_loss = total_loss + loss
        
        return total_loss / max(num_cont, 1)


# =============================================================================
# CAUSALTAB MODEL
# =============================================================================

class CausalTab(nn.Module):
    """
    CausalTab: Learning Causal Feature Interactions via Attention-Guided 
    Structural Discovery in Tabular Transformers
    
    Key innovations:
    1. Causal Attention Mask - Constrains attention to respect causal structure
    2. Differentiable Causal Discovery - Jointly learns causal DAG during training
    3. Counterfactual Regularization - Encourages reliance on causal features
    
    Architecture:
        1. Column Embedding for categorical features
        2. Causal Discovery Module (learns DAG)
        3. Transformer layers with Causal Attention Mask
        4. Concatenation with continuous features
        5. MLP for prediction
    """
    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        dim: int = 32,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 16,
        mlp_hidden_mults: Tuple[int, ...] = (4, 2),
        mlp_act: nn.Module = nn.ReLU(),
        num_classes: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        column_embed_dim: int = 8,
        # CausalTab specific parameters
        causal_threshold: float = 0.3,
        dag_loss_weight: float = 1.0,
        sparsity_loss_weight: float = 0.1,
        counterfactual_loss_weight: float = 0.5,
        use_causal_mask: bool = True
    ):
        super().__init__()
        
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.num_features = self.num_categories + num_continuous
        self.dim = dim
        self.use_causal_mask = use_causal_mask
        
        # Loss weights
        self.dag_loss_weight = dag_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight
        self.counterfactual_loss_weight = counterfactual_loss_weight
        
        # Column embedding for categorical features
        self.column_embedding = ColumnEmbedding(
            num_categories=categories,
            dim=dim,
            column_embed_dim=column_embed_dim
        )
        
        # Causal Discovery Module
        self.causal_discovery = CausalDiscoveryModule(
            num_features=self.num_categories,  # Only for categorical features in attention
            threshold=causal_threshold
        )
        
        # Causal Attention Mask
        self.causal_mask_module = CausalAttentionMask(self.num_categories)
        
        # Transformer layers with causal masking capability
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Continuous feature normalization
        if num_continuous > 0:
            self.cont_norm = nn.LayerNorm(num_continuous)
        
        # MLP input dimension
        mlp_input_dim = dim * self.num_categories + num_continuous
        
        # MLP layers
        mlp_layers = []
        dims = [mlp_input_dim] + [mlp_input_dim * m for m in mlp_hidden_mults]
        
        for i in range(len(dims) - 1):
            mlp_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                mlp_act,
                nn.Dropout(ff_dropout)
            ])
        
        mlp_layers.append(nn.Linear(dims[-1], num_classes))
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Counterfactual regularizer
        self.cf_regularizer = CounterfactualRegularizer(
            num_features=self.num_features
        )
        
        # Store for analysis
        self.attention_weights = None
        self.causal_graph = None
    
    def get_causal_mask(self) -> torch.Tensor:
        """Get the current causal attention mask."""
        adj = self.causal_discovery.get_adjacency_matrix(hard=False)
        return self.causal_mask_module(adj)
    
    def get_causal_graph(self, hard: bool = True) -> torch.Tensor:
        """Get the learned causal graph."""
        return self.causal_discovery.get_adjacency_matrix(hard=hard)
    
    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """Get auxiliary losses for training."""
        return {
            'dag_loss': self.causal_discovery.dag_loss() * self.dag_loss_weight,
            'sparsity_loss': self.causal_discovery.sparsity_loss() * self.sparsity_loss_weight
        }
    
    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        apply_causal_mask: bool = None
    ) -> torch.Tensor:
        """
        Args:
            x_cat: Categorical features (batch, num_cat_features)
            x_cont: Continuous features (batch, num_continuous)
            return_attention: Whether to return attention weights
            apply_causal_mask: Override for whether to apply causal mask
        
        Returns:
            Predictions of shape (batch, num_classes)
        """
        apply_mask = apply_causal_mask if apply_causal_mask is not None else self.use_causal_mask
        
        # Embed categorical features
        x = self.column_embedding(x_cat)
        
        # Get causal mask if using
        causal_mask = None
        if apply_mask:
            adj = self.causal_discovery.get_adjacency_matrix(hard=False)
            causal_mask = self.causal_mask_module(adj)
            self.causal_graph = adj.detach()
        
        # Pass through Transformer layers with causal mask
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn = block(x, causal_mask)
            attention_weights.append(attn)
        
        # Apply final norm
        x = self.norm(x)
        self.attention_weights = attention_weights
        
        # Flatten categorical embeddings
        x = x.flatten(start_dim=1)
        
        # Process continuous features
        if self.num_continuous > 0 and x_cont is not None:
            x_cont_normed = self.cont_norm(x_cont)
            x = torch.cat([x, x_cont_normed], dim=1)
        
        # MLP prediction
        out = self.mlp(x)
        
        if return_attention:
            return out, attention_weights
        
        return out
    
    def compute_total_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        x_cat: torch.Tensor,
        x_cont: Optional[torch.Tensor],
        criterion: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss including auxiliary losses.
        
        Returns:
            total_loss: Combined loss for optimization
            loss_dict: Dictionary of individual loss components
        """
        # Main prediction loss
        pred_loss = criterion(predictions, targets)
        
        # Auxiliary losses
        aux_losses = self.get_auxiliary_losses()
        
        # Counterfactual loss (optional, can be expensive)
        cf_loss = torch.tensor(0.0, device=predictions.device)
        if self.counterfactual_loss_weight > 0 and x_cont is not None:
            full_adj = self.causal_discovery.get_adjacency_matrix(hard=False)
            cf_loss = self.cf_regularizer(
                self, x_cat, x_cont, full_adj, predictions.detach()
            ) * self.counterfactual_loss_weight
        
        # Total loss
        total_loss = pred_loss + aux_losses['dag_loss'] + aux_losses['sparsity_loss'] + cf_loss
        
        loss_dict = {
            'prediction_loss': pred_loss,
            'dag_loss': aux_losses['dag_loss'],
            'sparsity_loss': aux_losses['sparsity_loss'],
            'counterfactual_loss': cf_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(
    model_type: str,
    categories: List[int],
    num_continuous: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'tabtransformer' or 'causaltab'
        categories: List of category counts per categorical feature
        num_continuous: Number of continuous features
        **kwargs: Additional model-specific parameters
    
    Returns:
        Instantiated model
    """
    if model_type.lower() == 'tabtransformer':
        return TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            **kwargs
        )
    elif model_type.lower() == 'causaltab':
        return CausalTab(
            categories=categories,
            num_continuous=num_continuous,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
