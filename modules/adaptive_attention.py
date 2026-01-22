import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PositionalEncoding


class AdaptiveAttention(nn.Module):
    """Attention mechanism that fuses adaptive sampling with quantile-conditioned attention.
    
    This module combines:
    1. Adaptive quantile sampling based on recent loss feedback
    2. Quantile-conditioned multi-head attention
    3. Dynamic attention weight adjustment based on sampling importance
    """
    
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1, 
                 max_len: int = 1000, adaptive_temp: float = 1.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.adaptive_temp = adaptive_temp

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Enhanced quantile conditioning
        self.quantile_proj_q = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.quantile_proj_k = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.quantile_proj_v = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Adaptive importance weighting
        self.importance_net = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # Attention modulation based on adaptive sampling
        self.attention_modulator = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),  # +1 for importance score
            nn.ReLU(),
            nn.Linear(d_model // 2, n_heads),  # One modulation per head
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Adaptive sampling state
        self.register_buffer('quantile_importance', torch.zeros(100))  # For 100 quantile bins
        self.register_buffer('importance_momentum', torch.tensor(0.99))

    def update_quantile_importance(self, quantiles: torch.Tensor, losses: torch.Tensor) -> None:
        """Update importance weights based on quantile-specific losses."""
        with torch.no_grad():
            # Bin quantiles and update importance
            bin_edges = torch.linspace(0.0, 1.0, 101, device=quantiles.device)
            bin_indices = torch.bucketize(quantiles, bin_edges) - 1
            bin_indices = torch.clamp(bin_indices, 0, 99)
            
            # Update running importance estimates
            for i, (bin_idx, loss) in enumerate(zip(bin_indices, losses)):
                current_importance = self.quantile_importance[bin_idx]
                updated_importance = (self.importance_momentum * current_importance + 
                                    (1 - self.importance_momentum) * loss)
                self.quantile_importance[bin_idx] = updated_importance

    def get_importance_weight(self, quantile: torch.Tensor) -> torch.Tensor:
        """Get importance weight for given quantile based on adaptive sampling."""
        with torch.no_grad():
            # Convert quantile to bin index
            bin_idx = torch.clamp((quantile * 100).long(), 0, 99)
            importance = self.quantile_importance[bin_idx]
            
            # Normalize importance to [0, 1] range
            max_importance = self.quantile_importance.max()
            if max_importance > 0:
                importance = importance / max_importance
            
            return importance

    def forward(self, x: torch.Tensor, quantile: torch.Tensor, 
                return_attention: bool = False) -> torch.Tensor:
        """Apply adaptive attention conditioned on quantile levels.

        Args:
            x: [B, T, d_model] time-series features
            quantile: [B, 1] or [B, Q] quantile levels
            return_attention: Whether to return attention weights for analysis
            
        Returns:
            Tensor of shape [B, T, d_model] after adaptive attention
            Optionally returns attention weights if return_attention=True
        """
        B, T, _ = x.shape
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Handle quantile broadcasting
        if quantile.dim() == 2 and quantile.shape[1] == 1:
            # Single quantile case
            q_embed = self.quantile_proj_q(quantile)  # [B, d_model]
            k_embed = self.quantile_proj_k(quantile)  # [B, d_model]
            v_embed = self.quantile_proj_v(quantile)  # [B, d_model]
            
            # Get importance weights for adaptive modulation
            importance_weights = torch.stack([
                self.get_importance_weight(q).unsqueeze(0) for q in quantile[:, 0]
            ])  # [B, 1]
            
            # Expand across time dimension
            q_embed = q_embed.unsqueeze(1).expand(B, T, -1)
            k_embed = k_embed.unsqueeze(1).expand(B, T, -1)
            v_embed = v_embed.unsqueeze(1).expand(B, T, -1)
            importance_weights = importance_weights.unsqueeze(1).expand(B, T, 1)
            
        else:
            # Multiple quantiles case - use average for attention computation
            avg_quantile = quantile.mean(dim=1, keepdim=True)  # [B, 1]
            
            q_embed = self.quantile_proj_q(avg_quantile)  # [B, d_model]
            k_embed = self.quantile_proj_k(avg_quantile)  # [B, d_model]
            v_embed = self.quantile_proj_v(avg_quantile)  # [B, d_model]
            
            # Get average importance weights
            importance_weights = torch.stack([
                self.get_importance_weight(q).unsqueeze(0) for q in avg_quantile[:, 0]
            ])  # [B, 1]
            
            # Expand across time dimension
            q_embed = q_embed.unsqueeze(1).expand(B, T, -1)
            k_embed = k_embed.unsqueeze(1).expand(B, T, -1)
            v_embed = v_embed.unsqueeze(1).expand(B, T, -1)
            importance_weights = importance_weights.unsqueeze(1).expand(B, T, 1)

        # Compute Q, K, V with quantile conditioning
        Q = self.W_q(x) + q_embed
        K = self.W_k(x) + k_embed
        V = self.W_v(x) + v_embed

        # Multi-head reshape
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Adaptive attention modulation based on importance
        # Simplified approach: use importance weights directly
        importance_expanded = importance_weights.unsqueeze(1)  # [B, 1, T, 1]
        importance_expanded = importance_expanded.expand(-1, self.n_heads, -1, -1)  # [B, H, T, 1]
        
        # Apply modulation to attention scores
        scores = scores * (1 + importance_expanded * self.adaptive_temp)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Residual + projection + norm
        out = self.W_o(out)
        result = self.layer_norm(x + out)
        
        if return_attention:
            return result, attn_weights
        return result


class AdaptiveAttentionBlock(nn.Module):
    """Complete block combining adaptive attention with feed-forward network."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1,
                 d_ff: int = 1024, adaptive_temp: float = 1.0):
        super().__init__()
        
        self.adaptive_attention = AdaptiveAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout,
            adaptive_temp=adaptive_temp
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, quantile: torch.Tensor, 
                return_attention: bool = False) -> torch.Tensor:
        """Forward pass through adaptive attention block."""
        if return_attention:
            attn_out, attn_weights = self.adaptive_attention(x, quantile, return_attention=True)
        else:
            attn_out = self.adaptive_attention(x, quantile)
            
        # Feed-forward with residual connection
        ff_out = self.feed_forward(attn_out)
        result = self.norm2(attn_out + ff_out)
        
        if return_attention:
            return result, attn_weights
        return result
