import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for time series data."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for even and odd positions separately
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sin to even positions (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd positions (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model] with positional encoding added
        """
        T = x.shape[1]
        return x + self.pe[:, :T, :]


class QuantileConditionedAttention(nn.Module):
    """Multi-head attention with quantile-conditioned queries, keys, and values.

    Different quantiles can focus on different temporal patterns (e.g., extremes on peaks).
    Enhanced with positional encoding and improved quantile broadcasting.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1, 
                 max_len: int = 1000):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Standard projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Enhanced quantile conditioning for all attention components
        self.quantile_proj_q = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.quantile_proj_k = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.quantile_proj_v = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Positional encoding for temporal awareness
        self.pos_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x: torch.Tensor, quantile: torch.Tensor) -> torch.Tensor:
        """Apply attention conditioned on quantile levels.

        Args:
            x: [B, T, d_model] time-series features.
            quantile: [B, 1] or [B, Q] quantile levels (will broadcast over time).
        Returns:
            Tensor of shape [B, T, d_model] after attention.
        """
        B, T, _ = x.shape
        
        # Add positional encoding for temporal awareness
        x = self.pos_encoding(x)
        
        # Improved quantile broadcasting
        if quantile.dim() == 2 and quantile.shape[1] == 1:
            # Single quantile case: [B, 1] -> [B, T, d_model]
            q_embed = self.quantile_proj_q(quantile)  # [B, d_model]
            k_embed = self.quantile_proj_k(quantile)  # [B, d_model]
            v_embed = self.quantile_proj_v(quantile)  # [B, d_model]
            
            # Expand across time dimension
            q_embed = q_embed.unsqueeze(1).expand(-1, T, -1)
            k_embed = k_embed.unsqueeze(1).expand(-1, T, -1)
            v_embed = v_embed.unsqueeze(1).expand(-1, T, -1)
            
        elif quantile.dim() == 2:
            # Multiple quantiles case: [B, Q] -> [B, T, d_model]
            # Use average quantile embedding for keys/values (they should be quantile-agnostic)
            avg_quantile = quantile.mean(dim=1, keepdim=True)  # [B, 1]
            
            # Process each quantile separately for queries
            q_embed_list = []
            for i in range(quantile.shape[1]):
                q_single = quantile[:, i:i+1]  # [B, 1]
                q_embed = self.quantile_proj_q(q_single)  # [B, d_model]
                q_embed_list.append(q_embed)
            
            # Stack and use first quantile for attention computation
            q_embed = torch.stack(q_embed_list, dim=1)  # [B, Q, d_model]
            k_embed = self.quantile_proj_k(avg_quantile)  # [B, d_model]
            v_embed = self.quantile_proj_v(avg_quantile)  # [B, d_model]
            
            # Handle query embedding for multiple quantiles
            if q_embed.shape[1] > 1:
                # Use first quantile for attention computation
                q_embed = q_embed[:, 0:1, :]  # [B, 1, d_model]
            
            # Expand across time dimension
            q_embed = q_embed.expand(-1, T, -1)
            k_embed = k_embed.unsqueeze(1).expand(-1, T, -1)
            v_embed = v_embed.unsqueeze(1).expand(-1, T, -1)
        else:
            raise ValueError(f"Unsupported quantile shape: {quantile.shape}")

        # Compute Q, K, V with enhanced quantile conditioning
        Q = self.W_q(x) + q_embed
        K = self.W_k(x) + k_embed
        V = self.W_v(x) + v_embed

        # Multi-head reshape
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Residual + projection + norm
        out = self.W_o(out)
        return self.layer_norm(x + out)
