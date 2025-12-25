import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileConditionedAttention(nn.Module):
    """Multi-head attention with quantile-conditioned queries.

    Different quantiles can focus on different temporal patterns (e.g., extremes on peaks).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
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

        # Quantile conditioning injected into queries
        self.quantile_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, quantile: torch.Tensor) -> torch.Tensor:
        """Apply attention conditioned on quantile levels.

        Args:
            x: [B, T, d_model] time-series features.
            quantile: [B, 1] or [B, Q] quantile levels (will broadcast over time).
        Returns:
            Tensor of shape [B, T, d_model] after attention.
        """
        B, T, _ = x.shape

        # Broadcast quantile embeddings to match time dimension
        q_embed = self.quantile_proj(quantile.unsqueeze(-1))  # [B, Q, d_model]
        if q_embed.dim() == 3 and q_embed.shape[1] == 1:
            q_embed = q_embed.expand(-1, T, -1)
        elif q_embed.shape[1] != T:
            # If quantiles are per-quantile (Q), expand across time
            q_embed = q_embed.unsqueeze(2).expand(-1, q_embed.shape[1], T, -1)
            q_embed = q_embed.reshape(B, T, self.d_model)

        # Compute Q, K, V
        Q = self.W_q(x) + q_embed
        K = self.W_k(x)
        V = self.W_v(x)

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
