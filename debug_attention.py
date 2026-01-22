#!/usr/bin/env python3
"""
Debug multiple quantile issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from modules.attention import QuantileConditionedAttention

def debug_multiple_quantiles():
    """Debug the multiple quantile issue"""
    print("=== Debugging Multiple Quantiles ===")
    
    attention = QuantileConditionedAttention(d_model=256, n_heads=4, max_len=200)
    
    B, T, d_model = 2, 48, 256
    
    # Test multiple quantiles
    x = torch.randn(B, T, d_model)
    q_multi = torch.rand(B, 3)  # [B, Q]
    
    print(f"Input shape: {x.shape}")
    print(f"Quantile shape: {q_multi.shape}")
    
    # Step through the attention forward pass
    print("\n--- Step 1: Positional encoding ---")
    x = attention.pos_encoding(x)
    print(f"After pos encoding: {x.shape}")
    
    print("\n--- Step 2: Quantile processing ---")
    avg_quantile = q_multi.mean(dim=1, keepdim=True)  # [B, 1]
    print(f"Avg quantile: {avg_quantile.shape}")
    
    # Process each quantile separately for queries (corrected)
    q_embed_list = []
    for i in range(q_multi.shape[1]):
        q_single = q_multi[:, i:i+1]  # [B, 1]
        q_embed = attention.quantile_proj_q(q_single)  # [B, d_model]
        q_embed_list.append(q_embed)
    
    q_embed = torch.stack(q_embed_list, dim=1)  # [B, Q, d_model]
    k_embed = attention.quantile_proj_k(avg_quantile)  # [B, d_model]
    v_embed = attention.quantile_proj_v(avg_quantile)  # [B, d_model]
    
    print(f"Q embed: {q_embed.shape}")
    print(f"K embed: {k_embed.shape}")
    print(f"V embed: {v_embed.shape}")
    
    print("\n--- Step 3: Handle multiple quantiles ---")
    if q_embed.shape[1] > 1:
        q_embed = q_embed[:, 0:1, :]  # [B, 1, d_model]
        print(f"Selected first quantile: {q_embed.shape}")
    
    print("\n--- Step 4: Expand across time ---")
    q_embed = q_embed.expand(-1, T, -1)
    k_embed = k_embed.unsqueeze(1).expand(-1, T, -1)
    v_embed = v_embed.unsqueeze(1).expand(-1, T, -1)
    
    print(f"Q embed expanded: {q_embed.shape}")
    print(f"K embed expanded: {k_embed.shape}")
    print(f"V embed expanded: {v_embed.shape}")
    
    print("\n--- Step 5: Linear projections ---")
    Q = attention.W_q(x) + q_embed
    K = attention.W_k(x) + k_embed
    V = attention.W_v(x) + v_embed
    
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    
    print("\n--- Step 6: Multi-head reshape ---")
    Q = Q.view(B, T, attention.n_heads, attention.d_k).transpose(1, 2)
    K = K.view(B, T, attention.n_heads, attention.d_k).transpose(1, 2)
    V = V.view(B, T, attention.n_heads, attention.d_k).transpose(1, 2)
    
    print(f"Q reshaped: {Q.shape}")
    print(f"K reshaped: {K.shape}")
    print(f"V reshaped: {V.shape}")

if __name__ == "__main__":
    debug_multiple_quantiles()
