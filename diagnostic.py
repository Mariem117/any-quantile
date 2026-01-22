#!/usr/bin/env python
"""
Diagnostic script to identify the root cause of zero predictions
Run this BEFORE training to test each component
"""

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

def test_attention_module():
    """Test if QuantileConditionedAttention works correctly"""
    print("\n" + "="*80)
    print("TEST 1: QuantileConditionedAttention")
    print("="*80)
    
    from modules.attention import QuantileConditionedAttention
    
    attn = QuantileConditionedAttention(d_model=256, n_heads=8, dropout=0.1)
    
    # Test input
    B, T = 4, 168
    x = torch.randn(B, T, 256) * 0.5
    quantile = torch.rand(B, 1)
    
    print(f"Input shape: {x.shape}")
    print(f"Quantile shape: {quantile.shape}")
    
    with torch.no_grad():
        output = attn(x, quantile)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    if output.abs().max() < 1e-8:
        print("❌ FAILED: Attention outputs zeros")
        return False
    
    print("✅ PASSED: Attention works")
    return True


def test_backbone():
    """Test if NBEATSAQATTENTION backbone works"""
    print("\n" + "="*80)
    print("TEST 2: NBEATSAQATTENTION Backbone")
    print("="*80)
    
    try:
        from modules import NBEATSAQATTENTION
    except ImportError:
        print("❌ Cannot import NBEATSAQATTENTION")
        print("   Please share your modules.py file")
        return False
    
    # Create backbone with your config params
    backbone = NBEATSAQATTENTION(
        dropout=0.1,
        layer_width=512,
        num_layers=3,
        num_blocks=15,
        share=False,
        size_in=168,
        size_out=48,
        n_heads=8
    )
    
    # Test with realistic normalized input
    B, T, Q = 4, 168, 10
    history = torch.randn(B, T) * 0.3  # Normalized around 0
    quantiles = torch.linspace(0.1, 0.9, Q)[None, :].expand(B, -1)
    
    print(f"Input history shape: {history.shape}")
    print(f"Input history range: [{history.min():.4f}, {history.max():.4f}]")
    print(f"Quantiles: {quantiles[0].tolist()}")
    
    with torch.no_grad():
        output = backbone(history, quantiles)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
    print(f"Output mean: {output.mean():.6f}")
    print(f"Output std: {output.std():.6f}")
    
    # Check expected shape
    if output.shape != torch.Size([B, 48, Q]) and output.shape != torch.Size([B, 48*Q]):
        print(f"⚠️  WARNING: Unexpected output shape")
        print(f"   Expected: [4, 48, 10] or [4, 480]")
        print(f"   Got: {output.shape}")
    
    # Check for zeros
    if output.abs().max() < 1e-8:
        print("\n❌ FAILED: Backbone outputs all zeros!")
        print("\nDebugging steps:")
        print("1. Check backbone initialization - print first layer weights:")
        for name, param in backbone.named_parameters():
            if 'weight' in name:
                print(f"   {name}: mean={param.mean():.6f}, std={param.std():.6f}")
                break
        
        print("\n2. Check if gradients flow:")
        history.requires_grad = True
        output = backbone(history, quantiles)
        loss = output.sum()
        loss.backward()
        print(f"   Input gradient norm: {history.grad.norm():.6f}")
        
        return False
    
    print("✅ PASSED: Backbone produces non-zero output")
    return True


def test_full_forward_pass():
    """Test the complete forward pass with normalization"""
    print("\n" + "="*80)
    print("TEST 3: Full Forward Pass (with normalization)")
    print("="*80)
    
    # Load config
    try:
        with open('config/nbeatsaq-attention-mhlv-fast.yaml') as f:
            cfg_dict = yaml.unsafe_load(f)
        cfg = OmegaConf.create(cfg_dict)
    except FileNotFoundError:
        print("⚠️  Config file not found, using default values")
        cfg = OmegaConf.create({
            'model': {
                'max_norm': True,
                'input_horizon_len': 168
            },
            'dataset': {
                'horizon_length': 48
            }
        })
    
    print(f"max_norm setting: {cfg.model.max_norm}")
    print(f"max_norm type: {type(cfg.model.max_norm)}")
    
    # Simulate realistic electricity data
    B = 4
    history = torch.tensor([
        [1000, 1200, 1500, 1800] * 42,  # Typical electricity demand
        [800, 900, 1100, 1300] * 42,
        [500, 600, 700, 800] * 42,
        [2000, 2500, 3000, 3500] * 42,
    ], dtype=torch.float32)
    
    target = torch.randn(B, 48) * 500 + 1500  # Target around 1500
    quantiles = torch.rand(B, 10)
    
    print(f"\nInput history range: [{history.min():.2f}, {history.max():.2f}]")
    print(f"Target range: [{target.min():.2f}, {target.max():.2f}]")
    
    # Normalize
    max_norm = cfg.model.max_norm
    if isinstance(max_norm, (float, int)):
        max_norm = bool(max_norm)
    
    if max_norm:
        x_max = history.abs().max(dim=-1, keepdims=True)[0]
        x_max = torch.clamp(x_max, min=1.0)
        print(f"x_max values: {x_max.squeeze()}")
    else:
        x_max = torch.ones(B, 1)
        print("No normalization applied")
    
    history_norm = history / x_max
    print(f"Normalized history range: [{history_norm.min():.4f}, {history_norm.max():.4f}]")
    
    # Test backbone
    try:
        from modules import NBEATSAQATTENTION
        backbone = NBEATSAQATTENTION(
            dropout=0.1, layer_width=512, num_layers=3, num_blocks=15,
            share=False, size_in=168, size_out=48, n_heads=8
        )
        
        with torch.no_grad():
            output = backbone(history_norm, quantiles)
        
        print(f"\nBackbone output shape: {output.shape}")
        print(f"Backbone output range: [{output.min():.6f}, {output.max():.6f}]")
        
        # Denormalize
        if output.dim() == 2 and output.shape[1] == 480:
            output = output.reshape(B, 48, 10)
        elif output.dim() == 2 and output.shape[1] == 10:
            output = output.unsqueeze(1).expand(-1, 48, -1)
        
        output_denorm = output * x_max.unsqueeze(-1)
        print(f"Denormalized output range: [{output_denorm.min():.2f}, {output_denorm.max():.2f}]")
        print(f"Expected (target) range: [{target.min():.2f}, {target.max():.2f}]")
        
        if output_denorm.abs().max() < 1e-6:
            print("\n❌ FAILED: Denormalized output is zero")
            return False
        
        print("✅ PASSED: Full forward pass works")
        return True
        
    except Exception as e:
        print(f"❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("DIAGNOSTIC TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Attention Module", test_attention_module()))
    results.append(("NBEATS Backbone", test_backbone()))
    results.append(("Full Forward Pass", test_full_forward_pass()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    if all(r[1] for r in results):
        print("\n🎉 All tests passed! Your model should train correctly.")
        print("\nNext steps:")
        print("1. Update your config: max_norm: 0.5 → max_norm: true")
        print("2. Replace shared_forward() with the fixed version")
        print("3. Run training: python run.py --config config/nbeatsaq-attention-mhlv-fast.yaml")
    else:
        print("\n⚠️  Some tests failed. Fix the failing components before training.")
        print("\nIf backbone test failed, please share your modules.py file.")


if __name__ == "__main__":
    main()