#!/usr/bin/env python3
"""
Simple test script to validate configuration without requiring all dependencies.
"""

from omegaconf import OmegaConf

def test_config():
    """Test the adaptive attention configuration."""
    print("🔍 Testing Adaptive Attention Configuration")
    print("=" * 50)
    
    # Load configuration
    cfg = OmegaConf.load('config/nbeatsaq-adaptive-attention-mhlv.yaml')
    
    # Basic validation
    print("✅ Configuration loaded successfully")
    print(f"📋 Model target: {cfg.model._target_}")
    print(f"📋 Dataset: {cfg.dataset.name}")
    print(f"📋 History length: {cfg.dataset.history_length}")
    print(f"📋 Horizon length: {cfg.dataset.horizon_length}")
    
    # Adaptive sampling config
    if hasattr(cfg.model, 'adaptive_sampling'):
        print("\n🎯 Adaptive Sampling Configuration:")
        sampling_cfg = cfg.model.adaptive_sampling
        print(f"   • Adaptive quantiles: {sampling_cfg.num_adaptive_quantiles}")
        print(f"   • Number of bins: {sampling_cfg.num_bins}")
        print(f"   • Momentum: {sampling_cfg.momentum}")
        print(f"   • Temperature: {sampling_cfg.temperature}")
        print(f"   • Min probability: {sampling_cfg.min_prob}")
    
    # Adaptive attention config
    if hasattr(cfg.model, 'adaptive_attention'):
        print("\n🧠 Adaptive Attention Configuration:")
        attention_cfg = cfg.model.adaptive_attention
        print(f"   • Model dimension: {attention_cfg.d_model}")
        print(f"   • Number of heads: {attention_cfg.n_heads}")
        print(f"   • Dropout: {attention_cfg.dropout}")
        print(f"   • Feed-forward dim: {attention_cfg.d_ff}")
        print(f"   • Adaptive temperature: {attention_cfg.adaptive_temp}")
        print(f"   • Number of blocks: {attention_cfg.num_blocks}")
    
    # Training config
    print("\n🏋️ Training Configuration:")
    print(f"   • Max epochs: {cfg.trainer.max_epochs}")
    print(f"   • Batch size: {cfg.dataset.train_batch_size}")
    print(f"   • Learning rate: {cfg.model.optimizer.lr}")
    print(f"   • Gradient clipping: {cfg.trainer.gradient_clip_val}")
    print(f"   • Warmup updates: {cfg.model.scheduler.warmup_updates}")
    
    # Backbone config
    print("\n🏗️ Backbone Configuration:")
    backbone_cfg = cfg.model.nn.backbone
    print(f"   • Type: {backbone_cfg._target_}")
    print(f"   • Number of blocks: {backbone_cfg.num_blocks}")
    print(f"   • Layer width: {backbone_cfg.layer_width}")
    print(f"   • Number of layers: {backbone_cfg.num_layers}")
    print(f"   • Dropout: {backbone_cfg.dropout}")
    
    print("\n✅ Configuration validation complete!")
    print("\n📖 Usage:")
    print("   python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml")
    print("   python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml --fast-dev-run")

if __name__ == "__main__":
    test_config()
