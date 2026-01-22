# Adaptive Attention Model Configuration Guide

## 🚀 **Quick Start**

### 1. **Basic Training**
```bash
python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml
```

### 2. **Fast Development Run**
```bash
python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml --fast-dev-run
```

### 3. **Resume from Checkpoint**
```bash
python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml --resume lightning_logs/nbeatsaq-adaptive-attention-mhlv/checkpoints/last.ckpt
```

## ⚙️ **Configuration Parameters**

### **Adaptive Sampling Parameters**
```yaml
adaptive_sampling:
  num_adaptive_quantiles: 3    # Number of adaptive quantiles per batch
  num_bins: 100               # Number of bins for importance tracking
  momentum: 0.99              # Momentum for importance updates (0.9-0.99)
  temperature: 1.0            # Temperature for sampling (0.5-2.0)
  min_prob: 0.001             # Minimum probability for each bin
```

### **Adaptive Attention Parameters**
```yaml
adaptive_attention:
  d_model: 256               # Attention dimension (128-512)
  n_heads: 8                 # Number of attention heads (4-16)
  dropout: 0.1               # Dropout rate (0.0-0.3)
  d_ff: 1024                 # Feed-forward dimension (512-2048)
  adaptive_temp: 1.0          # Temperature for attention modulation
  num_blocks: 2              # Number of attention blocks (1-4)
```

### **Model Architecture Parameters**
```yaml
nn:
  backbone:
    num_blocks: 30            # NBEATS blocks (20-50)
    layer_width: 512          # Layer width (256-1024)
    dropout: 0.1             # Dropout for regularization
```

## 🎯 **Performance Tuning Tips**

### **For Better Convergence:**
- Lower learning rate: `lr: 0.0005`
- Longer warmup: `warmup_updates: 800`
- Gradient accumulation: `accumulate_grad_batches: 2`

### **For Memory Efficiency:**
- Smaller batch size: `train_batch_size: 256`
- Fewer attention blocks: `num_blocks: 1`
- Smaller d_model: `d_model: 128`

### **For Better Performance:**
- More adaptive quantiles: `num_adaptive_quantiles: 5`
- Higher temperature: `adaptive_temp: 1.5`
- More attention heads: `n_heads: 12`

## 📊 **Monitoring**

### **Key Metrics to Watch:**
- `train/loss` - Training loss
- `val/loss` - Validation loss
- `val/crps` - Continuous Ranked Probability Score
- `val/coverage-0.9` - 90% coverage

### **TensorBoard Logs:**
```bash
tensorboard --logdir lightning_logs/nbeatsaq-adaptive-attention-mhlv
```

## 🔧 **Advanced Usage**

### **Custom Dataset**
```yaml
dataset:
  _target_: dataset.YourCustomDataModule
  # Your custom parameters
```

### **Different Backbone**
```yaml
nn:
  backbone:
    _target_: modules.NBEATSAQATTENTION  # Use attention backbone
    # Or other backbones
```

### **Learning Rate Scheduling**
```yaml
scheduler:
  _target_: schedulers.CosineAnnealing
  # Or other schedulers
```

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **CUDA Out of Memory**
   - Reduce `train_batch_size`
   - Reduce `layer_width` or `d_model`
   - Enable gradient accumulation

2. **Slow Convergence**
   - Lower learning rate
   - Increase warmup steps
   - Check data preprocessing

3. **Poor Performance**
   - Increase `num_adaptive_quantiles`
   - Adjust `adaptive_temp`
   - Try different `n_heads`

### **Debug Mode:**
```bash
python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml --fast-dev-run
```

## 📈 **Expected Results**

The adaptive attention model should show:
- **Faster convergence** than standard adaptive sampling
- **Better CRPS scores** due to attention mechanism
- **Improved coverage** across quantile ranges
- **More stable training** with importance weighting

## 🎉 **Success Indicators**

✅ Training loss decreases steadily  
✅ Validation loss follows training loss  
✅ CRPS improves over baseline  
✅ Coverage close to target levels  
✅ No gradient explosions or NaNs
