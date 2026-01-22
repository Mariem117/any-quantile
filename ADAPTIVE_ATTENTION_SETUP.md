# 🚀 Adaptive Attention Model - Complete Setup Guide

## 📁 **Files Created**

### **Core Model Files:**
- ✅ `modules/adaptive_attention.py` - Adaptive attention modules
- ✅ `model/models.py` - Updated with `AnyQuantileForecasterAdaptiveAttention`
- ✅ `modules/__init__.py` - Updated exports
- ✅ `model/__init__.py` - Updated exports

### **Configuration & Training:**
- ✅ `config/nbeatsaq-adaptive-attention-mhlv.yaml` - Main configuration file
- ✅ `train_adaptive_attention.py` - Training script
- ✅ `test_config.py` - Configuration validation script
- ✅ `test_adaptive_attention_simple.py` - Unit tests
- ✅ `ADAPTIVE_ATTENTION_GUIDE.md` - Comprehensive usage guide

## 🎯 **Quick Start Commands**

### **1. Test Configuration:**
```bash
python test_config.py
```

### **2. Run Unit Tests:**
```bash
python test_adaptive_attention_simple.py
```

### **3. Fast Development Run:**
```bash
python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml --fast-dev-run
```

### **4. Full Training:**
```bash
python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml
```

### **5. Resume Training:**
```bash
python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml --resume lightning_logs/nbeatsaq-adaptive-attention-mhlv/checkpoints/last.ckpt
```

## ⚙️ **Key Configuration Parameters**

### **Adaptive Sampling:**
- `num_adaptive_quantiles: 3` - Adaptive quantiles per batch
- `num_bins: 100` - Importance tracking bins
- `momentum: 0.99` - Update momentum
- `temperature: 1.0` - Sampling temperature

### **Adaptive Attention:**
- `d_model: 256` - Attention dimension
- `n_heads: 8` - Multi-head attention
- `num_blocks: 2` - Attention blocks
- `adaptive_temp: 1.0` - Modulation temperature

### **Training:**
- `max_epochs: 20` - Training epochs
- `train_batch_size: 512` - Batch size
- `lr: 0.0005` - Learning rate
- `warmup_updates: 800` - Warmup steps

## 🏗️ **Model Architecture**

```
Input History → NBEATS Backbone → Adaptive Attention Blocks → Enhanced Output → Forecast
                      ↓
              Adaptive Sampling → Importance Weights → Attention Modulation
```

### **Components:**
1. **NBEATS Backbone**: Base time series forecasting
2. **Adaptive Sampling**: Dynamic quantile selection based on loss
3. **Adaptive Attention**: Importance-weighted multi-head attention
4. **Residual Connections**: Stable training with skip connections

## 📊 **Expected Performance**

### **Advantages over Baseline:**
- **Better CRPS**: Improved quantile accuracy
- **Faster Convergence**: Adaptive focus on difficult regions
- **Stable Training**: Importance-weighted attention
- **Dynamic Learning**: Adapts to quantile-specific challenges

### **Monitoring Metrics:**
- `train/loss` - Training loss trend
- `val/loss` - Validation performance
- `val/crps` - Quantile forecasting quality
- `val/coverage-0.9` - Prediction interval coverage

## 🔧 **Customization Options**

### **Different Datasets:**
```yaml
dataset:
  _target_: dataset.YourCustomDataModule
  name: YOUR_DATASET
```

### **Model Variations:**
```yaml
# More attention blocks
adaptive_attention:
  num_blocks: 4

# Larger attention dimension
adaptive_attention:
  d_model: 512

# More adaptive quantiles
adaptive_sampling:
  num_adaptive_quantiles: 5
```

### **Training Optimizations:**
```yaml
# For larger models
trainer:
  accumulate_grad_batches: 4
  gradient_clip_val: 2.0

# For faster training
trainer:
  max_epochs: 10
  devices: 2
```

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **Memory Issues:**
   - Reduce `train_batch_size` to 256
   - Reduce `layer_width` to 256
   - Enable `accumulate_grad_batches`

2. **Slow Convergence:**
   - Lower `lr` to 0.0001
   - Increase `warmup_updates` to 1200
   - Check data preprocessing

3. **Poor Performance:**
   - Increase `num_adaptive_quantiles` to 5
   - Adjust `adaptive_temp` to 1.5
   - Try `n_heads: 12`

### **Debug Commands:**
```bash
# Test configuration only
python test_config.py

# Test model components
python test_adaptive_attention_simple.py

# Quick training test
python train_adaptive_attention.py --config config/nbeatsaq-adaptive-attention-mhlv.yaml --fast-dev-run
```

## 📈 **Results Interpretation**

### **Success Indicators:**
✅ Training loss decreases steadily  
✅ Validation loss follows training  
✅ CRPS improves over baseline  
✅ Coverage close to target (90%)  
✅ No NaN or gradient explosions  

### **TensorBoard Monitoring:**
```bash
tensorboard --logdir lightning_logs/nbeatsaq-adaptive-attention-mhlv
```

### **Key Plots to Watch:**
- Loss curves (train/val)
- Learning rate schedule
- Attention weight distributions
- Quantile importance evolution

## 🎉 **Next Steps**

1. **Run baseline comparison** with standard adaptive model
2. **Hyperparameter tuning** for your specific dataset
3. **Ablation studies** to test component contributions
4. **Production deployment** with model serving

## 📚 **References**

- Adaptive Sampling: Dynamic quantile selection based on loss feedback
- Multi-Head Attention: Parallel attention mechanisms
- NBEATS: Neural basis expansion analysis for time series
- Quantile Regression: Predicting conditional quantiles

---

**🎯 Ready to train your adaptive attention model!**
