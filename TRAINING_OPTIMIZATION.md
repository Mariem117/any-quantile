# Training Speed Optimization Analysis

## Current Configuration Analysis

### Major Bottlenecks Identified:

1. **Large Model Architecture**
   - `num_blocks: 15` - Very high, each block processes full sequence
   - `layer_width: 512` - Large hidden dimension
   - `n_heads: 8` - Attention heads add computational cost
   - **Total parameters: ~30.6M**

2. **Inefficient Data Loading**
   - `eval_step: 24` - Processing every 24th step creates large batches
   - `train_step: 1` - Very dense sampling
   - `num_workers: 4` - Could be increased for better I/O

3. **Suboptimal Training Settings**
   - `precision: 32` - Could use 16-bit for faster training
   - `max_epochs: 15` - Long training with current setup
   - `lr: 0.0001` - Conservative learning rate

## Optimization Strategies

### 1. Reduce Model Size (High Impact)
```yaml
model:
  nn:
    backbone:
      num_blocks: 8        # Reduce from 15 (-47% parameters)
      layer_width: 256      # Reduce from 512 (-50% parameters)
      # New total: ~8M parameters (-75% reduction)
```

### 2. Optimize Data Pipeline (Medium Impact)
```yaml
dataset:
  train_step: 24          # Reduce sampling frequency
  eval_step: 48           # Reduce validation frequency
  num_workers: 8           # Increase for better I/O
  train_batch_size: 256    # Increase if GPU memory allows
  eval_batch_size: 256
```

### 3. Use Mixed Precision (High Impact)
```yaml
trainer:
  precision: 16          # Use AMP for 2x speedup
```

### 4. Optimize Training Schedule (Medium Impact)
```yaml
model:
  optimizer:
    lr: 0.001            # Increase learning rate
trainer:
  max_epochs: 10         # Reduce epochs with faster convergence
  accumulate_grad_batches: 2  # Effective larger batch size
```

## Fast Configuration (Ready to Use)

Create `config/nbeatsaq-attention-mhlv-fast-optimized.yaml`:

```yaml
logging:
  path: ./lightning_logs/optimized
  name: nbeats-aq-optimized-seed=${random.seed}

dataset:
  _target_: dataset.ElectricityUnivariateDataModule
  name: MHLV
  num_workers: 8
  persistent_workers: true
  train_batch_size: 256
  eval_batch_size: 256
  history_length: 168
  horizon_length: 48
  split_boundaries: ["2006-01-01", "2017-01-01", "2018-01-01", "2019-01-01"]
  fillna: ffill
  train_step: 24          # Reduced from 1
  eval_step: 48           # Increased from 24

random:
  seed: [0, 1, 2]

trainer:
  max_epochs: 10          # Reduced from 15
  check_val_every_n_epoch: 1
  log_every_n_steps: 50    # More frequent logging
  devices: 1
  accelerator: gpu
  precision: 16           # Mixed precision
  accumulate_grad_batches: 2  # Effective batch size 512
  gradient_clip_val: 1.0

checkpoint:
  resume_ckpt: null
  save_top_k: 2

model:
  _target_: model.AnyQuantileForecaster
  input_horizon_len: ${dataset.history_length}
  loss:
    _target_: losses.MQNLoss
  max_norm: true
  q_sampling: random_in_batch
  q_distribution: uniform
  metric_space: original
  metric_clip: 10000

  nn:
    backbone:
      _target_: modules.NBEATSAQATTENTION
      dropout: 0.1
      layer_width: 256      # Reduced from 512
      num_layers: 3
      num_blocks: 8        # Reduced from 15
      share: false
      size_in: ${dataset.history_length}
      size_out: ${dataset.horizon_length}
      n_heads: 4          # Reduced from 8

  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001            # Increased from 0.0001
    weight_decay: 1e-4

  scheduler:
    _target_: schedulers.InverseSquareRoot
    warmup_updates: 200     # Reduced from 400
    warmup_end_lr: ${model.optimizer.lr}
```

## Expected Speed Improvements

### Configuration Changes Impact:
1. **Model Size**: 75% reduction (30.6M → ~8M parameters)
2. **Mixed Precision**: 2x speedup on RTX 3050
3. **Batch Size**: 2x larger effective batch (256×2 = 512)
4. **Data Loading**: 2x faster with 8 workers
5. **Training Steps**: 24x fewer due to train_step change

### Overall Expected Speedup: **8-12x faster**

## Alternative: Use Simpler Model

For even faster training, switch to simpler backbone:

```yaml
model:
  nn:
    backbone:
      _target_: modules.NBEATSAQCAT  # No attention, much faster
      layer_width: 256
      num_blocks: 6
```

## Monitoring Progress

Add these to track optimization:
```yaml
trainer:
  callbacks:
    - class_path: pytorch_lightning.callbacks.LearningRateMonitor
      init_args:
        logging_interval: step
    - class_path: pytorch_lightning.callbacks.ModelSummary
      init_args:
        max_depth: 2
```

## Quick Test Command

```bash
docker run --gpus all --rm -it --shm-size=8g \
  -e TORCH_FLOAT32_MATMUL_PRECISION=medium \
  -v "${PWD}:/workspace/any-quantile" \
  -w /workspace/any-quantile any_quantile:latest \
  python run.py --config=config/nbeatsaq-attention-mhlv-fast-optimized.yaml
```

This should be **8-12x faster** than the current configuration.
