with open('/workspace/any-quantile/run.py', 'r') as f:
    content = f.read()

# Define the correct imports
new_imports = '''import argparse
import yaml
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.checkpointing import get_checkpoint_path
from utils.model_factory import instantiate
'''

# Find where imports end (first empty line or comment after imports)
lines = content.split('\n')
import_end = 0
for i, line in enumerate(lines):
    if line.strip() == '' and i > 5:  # Found empty line after imports
        import_end = i
        break
    if line.startswith('# NumPy'):
        import_end = i
        break

# Rebuild the file
new_content = new_imports + '\n' + '\n'.join(lines[import_end:])

with open('/workspace/any-quantile/run.py', 'w') as f:
    f.write(new_content)

print("Fixed run.py!")
