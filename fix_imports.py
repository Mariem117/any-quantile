with open('/workspace/any-quantile/run.py', 'r') as f:
    lines = f.readlines()

# Find the line with the TensorBoardLogger import and add pl import before it
for i, line in enumerate(lines):
    if 'from pytorch_lightning.loggers import TensorBoardLogger' in line:
        lines.insert(i, 'import pytorch_lightning as pl\n')
        break

with open('/workspace/any-quantile/run.py', 'w') as f:
    f.writelines(lines)

print("Fixed!")
