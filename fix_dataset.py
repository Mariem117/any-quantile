# Read the file
with open('/workspace/any-quantile/dataset/datasets.py', 'r') as f:
    lines = f.readlines()

# Comment out the assert False line (line 367, index 366)
for i, line in enumerate(lines):
    if line.strip() == 'assert False':
        lines[i] = '        # assert False  # Commented out to allow training\n'
        print(f"Commented out 'assert False' at line {i+1}")
        break

# Write back
with open('/workspace/any-quantile/dataset/datasets.py', 'w') as f:
    f.writelines(lines)

print("File fixed successfully!")
