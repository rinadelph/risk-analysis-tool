from pathlib import Path

# Create necessary folders
folders = [
    'raw',
    'clean',
    'working',
    'analysis/single',
    'analysis/batch'
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True) 