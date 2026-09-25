# =============================================================================
# PlantCare AI — Google Colab Training Script
# =============================================================================
# HOW TO USE:
#   1. Open Google Colab: https://colab.research.google.com
#   2. Runtime -> Change runtime type -> T4 GPU  (MUST DO FIRST)
#   3. Create a new notebook
#   4. Paste each CELL block below into its own Colab cell
#   5. Run cells top to bottom -- do NOT skip any
# =============================================================================


# ---- CELL 1 -- Verify GPU ---------------------------------------------------
import torch
print('=== GPU Sanity Check ===')
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
    print(f'VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    raise RuntimeError(
        'No GPU detected.\n'
        'Go to Runtime > Change runtime type > Hardware accelerator > T4 GPU\n'
        'Then re-run from Cell 1.'
    )


# ---- CELL 2 -- Upload and extract your project zip -------------------------
# Zip your entire project folder (whatappp/) on your PC first:
#   Windows: right-click folder -> Send to -> Compressed (zipped) folder
# Then run this cell and choose the zip when the upload dialog appears.

from google.colab import files
import os, zipfile, glob

print('Click Choose Files and select your project zip (whatappp.zip):')
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]

print(f'Extracting {zip_name}...')
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall('/content/')

# Find extracted project root
candidates = glob.glob('/content/*/src/train.py') + glob.glob('/content/src/train.py')
if not candidates:
    raise RuntimeError('Could not find src/train.py after extraction. Check your zip structure.')

project_root = os.path.dirname(os.path.dirname(candidates[0]))
os.chdir(project_root)
print(f'Working directory set to: {os.getcwd()}')
os.system('ls')


# ---- CELL 3 -- Install dependencies ----------------------------------------
import subprocess
subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'], check=True)

import torch, torchvision, sklearn, PIL, numpy, pandas, matplotlib
print(f'torch       {torch.__version__}')
print(f'torchvision {torchvision.__version__}')
print(f'sklearn     {sklearn.__version__}')
print('All packages OK')


# ---- CELL 4a -- Download PlantVillage via Kaggle API (PREFERRED) -----------
# Get your kaggle.json from: https://www.kaggle.com/settings -> API -> Create New Token
# Upload it when prompted:

from google.colab import files as colab_files
import os

print('Upload your kaggle.json:')
kag_uploaded = colab_files.upload()
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
os.system('cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json')
os.system('pip install -q kaggle')
os.system('kaggle datasets download -d emmarex/plantdisease -p /tmp/plantvillage --unzip')
print('Download complete.')
os.system('find /tmp/plantvillage -maxdepth 3 -type d | head -20')


# ---- CELL 4b -- OR: Upload a manually downloaded PlantVillage zip (ALTERNATIVE)
# Run this ONLY if you did NOT run Cell 4a.
# Download the zip from: https://www.kaggle.com/datasets/emmarex/plantdisease
#
# from google.colab import files as colab_files
# import os
# print('Upload PlantVillage zip:')
# pv_uploaded = colab_files.upload()
# pv_zip = list(pv_uploaded.keys())[0]
# os.system(f'unzip -q "{pv_zip}" -d /tmp/plantvillage')
# print('Extraction complete.')
# os.system('find /tmp/plantvillage -maxdepth 3 -type d | head -20')


# ---- CELL 5 -- Link PlantVillage into data/raw/ ----------------------------
import os, shutil
from pathlib import Path

def find_dataset_root(search_root):
    for p in sorted(Path(search_root).rglob('*')):
        if p.is_dir():
            children = [c for c in p.iterdir() if c.is_dir()]
            if len(children) >= 10:
                print(f'Found dataset root: {p}  ({len(children)} class dirs)')
                return p
    return None

dataset_root = find_dataset_root('/tmp/plantvillage')
if dataset_root is None:
    raise RuntimeError('Dataset root not found. Check /tmp/plantvillage manually.')

raw_dir = Path('data/raw')
raw_dir.mkdir(parents=True, exist_ok=True)

class_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir())
print(f'Linking {len(class_dirs)} class directories...')

for cls_dir in class_dirs:
    target = raw_dir / cls_dir.name
    if not target.exists():
        try:
            os.symlink(cls_dir.resolve(), target)
        except Exception:
            shutil.copytree(str(cls_dir), str(target))

total_imgs = sum(1 for f in raw_dir.rglob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png'})
print(f'Total images in data/raw: {total_imgs:,}')
assert total_imgs >= 10000, f'Only {total_imgs} images -- expected ~54,000. Check dataset.'
print('Real PlantVillage data confirmed.')


# ---- CELL 6 -- Dataset sanity check and create splits ----------------------
import sys, json, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s - %(message)s')

from pathlib import Path
from src.preprocessing import scan_dataset, create_stratified_splits
import numpy as np

print('Scanning dataset...')
paths, labels, class_names = scan_dataset('data/raw')

label_arr = np.array(labels)
counts = np.bincount(label_arr)
print(f'Total images             : {len(paths):,}')
print(f'Classes                  : {len(class_names)}')
print(f'Min/Max/Mean per class   : {counts.min()} / {counts.max()} / {counts.mean():.0f}')

# Remove stale split_info from old mock run (86-image test set)
stale = Path('data/processed/split_info.json')
if stale.exists():
    with open(stale) as f:
        old = json.load(f)
    if old.get('total_images', 0) < 10000:
        stale.unlink()
        print('Removed stale split_info.json (was from 570-image mock run).')

print('Creating fresh 70/15/15 stratified splits...')
split_info = create_stratified_splits(
    paths, labels, class_names,
    train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
    seed=42,
    save_path='data/processed/split_info.json',
)
print(f'Train : {split_info["train_size"]:,}')
print(f'Val   : {split_info["val_size"]:,}')
print(f'Test  : {split_info["test_size"]:,}')
print('Splits saved.')


# ---- CELL 7 -- Train ResNet50  (expected: ~15-25 min on T4) ----------------
from src.train import train_model

print('TRAINING: ResNet50')
resnet_result = train_model(
    model_name='resnet50',
    epochs=15,
    batch_size=64,
    lr=0.0005,
    weight_decay=0.0001,
    seed=42,
)
print(f'ResNet50 done  --  Best Val F1: {resnet_result["best_metrics"]["val_f1"]:.4f}  '
      f'({resnet_result["total_time_sec"]/60:.1f} min)')


# ---- CELL 8 -- Train EfficientNet-B0  (expected: ~10-18 min on T4) ---------
from src.train import train_model

print('TRAINING: EfficientNet-B0')
effnet_result = train_model(
    model_name='efficientnet_b0',
    epochs=15,
    batch_size=64,
    lr=0.0005,
    weight_decay=0.0001,
    seed=42,
)
print(f'EfficientNet-B0 done  --  Best Val F1: {effnet_result["best_metrics"]["val_f1"]:.4f}  '
      f'({effnet_result["total_time_sec"]/60:.1f} min)')


# ---- CELL 9 -- Train ViT-B/16  (expected: ~25-40 min on T4) ----------------
from src.train import train_model

print('TRAINING: ViT-B/16')
vit_result = train_model(
    model_name='vit_b_16',
    epochs=15,
    batch_size=32,
    lr=0.0001,
    weight_decay=0.00005,
    seed=42,
)
print(f'ViT-B/16 done  --  Best Val F1: {vit_result["best_metrics"]["val_f1"]:.4f}  '
      f'({vit_result["total_time_sec"]/60:.1f} min)')


# ---- CELL 10 -- Full Evaluation (Clean + 9 Robustness Variants) -------------
from src.evaluate import run_full_evaluation

print('Running full evaluation suite (clean + 9 robustness variants)...')
eval_results = run_full_evaluation(
    model_names=['resnet50', 'efficientnet_b0', 'vit_b_16'],
    seed=42,
)

print('\n=== CLEAN TEST SET RESULTS ===')
for model_name, metrics in eval_results['clean_evaluations'].items():
    print(f'  {model_name:22s}  Acc={metrics["accuracy"]:.4f}  '
          f'F1={metrics["f1_macro"]:.4f}  '
          f'Latency={metrics["mean_latency_ms"]:.1f}ms  '
          f'ECE={metrics["expected_calibration_error"]:.4f}')


# ---- CELL 11 -- Print full results table (COPY THIS ENTIRE OUTPUT) ----------
import pandas as pd

df = pd.read_csv('results/metrics/comparison_table.csv')
clean_df = df[df['Condition'] == 'Clean (Baseline)']

print('PLANTCARE AI -- FINAL RESULTS SUMMARY')
print('=' * 70)
print(f'Dataset  : PlantVillage  {split_info["total_images"]:,} images  {split_info["num_classes"]} classes')
print(f'Split    : {split_info["train_size"]:,} train / {split_info["val_size"]:,} val / {split_info["test_size"]:,} test')
print(f'Device   : {torch.cuda.get_device_name(0)}')
print(f'PyTorch  : {torch.__version__}')
print()
print(f'{"Model":<22} {"Accuracy":>10} {"Macro F1":>10} {"Precision":>10} {"Recall":>10} {"Latency ms":>11}')
print('-' * 76)
for model_name, m in eval_results['clean_evaluations'].items():
    print(f'{model_name:<22} {m["accuracy"]:>10.4f} {m["f1_macro"]:>10.4f} '
          f'{m["precision_macro"]:>10.4f} {m["recall_macro"]:>10.4f} {m["mean_latency_ms"]:>11.1f}')

print()
print('Training history:')
for name, log in [('resnet50', resnet_result), ('efficientnet_b0', effnet_result), ('vit_b_16', vit_result)]:
    print(f'  {name}: {log["epochs_trained"]} epochs | {log["total_time_sec"]/60:.1f} min | '
          f'best val F1 = {log["best_metrics"]["val_f1"]:.4f}')

print()
print('Robustness (accuracy drop vs clean baseline):')
clean_acc = dict(zip(clean_df['Model'], clean_df['Accuracy']))
rob_df = df[df['Condition'] != 'Clean (Baseline)'].copy()
rob_df['Acc Drop'] = rob_df.apply(
    lambda r: round(clean_acc.get(r['Model'], 0) - r['Accuracy'], 4), axis=1
)
pivot = rob_df.pivot_table(index='Condition', columns='Model', values='Acc Drop')
print(pivot.to_string())


# ---- CELL 12 -- Download all results + checkpoints -------------------------
import zipfile, os
from google.colab import files as colab_files
from pathlib import Path

output_zip = '/tmp/plantcare_results.zip'
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in Path('results/metrics').glob('*'):
        if f.is_file():
            zf.write(f, f'results/metrics/{f.name}')
            print(f'  + {f.name}')
    for f in Path('results/graphs').glob('*.png'):
        zf.write(f, f'results/graphs/{f.name}')
        print(f'  + graphs/{f.name}')
    for m in ['resnet50', 'efficientnet_b0', 'vit_b_16']:
        ckpt = Path(f'models/{m}/best_model.pth')
        if ckpt.exists():
            zf.write(ckpt, f'models/{m}/best_model.pth')
            print(f'  + models/{m}/best_model.pth  ({ckpt.stat().st_size/1e6:.0f} MB)')

print(f'Zip size: {os.path.getsize(output_zip)/1e6:.0f} MB')
print('Downloading...')
colab_files.download(output_zip)
