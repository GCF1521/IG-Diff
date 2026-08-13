#!/bin/bash
# Download AV2 pkl data from HF-Mirror and convert to av2_dataset.py compatible format.
#
# Usage:
#   bash download_and_convert_av2.sh
#
# Prerequisites:
#   pip install huggingface_hub
#
# This downloads ~499 pkl files (~20MB) from saeedrmd/trajectory-prediction-argoverse2
# on HF-Mirror, then converts them using convert_pkl_to_av2.py.

set -e

export HF_ENDPOINT=https://hf-mirror.com

PKL_DIR="av2_pkl_raw"
CONVERTED_DIR="av2_converted"

echo "=== Step 1: Download pkl data from HF-Mirror ==="
echo "Endpoint: $HF_ENDPOINT"
echo "Repo: saeedrmd/trajectory-prediction-argoverse2"
echo ""

pip install -q huggingface_hub

python3 -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
path = snapshot_download(
    repo_id='saeedrmd/trajectory-prediction-argoverse2',
    repo_type='dataset',
    local_dir='$PKL_DIR',
    resume_download=True,
)
print(f'Downloaded to: {path}')
"

echo ""
echo "=== Step 2: Convert pkl to av2 format ==="
python3 convert_pkl_to_av2.py -i "$PKL_DIR" -o "$CONVERTED_DIR"

echo ""
echo "=== Step 3: Update config ==="
echo "Set in config/default.yaml:"
echo "  data:"
echo "    train_dir: $CONVERTED_DIR/"
echo "    val_dir: $CONVERTED_DIR/"
echo "    map_dir: null"

echo ""
echo "=== Done! ==="
echo "Converted data ready at: $CONVERTED_DIR/"
echo "You can now run: python -m src.train --config config/default.yaml"