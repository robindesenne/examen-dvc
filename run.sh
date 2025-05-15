#!/usr/bin/env bash
set -euo pipefail

echo "⏱️  Total pipeline:"
time (
    # 1. SPLIT
    python src/data/split.py --raw_path data/raw/raw.csv --out_dir data/processed
    # 2. NORMALISE
    python src/data/normalize.py
    # 3. GRID SEARCH
    python src/models/grid_search.py
    # 4. TRAIN
    python src/models/train.py
    # 5. EVALUATE
    python src/models/evaluate.py
)
