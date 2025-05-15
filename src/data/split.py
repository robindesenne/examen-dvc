#!/usr/bin/env python
"""
Split du dataset brut en 4 fichiers :
  X_train.csv, X_test.csv, y_train.csv, y_test.csv

Exemple CLI :
python src/data/split.py \
  --raw_path data/raw/raw.csv \
  --out_dir data/processed \
  --test_size 0.2 \
  --seed 42
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, cast

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.raw_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {args.raw_path}")

    df = pd.read_csv(args.raw_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_tr, X_te, y_tr, y_te = cast(
        Tuple[DataFrame, DataFrame, Series, Series],
        train_test_split(X, y, test_size=args.test_size, random_state=args.seed),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X_tr.to_csv(args.out_dir / "X_train.csv", index=False)
    X_te.to_csv(args.out_dir / "X_test.csv", index=False)
    y_tr.to_csv(args.out_dir / "y_train.csv", index=False)
    y_te.to_csv(args.out_dir / "y_test.csv", index=False)

    logging.info("✔️  Split terminé : fichiers écrits dans %s", args.out_dir)


if __name__ == "__main__":
    main()
