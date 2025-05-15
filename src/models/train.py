#!/usr/bin/env python
"""
Entraîne le XGBRegressor final avec les meilleurs hyper-paramètres.
→ models/gbr_model.pkl  (on garde le même nom pour ne pas casser la pipeline)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--X_train", type=Path, default="data/processed/X_train_scaled.csv")
    p.add_argument("--y_train", type=Path, default="data/processed/y_train.csv")
    p.add_argument("--params", type=Path, default="models/data/best_params.pkl")
    p.add_argument("--out", type=Path, default="models/models/gbr_model.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X = pd.read_csv(args.X_train)
    y = pd.read_csv(args.y_train).squeeze()
    best_params = joblib.load(args.params)

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        **best_params,
    )
    model.fit(X, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)

    logging.info("✔️  Modèle XGB entraîné et sauvegardé : %s", args.out)


if __name__ == "__main__":
    main()
