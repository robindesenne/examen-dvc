#!/usr/bin/env python
"""
Hyper-param tuning avec GridSearchCV sur XGBRegressor.
Sortie : models/artefacts/best_params.pkl
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--X_train", type=Path, default="data/processed/X_train_scaled.csv")
    p.add_argument("--y_train", type=Path, default="data/processed/y_train.csv")
    p.add_argument("--out", type=Path, default="models/artefacts/best_params.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X = pd.read_csv(args.X_train)
    y = pd.read_csv(args.y_train).squeeze()

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",  # rapide sur CPU
    )

    param_grid = {
        "n_estimators": [300, 500, 800],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "gamma": [0, 1],
    }

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(r2_score),
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_params_, args.out)

    logging.info(
        "✔️  GridSearch XGB terminé : meilleur R²=%.4f | params=%s",
        gs.best_score_,
        gs.best_params_,
    )


if __name__ == "__main__":
    main()
