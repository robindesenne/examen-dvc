#!/usr/bin/env python
"""
Hyper-param tuning avec GridSearchCV sur GradientBoostingRegressor
→ best_params.pkl + trace du meilleur score R².
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV

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
    y = pd.read_csv(args.y_train).squeeze()  # Series

    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 4],
    }

    gs = GridSearchCV(
        model, param_grid, scoring=make_scorer(r2_score), cv=5, n_jobs=-1, verbose=1
    )
    gs.fit(X, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_params_, args.out)

    logging.info(
        "✔️  GridSearch terminé : meilleur R² = %.4f | params : %s",
        gs.best_score_,
        gs.best_params_,
    )


if __name__ == "__main__":
    main()
