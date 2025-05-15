#!/usr/bin/env python
"""
Évalue le modèle, génère :
  - data/prediction.csv
  - metrics/scores.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--X_test", type=Path, default="data/processed/X_test_scaled.csv")
    p.add_argument("--y_test", type=Path, default="data/processed/y_test.csv")
    p.add_argument("--model", type=Path, default="models/gbr_model.pkl")
    p.add_argument("--pred_out", type=Path, default="data/prediction.csv")
    p.add_argument("--metrics_out", type=Path, default="metrics/scores.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X = pd.read_csv(args.X_test)
    y_true = pd.read_csv(args.y_test).squeeze()
    model = joblib.load(args.model)


    y_pred = model.predict(X)

    # -- Sauvegarde prédictions
    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.pred_out, index=False)

    # -- Métriques
    scores = {"mse": mean_squared_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(scores, f, indent=2)

    logging.info("✔️  Évaluation OK : r2=%.4f | mse=%.4f", scores["r2"], scores["mse"])


if __name__ == "__main__":
    main()
