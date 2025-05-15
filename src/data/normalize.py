#!/usr/bin/env python
"""
Normalisation des features via un ColumnTransformer :
  - StandardScaler par défaut
  - RobustScaler pour les colonnes contenant 'flow' ou 'density'

Sorties :
  X_train_scaled.csv, X_test_scaled.csv
  + artefacts/pipe_scaler.pkl
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def enrich_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme la colonne 'date' (s’il y en a une) en features horaires."""
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = df["date"].dt.hour
        df["dayofweek"] = df["date"].dt.dayofweek
        df.drop(columns="date", inplace=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--scaler_out", type=Path, default=Path("models/artefacts/pipe_scaler.pkl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X_tr = pd.read_csv(args.in_dir / "X_train.csv")
    X_te = pd.read_csv(args.in_dir / "X_test.csv")

    X_tr = enrich_datetime(X_tr)
    X_te = enrich_datetime(X_te)

    numeric_cols = X_tr.columns.tolist()
    robust_cols = [c for c in numeric_cols if "flow" in c or "density" in c]
    standard_cols = [c for c in numeric_cols if c not in robust_cols]

    if not robust_cols:
        logging.info("Aucune colonne 'flow'/'density' trouvée : 100%% StandardScaler.")

    preprocessor = ColumnTransformer(
        [("std", StandardScaler(), standard_cols), ("robust", RobustScaler(), robust_cols)],
        remainder="drop",
    )
    pipe = Pipeline([("scaler", preprocessor)])

    X_tr_scaled = pipe.fit_transform(X_tr)
    X_te_scaled = pipe.transform(X_te)

    # reconstruction DataFrame dans le même ordre que numeric_cols
    X_tr_df = pd.DataFrame(X_tr_scaled, columns=standard_cols + robust_cols)
    X_te_df = pd.DataFrame(X_te_scaled, columns=standard_cols + robust_cols)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X_tr_df.to_csv(args.out_dir / "X_train_scaled.csv", index=False)
    X_te_df.to_csv(args.out_dir / "X_test_scaled.csv", index=False)

    args.scaler_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.scaler_out)

    logging.info("✔️  Normalisation terminée, artefact sauvegardé : %s", args.scaler_out)


if __name__ == "__main__":
    main()
