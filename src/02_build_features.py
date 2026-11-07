"""Step 11: build one row per patient and attach the outcome.

Run from the project root:  python src\\02_build_features.py
"""
from pathlib import Path

import pandas as pd

from features import build_features

PROCESSED = Path("data/processed")


def main():
    long_df = pd.read_csv(PROCESSED / "long.csv.gz")
    static = pd.read_csv(PROCESSED / "static.csv", index_col="RecordID")
    outcomes = pd.read_csv(PROCESSED / "outcomes.csv", index_col="RecordID")

    X = build_features(long_df, static)

    # Only the outcome we want to predict. SAPS-I, SOFA, Length_of_stay and Survival
    # are NOT used as inputs: they are scores or results that would leak the answer.
    y = outcomes.loc[X.index, "In-hospital_death"].rename("death")

    table = X.join(y)
    table.to_csv(PROCESSED / "features.csv")

    value_cols = [c for c in X.columns if not c.endswith("_count")]
    missing_share = X[value_cols].isna().mean().mean()

    print(f"Rows (patients): {table.shape[0]}")
    print(f"Columns: {table.shape[1]} ({X.shape[1]} inputs + 1 target)")
    print(f"Deaths: {int(y.sum())} of {len(y)} ({y.mean():.1%})")
    print(f"Share of input cells that are missing: {missing_share:.1%}")
    print("\nFirst 5 rows, a few columns:")
    show = [c for c in ["Age", "ICUType", "HR_mean", "HR_count", "TroponinI_mean", "death"] if c in table.columns]
    print(table[show].head().to_string())
    print("\nSaved data/processed/features.csv")


if __name__ == "__main__":
    main()
    