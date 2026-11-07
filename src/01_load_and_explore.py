"""Step 1: load the PhysioNet/CinC 2012 ICU data (set-a) and look at missingness.

Run from the project root:  py src/01_load_and_explore.py
Needs: pandas, numpy, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data/raw/set-a")
OUTCOMES = Path("data/raw/Outcomes-a.txt")
PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures")

# Descriptors recorded once at admission (time 00:00). -1 means "not recorded".
STATIC = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]


def load_patient(path):
    """One file = one patient. Columns: Time (HH:MM), Parameter, Value."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    hm = df["Time"].str.split(":", expand=True).astype(int)
    df["minutes"] = hm[0] * 60 + hm[1]

    is_static = df["Parameter"].isin(STATIC) & (df["minutes"] == 0)
    static = df[is_static].drop_duplicates("Parameter").set_index("Parameter")["Value"]
    series = df[~is_static].copy()
    return static, series


def load_all():
    statics, series_list = [], []
    for path in sorted(DATA.glob("*.txt")):
        static, series = load_patient(path)
        rid = int(static["RecordID"])
        statics.append(static.drop("RecordID").rename(rid))
        series["RecordID"] = rid
        series_list.append(series)

    static_df = pd.DataFrame(statics).replace(-1, np.nan)
    static_df.index.name = "RecordID"
    long_df = pd.concat(series_list, ignore_index=True)
    return static_df, long_df


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    static, long_df = load_all()
    outcomes = pd.read_csv(OUTCOMES).set_index("RecordID")

    n_patients = len(static)
    print(f"Patients: {n_patients}")
    print(f"Measurements: {len(long_df):,}")
    print(f"In-hospital death rate: {outcomes['In-hospital_death'].mean():.1%}")

    # How much of each variable is actually observed?
    per_patient = long_df.groupby(["Parameter", "RecordID"]).size()
    coverage = (per_patient.groupby("Parameter").size() / n_patients).sort_values()
    median_n = per_patient.groupby("Parameter").median()
    summary = pd.DataFrame(
        {"share_of_patients_with_any_value": coverage, "median_measurements": median_n}
    ).sort_values("share_of_patients_with_any_value")
    print("\nVariable coverage (lowest first):")
    print(summary.round(2).to_string())

    # Save for the next steps
    long_df.to_csv(PROCESSED / "long.csv.gz", index=False)
    static.to_csv(PROCESSED / "static.csv")
    outcomes.to_csv(PROCESSED / "outcomes.csv")

    # Figure for the README
    ax = summary["share_of_patients_with_any_value"].plot.barh(figsize=(7, 9))
    ax.set_xlabel("Share of patients with at least one measurement")
    ax.set_title("How often is each ICU variable observed?")
    plt.tight_layout()
    plt.savefig(FIGURES / "coverage.png", dpi=150)
    print("\nSaved data/processed/*.csv and reports/figures/coverage.png")


if __name__ == "__main__":
    main()