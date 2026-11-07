"""Turn the irregular long-format ICU data into one row per patient.

Why this is a separate module: later we delete measurements on purpose and rebuild
the table, so the same function is reused by the missing-data experiment.
"""
import numpy as np
import pandas as pd

STATS = ["mean", "min", "max", "last", "count"]


def build_features(long_df, static_df):
    """long_df: RecordID, Parameter, Value, minutes (one row per measurement).
    static_df: one row per patient (Age, Gender, Height, ICUType, Weight), index RecordID.
    Returns one row per patient: static columns + <variable>_<stat> columns.
    """
    df = long_df.copy()
    df = df[df["Value"] >= 0]  # -1 means "not recorded" in this dataset
    df = df.sort_values(["RecordID", "Parameter", "minutes"])

    agg = df.groupby(["RecordID", "Parameter"])["Value"].agg(
        mean="mean", min="min", max="max", last="last", count="count"
    )
    wide = agg.unstack("Parameter")
    wide.columns = [f"{param}_{stat}" for stat, param in wide.columns]

    # Patients with no measurements at all must still appear (as all-missing rows)
    wide = wide.reindex(static_df.index)

    # A count of 0 means "never measured": count columns are 0, not missing
    count_cols = [c for c in wide.columns if c.endswith("_count")]
    wide[count_cols] = wide[count_cols].fillna(0)

    return static_df.join(wide)