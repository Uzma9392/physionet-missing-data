"""Step 13: how does missing data affect prediction, and which methods cope best?

For each missingness rate we delete measurements from the raw (long) data, rebuild the
per-patient table, and score every method with 5-fold cross-validation (AUROC).

Two ways of deleting:
  random  - every measurement has the same chance of being deleted (MCAR)
  biased  - extreme values (far from the variable's typical range, i.e. often the
            sickest patients) are more likely to be deleted (a simple MNAR pattern)

Run from the project root:
    python src\\04_missingness_experiment.py            (full run, 3 repeats)
    python src\\04_missingness_experiment.py --repeats 1 (faster test)
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGURES = REPORTS / "figures"
SEED = 42
RATES = [0.1, 0.3, 0.5, 0.7]
MECHANISMS = ["random", "biased"]


def solve_intercept(score, target):
    """Find a so that mean(sigmoid(a + score)) equals the target deletion rate."""
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + score)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def delete_measurements(long_df, rate, mechanism, rng):
    """Return a copy of long_df with about `rate` of the measurements removed."""
    if mechanism == "random":
        keep = rng.random(len(long_df)) >= rate
    else:
        grp = long_df.groupby("Parameter")["Value"]
        z = ((long_df["Value"] - grp.transform("mean")) / grp.transform("std")).abs()
        z = z.fillna(0).to_numpy()
        p_drop = 1 / (1 + np.exp(-(solve_intercept(1.5 * z, rate) + 1.5 * z)))
        keep = rng.random(len(long_df)) >= p_drop
    return long_df[keep]


def make_models(with_mice):
    models = {
        "Median fill + logistic": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "Median fill + missing flags + logistic": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "Gradient boosting (native NaN)": HistGradientBoostingClassifier(random_state=SEED),
    }
    if with_mice:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        models["Iterative imputation + logistic"] = make_pipeline(
            IterativeImputer(max_iter=3, random_state=SEED),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )
    return models


def prepare_xy(long_df, static, outcomes):
    X = build_features(long_df, static)
    y = outcomes.loc[X.index, "In-hospital_death"]
    X = pd.get_dummies(X, columns=["ICUType"], dtype=float)
    return X, y


def plot(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, mech in zip(axes, MECHANISMS):
        for method, sub in results.groupby("method"):
            base = sub[sub["mechanism"] == "none"]
            part = pd.concat([base, sub[sub["mechanism"] == mech]])
            g = part.groupby("rate")["auroc"].agg(["mean", "std"]).fillna(0)
            ax.errorbar(g.index * 100, g["mean"], yerr=g["std"], marker="o", capsize=3, label=method)
        ax.set_title(f"Deletion pattern: {mech}")
        ax.set_xlabel("Measurements deleted (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC (5-fold cross-validation)")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Prediction of in-hospital death as ICU measurements are deleted")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3, help="random repeats per condition")
    parser.add_argument("--mice", action="store_true", help="also run iterative imputation (slow)")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    long_df = pd.read_csv(PROCESSED / "long.csv.gz")
    long_df = long_df[long_df["Value"] >= 0]
    static = pd.read_csv(PROCESSED / "static.csv", index_col="RecordID")
    outcomes = pd.read_csv(PROCESSED / "outcomes.csv", index_col="RecordID")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = make_models(args.mice)

    conditions = [("none", 0.0, 0)]
    for mech in MECHANISMS:
        for rate in RATES:
            for rep in range(args.repeats):
                conditions.append((mech, rate, rep))

    rows, start = [], time.time()
    for i, (mech, rate, rep) in enumerate(conditions, 1):
        if mech == "none":
            degraded = long_df
        else:
            rng = np.random.default_rng(1000 * rep + int(rate * 100) + (7 if mech == "biased" else 0))
            degraded = delete_measurements(long_df, rate, mech, rng)
        X, y = prepare_xy(degraded, static, outcomes)
        missing = X[[c for c in X.columns if not c.endswith("_count")]].isna().mean().mean()
        for name, model in models.items():
            auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
            rows.append(
                {"mechanism": mech, "rate": rate, "repeat": rep, "method": name,
                 "auroc": auc, "missing_cells": missing}
            )
        last = [r for r in rows[-len(models):]]
        print(
            f"[{i}/{len(conditions)}] {mech:6s} {rate:.0%} rep {rep}  "
            + "  ".join(f"{r['auroc']:.3f}" for r in last)
            + f"   ({(time.time() - start) / 60:.1f} min)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(REPORTS / "missingness_results.csv", index=False)

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["mechanism", "rate", "method"])["auroc"].agg(["mean", "std"]).round(3)
    )
    print("\nAUROC by condition (mean and std over repeats):")
    print(summary.to_string())
    plot(results, FIGURES / "auroc_vs_missingness.png")
    print("\nSaved reports/missingness_results.csv and reports/figures/auroc_vs_missingness.png")


if __name__ == "__main__":
    main()
    