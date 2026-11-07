"""Step 12: baseline models on the FULL data (no values removed yet).

These scores are the reference point: later we delete measurements on purpose and
see how far performance falls below this.

Run from the project root:  python src\\03_baseline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
SEED = 42


def load_xy():
    table = pd.read_csv(PROCESSED / "features.csv", index_col="RecordID")
    y = table["death"]
    X = table.drop(columns="death")
    # ICU type is a category (1 to 4), not a quantity: turn it into yes/no columns
    X = pd.get_dummies(X, columns=["ICUType"], dtype=float)
    return X, y


def main():
    REPORTS.mkdir(exist_ok=True)
    X, y = load_xy()
    print(f"{X.shape[0]} patients, {X.shape[1]} input columns, death rate {y.mean():.1%}")

    models = {
        # Needs complete numbers, so missing values are filled with the column median
        "Logistic regression (median fill)": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        # Handles missing values by itself
        "Gradient boosting (native NaN)": HistGradientBoostingClassifier(random_state=SEED),
    }

    # 5-fold cross-validation: train on 4/5 of patients, test on the other 1/5, five times
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        res = cross_validate(
            model, X, y, cv=cv, scoring=["roc_auc", "average_precision"]
        )
        rows.append(
            {
                "model": name,
                "AUROC_mean": res["test_roc_auc"].mean(),
                "AUROC_std": res["test_roc_auc"].std(),
                "AUPRC_mean": res["test_average_precision"].mean(),
                "AUPRC_std": res["test_average_precision"].std(),
            }
        )
        print(
            f"{name}: AUROC {rows[-1]['AUROC_mean']:.3f} +/- {rows[-1]['AUROC_std']:.3f}, "
            f"AUPRC {rows[-1]['AUPRC_mean']:.3f} +/- {rows[-1]['AUPRC_std']:.3f}"
        )

    print(f"\n(A model that guesses at random would score AUROC 0.50 and AUPRC {y.mean():.2f}.)")
    pd.DataFrame(rows).round(4).to_csv(REPORTS / "baseline_results.csv", index=False)
    print("Saved reports/baseline_results.csv")


if __name__ == "__main__":
    main()