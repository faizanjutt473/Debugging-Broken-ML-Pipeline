"""
ablation_study.py
-----------------
Applies the fixes ONE AT A TIME (cumulatively) so you can see how reported
performance changes after every correction.

    Step 0  Broken pipeline (all flaws)
    Step 1  + remove the leaky feature
    Step 2  + remove duplicate rows before splitting
    Step 3  + imputer and scaler inside a Pipeline (fit on training data only)
    Step 4  + class_weight='balanced'
    Step 5  + chronological (time-based) split
    Step 6  + hyperparameters tuned with TimeSeriesSplit CV, test set used once
            (= fixed_pipeline.py)

Saves results/ablation.csv
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_generator import LEAKY_COL, TARGET, TIME_COL, generate_dataset
from utils import compute_metrics

warnings.filterwarnings("ignore")

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
OUTPUT_DIR = Path("results")

STEPS = [
    ("0. Broken pipeline",              dict(leak=False, dedup=False, prep=False, balanced=False, chrono=False, cv=False)),
    ("1. + remove leaky feature",       dict(leak=True,  dedup=False, prep=False, balanced=False, chrono=False, cv=False)),
    ("2. + remove duplicates",          dict(leak=True,  dedup=True,  prep=False, balanced=False, chrono=False, cv=False)),
    ("3. + impute/scale in Pipeline",   dict(leak=True,  dedup=True,  prep=True,  balanced=False, chrono=False, cv=False)),
    ("4. + class_weight='balanced'",    dict(leak=True,  dedup=True,  prep=True,  balanced=True,  chrono=False, cv=False)),
    ("5. + chronological split",        dict(leak=True,  dedup=True,  prep=True,  balanced=True,  chrono=True,  cv=False)),
    ("6. + CV tuning (fixed pipeline)", dict(leak=True,  dedup=True,  prep=True,  balanced=True,  chrono=True,  cv=True)),
]


def run_variant(leak, dedup, prep, balanced, chrono, cv) -> dict:
    df = generate_dataset()
    if dedup:
        df = df.drop_duplicates().reset_index(drop=True)

    drop_cols = [TARGET, TIME_COL] + ([LEAKY_COL] if leak else [])
    X, y = df.drop(columns=drop_cols), df[TARGET]
    class_weight = "balanced" if balanced else None

    # Flawed preprocessing: imputer fitted on all rows BEFORE the split
    if not prep:
        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)

    if chrono:
        cut = int(len(df) * 0.8)
        X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    def make_model(c=1.0):
        lr = LogisticRegression(C=c, class_weight=class_weight, max_iter=1000)
        if not prep:
            return lr
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()), ("model", lr)])

    if cv:
        search = GridSearchCV(make_model(), {"model__C": C_GRID}, scoring="average_precision",
                              cv=TimeSeriesSplit(n_splits=5), n_jobs=-1).fit(X_tr, y_tr)
        best = search
    else:  # flawed: choose C by looking at the test set
        best, best_acc = None, -1.0
        for c in C_GRID:
            m = make_model(c).fit(X_tr, y_tr)
            acc = accuracy_score(y_te, m.predict(X_te))
            if acc > best_acc:
                best, best_acc = m, acc

    return compute_metrics(y_te, best.predict(X_te), best.predict_proba(X_te)[:, 1])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = {name: run_variant(**cfg) for name, cfg in STEPS}
    table = pd.DataFrame(rows).T.round(4)
    table.to_csv(OUTPUT_DIR / "ablation.csv")
    print(table.to_string())
    print(f"\nSaved: {OUTPUT_DIR / 'ablation.csv'}")


if __name__ == "__main__":
    main()
