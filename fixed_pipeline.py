"""
fixed_pipeline.py
-----------------
CORRECTED pipeline. Each fix maps to a flaw in broken_pipeline.py.

 1. Leaky feature detected (correlation check) and removed
 2. Duplicates removed BEFORE splitting
 3. Imputer + scaler live inside a Pipeline, fitted on training folds only
 4. Chronological (time-based) split instead of a random one
 5. Cross-validation respects time order (TimeSeriesSplit)
 6. StandardScaler applied to all features
 7. class_weight='balanced' handles the rare positive class
 8. Hyperparameters chosen by CV on TRAIN only; test set used once
 9. Metrics: precision, recall, F1, ROC-AUC, PR-AUC (not just accuracy)
"""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_generator import TARGET, TIME_COL, generate_dataset
from utils import compute_metrics, find_leaky_features, print_metrics, report_data_quality

TEST_FRACTION = 0.20


def run_fixed_pipeline(verbose: bool = True) -> dict:
    df = generate_dataset()
    if verbose:
        report_data_quality(df, TARGET)

    # FIX 2: remove duplicates before any split
    df = df.drop_duplicates().reset_index(drop=True)

    # FIX 4: sort by time, then split chronologically (train = past, test = future)
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df.drop(columns=[TARGET, TIME_COL]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET, TIME_COL]), test_df[TARGET]

    # FIX 1: detect and drop leaky features (check on TRAIN data only)
    leaky = find_leaky_features(X_train.fillna(X_train.median()), y_train, threshold=0.5)
    if verbose:
        print(f"\nLeaky features detected: {leaky}")
    X_train, X_test = X_train.drop(columns=leaky), X_test.drop(columns=leaky)

    # FIX 3 + 6 + 7: preprocessing and model in ONE pipeline
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )

    # FIX 5 + 8: tune with time-aware CV on the training set; metric suits imbalance
    search = GridSearchCV(
        pipe,
        param_grid={"model__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=5),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    # Test set is touched exactly once
    y_pred = search.predict(X_test)
    y_proba = search.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    if verbose:
        print(f"Best C (from CV)       : {search.best_params_['model__C']}")
        print_metrics("FIXED PIPELINE - full metrics", metrics, y_test, y_pred)
    return metrics


if __name__ == "__main__":
    run_fixed_pipeline()
