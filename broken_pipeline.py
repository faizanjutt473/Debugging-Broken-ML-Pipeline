"""
broken_pipeline.py
------------------
ORIGINAL (BROKEN) pipeline. Every flaw is marked with a "FLAW n" comment.
Do NOT use this code in real projects. It exists to be debugged.

Flaws
-----
 1. Target leakage            - 'cancellation_request_logged' is only known after churn
 2. Duplicate rows            - copies end up in both train and test
 3. Preprocessing leakage     - imputer fitted on ALL data before the split
 4. Data-ordering bug         - time-ordered data split randomly (future leaks into past)
 5. No stratification         - rare class may be unevenly split
 6. Unscaled features         - annual_income (~60,000) vs support_calls (~2) in Logistic Regression
 7. Class imbalance ignored   - ~6% positives, no class_weight / resampling
 8. Hyperparameters tuned on the TEST set
 9. Wrong metric              - accuracy only, on imbalanced data
"""

import warnings

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_generator import LEAKY_COL, TARGET, TIME_COL, generate_dataset
from utils import compute_metrics, print_metrics

warnings.filterwarnings("ignore")  # hides convergence warnings (also a bad habit)

RANDOM_STATE = 42


def run_broken_pipeline(verbose: bool = True) -> dict:
    df = generate_dataset()

    # FLAW 2: duplicates are never removed
    # FLAW 1: the leaky column stays in the feature set
    X = df.drop(columns=[TARGET, TIME_COL])
    y = df[TARGET]

    # FLAW 3: imputer learns the median from the WHOLE dataset (test data included)
    X = SimpleImputer(strategy="median").fit_transform(X)

    # FLAW 4 + 5: random shuffle of time-ordered rows, and no stratify=y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
    )

    # FLAW 6: no scaling.  FLAW 7: no class_weight.
    # FLAW 8: the "best" C is chosen by looking at the TEST set.
    best_model, best_acc = None, -1.0
    for c in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=c, max_iter=1000)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_model, best_acc = model, acc

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    if verbose:
        # FLAW 9: the headline number is accuracy
        print(f"[BROKEN] Reported accuracy: {metrics['accuracy']:.2%}")
        print_metrics("BROKEN PIPELINE - full metrics", metrics, y_test, y_pred)
    return metrics


if __name__ == "__main__":
    run_broken_pipeline()