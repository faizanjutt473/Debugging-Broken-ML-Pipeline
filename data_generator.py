"""
data_generator.py
-----------------
Creates a synthetic customer-churn dataset that deliberately contains the
conditions needed to demonstrate common ML pipeline mistakes:

    * Highly imbalanced target (~6% churn)
    * A feature that leaks the target (cancellation_request_logged)
    * Duplicate rows
    * Missing values
    * Features on very different scales (annual_income vs support_calls)
    * Rows ordered by time, with the churn pattern drifting over time
"""

import numpy as np
import pandas as pd

TARGET = "churned"
TIME_COL = "signup_date"
LEAKY_COL = "cancellation_request_logged"


def generate_dataset(n_rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Rows are ordered by time (oldest -> newest)
    dates = pd.date_range("2021-01-01", periods=n_rows, freq="h")
    t = np.linspace(0, 1, n_rows)  # 0 = oldest, 1 = newest

    df = pd.DataFrame(
        {
            TIME_COL: dates,
            "tenure_months": rng.integers(1, 73, n_rows),
            "monthly_charges": rng.normal(70, 25, n_rows).clip(10, 200),
            "annual_income": rng.lognormal(11, 0.5, n_rows),  # ~60,000 (large scale)
            "support_calls": rng.poisson(2, n_rows),
            "age": rng.normal(40, 12, n_rows).clip(18, 90),
            "usage_hours": rng.normal(120, 40, n_rows).clip(0, None),
        }
    )

    # True churn mechanism. The effect of monthly_charges GROWS over time
    # (concept drift), so a random split overestimates real-world quality.
    logit = (
        -3.0
        + 0.45 * df["support_calls"]
        - 0.03 * df["tenure_months"]
        + (0.005 + 0.035 * t) * (df["monthly_charges"] - 70)
        - 0.004 * (df["usage_hours"] - 120)
    )
    prob = 1 / (1 + np.exp(-logit))
    df[TARGET] = (rng.random(n_rows) < prob).astype(int)

    # Leaky feature: only exists AFTER the customer has already churned
    leak_prob = np.where(df[TARGET] == 1, 0.97, 0.01)
    df[LEAKY_COL] = (rng.random(n_rows) < leak_prob).astype(int)

    # Missing values (random)
    for col, rate in [("monthly_charges", 0.08), ("usage_hours", 0.05), ("age", 0.05)]:
        df.loc[rng.random(n_rows) < rate, col] = np.nan

    # Duplicate rows (~10%), placed next to the originals so time order is kept
    dup = df.sample(frac=0.10, random_state=seed)
    df = pd.concat([df, dup]).sort_values(TIME_COL, kind="stable").reset_index(drop=True)

    return df


if __name__ == "__main__":
    data = generate_dataset()
    print(data.shape)
    print(data[TARGET].value_counts(normalize=True).round(3))
