"""
Shared helpers used by 05_model_improvement.ipynb and app.py.

Keeping them in one file means the notebook (which trains the model) and the
app (which uses it) prepare customer data in exactly the same way.
"""
import numpy as np
import pandas as pd

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]

# number columns that get imputed and scaled
NUM_COLS = ["tenure", "MonthlyCharges", "num_addons", "log_tenure"]


def add_features(customers):
    """Add the new 'engineered' columns. Each one is a pattern we found in the EDA."""
    c = customers.copy()

    # how many of the 6 add-ons the customer has (more add-ons = more loyal)
    c["num_addons"] = (c[ADDONS] == "Yes").sum(axis=1)

    # brand-new customer (first 6 months had 53% churn)
    c["is_new"] = (c["tenure"] <= 6).astype(int)

    # fiber customer with no TechSupport and no OnlineSecurity (the riskiest service setup)
    c["fiber_no_support"] = ((c["InternetService"] == "Fiber optic")
                             & (c["TechSupport"] == "No")
                             & (c["OnlineSecurity"] == "No")).astype(int)

    # month-to-month AND fiber (55% churn in the EDA grid)
    c["m2m_fiber"] = ((c["Contract"] == "Month-to-month")
                      & (c["InternetService"] == "Fiber optic")).astype(int)

    # month-to-month AND in the first year
    c["m2m_new"] = ((c["Contract"] == "Month-to-month") & (c["tenure"] <= 12)).astype(int)

    # churn drops fast in the first months, then levels off; log captures that curve
    c["log_tenure"] = np.log1p(c["tenure"])

    return c


def encode(customers, categories, columns=None):
    """Drop unused columns, add features, turn text into 0/1 columns."""
    X = customers.drop(columns=["Churn", "customerID", "TotalCharges"], errors="ignore")
    X = add_features(X)

    # remind every text column of ALL its possible values, so encoding is always the same
    for col, levels in categories.items():
        X[col] = pd.Categorical(X[col], categories=levels)

    X = pd.get_dummies(X, drop_first=True, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def prepare(customers, saved):
    """Full preparation with the imputer and scaler learned in the notebook."""
    X = encode(customers, saved["categories"], saved["columns"])
    X[NUM_COLS] = saved["imputer"].transform(X[NUM_COLS])
    X[NUM_COLS] = saved["scaler"].transform(X[NUM_COLS])
    return X
