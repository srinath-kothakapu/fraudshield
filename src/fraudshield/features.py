"""Feature engineering shared by training and serving (single source of truth)."""
import numpy as np
import pandas as pd

from .data import CATEGORIES

NUMERIC = ["amount", "hour", "account_age_days", "is_foreign", "card_present", "txn_count_24h",
           "dist_from_home_km", "avg_amount_30d", "device_new",
           "amount_ratio", "log_amount", "is_night", "new_account", "foreign_cnp", "velocity_high"]
RAW_REQUIRED = ["amount", "hour", "account_age_days", "merchant_category", "is_foreign",
                "card_present", "txn_count_24h", "dist_from_home_km", "avg_amount_30d", "device_new"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[RAW_REQUIRED].copy()
    out["amount_ratio"] = out["amount"] / out["avg_amount_30d"].clip(lower=1.0)
    out["log_amount"] = np.log1p(out["amount"])
    out["is_night"] = (out["hour"] < 6).astype(int)
    out["new_account"] = (out["account_age_days"] < 30).astype(int)
    out["foreign_cnp"] = ((out["is_foreign"] == 1) & (out["card_present"] == 0)).astype(int)
    out["velocity_high"] = (out["txn_count_24h"] >= 6).astype(int)
    for c in CATEGORIES:  # fixed one-hot => identical columns at train and serve time
        out[f"cat_{c}"] = (out["merchant_category"] == c).astype(int)
    return out.drop(columns=["merchant_category"])
