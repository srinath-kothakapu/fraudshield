"""Synthetic but realistic transaction generator.

No external download needed, so the repo is fully reproducible. The generator
embeds real-world fraud patterns (night-time, new accounts, foreign + card-not-
present, velocity bursts, high amount) with noise and class imbalance (~1.5%),
and supports *drift* to demo monitoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED

CATEGORIES = ["grocery", "restaurant", "travel", "electronics", "fashion", "gaming", "fuel", "utilities"]
CAT_RISK = {"grocery": -0.6, "restaurant": -0.3, "travel": 0.5, "electronics": 0.9,
            "fashion": 0.1, "gaming": 0.7, "fuel": -0.4, "utilities": -0.8}


def generate_transactions(n: int = 120_000, fraud_rate_target: float = 0.015,
                          seed: int = SEED, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(CATEGORIES, n, p=[.24, .18, .08, .1, .14, .06, .12, .08])
    hour = np.where(rng.random(n) < 0.1, rng.integers(0, 6, n), rng.integers(6, 24, n))
    account_age_days = rng.gamma(2.0, 400, n).astype(int) + 1
    amount = np.exp(rng.normal(3.6, 1.0, n))
    if drift:  # inflation + new high-ticket merchants
        amount = amount * 1.9
    is_foreign = rng.random(n) < (0.12 if not drift else 0.28)
    card_present = rng.random(n) < 0.62
    txn_24h = rng.poisson(1.8, n) + (rng.random(n) < 0.03) * rng.integers(4, 14, n)
    dist_from_home_km = np.where(is_foreign, rng.gamma(2, 600, n), rng.gamma(1.5, 12, n))
    avg_amount_30d = np.exp(rng.normal(3.5, 0.6, n))
    amount_ratio = amount / avg_amount_30d
    device_new = rng.random(n) < 0.08
    cat_risk = np.array([CAT_RISK[c] for c in cat])

    night = hour < 6
    fcnp = is_foreign & ~card_present
    ratio_hi = np.clip(np.log(amount_ratio), 0, 4)
    logit = (-7.0
             + 1.5 * night
             + 1.6 * (account_age_days < 30)
             + 1.4 * fcnp
             + 0.6 * np.log1p(txn_24h)
             + 0.9 * ratio_hi
             + 2.0 * device_new
             + 0.5 * np.minimum(dist_from_home_km, 1500) / 300
             + 1.5 * cat_risk
             # non-linear interactions: trees can exploit these, linear models cannot
             + 2.2 * (night & fcnp)
             + 1.8 * ((account_age_days < 30) & (ratio_hi > 1.5))
             + 1.6 * (device_new & (txn_24h >= 6)))
    # calibrate intercept so overall rate ~ target
    shift = 0.0
    for _ in range(25):
        p = 1 / (1 + np.exp(-(logit + shift)))
        shift += np.log(fraud_rate_target / max(p.mean(), 1e-9)) * 0.9
    p = 1 / (1 + np.exp(-(logit + shift)))
    is_fraud = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "amount": amount.round(2), "hour": hour, "account_age_days": account_age_days,
        "merchant_category": cat, "is_foreign": is_foreign.astype(int),
        "card_present": card_present.astype(int), "txn_count_24h": txn_24h,
        "dist_from_home_km": dist_from_home_km.round(1), "avg_amount_30d": avg_amount_30d.round(2),
        "device_new": device_new.astype(int), "is_fraud": is_fraud,
    })
    return df


def time_based_split(df: pd.DataFrame, train=0.7, val=0.15):
    """Rows are treated as time-ordered; split chronologically to avoid leakage."""
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()
