"""Data-drift monitoring with Population Stability Index (PSI) + KS test.

Rule of thumb for PSI: <0.1 stable, 0.1-0.25 moderate shift, >0.25 retrain.

    python -m fraudshield.drift
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import MODEL_DIR, REPORT_DIR


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:  # near-constant / binary feature
        cuts = np.array([-np.inf, np.median(expected) + 1e-9, np.inf])
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual, cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in reference.columns:
        v = psi(reference[c].values, current[c].values)
        ks = ks_2samp(reference[c], current[c])
        status = "RETRAIN" if v > 0.25 else "WATCH" if v > 0.1 else "OK"
        rows.append({"feature": c, "psi": round(v, 4), "ks_stat": round(ks.statistic, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def main():
    from .data import generate_transactions
    from .features import build_features
    ref = pd.read_csv(MODEL_DIR / "reference.csv")
    stable = build_features(generate_transactions(20_000, seed=7))
    drifted = build_features(generate_transactions(20_000, seed=8, drift=True))
    for name, cur in [("STABLE batch", stable), ("DRIFTED batch", drifted)]:
        r = drift_report(ref, cur)
        print(f"\n== {name}: {(r.status != 'OK').sum()} features flagged ==")
        print(r.head(6).to_string(index=False))
        r.to_csv(REPORT_DIR / f"drift_{name.split()[0].lower()}.csv", index=False)


if __name__ == "__main__":
    main()
