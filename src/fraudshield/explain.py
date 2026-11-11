"""Global (SHAP) and local (per-transaction reason codes) explainability.

    python -m fraudshield.explain
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import MODEL_DIR, MODEL_PATH, REPORT_DIR


def _base_estimator(cal):
    return cal.calibrated_classifiers_[0].estimator.estimator


class Explainer:
    def __init__(self, cal=None):
        self.cal = cal or joblib.load(MODEL_PATH)
        self.base = _base_estimator(self.cal)
        self.exp = shap.TreeExplainer(self.base)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        v = self.exp.shap_values(X)
        return v[1] if isinstance(v, list) else v

    def reason_codes(self, X: pd.DataFrame, k: int = 3):
        """Top-k features pushing each row toward 'fraud' (adverse-action style reasons)."""
        sv = self.shap_values(X)
        out = []
        for i in range(len(X)):
            idx = np.argsort(-sv[i])[:k]
            out.append([{"feature": X.columns[j], "value": float(X.iloc[i, j]), "impact": round(float(sv[i, j]), 3)}
                        for j in idx if sv[i, j] > 0])
        return out


def main():
    ex = Explainer()
    ref = pd.read_csv(MODEL_DIR / "reference.csv").head(2000)
    sv = ex.shap_values(ref)
    plt.figure(); shap.summary_plot(sv, ref, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(REPORT_DIR / "shap_summary.png", dpi=130, bbox_inches="tight"); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=ref.columns).sort_values(ascending=False).head(10)
    print(imp.round(4).to_string())


if __name__ == "__main__":
    main()
