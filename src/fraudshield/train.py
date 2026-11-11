"""Train, compare, calibrate, threshold-tune, and persist the model.

    python -m fraudshield.train
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import __version__
from .config import COSTS, META_PATH, MODEL_DIR, MODEL_PATH, REPORT_DIR, SEED
from .data import generate_transactions, time_based_split
from .evaluate import best_threshold, report, total_cost
from .features import build_features


def _models(pos_weight: float) -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="aucpr",
                                 n_jobs=-1, random_state=SEED, tree_method="hist"),
    }


def main(n: int = 120_000) -> dict:
    MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    df = generate_transactions(n)
    tr, va, te = time_based_split(df)
    Xtr, Xva, Xte = (build_features(d) for d in (tr, va, te))
    ytr, yva, yte = tr.is_fraud.values, va.is_fraud.values, te.is_fraud.values
    print(f"rows train/val/test = {len(tr)}/{len(va)}/{len(te)} | fraud rate = {df.is_fraud.mean():.3%}")

    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    results, fitted = {}, {}
    for name, m in _models(pos_weight).items():
        t0 = time.time(); m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        results[name] = report(yva, p, va.amount.values, best_threshold(yva, p, va.amount.values)[0])
        results[name]["train_seconds"] = round(time.time() - t0, 1)
        fitted[name] = m
        print(f"{name:14s} PR-AUC={results[name]['pr_auc']:.3f} ROC-AUC={results[name]['roc_auc']:.3f} "
              f"savings={results[name]['savings_pct']:.1f}%")

    best_name = max(results, key=lambda k: results[k]["pr_auc"])  # select on validation PR-AUC
    print("selected:", best_name)

    # Calibrate probabilities (needed so the cost-optimal threshold is meaningful)
    base = _models(pos_weight)[best_name]
    base.fit(Xtr, ytr)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(Xva, yva)
    pva = cal.predict_proba(Xva)[:, 1]
    thr, _ = best_threshold(yva, pva, va.amount.values)

    pte = cal.predict_proba(Xte)[:, 1]  # untouched test set: reported once
    test = report(yte, pte, te.amount.values, thr)
    print("TEST:", json.dumps({k: round(v, 4) for k, v in test.items()}, indent=1))

    # ---- plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for name, m in fitted.items():
        pr, rc, _ = precision_recall_curve(yte, m.predict_proba(Xte)[:, 1]); ax[0].plot(rc, pr, label=name)
    ax[0].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision"); ax[0].legend()
    ths = np.linspace(0.01, 0.95, 120)
    ax[1].plot(ths, [total_cost(yva, pva, va.amount.values, t) for t in ths])
    ax[1].axvline(thr, color="r", ls="--", label=f"optimal t={thr:.2f}")
    ax[1].set(title="Business cost vs threshold (val)", xlabel="Threshold", ylabel="Cost (USD)"); ax[1].legend()
    fr, mp = calibration_curve(yte, pte, n_bins=10, strategy="quantile")
    ax[2].plot(mp, fr, "o-", label="calibrated"); ax[2].plot([0, 1], [0, 1], "k:")
    ax[2].set(title="Calibration (test)", xlabel="Predicted", ylabel="Observed"); ax[2].legend()
    plt.tight_layout(); plt.savefig(REPORT_DIR / "evaluation.png", dpi=130); plt.close()

    # ---- persist model + reference stats for drift monitoring
    joblib.dump(cal, MODEL_PATH)
    Xtr.sample(5000, random_state=SEED).to_csv(MODEL_DIR / "reference.csv", index=False)
    meta = {"version": __version__, "model": best_name, "threshold": thr, "features": list(Xtr.columns),
            "validation": results, "test": test, "costs": COSTS.__dict__}
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    main()
