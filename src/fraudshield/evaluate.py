"""Metrics that matter for imbalanced, cost-asymmetric problems."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

from .config import COSTS


def total_cost(y, p, amount, threshold, c=COSTS) -> float:
    pred = p >= threshold
    fn = (y == 1) & ~pred
    fp = (y == 0) & pred
    tp = (y == 1) & pred
    # missed fraud loses the amount + chargeback; every flagged txn costs a review
    return float((amount[fn] + c.chargeback_fee).sum()
                 + fp.sum() * (c.review_cost + c.friction_cost)
                 + tp.sum() * c.review_cost)


def best_threshold(y, p, amount, c=COSTS):
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.9995, 400)))
    costs = np.array([total_cost(y, p, amount, t, c) for t in grid])
    i = int(costs.argmin())
    return float(grid[i]), float(costs[i])


def report(y, p, amount, threshold) -> dict:
    pred = (p >= threshold).astype(int)
    baseline_cost = total_cost(y, p, amount, np.inf)  # approve everything
    cost = total_cost(y, p, amount, threshold)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "threshold": float(threshold),
        "alert_rate": float(pred.mean()),
        "cost_no_model": baseline_cost,
        "cost_with_model": cost,
        "savings": baseline_cost - cost,
        "savings_pct": float(100 * (baseline_cost - cost) / baseline_cost),
    }
