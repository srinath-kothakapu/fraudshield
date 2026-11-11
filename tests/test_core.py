import numpy as np
import pandas as pd

from fraudshield.data import generate_transactions, time_based_split
from fraudshield.drift import drift_report, psi
from fraudshield.evaluate import best_threshold, total_cost
from fraudshield.features import build_features


def test_generator_is_reproducible_and_imbalanced():
    a, b = generate_transactions(5000, seed=1), generate_transactions(5000, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert 0.005 < a.is_fraud.mean() < 0.03


def test_time_split_has_no_overlap_and_keeps_order():
    df = generate_transactions(1000)
    tr, va, te = time_based_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.index.max() < va.index.min() < te.index.min()


def test_features_have_stable_columns_for_single_row():
    df = generate_transactions(500)
    full, one = build_features(df), build_features(df.iloc[[0]])
    assert list(full.columns) == list(one.columns)
    assert not full.isna().any().any()


def test_cost_model_basics():
    y = np.array([1, 0, 0, 1]); p = np.array([.9, .8, .1, .2]); amt = np.array([100., 50, 20, 10])
    assert total_cost(y, p, amt, np.inf) == (100 + 15) + (10 + 15)  # approve all => lose all fraud
    perfect = total_cost(y, np.array([1, 0, 0, 1.]), amt, 0.5)
    assert perfect < total_cost(y, p, amt, np.inf)


def test_best_threshold_beats_extremes():
    df = generate_transactions(20000, seed=3)
    rng = np.random.default_rng(0)
    p = np.clip(df.is_fraud.values * 0.5 + rng.random(len(df)) * 0.5, 0, 1)
    t, c = best_threshold(df.is_fraud.values, p, df.amount.values)
    assert c <= total_cost(df.is_fraud.values, p, df.amount.values, np.inf)


def test_psi_detects_shift_but_not_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 10000)
    assert psi(a, rng.normal(0, 1, 10000)) < 0.05
    assert psi(a, rng.normal(1.0, 1, 10000)) > 0.25


def test_drift_report_flags_shifted_feature():
    ref = build_features(generate_transactions(8000, seed=1))
    cur = build_features(generate_transactions(8000, seed=2, drift=True))
    r = drift_report(ref, cur).set_index("feature")
    assert r.loc["amount", "status"] != "OK"
