"""FastAPI service: /predict returns fraud probability, decision and reason codes.

    uvicorn fraudshield.api:app --reload
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import META_PATH, MODEL_PATH
from .explain import Explainer
from .features import build_features

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run `make train` first.")
    STATE["model"] = joblib.load(MODEL_PATH)
    STATE["meta"] = json.loads(META_PATH.read_text())
    STATE["explainer"] = Explainer(STATE["model"])
    yield
    STATE.clear()


app = FastAPI(title="FraudShield", version="1.0.0", lifespan=lifespan)

Category = Literal["grocery", "restaurant", "travel", "electronics", "fashion", "gaming", "fuel", "utilities"]


class Transaction(BaseModel):
    amount: float = Field(gt=0, examples=[420.0])
    hour: int = Field(ge=0, le=23, examples=[3])
    account_age_days: int = Field(ge=0, examples=[12])
    merchant_category: Category = "electronics"
    is_foreign: int = Field(ge=0, le=1, examples=[1])
    card_present: int = Field(ge=0, le=1, examples=[0])
    txn_count_24h: int = Field(ge=0, examples=[7])
    dist_from_home_km: float = Field(ge=0, examples=[2400.0])
    avg_amount_30d: float = Field(gt=0, examples=[35.0])
    device_new: int = Field(ge=0, le=1, examples=[1])


class Decision(BaseModel):
    fraud_probability: float
    decision: Literal["BLOCK", "APPROVE"]
    threshold: float
    reasons: list[dict]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["meta"]["model"], "version": STATE["meta"]["version"]}


@app.get("/model-info")
def model_info():
    m = STATE["meta"]
    return {"model": m["model"], "threshold": m["threshold"], "test_metrics": m["test"]}


@app.post("/predict", response_model=Decision)
def predict(txn: Transaction):
    t0 = time.perf_counter()
    try:
        X = build_features(pd.DataFrame([txn.model_dump()]))[STATE["meta"]["features"]]
    except Exception as e:  # pragma: no cover
        raise HTTPException(422, f"feature error: {e}")
    p = float(STATE["model"].predict_proba(X)[0, 1])
    thr = STATE["meta"]["threshold"]
    block = p >= thr
    return Decision(fraud_probability=round(p, 5), decision="BLOCK" if block else "APPROVE", threshold=round(thr, 4),
                    reasons=STATE["explainer"].reason_codes(X)[0] if block else [],
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2))
