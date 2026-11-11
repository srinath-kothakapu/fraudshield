import pytest
from fastapi.testclient import TestClient

from fraudshield.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `make train` first")

RISKY = dict(amount=900, hour=3, account_age_days=5, merchant_category="electronics", is_foreign=1,
             card_present=0, txn_count_24h=9, dist_from_home_km=3000, avg_amount_30d=30, device_new=1)
SAFE = dict(amount=25, hour=14, account_age_days=1500, merchant_category="grocery", is_foreign=0,
            card_present=1, txn_count_24h=1, dist_from_home_km=3, avg_amount_30d=40, device_new=0)


@pytest.fixture(scope="module")
def client():
    from fraudshield.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_risky_blocked_with_reasons_and_safe_approved(client):
    r = client.post("/predict", json=RISKY).json()
    s = client.post("/predict", json=SAFE).json()
    assert r["decision"] == "BLOCK" and len(r["reasons"]) > 0
    assert s["decision"] == "APPROVE" and s["reasons"] == []
    assert r["fraud_probability"] > s["fraud_probability"]


def test_validation_rejects_bad_input(client):
    assert client.post("/predict", json={**SAFE, "hour": 99}).status_code == 422
    assert client.post("/predict", json={**SAFE, "merchant_category": "casino"}).status_code == 422
