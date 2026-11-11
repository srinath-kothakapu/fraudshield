"""Central configuration. Business costs drive the decision threshold."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "metadata.json"

SEED = 42


@dataclass(frozen=True)
class CostConfig:
    """Business cost model (USD).

    A missed fraud (false negative) costs the transaction amount plus a
    chargeback fee. A false alarm (false positive) costs a manual review
    plus a customer-friction penalty (declined legit customer may churn).
    """
    chargeback_fee: float = 15.0
    review_cost: float = 4.0
    friction_cost: float = 6.0


COSTS = CostConfig()
