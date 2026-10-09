"""Regression downstream: Ridge + Gradient Boosting, đo R²."""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from .common import fit_and_score

MODELS = {'ridge': Ridge, 'gradient_boosting': GradientBoostingRegressor}


def evaluate(algorithm, train, test, target, seed, params):
    """Huấn luyện trên train, trả về R² trên test (cả hai có thể đã bị làm bẩn — Scenario 3)."""
    model = MODELS[algorithm](random_state=seed, **params)
    return fit_and_score(model, train, test, target, r2_score)
