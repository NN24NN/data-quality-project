"""Classification downstream: Logistic Regression + Random Forest, đo F1-macro."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .common import fit_and_score

MODELS = {'logistic_regression': LogisticRegression, 'random_forest': RandomForestClassifier}


def evaluate(algorithm, train, test, target, seed, params):
    """Huấn luyện trên train, trả về F1-macro trên test (cả hai có thể đã bị làm bẩn — Scenario 3)."""
    model = MODELS[algorithm](random_state=seed, **params)
    return fit_and_score(model, train, test, target, lambda y, pred: f1_score(y, pred, average='macro'))
