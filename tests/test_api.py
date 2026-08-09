"""
tests/test_api.py — Unit Tests for the Churn Prediction API (Session 8)
=========================================================================
RUN:
    pytest tests/test_api.py -v

RUN VIA CI/CD:
    These tests run automatically on every git push (see .github/workflows/ci.yml).
    If any test fails, the pipeline stops and the Docker image is never built.

WHY WE MOCK THE MODEL:
    Tests should not depend on a live MLflow server being reachable.
    We patch mlflow.sklearn.load_model BEFORE importing the app, so the
    app's startup event uses our fake model instead of trying to connect
    to a real MLflow Registry.
"""

import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Mock model setup ──────────────────────────────────────────────────────────
# predict_proba returns [[P(not churn), P(churn)]] for a single sample
mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.65, 0.35]])

# Patch BEFORE importing the app — the startup event runs load_model()
# which calls mlflow.sklearn.load_model internally.
with patch("mlflow.sklearn.load_model", return_value=mock_model), \
     patch("mlflow.set_tracking_uri"), \
     patch("mlflow.tracking.MlflowClient") as MockClient:

    # get_latest_versions must return something with a .version attribute
    mock_version = MagicMock()
    mock_version.version = "1"
    MockClient.return_value.get_latest_versions.return_value = [mock_version]

    from api.main import app


# ── Client fixture ────────────────────────────────────────────────────────────
# IMPORTANT: TestClient must be used as a context manager for FastAPI's
# @app.on_event("startup") handler to actually run. Without the `with`
# block, the startup event never fires and `model` stays None, causing
# every /predict call to return 503 regardless of mocking.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Sample valid request body ────────────────────────────────────────────────
SAMPLE_REQUEST = {
    "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
    "tenure": 24, "PhoneService": 1, "MultipleLines": 0,
    "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 1,
    "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 1,
    "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
    "PaymentMethod": 2, "MonthlyCharges": 65.5, "TotalCharges": 1572.0,
}


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_root_endpoint_returns_200(client):
    """The root endpoint should return a friendly landing message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint_returns_200(client):
    """The health endpoint should confirm the service and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_name" in data
    assert data["model_name"] == "churn-prediction-xgboost"


def test_predict_returns_valid_response(client):
    """A valid request should return a well-formed prediction."""
    response = client.post("/predict", json=SAMPLE_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert "confidence" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["prediction"] in ["Will churn", "Will not churn"]
    assert data["confidence"] in ["High", "Medium", "Low"]


def test_predict_rejects_missing_field(client):
    """Removing a required field should trigger automatic 422 validation."""
    bad_request = SAMPLE_REQUEST.copy()
    del bad_request["tenure"]
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


def test_predict_rejects_wrong_type(client):
    """Sending a string where a number is expected should trigger 422."""
    bad_request = SAMPLE_REQUEST.copy()
    bad_request["MonthlyCharges"] = "not_a_number"
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


def test_predict_churn_probability_range(client):
    """Churn probability must always be a valid probability."""
    response = client.post("/predict", json=SAMPLE_REQUEST)
    data = response.json()
    prob = data["churn_probability"]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} is outside [0, 1]"


def test_predict_high_risk_customer(client):
    """
    A high-risk profile (short tenure, month-to-month, high charges)
    should still return a structurally valid response even with a
    mocked model — this test checks the request/response contract,
    not real model accuracy.
    """
    high_risk_request = SAMPLE_REQUEST.copy()
    high_risk_request.update({
        "tenure": 2,
        "Contract": 0,
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.0,
    })
    response = client.post("/predict", json=high_risk_request)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data


def test_docs_endpoint_available(client):
    """Swagger UI should be reachable at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200