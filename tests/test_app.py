import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

# Mock BigQuery Client before importing the app
with patch("google.cloud.bigquery.Client") as mock_client_cls:
    mock_client_instance = MagicMock()
    mock_client_instance.project = "mock-gcp-project"
    mock_client_cls.return_value = mock_client_instance
    
    from main import app, PredictionInput, bq_client

client = TestClient(app)


def test_app_exists():
    assert app is not None

def test_feature_count():
    fields = getattr(PredictionInput, "model_fields", getattr(PredictionInput, "__fields__", {}))
    assert len(fields) == 23

def test_target_not_in_features():
    fields = getattr(PredictionInput, "model_fields", getattr(PredictionInput, "__fields__", {}))
    assert "default_payment_next_month" not in fields


def test_predict_endpoint_success():
    """
    Tests the /predict endpoint by intercepting the initialized bq_client.
    """

    mock_row = {
        "predicted_default_payment_next_month": 1,
        "predicted_default_payment_next_month_probs": [
            {"label": "1", "prob": 0.72},
            {"label": "0", "prob": 0.28}
        ]
    }
    
    mock_query_job = MagicMock()
    mock_query_job.result.return_value = [mock_row]
    bq_client.query.return_value = mock_query_job

    valid_payload = {
        "limit_balance": 50000.0, "sex": "1", "education_level": "2", "marital_status": "1", "age": 35,
        "pay_0": 0.0, "pay_2": 0.0, "pay_3": 0.0, "pay_4": 0.0, "pay_5": 0.0, "pay_6": 0.0,
        "bill_amt_1": 10000.0, "bill_amt_2": 9500.0, "bill_amt_3": 9200.0, "bill_amt_4": 8000.0, "bill_amt_5": 7500.0, "bill_amt_6": 7000.0,
        "pay_amt_1": 2000.0, "pay_amt_2": 2000.0, "pay_amt_3": 2000.0, "pay_amt_4": 1500.0, "pay_amt_5": 1500.0, "pay_amt_6": 1500.0
    }

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    
    json_data = response.json()
    assert json_data["predicted_default_payment_next_month"] == 1
    assert len(json_data["probabilities"]) == 2
    assert json_data["probabilities"][0]["label"] == 1
    assert json_data["probabilities"][0]["prob"] == 0.72

    bq_client.query.assert_called_once()
