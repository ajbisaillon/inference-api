from main import app, PredictionInput

def test_app_exists():
    assert app is not None

def test_feature_count():
    assert len(PredictionInput.__fields__) == 23

def test_target_not_in_features():
    assert "default_payment_next_month" not in PredictionInput.__fields__