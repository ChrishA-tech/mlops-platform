import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops.train import load_cfg, run
from mlops.data import load_clean_validate, clean, load_raw
from mlops.features import engineer, split_xy
from mlops.monitor import drift_report, simulate_shift, psi

CFG = load_cfg()


def test_clean_binarises_target_and_fixes_totalcharges():
    df = clean(load_raw(CFG["dataset"]["source"]), CFG)
    assert set(df["Churn"].unique()) <= {0, 1}
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert CFG["dataset"]["id_col"] not in df.columns


def test_schema_validation_passes_on_clean_data():
    df = load_clean_validate(CFG)
    assert len(df) > 7000


def test_engineer_is_pure_and_adds_features():
    df = load_clean_validate(CFG)
    out = engineer(df)
    for c in ("avg_monthly_spend", "n_services", "tenure_bucket"):
        assert c in out.columns
    # purity: calling twice yields identical frames
    pd.testing.assert_frame_equal(engineer(df), out)


def test_no_target_leakage_in_features():
    X, y = split_xy(engineer(load_clean_validate(CFG)))
    assert "Churn" not in X.columns


def test_monitor_fires_on_injected_shift():
    df = engineer(load_clean_validate(CFG))
    X, _ = split_xy(df)
    rep = drift_report(X, simulate_shift(X))
    assert rep["n_drifted"] >= 1
    # and does NOT fire comparing a sample to itself
    base = drift_report(X, X.sample(frac=0.5, random_state=0))
    assert base["share_drifted"] <= 0.2


def test_psi_zero_for_identical():
    import numpy as np
    assert psi(np.arange(1000), np.arange(1000)) < 1e-6


@pytest.mark.slow
def test_train_smoke_and_serve(tmp_path, monkeypatch):
    monkeypatch.setenv("HPO_N_TRIALS", "2")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    cfg = load_cfg()
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path/'mlf.db'}"
    cfg["hpo"]["n_trials"] = 2
    m = run(cfg)
    assert 0.55 <= m["test_roc_auc"] <= 1.0
    # serving loads the freshly trained artifacts
    from serving.app import app
    c = TestClient(app)
    assert c.get("/healthz").json()["status"] == "ok"
    rec = {"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
           "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
           "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
           "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
           "Contract": "Month-to-month", "PaperlessBilling": "Yes",
           "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 150.0}
    r = c.post("/predict", json={"features": rec}).json()
    assert 0.0 <= r["churn_probability"] <= 1.0
