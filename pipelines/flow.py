"""Prefect orchestration: ingest -> validate -> train -> monitor, as one flow.

Run locally with:  python pipelines/flow.py
Schedule in Prefect Cloud/Server by deploying this flow.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prefect import flow, task  # type: ignore
from mlops.train import load_cfg, run
from mlops.data import load_clean_validate
from mlops.features import engineer, split_xy
from mlops.monitor import drift_report, simulate_shift


@task(retries=2, retry_delay_seconds=5)
def validate_data(cfg):
    df = load_clean_validate(cfg)
    return len(df)


@task
def train_model(cfg):
    return run(cfg)


@task
def check_drift(cfg):
    X, _ = split_xy(engineer(load_clean_validate(cfg)))
    # In production `current` is the latest scored batch; here we simulate a shift.
    return drift_report(X, simulate_shift(X))


@flow(name="mlops-platform-retrain")
def retrain_flow(config_path: str = "conf/config.yaml"):
    cfg = load_cfg(config_path)
    n = validate_data(cfg)
    metrics = train_model(cfg)
    drift = check_drift(cfg)
    print({"rows": n, "metrics": metrics, "drift_share": drift["share_drifted"]})
    return metrics


if __name__ == "__main__":
    retrain_flow()
