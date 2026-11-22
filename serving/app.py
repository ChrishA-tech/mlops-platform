"""FastAPI model service: health/readiness probes, Prometheus metrics, versioned model.

Applies the same `engineer()` features as training, so train/serve cannot diverge.
"""
from __future__ import annotations
import json, os, time, logging
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlops.features import engineer  # noqa: E402

log = logging.getLogger("serving"); logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
ART = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

MODEL = joblib.load(ART / "model.joblib")
SCHEMA = json.load(open(ART / "schema.json"))
COLS = SCHEMA["columns"]

REQS = Counter("predict_requests_total", "Prediction requests", ["status"])
LAT = Histogram("predict_latency_seconds", "Prediction latency",
                buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1))
app = FastAPI(title="mlops-platform serving", version="1.0.0")


class Record(BaseModel):
    features: dict  # raw customer attributes


@app.get("/healthz")
def healthz(): return {"status": "ok"}


@app.get("/readyz")
def readyz(): return {"status": "ready", "n_features": len(COLS)}


@app.get("/metrics")
def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(rec: Record):
    t = time.perf_counter()
    try:
        df = engineer(pd.DataFrame([rec.features])).reindex(columns=COLS)
        p = float(MODEL.predict_proba(df)[0, 1])
    except Exception:
        REQS.labels("error").inc(); log.exception("prediction failed")
        raise HTTPException(500, "prediction failed")
    LAT.observe(time.perf_counter() - t); REQS.labels("ok").inc()
    return {"churn_probability": round(p, 6), "flag": p >= THRESHOLD, "model_version": "1.0.0"}
