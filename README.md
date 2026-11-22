# mlops-platform: an end-to-end MLOps pipeline

A production-shaped machine-learning platform built around one principle: **the
model is the easy part — validation, reproducibility, serving and monitoring are
the job.** It takes a real dataset from raw CSV to a registered model served
behind an API with drift monitoring, wired together by orchestration, CI/CD and
infrastructure-as-code.

The modelling payload is customer churn (IBM Telco, 7,043 customers, 26.5% churn)
but the dataset is **swappable** via `conf/config.yaml` — point `source` at a
larger file (e.g. NYC taxi parquet) and the pipeline is unchanged.

## What it does, end to end
1. **Ingest + validate** — `pandera` schema contract catches dirty data loudly
   (the Telco set ships blank `TotalCharges` strings; the pipeline coerces and
   checks ranges before anything trains).
2. **Feature engineering** — pure, leakage-safe derived features (spend ratios,
   service counts, tenure buckets); the same logic exists in **PySpark** for
   cluster-scale data.
3. **Train + tune** — `Optuna` HPO over a LightGBM pipeline, PR-AUC under
   stratified CV, chosen on validation and **scored once on test**.
4. **Track + register** — every run logged to **MLflow** (SQLite backend); the
   best model is versioned in the **MLflow Model Registry** (`churn-lgbm`).
5. **Serve** — a **FastAPI** service (health/readiness probes, **Prometheus**
   metrics) applying the identical feature logic, in a hardened container on
   **Kubernetes** (Deployment, Service, HPA).
6. **Monitor** — a native **PSI + KS** drift report against the training
   baseline, plus an optional Evidently HTML report.
7. **Orchestrate + ship** — a **Prefect** flow chains ingest → validate → train →
   monitor; **GitHub Actions** runs the whole thing on every push; **Terraform**
   describes the S3 + ECR footprint.

## Verified results
- **Model:** test ROC-AUC **0.835**, PR-AUC 0.647, Brier 0.159 (churn is a hard,
  well-calibrated target; the number is honest, not tuned on test).
- **Registry:** `churn-lgbm` version 1 registered to the MLflow registry on a
  SQLite backend (a file store cannot register — the project uses the DB backend
  deliberately).
- **Drift monitor:** injecting a covariate shift on 2 features flags exactly
  those 2 of 7 numeric features; a control resample flags 0. The monitor fires on
  real shift and stays quiet otherwise.
- **Serving:** 2,000 concurrent requests at concurrency 32 with **0 errors**;
  each request runs the full feature pipeline (latency is dominated by per-row
  pandas work — batching is the obvious next step, noted honestly).
- **CI:** trains (HPO + MLflow + registry), runs 7 tests, builds the Docker
  image, and validates the Kubernetes manifests against the 1.29 schema.

## Not verified in this environment
`docker build`, a live Kubernetes deployment, the Prefect scheduler, and
`terraform apply` are authored and statically checked, not run here. Commands to
run them yourself are below.

## Tools used
Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, Optuna, MLflow
(tracking + registry), pandera, PySpark, FastAPI, Uvicorn, Pydantic, Prometheus,
Evidently, SciPy, Prefect, Docker, Kubernetes, Terraform, GitHub Actions, pytest,
joblib, Grafana.

## Run it
```bash
pip install -e ".[dev]"
python -m mlops.train              # HPO -> MLflow -> registry -> artifacts/
pytest -q                          # validation, features, drift, serving
uvicorn serving.app:app --port 8000
# orchestration / infra (optional extras)
pip install -e ".[orchestration]" && python pipelines/flow.py
docker build -t mlops-platform:1.0.0 . && kubectl apply -f k8s/
terraform -chdir=infra/terraform init && terraform -chdir=infra/terraform validate
```

## Layout
`src/mlops/` data · features · train · monitor  ·  `serving/` FastAPI app  ·
`pipelines/` Prefect flow  ·  `k8s/` manifests  ·  `infra/terraform/` IaC  ·
`monitoring/` Prometheus + Grafana  ·  `conf/config.yaml` swappable dataset.
