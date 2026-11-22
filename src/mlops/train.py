"""Training: Optuna HPO over LightGBM, logged to MLflow, best model registered.

Metric is PR-AUC (average precision) under a stratified CV, chosen on a held-out
validation split and scored once on test — no tuning on test.
"""
from __future__ import annotations
import json, os, warnings
from pathlib import Path
import numpy as np
import joblib
import yaml
import optuna
import mlflow
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier

from mlops.data import load_clean_validate
from mlops.features import engineer, split_xy, build_preprocessor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ART = Path("artifacts"); ART.mkdir(exist_ok=True)


def load_cfg(path="conf/config.yaml") -> dict:
    return yaml.safe_load(open(path))


def _splits(cfg):
    df = engineer(load_clean_validate(cfg))
    X, y = split_xy(df)
    s = cfg["split"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=s["test_size"],
                                              stratify=y, random_state=s["random_state"])
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=s["valid_size"],
                                              stratify=y_tr, random_state=s["random_state"])
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def _make_pipe(params, X, n_jobs=-1):
    return Pipeline([("prep", build_preprocessor(X)),
                     ("clf", LGBMClassifier(**params, verbose=-1, n_jobs=n_jobs))])


def run(cfg: dict) -> dict:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    X_tr, X_va, X_te, y_tr, y_va, y_te = _splits(cfg)
    folds = StratifiedKFold(cfg["hpo"]["cv_folds"], shuffle=True, random_state=42)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 600),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            class_weight="balanced",
        )
        pipe = _make_pipe(params, X_tr, n_jobs=1)  # avoid thread oversubscription under CV parallelism
        return cross_val_score(pipe, X_tr, y_tr, cv=folds,
                               scoring=cfg["hpo"]["metric"], n_jobs=-1).mean()

    n_trials = int(os.environ.get("HPO_N_TRIALS", cfg["hpo"]["n_trials"]))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials,
                   timeout=cfg["hpo"]["timeout_s"], show_progress_bar=False)

    best = {**study.best_params, "class_weight": "balanced"}
    with mlflow.start_run(run_name="lgbm-hpo-best") as mlrun:
        mlflow.log_params(best)
        mlflow.log_metric("cv_pr_auc", study.best_value)
        pipe = _make_pipe(best, X_tr).fit(X_tr, y_tr)
        va_p = pipe.predict_proba(X_va)[:, 1]
        val_metrics = {"val_pr_auc": average_precision_score(y_va, va_p),
                       "val_roc_auc": roc_auc_score(y_va, va_p)}
        # Refit on train+valid, score once on test.
        import pandas as pd
        pipe = _make_pipe(best, X_tr).fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
        te_p = pipe.predict_proba(X_te)[:, 1]
        test_metrics = {"test_pr_auc": average_precision_score(y_te, te_p),
                        "test_roc_auc": roc_auc_score(y_te, te_p),
                        "test_brier": brier_score_loss(y_te, te_p)}
        mlflow.log_metrics({**val_metrics, **test_metrics})
        joblib.dump(pipe, ART / "model.joblib")
        X_te.to_csv(ART / "reference.csv", index=False)  # baseline for drift monitoring
        feat_meta = {"n_features_in": X_tr.shape[1], "columns": list(X_tr.columns)}
        json.dump(feat_meta, open(ART / "schema.json", "w"), indent=2)
        metrics = {"cv_pr_auc": round(study.best_value, 4),
                   **{k: round(v, 4) for k, v in {**val_metrics, **test_metrics}.items()}}
        json.dump(metrics, open(ART / "metrics.json", "w"), indent=2)
        mlflow.log_artifacts(str(ART))
        # DB-backed store (sqlite) enables the model registry; cloudpickle avoids
        # skops' untrusted-type rejection of the LightGBM booster.
        mlflow.sklearn.log_model(
            pipe, name="model", registered_model_name=cfg["model"]["registry_name"],
            serialization_format="cloudpickle")
    return metrics


if __name__ == "__main__":
    m = run(load_cfg())
    print(json.dumps(m, indent=2))
