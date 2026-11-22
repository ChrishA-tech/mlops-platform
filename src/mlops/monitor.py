"""Data-drift monitoring: PSI + KS test per feature against the training baseline.

Native (scipy) so it runs in CI with no heavy dependency; an Evidently HTML
report is available via the optional ``drift`` extra for a richer view.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. >0.2 is the usual 'material drift' threshold."""
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    e = np.histogram(expected, bins=qs)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=qs)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 psi_threshold: float = 0.2) -> dict:
    num = reference.select_dtypes(include=[np.number]).columns
    rows, drifted = [], 0
    for col in num:
        if col not in current.columns:
            continue
        p = psi(reference[col].values, current[col].values)
        ks = ks_2samp(reference[col].dropna(), current[col].dropna())
        flag = bool(p > psi_threshold or ks.pvalue < 0.01)
        drifted += flag
        rows.append({"feature": col, "psi": round(p, 4),
                     "ks_pvalue": round(float(ks.pvalue), 4), "drift": flag})
    return {"n_features": len(rows), "n_drifted": drifted,
            "share_drifted": round(drifted / max(len(rows), 1), 3),
            "dataset_drift": drifted / max(len(rows), 1) > 0.3,
            "features": rows}


def simulate_shift(df: pd.DataFrame, cols=("MonthlyCharges", "tenure"), scale=1.4) -> pd.DataFrame:
    """Inject a covariate shift, for tests and demos of the monitor firing."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c] * scale
    return out


def evidently_report(reference, current, path="reports/drift.html"):  # pragma: no cover
    """Optional richer report (needs the `drift` extra: pip install -e '.[drift]')."""
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=reference, current_data=current)
    rep.save_html(path)
    return path
