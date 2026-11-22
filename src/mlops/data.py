"""Data ingestion, cleaning and schema validation.

The schema validation is the point: real data arrives dirty (here, blank
``TotalCharges`` strings), and the pipeline must fail loudly on contract
violations rather than silently training on garbage.
"""
from __future__ import annotations
import pandas as pd
import pandera as pa
from pandera import Column, Check


def load_raw(source: str) -> pd.DataFrame:
    return pd.read_csv(source)


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Deterministic cleaning: coerce types, drop the ID, binarise the target."""
    df = df.copy()
    ds = cfg["dataset"]
    # Telco quirk: TotalCharges holds blank strings for brand-new customers.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    if ds["id_col"] in df.columns:
        df = df.drop(columns=[ds["id_col"]])
    tgt, pos = ds["target"], ds["positive_label"]
    if not pd.api.types.is_numeric_dtype(df[tgt]):
        df[tgt] = (df[tgt].astype("string") == pos).astype(int)
    return df


def validated_schema() -> pa.DataFrameSchema:
    """Post-clean contract. Extend per dataset; unknown columns are allowed."""
    return pa.DataFrameSchema(
        {
            "tenure": Column(int, Check.in_range(0, 200), nullable=False),
            "MonthlyCharges": Column(float, Check.ge(0), nullable=False),
            "TotalCharges": Column(float, Check.ge(0), nullable=False),
            "Churn": Column(int, Check.isin([0, 1]), nullable=False),
        },
        strict=False,
        coerce=True,
    )


def load_clean_validate(cfg: dict) -> pd.DataFrame:
    df = clean(load_raw(cfg["dataset"]["source"]), cfg)
    return validated_schema().validate(df)
