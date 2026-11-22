"""Feature engineering + a leakage-safe sklearn preprocessor.

A PySpark equivalent of the derived features lives in ``spark_features`` so the
same logic scales to a cluster for large datasets; the sklearn path is used for
training and serving.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

TARGET = "Churn"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features. Pure function of a row, so train/serve cannot diverge."""
    df = df.copy()
    if {"TotalCharges", "tenure"}.issubset(df.columns):
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    if {"MonthlyCharges", "TotalCharges"}.issubset(df.columns):
        df["charges_gap"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    service_cols = [c for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                "TechSupport", "StreamingTV", "StreamingMovies"] if c in df.columns]
    if service_cols:
        df["n_services"] = (df[service_cols] == "Yes").sum(axis=1)
    if "tenure" in df.columns:
        df["tenure_bucket"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 1000],
                                     labels=["0-6", "6-12", "12-24", "24-48", "48+"]).astype(str)
    return df


def split_xy(df: pd.DataFrame):
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)], remainder="drop")


def spark_features(spark_df):  # pragma: no cover - requires a Spark session
    """Same derived features in PySpark, for cluster-scale datasets."""
    from pyspark.sql import functions as F
    out = spark_df.withColumn("avg_monthly_spend",
                              F.col("TotalCharges") / F.greatest(F.col("tenure"), F.lit(1)))
    out = out.withColumn("charges_gap", F.col("MonthlyCharges") - F.col("avg_monthly_spend"))
    return out
