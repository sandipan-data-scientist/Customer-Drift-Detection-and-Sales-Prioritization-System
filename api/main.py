"""
FastAPI application for the Transaction Anomaly Detection service.

Endpoints:
    GET  /health                     - health check
    GET  /summary                    - executive summary stats
    GET  /customers                  - full tiered customer list (paginated)
    GET  /customers/{customer_id}    - single customer detail
    GET  /tiers                      - tier counts
    POST /predict                    - score a new customer record
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- app setup ---- #
app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="Flags at-risk business accounts and assigns prioritization tiers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- load artefacts at startup ---- #
BASE_DIR   = Path(__file__).parent.parent / "outputs"
MODEL_PATH = BASE_DIR / "isolation_forest.pkl"
SCALER_PATH= BASE_DIR / "scaler.pkl"
DATA_PATH  = BASE_DIR / "tiered_customers.csv"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df     = pd.read_csv(DATA_PATH)

MODEL_FEATURES = [
    "days_since_last_transaction",
    "total_lifetime_spend",
    "avg_monthly_spend",
    "spend_last_30_days",
    "drop_in_spend_vs_30_day_avg",
    "recency_score",
    "tx_frequency_per_month",
    "avg_basket_size",
    "active_months",
]


# ---- pydantic schemas ---- #
class CustomerFeatures(BaseModel):
    days_since_last_transaction: float
    total_lifetime_spend: float
    avg_monthly_spend: float
    spend_last_30_days: float
    drop_in_spend_vs_30_day_avg: float
    recency_score: float
    tx_frequency_per_month: float
    avg_basket_size: float
    active_months: float


class PredictionResponse(BaseModel):
    anomaly_flag: int          # -1 anomaly, +1 normal
    anomaly_score: float
    is_at_risk: bool
    recommended_tier: str


# ---- helper ---- #
def _assign_tier(score: float, lifetime_spend: float) -> str:
    value_threshold = df["total_lifetime_spend"].quantile(0.75)
    risk_threshold  = df.loc[df["anomaly_flag"] == -1, "anomaly_score"].median()

    high_value = lifetime_spend >= value_threshold
    high_risk  = score < risk_threshold

    if high_risk and high_value:
        return "Tier 1: High Risk / High Value"
    elif not high_risk and high_value:
        return "Tier 2: Low Risk / High Value"
    elif high_risk and not high_value:
        return "Tier 3: High Risk / Low Value"
    else:
        return "Tier 4: Low Risk / Low Value"


# ---- endpoints ---- #
@app.get("/health")
def health():
    return {"status": "ok", "records_loaded": len(df)}


@app.get("/summary")
def summary():
    flagged      = df[df["anomaly_flag"] == -1]
    tier_counts  = df["priority_tier"].value_counts().to_dict()
    at_risk_pct  = round(len(flagged) / len(df) * 100, 2)
    t1_value     = float(df[df["priority_tier"]=="Tier 1: High Risk / High Value"]["total_lifetime_spend"].sum())

    return {
        "total_customers"    : len(df),
        "at_risk_count"      : len(flagged),
        "at_risk_pct"        : at_risk_pct,
        "tier_counts"        : tier_counts,
        "tier1_combined_value": t1_value,
    }


@app.get("/customers")
def get_customers(
    tier: Optional[str] = Query(None, description="Filter by tier name"),
    limit: int = Query(50, le=500),
    offset: int = Query(0)
):
    result = df.copy()
    if tier:
        result = result[result["priority_tier"].str.lower().str.contains(tier.lower())]
    result = result.sort_values("anomaly_score")
    total  = len(result)
    result = result.iloc[offset: offset + limit]
    return {
        "total"   : total,
        "limit"   : limit,
        "offset"  : offset,
        "data"    : result.fillna(0).to_dict(orient="records"),
    }


@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):
    row = df[df["CustomerID"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"CustomerID {customer_id} not found.")
    return row.fillna(0).iloc[0].to_dict()


@app.get("/tiers")
def get_tiers():
    return df["priority_tier"].value_counts().to_dict()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    X = pd.DataFrame([features.dict()])[MODEL_FEATURES]
    X_scaled = scaler.transform(X)

    flag  = int(model.predict(X_scaled)[0])
    score = float(model.decision_function(X_scaled)[0])

    is_at_risk = flag == -1
    tier = _assign_tier(score, features.total_lifetime_spend) if is_at_risk else "No Action"

    return PredictionResponse(
        anomaly_flag=flag,
        anomaly_score=round(score, 6),
        is_at_risk=is_at_risk,
        recommended_tier=tier,
    )