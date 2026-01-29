from pathlib import Path
from typing import List

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pydantic import ConfigDict


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / ".." / "training" / "churn_model.h5").resolve()
PREPROCESS_PATH = (BASE_DIR / ".." / "training" / "preprocess_meta.json").resolve()
PREDICTIONS_DB = (BASE_DIR / ".." / "data" / "predictions.db").resolve()

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


class UserFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    user_id: str = Field(alias="customerID")
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: str | float = Field(default=0)


class PredictionResponse(BaseModel):
    user_id: str
    churn_probability: float
    risk_level: str


app = FastAPI(title="Churn Prediction API")


@app.on_event("startup")
def load_artifacts() -> None:
    try:
        app.state.model = tf.keras.models.load_model(MODEL_PATH)
        app.state.preprocess_meta = json.loads(PREPROCESS_PATH.read_text())
        PREDICTIONS_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(PREDICTIONS_DB)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_id TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        raise RuntimeError(f"Failed to load model or preprocessing metadata: {exc}") from exc


def risk_level_from_prob(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.6:
        return "medium"
    return "high"


def transform_with_meta(df: pd.DataFrame, meta: dict) -> np.ndarray:
    mean = pd.Series(meta["numeric_mean"])
    std = pd.Series(meta["numeric_std"]).replace(0, 1)
    df[meta["numeric_columns"]] = (df[meta["numeric_columns"]] - mean) / std
    encoded = pd.get_dummies(
        df[CATEGORICAL_COLUMNS + meta["numeric_columns"]], drop_first=False
    )
    encoded = encoded.reindex(columns=meta["feature_columns"], fill_value=0)
    return encoded.values.astype(float)


def preprocess_users(users: List[UserFeatures]) -> np.ndarray:
    rows = []
    for user in users:
        row = user.model_dump(by_alias=True)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    return transform_with_meta(df, app.state.preprocess_meta)


def store_predictions(results: List[PredictionResponse], payloads: List[dict]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (now, res.user_id, res.churn_probability, res.risk_level, json.dumps(payload))
        for res, payload in zip(results, payloads)
    ]
    conn = sqlite3.connect(PREDICTIONS_DB)
    conn.executemany(
        """
        INSERT INTO predictions (created_at, user_id, churn_probability, risk_level, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def parse_range(range_str: str) -> timedelta:
    value = int(range_str[:-1])
    unit = range_str[-1]
    if unit == "d":
        return timedelta(days=value)
    if unit == "h":
        return timedelta(hours=value)
    raise ValueError("range must end with 'd' or 'h', e.g. 7d or 24h")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: UserFeatures) -> PredictionResponse:
    try:
        X = preprocess_users([features])
        prob = float(app.state.model.predict(X, verbose=0)[0][0])
        result = PredictionResponse(
            user_id=features.user_id,
            churn_probability=round(prob, 4),
            risk_level=risk_level_from_prob(prob),
        )
        store_predictions([result], [features.model_dump(by_alias=True)])
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/batch_predict", response_model=List[PredictionResponse])
def batch_predict(payload: object = Body(...)) -> List[PredictionResponse]:
    if payload is None:
        raise HTTPException(status_code=400, detail="Empty batch")

    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        if "payload" in payload:
            raw_items = payload["payload"]
        elif "items" in payload:
            raw_items = payload["items"]
        else:
            raise HTTPException(status_code=400, detail="Expected list or payload/items key")
    else:
        raise HTTPException(status_code=400, detail="Invalid request body")

    if not raw_items:
        raise HTTPException(status_code=400, detail="Empty batch")

    try:
        users = [UserFeatures.model_validate(item) for item in raw_items]
        X = preprocess_users(users)
        probs = app.state.model.predict(X, verbose=0).reshape(-1)
        responses = []
        for user, prob in zip(users, probs):
            responses.append(
                PredictionResponse(
                    user_id=user.user_id,
                    churn_probability=round(float(prob), 4),
                    risk_level=risk_level_from_prob(float(prob)),
                )
            )
        store_predictions(responses, [u.model_dump(by_alias=True) for u in users])
        return responses
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predictions")
def get_predictions(range: str = Query(default="7d")):
    try:
        delta = parse_range(range)
        cutoff = datetime.now(timezone.utc) - delta
        conn = sqlite3.connect(PREDICTIONS_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT created_at, user_id, churn_probability, risk_level
            FROM predictions
            WHERE created_at >= ?
            ORDER BY created_at DESC
            """,
            (cutoff.isoformat(),),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
