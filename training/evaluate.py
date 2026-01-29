import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


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
TARGET_COLUMN = "Churn"


def stratified_split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for label in np.unique(y):
        label_idxs = np.where(y == label)[0]
        rng.shuffle(label_idxs)
        split = int(len(label_idxs) * (1 - test_size))
        train_idx.append(label_idxs[:split])
        val_idx.append(label_idxs[split:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df


def transform_with_meta(df: pd.DataFrame, meta: dict) -> np.ndarray:
    df = df.copy()
    mean = pd.Series(meta["numeric_mean"])
    std = pd.Series(meta["numeric_std"]).replace(0, 1)
    df[meta["numeric_columns"]] = (df[meta["numeric_columns"]] - mean) / std
    encoded = pd.get_dummies(
        df[CATEGORICAL_COLUMNS + meta["numeric_columns"]], drop_first=False
    )
    encoded = encoded.reindex(columns=meta["feature_columns"], fill_value=0)
    return encoded.values.astype(float)


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])

    if tps[-1] == 0 or fps[-1] == 0:
        return 0.0

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    return float(np.trapz(tpr, fpr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate churn prediction model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "churn_dataset.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "churn_model.h5",
    )
    parser.add_argument(
        "--preprocess",
        type=Path,
        default=Path(__file__).resolve().parent / "preprocess_meta.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = prepare_dataframe(df)

    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).values
    features_df = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]

    meta = json.loads(args.preprocess.read_text())
    X_all = transform_with_meta(features_df, meta)

    _, X_val, _, y_val = stratified_split(X_all, y, test_size=0.2, seed=args.seed)

    model = tf.keras.models.load_model(args.model)
    probs = model.predict(X_val, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_val, probs)
    precision, recall = precision_recall(y_val, preds)

    print("Evaluation on validation split")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
