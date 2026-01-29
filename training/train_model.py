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


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


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


def compute_class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = {}
    for cls in classes:
        count = np.sum(y == cls)
        weights[int(cls)] = len(y) / (len(classes) * count)
    return weights


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df


def fit_preprocessor(df: pd.DataFrame):
    df = df.copy()
    numeric_mean = df[NUMERIC_COLUMNS].mean()
    numeric_std = df[NUMERIC_COLUMNS].std().replace(0, 1)
    df[NUMERIC_COLUMNS] = (df[NUMERIC_COLUMNS] - numeric_mean) / numeric_std
    encoded = pd.get_dummies(df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS], drop_first=False)
    feature_columns = encoded.columns.tolist()
    meta = {
        "feature_columns": feature_columns,
        "numeric_columns": NUMERIC_COLUMNS,
        "numeric_mean": numeric_mean.to_dict(),
        "numeric_std": numeric_std.to_dict(),
    }
    return encoded.values.astype(float), meta


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "churn_dataset.csv",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path(__file__).resolve().parent / "churn_model.h5",
    )
    parser.add_argument(
        "--preprocess-out",
        type=Path,
        default=Path(__file__).resolve().parent / "preprocess_meta.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = pd.read_csv(args.data)
    df = prepare_dataframe(df)

    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).values
    features_df = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]

    X_all, meta = fit_preprocessor(features_df)

    X_train, X_val, y_train, y_val = stratified_split(
        X_all, y, test_size=0.2, seed=args.seed
    )

    class_weight_dict = compute_class_weights(y_train)

    model = build_model(input_dim=X_train.shape[1])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=128,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1,
    )

    eval_results = model.evaluate(X_val, y_val, verbose=0)
    metrics = dict(zip(model.metrics_names, eval_results))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    args.preprocess_out.write_text(json.dumps(meta, indent=2))

    print("Training complete")
    print(f"Validation metrics: {metrics}")
    print(f"Saved model to {args.model_out}")
    print(f"Saved preprocessing metadata to {args.preprocess_out}")


if __name__ == "__main__":
    main()
