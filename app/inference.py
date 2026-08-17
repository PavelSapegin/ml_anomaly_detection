import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from src.model import Autoencoder

from .schemas import PredictionResponse, Transaction


def predict_batch(
    transactions: list[Transaction],
    model: Autoencoder,
    preprocessing_pipeline: Pipeline,
    threshold: float,
) -> list[PredictionResponse]:

    features_df = pd.DataFrame([t.model_dump() for t in transactions])
    ids = ids = (
        features_df.pop("transaction_id")
        if "transaction_id" in features_df.columns
        else pd.Series([None] * len(transactions))
    )

    X = preprocessing_pipeline.transform(features_df)
    X_tensor = torch.tensor(
        X.values if hasattr(X, "values") else X, dtype=torch.float32
    )

    with torch.no_grad():
        decoded = model(X_tensor)
        errors = torch.mean((X_tensor - decoded) ** 2, dim=1).numpy()

    return [
        PredictionResponse(
            transaction_id=int(id_) if pd.notna(id_) and id_ is not None else None,
            is_fraud=bool(error > threshold),
            anomaly_score=float(error),
            threshold=threshold,
        )
        for id_, error in zip(ids, errors)
    ]
