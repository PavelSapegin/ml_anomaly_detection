from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: int | None
    Time: int = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


class PredictionResponse(BaseModel):
    transaction_id: int | None
    is_fraud: bool = Field(description="Flag - transaction is fraud")
    anomaly_score: float = Field(description="Autoencoder's anomaly score")
    threshold: float = Field(description="Threshold, which used for anomaly detection")


class BatchRequest(BaseModel):
    transactions: list[Transaction]


class BatchResponse(BaseModel):
    responses: list[PredictionResponse]
