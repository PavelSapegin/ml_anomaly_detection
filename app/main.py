import joblib
from fastapi import FastAPI, HTTPException

from src.model import load_model

from .config import config
from .inference import predict_batch
from .schemas import BatchRequest, BatchResponse

app = FastAPI()

preprocessing_pipeline = joblib.load(config.COLUMN_TRANSFORMER_PATH)
model = load_model(config.MODEL_WEIGHTS_PATH, config.INPUT_DIM, config.LATENT_DIM)

@app.post("/predict", response_model=BatchResponse)
def predict(request: BatchRequest):
    try:
        responses = predict_batch(request.transactions,
                                  model,
                                  preprocessing_pipeline,
                                  config.THRESHOLD)
    except (KeyError, ValueError) as e:
       raise HTTPException(status_code=400, detail=f"Preprocessing/inference error:\
                            {str(e)}")


    return BatchResponse(responses=responses)
