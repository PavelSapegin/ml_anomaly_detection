from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request

from src.model import load_model

from .config import config
from .inference import predict_batch
from .schemas import BatchRequest, BatchResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.preprocessing_pipeline = joblib.load(config.COLUMN_TRANSFORMER_PATH)
    app.state.model = load_model(config.MODEL_WEIGHTS_PATH,
                                 config.INPUT_DIM,
                                 config.LATENT_DIM)

    yield

    app.state.preprocessing_pipeline = None
    app.state.model =None

app = FastAPI(lifespan=lifespan)



@app.post("/predict", response_model=BatchResponse)
def predict(payload: BatchRequest, request: Request):
    try:
        responses = predict_batch(
            payload.transactions,
            request.app.state.model,
            request.app.state.
            preprocessing_pipeline,
            config.THRESHOLD
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=
            "Preprocessing/inference error: "
            f"{e}",
        ) from e

    return BatchResponse(responses=responses)
