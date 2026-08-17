from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    THRESHOLD: float = 3.53
    INPUT_DIM: int = 31
    LATENT_DIM: int = 8
    EXPECTED_RAW_COLUMNS: list[str] = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
    ]

    MODELS_DIR: Path = BASE_DIR / "models"

    @property
    def COLUMN_TRANSFORMER_PATH(self) -> Path:
        return self.MODELS_DIR / "preprocessing_pipeline.joblib"

    @property
    def MODEL_WEIGHTS_PATH(self) -> Path:
        return self.MODELS_DIR / "autoencoder.pth"

    class Config:
        env_file = ".env"


config = Settings()
