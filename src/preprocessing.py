import numpy as np
import pandas as pd


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["Hour"] = (df["Time"] % 86_400) / 3600
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df = df.drop(columns=["Time", "Hour"])
    df["Amount_log"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Amount"])

    return df
