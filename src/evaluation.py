import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score


def evaluate(model: BaseEstimator, x_val: pd.DataFrame, y_val: pd.Series):

    anomaly_score = -model.score_samples(x_val)

    pr_auc = average_precision_score(y_val, anomaly_score)

    return pr_auc


def evaluate_autoencoder(model: nn.Module,
                        X_val: torch.Tensor,
                        y_val: pd.Series,
                        device: torch.device | str = "cpu") -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        X_val_tensor = X_val.to(device)
        decoded = model(X_val_tensor)
        reconstruction_error = torch.mean((X_val_tensor - decoded)**2,1).cpu().numpy()
        pr_auc = average_precision_score(y_val, reconstruction_error)

    return pr_auc, reconstruction_error
