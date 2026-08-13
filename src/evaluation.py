import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score


def evaluate(model: BaseEstimator, x_val: pd.DataFrame, y_val: pd.Series):

    anomaly_score = -model.score_samples(x_val)

    pr_auc = average_precision_score(y_val, anomaly_score)

    return pr_auc
