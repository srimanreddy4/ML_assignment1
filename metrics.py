from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    return np.sum(y_hat == y)/y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    # Precision = True Positives / (True Positives + False Positive)
    assert y_hat.size == y.size
    cls_series = pd.Series([cls] * len(y_hat))
    true_positive = np.sum((y == y_hat) & (y == cls_series))
    true_predicted = np.sum(y_hat == cls_series)
    prec = float(true_positive / true_predicted) if true_predicted > 0 else 0.0
    return prec


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    # Recall = True Positives / (True Positives + False Negatives)
    assert y_hat.size == y.size
    cls_series = pd.Series([cls] * len(y_hat))
    true_positive = np.sum((y == y_hat) & (y == cls_series))
    true_actual = np.sum(y == cls_series)
    rec = float(true_positive / true_actual) if true_actual > 0 else 0.0
    return rec


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Calculates the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    return np.sqrt(np.mean((y_hat - y)**2))

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Calculates the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    return np.mean(np.abs(y_hat - y))