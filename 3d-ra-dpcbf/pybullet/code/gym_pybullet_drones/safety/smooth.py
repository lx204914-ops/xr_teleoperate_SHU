"""Numerically stable smooth positive-part helpers."""

import numpy as np


def softplus(value, beta: float = 10.0):
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    scaled = beta * np.asarray(value, dtype=float)
    result = (
        np.maximum(scaled, 0.0)
        + np.log1p(np.exp(-np.abs(scaled)))
    ) / beta
    return float(result) if result.ndim == 0 else result


def sigmoid(value, beta: float = 10.0):
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    scaled = beta * np.asarray(value, dtype=float)
    result = np.empty_like(scaled, dtype=float)
    positive = scaled >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-scaled[positive]))
    exp_value = np.exp(scaled[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return float(result) if result.ndim == 0 else result

