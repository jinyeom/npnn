from typing import Tuple, Optional

import numpy as np


__all__ = [
    "sigmoid",
    "tanh",
    "relu",
    "softmax",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies the element-wise sigmoid function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        Function output.
    """
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Applies the element-wise hyperbolic tangent function. Note that this function is a
    simple wrapper around `numpy.tanh()`.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        Function output.
    """
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Applies the element-wise rectified linear unit function.

    Parameters:
        x (np.ndarray): Input array.
    
    Returns:
        Function output.
    """
    return x * (x > 0)


def softmax(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Applies a softmax function along the argument dimension `axis`.

    Parameters:
        x (np.ndarray): Input array.
        axis (Optional[int]): A dimension along which softmax will be computed.
    
    Returns:
        Function output.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)
