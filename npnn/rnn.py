import numpy as np
from npne.nn import Module
from npne.nn.functional import sigmoid, tanh


__all__ = ["RNN", "LSTM"]


class RNN(Module):
    """
    A vanilla recurrent neural network cell `Module` with hyperbolic tangent
    nonlinearity.

    Parameters:
        in_dim (int): Size of the input feature vector.
        hid_dim (int): Size of the hidden state vector.
    """

    def __init__(self, in_dim: int, hid_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.new_param("init_h", (hid_dim,))
        self.new_param("weight", (in_dim + hid_dim, hid_dim))
        self.new_param("bias", (hid_dim,))

    def reset(self) -> None:
        self.h = np.array(self.init_h)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xh = np.concatenate([x, self.h])
        self.h = tanh(xh @ self.weight + self.bias)
        return self.h


class LSTM(Module):
    """
    A long short-term memory (LSTM) cell.

    Parameters:
        in_dim (int): Size of the input feature vector.
        hid_dim (int): Size of the hidden state vector.
    """

    def __init__(self, in_dim: int, hid_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.new_param("init_c", (hid_dim,))
        self.new_param("init_h", (hid_dim,))
        self.new_param("weight", (in_dim + hid_dim, 4 * hid_dim))
        self.new_param("bias", (4 * hid_dim,))

    def reset(self) -> None:
        self.c = np.array(self.init_c)
        self.h = np.array(self.init_h)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xh = np.concatenate([x, self.h])
        z = xh @ self.weight + self.bias
        fioc = np.split(z, 4)
        f = sigmoid(fioc[0])
        i = sigmoid(fioc[1])
        o = sigmoid(fioc[2])
        c = tanh(fioc[3])
        self.c = f * self.c + i * c
        self.h = o * tanh(self.c)
        return self.h
