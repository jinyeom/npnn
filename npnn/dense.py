import numpy as np

from .module import Module


__all__ = ["Dense", "Plastic"]


class Dense(Module):
    """
    A `Module` that applies a affine transformation to the incoming data.

    Parameters:
        in_dim (int): Size of the input feature vector.
        out_dim (int): Size of the output feature vector.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.new_param("weight", (in_dim, out_dim))
        self.new_param("bias", (out_dim,))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias


class Plastic(Module):
    """
    A nonlinear dense layer with Hebbian plasticity.

    Parameters:
        in_dim (int): Size of the input feature vector.
        out_dim (int): Size of the output feature vector.
        eta (float): Learning rate of the Hebbian plasticity.
        activ (Module): Nonlinear activation function.
    """

    def __init__(self, in_dim: int, out_dim: int, eta: float, activ: Module) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eta = eta
        self.new_param("weight", (in_dim, out_dim))
        self.new_param("hebb_coef", (in_dim, out_dim))
        self.new_param("bias", (out_dim,))
        self.new_module("activ", activ)

    def reset(self) -> None:
        self.hebb = np.zeros((self.in_dim, self.out_dim), dtype=self.dtype)
        return None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        weight = self.weight + self.hebb_coef * self.hebb
        y = self.activ(x @ weight + self.bias)
        delta = self.eta * np.outer(x, y)
        self.hebb = (1 - self.eta) * self.hebb + delta
        return y
