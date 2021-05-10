from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np


class Module:
    """
    A base class for neural network modules, whose weights are assigned by transcribing
    a weight vector.
    """

    dtype: np.ndarray = np.float32

    def __init__(self) -> None:
        self._parameters = []
        self._children = []

    @property
    def parameters(self) -> List[np.ndarray]:
        """
        List of all parameters, including those in children modules.
        """
        parameters = self._parameters
        for child in self._children:
            parameters.extend(child.parameters)
        return parameters

    @property
    def blueprint(self) -> List[Tuple[int, ...]]:
        """
        List of shapes of all parameters.
        """
        return [p.shape for p in self.parameters]

    @property
    def children(self) -> List[Module]:
        """
        List of all children modules.
        """
        return self._children

    def __len__(self) -> int:
        """
        Computes the total number of parameters. Useful for creating a weight vector.

        Returns:
            Total number of parameters.
        """
        return sum(p.size for p in self.parameters)

    def new_param(self, name: str, shape: Sequence[int]) -> None:
        """
        Create and add a new named parameter given the argument shape.

        Parameters:
            name (str): Name of the new parameter.
            shape (Sequence[int]): Shape of the new parameter.
        """
        if hasattr(self, name):
            raise ValueError(f"invalid parameter name: {name}")
        param = np.empty(shape, dtype=self.dtype)
        setattr(self, name, param)
        self._parameters.append(param)

    def new_module(self, name: str, module: Module) -> None:
        """
        Add a new module as an attribute under the argument name.

        Parameters:
            name (str): Name of the new child module
            module (Module): Module to be added as a new child module.
        """
        if hasattr(self, name):
            raise ValueError(f"invalid module name: {name}")
        setattr(self, name, module)
        self._children.append(module)

    def transcribe(self, src: np.ndarray) -> np.ndarray:
        """
        Copy the data in the source weight vector into each parameter in the module, and
        return the remaining part of the vector, an empty array when its size is the
        same as that of the target module.

        Parameters:
            src (np.ndarray): Source weight vector to copy the data from.

        Returns:
            Remaining part of the source weight vector. An empty NumPy array when the
            source weight vector has the same size as the target module.
        """
        for dst in self.parameters:
            src_slice, src = src[: dst.size], src[dst.size :]
            np.copyto(dst, src_slice.reshape(dst.shape).astype(dst.dtype))
        return src

    def reset(self) -> None:
        """
        Reset all the hidden states. Optional, but useful for implementing RNNs.
        """
        pass

    def __call__(self) -> None:
        """
        Forward pass input arrays and return output arrays. Implement this method to
        inherit from the `Module` class.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
