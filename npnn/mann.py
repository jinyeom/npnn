import numpy as np

__all__ = ["MemoryTape"]


class MemoryTape:
    """
    Infinite memory tape for the Neural Turing Machine.

    Parameters:
        mem_dim (int): Size of each memory vector.
    """

    def __init__(self, mem_dim: int, dtype: np.dtype = np.float32) -> None:
        self.mem_dim = mem_dim
        self.dtype = dtype
        self._buffer = np.zeros((1, mem_dim), dtype=dtype)
        self._head = 0

    def __len__(self) -> int:
        """
        Returns:
            The current memory buffer size.
        """
        return self._buffer.shape[0]

    def __getitem__(self, addr: int) -> np.ndarray:
        """
        Parameters:
            addr (int): Memory address (buffer index) to read from.

        Returns:
            A copy of the memory content at the argument location.
        """
        return np.array(self._buffer[addr])

    def write(self, data: np.ndarray, forget: np.ndarray) -> None:
        """
        Write `data` to the current address, while forgetting the old data.

        Parameters:
            data (np.ndarray): Incoming data to be written on the current address.
            forget (np.ndarray): Forget vector for blending old data with new data.
        """
        old_data = forget * self._buffer[self._head]
        new_data = (1 - forget) * data
        self._buffer[self._head] = old_data + new_data

    def content_jump(self, jump: np.ndarray, data: np.ndarray) -> None:
        """
        If the `jump` value is higher than 0.5, jump to a memory location that is the
        closest to `data` in Euclidean space.

        Parameters:
            jump (np.ndarray): A float that determines whether to perform content jump.
            data (np.ndarray): Incoming data vector that determines where to jump to.
        """
        if len(jump) != 1:
            raise ValueError("`jump` must be a one dimensional vector")
        if jump > 0.5:
            euc_dist = np.abs(self._buffer - data).sum(axis=1)
            self._head = np.argmin(euc_dist)

    def shift(self, shift_vec: np.ndarray) -> None:
        """
        Shift the pointer head up (argmax is 0), down (argmax is 2) or leave it
        unchanged (argmax is 1). If the pointer head goes out of bound, double
        the memory capacity in the corresponding direction.

        Parameters:
            shift_vec (np.ndarray): A vector that determines the shift behaviour based
                on the index of the largest element: shift left when `argmax(shift_vec)`
                is 0, shift right when it's 2, and do nothing otherwise.
        """
        if len(shift_vec) != 3:
            raise ValueError("`shift_vec` must be a three dimensional vector")
        shift = np.argmax(shift_vec) - 1
        self._head += shift
        if self._head < 0:
            new_buffer = np.zeros_like(self._buffer)
            self._buffer = np.concatenate([new_buffer, self._buffer], axis=0)
            self._head += len(new_buffer)
        elif self._head > len(self._buffer) - 1:
            new_buffer = np.zeros_like(self._buffer)
            self._buffer = np.concatenate([self._buffer, new_buffer], axis=0)

    def read(self) -> np.ndarray:
        """
        Read and return a copy of the data vector at the current address.

        Returns:
            A copy of the data vector at the current address.
        """
        return self[self._head]

    def __call__(
        self,
        data: np.ndarray,
        forget: np.ndarray,
        jump: np.ndarray,
        shift_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Write the incoming data to memory and read from a new location determined by the
        content jump and shift operations.

        Parameters:
            data (np.ndarray): Incoming data to be written on the current address.
            forget (np.ndarray): Forget vector for blending old data with new data.
            jump (np.ndarray): A float that determines whether to perform content jump.
            shift_vec (np.ndarray): A vector that determines the shift behaviour.

        Returns:
            Data vector at the next memory address.
        """
        self.write(data, forget)
        self.content_jump(jump, data)
        self.shift(shift_vec)
        return self.read()
