from unittest import TestCase
import numpy as np

from npnn import MemoryTape


class TestMemoryTape(TestCase):
    def test_memory_tape__write(self) -> None:
        mt = MemoryTape(10)
        data = np.random.rand(10).astype(np.float32)
        forget = np.zeros(10)
        mt.write(data, forget)

        self.assertTrue(np.array_equal(mt[0], data))
        self.assertEqual(len(mt), 1)

    def test_memory_tape__write_forget(self) -> None:
        mt = MemoryTape(10)

        data1 = np.random.rand(10).astype(np.float32)
        forget = np.zeros(10)
        mt.write(data1, forget)

        data2 = np.random.rand(10).astype(np.float32)
        forget = np.full_like(forget, 0.5)
        mt.write(data2, forget)

        self.assertTrue(np.array_equal(mt[0], (data1 + data2) / 2))
        self.assertEqual(len(mt), 1)

    def test_memory_tape__content_jump(self) -> None:
        mt = MemoryTape(10)
        data = np.random.rand(4, 10).astype(np.float32)
        mt._buffer = np.array(data)

        mt.content_jump(np.array([1]), data[2])
        self.assertEqual(mt._head, 2)
        self.assertTrue(np.array_equal(data[2], mt.read()))

    def test_memory_tape__shift(self) -> None:
        mt = MemoryTape(10)

        mt.shift(np.array([0, 1, 0]))
        self.assertEqual(mt._head, 0)

        mt.shift(np.array([0, 2, 5]))
        self.assertEqual(mt._head, 1)
        self.assertEqual(len(mt), 2)

        mt.shift(np.array([1, 0, 0]))
        mt.shift(np.array([1, 0, 0]))
        self.assertEqual(mt._head, 1)
        self.assertEqual(len(mt), 4)
