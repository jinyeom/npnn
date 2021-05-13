import unittest

import numpy as np

from npnn import Dense


class TestModule(unittest.TestCase):
    def test_dense_transcribe(self) -> None:
        in_dim = 3
        out_dim = 4
        dtype = np.float32

        module = Dense(in_dim, out_dim)
        params_copy = np.concatenate([p.ravel() for p in module.parameters])
        theta = np.random.rand(len(module)).astype(dtype)
        theta_rest = module.transcribe(theta)
        params = np.concatenate([p.ravel() for p in module.parameters])

        self.assertFalse(np.array_equal(params, params_copy))
        self.assertTrue(np.array_equal(params, theta))
        self.assertEqual(len(theta_rest), 0)

    def test_dense_transcribe_with_remainder(self) -> None:
        dtype = np.float32
        module = Dense(3, 4)
        params_copy = np.concatenate([p.ravel() for p in module.parameters])
        theta_len = len(module) + 100
        theta = np.random.rand(theta_len).astype(dtype)
        theta_rest = module.transcribe(theta)
        params = np.concatenate([p.ravel() for p in module.parameters])

        self.assertFalse(np.array_equal(params, params_copy))
        self.assertTrue(np.array_equal(params, theta[: len(module)]))
        self.assertTrue(np.array_equal(theta[-100:], theta_rest))

    def test_dense_forward_with_zeros(self) -> None:
        in_dim = 3
        out_dim = 5
        dtype = np.float32

        module = Dense(in_dim, out_dim)
        theta = np.zeros(len(module), dtype=dtype)
        module.transcribe(theta)

        x = np.random.rand(3).astype(dtype)
        y = module(x)
        y_targ = np.zeros(out_dim)

        self.assertEqual(len(y), out_dim)
        self.assertTrue(np.array_equal(y, y_targ))

    def test_dense_forward_with_ones(self) -> None:
        in_dim = 3
        out_dim = 5
        dtype = np.float32

        module = Dense(in_dim, out_dim)
        theta = np.ones(len(module), dtype=dtype)
        module.transcribe(theta)

        x = np.random.rand(3).astype(dtype)
        y = module(x)
        y_targ = np.array(out_dim * [x.sum() + 1], dtype=dtype)

        self.assertEqual(len(y), out_dim)
        self.assertTrue(np.array_equal(y, y_targ))
