import unittest
import hrr

import numpy as np  # we'll compare our results to numpy output
# later, the main package will be the numpy versions


class TestVector(unittest.TestCase):

    def test_pow(self):
        np_vec = np.random.normal(0, 1 / 512, 512)
        hrr_vec = hrr.Vector(np_vec)
        hrr_res = hrr_vec**2
        np_res = np.fft.ifft(np.power(np.fft.fft(np_vec), 2))
        np.testing.assert_allclose(np.array(hrr_res.values), np_res)

    def test_mul(self):
        np_vec_A = np.random.normal(0, 1 / 512, 512)
        np_vec_B = np.random.normal(0, 1 / 512, 512)
        hrr_vec_A = hrr.Vector(np_vec_A)
        hrr_vec_B = hrr.Vector(np_vec_B)
        np.testing.assert_allclose(hrr_vec_A * hrr_vec_B,
                                   np.dot(np_vec_A, np_vec_B))

    def test_truediv(self):
        np_vec = np.random.normal(0, 1 / 512, 512)
        hrr_vec = hrr.Vector(np_vec)
        divisor = np.random.uniform(0, 1)
        hrr_res = hrr_vec / divisor
        np.testing.assert_allclose(np.array(hrr_res.values),
                                   np.true_divide(np_vec, divisor))

    def test_floordiv(self):
        np_vec = np.random.normal(0, 1 / 512, 512)
        hrr_vec = hrr.Vector(np_vec)
        divisor = np.random.uniform(0, 1)
        hrr_res = hrr_vec // divisor
        np.testing.assert_allclose(np.array(hrr_res.values),
                                   np.floor_divide(np_vec, divisor))


class TestHRR(unittest.TestCase):

    def test_convolve(self):
        np_vec_A = np.random.normal(0, 1 / 512, 512)
        np_vec_B = np.random.normal(0, 1 / 512, 512)
        hrr_vec_A = hrr.HRR(np_vec_A)
        hrr_vec_B = hrr.HRR(np_vec_B)
        hrr_res = hrr_vec_A.encode(hrr_vec_B)
        np_res = np.fft.ifft(np.multiply(
                             np.fft.fft(np_vec_A), np.fft.fft(np_vec_B)))
        np.testing.assert_allclose(hrr_res.values, np_res)


if __name__ == '__main__':
    unittest.main()