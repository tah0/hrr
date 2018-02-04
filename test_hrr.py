import unittest
import hrr

from itertools import chain
from functools import reduce

import numpy as np  # numpy.testing for comparing
# hrr output with corresponding numpy functions

# later, the main package will be the numpy f'n versions


def _np_power(A=np.array, B=int):
    return np.fft.ifft(np.power(np.fft.fft(A), B))


def _np_dot(A=np.array, B=np.array):
    return np.dot(A, B)


def _np_circ_convolve(A=np.array, B=np.array):
    return np.fft.ifft(np.multiply(np.fft.fft(A), np.fft.fft(B)))


def _np_circ_correlate(A=np.array, B=np.array):
    return np.fft.ifft(np.fft.fft(A) * np.fft.fft(B).conj()).real


def _np_getClosest(item=np.array, memoryDict=dict, howMany=3,
                   likenessFn=lambda x, y: np.dot(x, y)):
    dists = {key: likenessFn(item, value) for key, value in memoryDict.items()}
    sortedDists = sorted(dists.keys(),
                         key=(lambda key: dists[key]), reverse=True)
    return {k: round(dists[k], 5) for k in
            sortedDists[:min(howMany, len(memoryDict))]}


def _np_make_Sequence_ab(seq=list, alpha=None, beta=None):
    alpha_elems = (p[0] * p[1] for p in zip(alpha, seq))
    beta_elems = (p[0] * p[1]
                  for p in zip(beta, (seq[i] * seq[i + 1]
                                      for i in range(len(seq) - 1))))
    return sum(chain(alpha_elems, beta_elems))  # chain just joins generators


def _np_make_Sequence_triangle(seq=list):
    return seq[0] + sum((reduce(lambda x, y: x.encode(y), seq[:e])
                         for e in range(2, len(seq) + 1)))


def _np_make_Sequence_positional(seq=list, p=None):
    return sum((p ** (i + 1)).encode(seq[i]) for i in range(0, len(seq)))


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

    def test_correlate(self):
        np_vec_A = np.random.normal(0, 1 / 512, 512)
        np_vec_B = np.random.normal(0, 1 / 512, 512)
        hrr_vec_A = hrr.HRR(np_vec_A)
        hrr_vec_B = hrr.HRR(np_vec_B)
        hrr_res = hrr_vec_A.decode(hrr_vec_B)
        np_res = np.fft.ifft(np.fft.fft(np_vec_A) *
                             np.fft.fft(np_vec_B).conj()).real
        np.testing.assert_allclose(hrr_res.values, np_res)


class TestHRRStructures(unittest.TestCase):

    def test_HRR_Sequence_ab(self):
        # TODO: doesn't yet determine the end of a sequence naturally
        # set up for making and decoding a Sequence
        seq_order = ['a', 'b', 'c']
        # autoassociative memories
        np_memory = {i: np.random.normal(0, 1 / 512, 512) for i in seq_order}
        hrr_memory = {i: hrr.HRR(np_memory[i]) for i in seq_order}
        alpha = [x / len(seq_order) for x in
                 range(1, len(seq_order) + 1)][::-1]
        beta = [x / (len(seq_order) - 1) for x in
                range(1, len(seq_order))][::-1]
        np_seq_ab = list(_np_make_Sequence_ab([np_memory[i] for i in seq_order],
                                               alpha, beta))[0]
        hrr_seq_ab = list(hrr.makeSequence([hrr_memory[i] for i in seq_order],
                                           encoding='ab',
                                           alpha=alpha,
                                           beta=beta))[0]
        # test that the Seq reps are the same
        np.testing.assert_allclose(hrr_seq_ab, np_seq_ab)
        # for the decoding & comparison loop below, set out starting rep as the
        # full encoded sequence
        hrr_curr = hrr_seq_ab
        np_curr = np_seq_ab
        # TODO: do I need to include alpha, beta terms when factoring out???
        for i in range(len(seq_order)):
            # decode the Seq reps per item, testing that all outputs are equal
            # retrieve strongest component and compare method dot products
            hrr_item = list(hrr.getClosest(hrr_curr, hrr_memory, howMany=1))[0]
            np_item = list(_np_getClosest(np_curr, np_memory, howMany=1))[0]
            np.testing.assert_allclose(hrr_item, np_item)
            # "correlate out" the cleaned up component from the seq rep
            # and test that  the resulting traces are equal
            hrr_curr = hrr_curr.decode(hrr_memory[hrr_item.keys])
            np_curr = _np_circ_correlate(np_curr, np_memory[np_item.keys])
            np.testing.assert_allclose(hrr_curr, np_curr)

    def test_HRR_Sequence_triangle(self):
        # TODO: doesn't yet determine the end of a sequence naturally
        # set up for making and decoding a Sequence
        seq_order = ['a', 'b', 'c']
        # autoassociative memories
        np_memory = {i: np.random.normal(0, 1 / 512, 512) for i in seq_order}
        hrr_memory = {i: hrr.HRR(np_memory[i]) for i in seq_order}
        # test that the Seq reps are the same
        np_seq_ab = list(_np_make_Sequence_triangle([np_memory[i] for i in seq_order]))[0]
        hrr_seq_ab = list(hrr.makeSequence([hrr_memory[i] for i in seq_order],
                                           encoding='triangle'))[0]
        np.testing.assert_allclose(hrr_seq_ab, np_seq_ab)
        # for the decoding & comparison loop below, set out starting rep as the
        # full encoded sequence
        hrr_curr = hrr_seq_ab
        np_curr = np_seq_ab
        hrr_factor = []
        np_factor = []
        for i in range(len(seq_order)):
            # decode the Seq reps per item, testing that all outputs are equal
            # retrieve strongest component and compare method dot products
            hrr_item = list(hrr.getClosest(hrr_curr, hrr_memory, howMany=1))[0]
            np_item = list(_np_getClosest(np_curr, np_memory, howMany=1))[0]
            np.testing.assert_allclose(hrr_item, np_item)
            # "correlate out" the cleaned up component from the seq rep
            # and test that  the resulting traces are equal
            if hrr_factor and np_factor:
                hrr_factor = hrr_factor.encode(hrr_item)
                np_factor = _np_circ_convolve(np_factor, np_item)
            else:
                hrr_factor = hrr_item
                np_factor = np_item
            hrr_curr = hrr_curr.decode(hrr_factor)
            np_curr = _np_circ_correlate(np_curr, np_factor)
            np.testing.assert_allclose(hrr_curr, np_curr)

    def test_HRR_Sequence_positional(self):
        # TODO: doesn't yet determine the end of a sequence naturally
        # set up for making and decoding a Sequence
        seq_order = ['a', 'b', 'c']
        # autoassociative memories
        np_memory = {i: np.random.normal(0, 1 / 512, 512) for i in seq_order}
        hrr_memory = {i: hrr.HRR(np_memory[i]) for i in seq_order}
        p_np = np.random.normal(0, 1 / 512, 512)
        p_hrr = hrr.HRR(p_np)
        np_seq_ab = list(_np_make_Sequence_positional([np_memory[i] for i in seq_order],
                                               p=p_np))[0]
        hrr_seq_ab = list(hrr.makeSequence([hrr_memory[i] for i in seq_order],
                                           encoding='positional',
                                           p=p_hrr))[0]
        # test that the Seq reps are the same
        np.testing.assert_allclose(hrr_seq_ab, np_seq_ab)
        # for the decoding & comparison loop below, set out starting rep as the
        # full encoded sequence
        hrr_curr = hrr_seq_ab
        np_curr = np_seq_ab
        # TODO: do I need to include p terms when factoring out???
        for i in range(len(seq_order)):
            # decode the Seq reps per item, testing that all outputs are equal
            # retrieve strongest component and compare method dot products
            hrr_item = list(hrr.getClosest(hrr_curr, hrr_memory, howMany=1))[0]
            np_item = list(_np_getClosest(np_curr, np_memory, howMany=1))[0]
            np.testing.assert_allclose(hrr_item, np_item)
            # "correlate out" the cleaned up component from the seq rep
            # and test that  the resulting traces are equal
            hrr_curr = hrr_seq_ab.decode(hrr_memory[hrr_item.keys])
            np_curr = _np_circ_correlate(np_curr, np_memory[np_item.keys])
            np.testing.assert_allclose(hrr_curr, np_curr)

    def test_HRR_Stack(self):
        pass


if __name__ == '__main__':
    unittest.main()