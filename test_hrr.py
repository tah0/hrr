#! /usr/bin/env python3

import unittest
import hrr

from itertools import chain, islice, combinations
from functools import reduce

import numpy as np  # for comparing hrr output to

# output of corresponding numpy functions, using np.testing

# later, the main package will use the numpy f'n versions (and test using the
# current functions?)
from scipy import spatial


def _np_power(A=np.ndarray, B=int):
    return np.fft.ifft(np.power(np.fft.fft(A), B))


def _np_dot(A=np.ndarray, B=np.ndarray):
    return np.dot(A, B)


def _np_circ_convolve(A=np.ndarray, B=np.ndarray):
    return np.fft.ifft(np.multiply(np.fft.fft(A), np.fft.fft(B)))


def _np_circ_decode(A=np.ndarray, B=np.ndarray):
    return np.fft.ifft(np.multiply(np.fft.fft(A), 
                                   np.fft.fft(np.array([B[-i % len(B)] for i in range(len(B))]))
                                  )
                      )


def _np_circ_correlate(A=np.ndarray, B=np.ndarray):
    return np.fft.ifft(np.fft.fft(A) * np.fft.fft(B).conj()).real


def _np_getClosest(item=np.ndarray, memoryDict=dict, howMany=1,
                   similarityFn=lambda x, y: np.dot(x, y)):
    dists = {key: similarityFn(item, value)
             for key, value in memoryDict.items()}
    sortedDists = sorted(dists.keys(),
                         key=(lambda key: dists[key]), reverse=True)
    return sortedDists[:howMany]
    # return {k: round(dists[k], 5) for k in
    #         sortedDists[:min(howMany, len(memoryDict))]}


def _np_make_Sequence_ab(seq=list, alpha=None, beta=None):
    alpha_elems = (p[0] * p[1] for p in zip(seq, alpha))
    beta_elems = (p[0] * p[1] for p in
                  zip((_np_circ_convolve(seq[i], seq[i + 1])
                       for i in range(len(seq) - 1)), beta))
    return sum(chain(alpha_elems, beta_elems))  # chain joins generators


def _np_make_Sequence_triangle(seq=list):
    return seq[0] + sum((reduce(lambda x, y: _np_circ_convolve(x, y), seq[:e])
                         for e in range(2, len(seq) + 1)))


def _np_make_Sequence_positional(seq=list, p=None):
    return sum(_np_circ_convolve(_np_power(p, i + 1), seq[i])
               for i in range(0, len(seq)))


_np_make_Stack = _np_make_Sequence_positional


def _np_push(stack: np.ndarray, item: np.ndarray, p: np.ndarray):
    stack = item + _np_circ_convolve(p, stack)


def _np_stackTop(stack: np.ndarray, memory: dict,
                 similarityFn=lambda x, y: x * y) -> np.ndarray:
    return _np_getClosest(stack, memory, howMany=1,
                          similarityFn=similarityFn)[0]


def _np_stackPop(stack: np.ndarray, memory: dict,
                 p: np.ndarray, similarityFn=lambda x, y: x * y) -> np.ndarray:
    out = memory[_np_stackTop(stack, memory, similarityFn)]
    stack = _np_circ_decode((stack - out), p)
    return out


def _np_makeFrame(frame_label: np.ndarray, *args) -> np.ndarray:
    for arg in args:
        for elem in arg:
            if type(elem) != np.ndarray:
                return _np_makeFrame(*elem)
    return sum([frame_label] + [_np_circ_convolve(t[0], t[1]) for t in args])


class TestVector(unittest.TestCase):

    def test_pow(self):
        np_vec = np.random.normal(0, np.sqrt(1 / 512), 512)
        hrr_vec = hrr.Vector(np_vec)
        hrr_res = hrr_vec**2
        np_res = np.fft.ifft(np.power(np.fft.fft(np_vec), 2))
        np.testing.assert_allclose(np.array(hrr_res.values), np_res)

    def test_mul(self):
        np_vec_A = np.random.normal(0, np.sqrt(1 / 512), 512)
        np_vec_B = np.random.normal(0, np.sqrt(1 / 512), 512)
        hrr_vec_A = hrr.Vector(np_vec_A)
        hrr_vec_B = hrr.Vector(np_vec_B)
        np.testing.assert_allclose(hrr_vec_A * hrr_vec_B,
                                   np.dot(np_vec_A, np_vec_B))

    def test_truediv(self):
        np_vec = np.random.normal(0, np.sqrt(1 / 512), 512)
        hrr_vec = hrr.Vector(np_vec)
        divisor = np.random.uniform(0, 1)
        hrr_res = hrr_vec / divisor
        np.testing.assert_allclose(np.array(hrr_res.values),
                                   np.true_divide(np_vec, divisor))

    def test_floordiv(self):
        np_vec = np.random.normal(0, np.sqrt(1 / 512), 512)
        hrr_vec = hrr.Vector(np_vec)
        divisor = np.random.uniform(0, 1)
        hrr_res = hrr_vec // divisor
        np.testing.assert_allclose(np.array(hrr_res.values),
                                   np.floor_divide(np_vec, divisor))


class TestHRR(unittest.TestCase):
    def test_convolve(self):
        np_vec_A = np.random.normal(0, np.sqrt(1 / 512), 512)
        np_vec_B = np.random.normal(0, np.sqrt(1 / 512), 512)
        hrr_vec_A = hrr.HRR(np_vec_A)
        hrr_vec_B = hrr.HRR(np_vec_B)
        hrr_res = hrr_vec_A.encode(hrr_vec_B)
        np_res = _np_circ_convolve(np_vec_A, np_vec_B)
        np.testing.assert_allclose(hrr_res.values, np_res)

    def test_compose(self):
        np_vec_A = np.random.normal(0, np.sqrt(1 / 512), 512)
        np_vec_B = np.random.normal(0, np.sqrt(1 / 512), 512)
        hrr_vec_A = hrr.HRR(np_vec_A)
        hrr_vec_B = hrr.HRR(np_vec_B)
        hrr_res = hrr_vec_A.compose(hrr_vec_B)
        np_res = np_vec_A + np_vec_B
        np.testing.assert_allclose(hrr_res.values, np_res)

    def test_correlate(self):
        np_vec_A = np.random.normal(0, np.sqrt(1 / 512), 512)
        np_vec_B = np.random.normal(0, np.sqrt(1 / 512), 512)
        hrr_vec_A = hrr.HRR(np_vec_A)
        hrr_vec_B = hrr.HRR(np_vec_B)
        hrr_res = hrr_vec_A.correlate(hrr_vec_B)
        np_res = _np_circ_correlate(np_vec_A, np_vec_B)
        np.testing.assert_allclose(hrr_res.values, np_res)

    def test_decode(self):
        np_vec_A = np.random.normal(0, np.sqrt(1 / 512), 512)
        np_vec_B = np.random.normal(0, np.sqrt(1 / 512), 512)
        hrr_vec_A = hrr.HRR(np_vec_A)
        hrr_vec_B = hrr.HRR(np_vec_B)
        np_enc = _np_circ_convolve(np_vec_A, np_vec_B)
        hrr_enc = hrr_vec_A.encode(hrr_vec_B)
        np.testing.assert_allclose(hrr_enc.values, np_enc)
        hrr_res = hrr_enc.decode(hrr_vec_B)
        np_res = _np_circ_decode(np_enc, np_vec_B)
        np.testing.assert_allclose(hrr_res.values, np_res)


class TestHRRStructures(unittest.TestCase):

    def test_HRR_Sequence(self):
        # set up for making and decoding a Sequence
        seq_order = ['a', 'b', 'c']
        # autoassociative memories
        np_memory = {i: np.random.normal(0, np.sqrt(1 / 512), 512) for i in seq_order}
        hrr_memory = {i: hrr.HRR(np_memory[i]) for i in seq_order}
        np_seq_elems = [np_memory[i] for i in seq_order]
        hrr_seq_elems = [hrr_memory[i] for i in seq_order]

        def ab():
            # make decreasing alpha values, 1 for each element in sequence
            alpha = [x / len(seq_order)
                     for x in range(1, len(seq_order) + 1)][::-1]
            # same for betas, 1 for each space between elements
            beta = [x / (len(seq_order) - 1)
                    for x in range(1, len(seq_order))][::-1]
            # gen sequence reps and confirm they're the same
            np_seq = _np_make_Sequence_ab([np_memory[i] for i in seq_order],
                                          alpha, beta)
            hrr_seq = hrr.makeSequence([hrr_memory[i] for i in seq_order],
                                       encoding='ab', alpha=alpha, beta=beta)
            hrr_curr = hrr_seq
            np_curr = np_seq
            np.testing.assert_allclose(hrr_curr.values, np_curr)
            for i in seq_order:
                # decode the Seq reps item by item; test outputs are equal
                # retrieve strongest component and compare dot products
                # print("current = ", i)
                hrr_item = list(hrr.getClosest(hrr_curr, hrr_memory, howMany=1).keys())[0]
                np_item = _np_getClosest(np_curr, np_memory, howMany=1)[0]
                np.testing.assert_allclose(hrr_memory[hrr_item].values,
                                           np_memory[np_item])
                # "correlate out" the cleaned up component from the seq rep
                # and check resulting traces are equal
                hrr_curr = hrr_curr.decode(hrr_memory[hrr_item])
                np_curr = _np_circ_decode(np_curr, np_memory[np_item])
                np.testing.assert_allclose(hrr_curr.values, np_curr)

        def tri():
            np_seq = _np_make_Sequence_triangle(np_seq_elems)
            hrr_seq = hrr.makeSequence(hrr_seq_elems, encoding='triangle')
            np.testing.assert_allclose(np_seq, np.array(hrr_seq.values))
            # make a, a*b, a*b*c, etc reps from sequences
            hrr_decode_elems = [hrr_seq_elems[0]] +\
                [reduce(lambda x, y: x.encode(y), hrr_seq_elems[:e])
                    for e in range(2, len(hrr_seq_elems) + 1)]
            np_decode_elems = [np_seq_elems[0]] +\
                [reduce(lambda x, y: _np_circ_convolve(x, y), np_seq_elems[:e])
                    for e in range(2, len(np_seq_elems) + 1)]
            #
            for i in range(max(len(hrr_decode_elems), len(np_decode_elems))):
                np.testing. \
                    assert_allclose(np.array(hrr_decode_elems[i].values),
                                    np_decode_elems[i])
            np_curr = np_seq
            hrr_curr = hrr_seq
            for d in range(len(np_decode_elems)):
                np_item = _np_getClosest(np_curr, np_memory)[0]
                hrr_item = list(hrr.getClosest(hrr_curr, hrr_memory, howMany=1).keys())[0]
                np.testing.assert_allclose(
                    np.array(hrr_memory[hrr_item].values), np_memory[np_item])
                np_curr = _np_circ_decode(np_curr, np_decode_elems[d])
                hrr_curr = hrr_curr.decode(hrr_decode_elems[d])
                np.testing.assert_allclose(np_curr, np.array(hrr_curr.values))

        def positional():
            p_np = np.random.normal(0, np.sqrt(1 / 512), 512)
            p_hrr = hrr.HRR(p_np)
            np.testing.assert_allclose(p_np, np.array(p_hrr.values))
            np_seq = _np_make_Sequence_positional([np_memory[i]
                                                   for i in seq_order], p=p_np)
            hrr_seq = hrr.makeSequence([hrr_memory[i]
                                        for i in seq_order],
                                       encoding='positional', p=p_hrr)
            np.testing.assert_allclose(np_seq, np.array(hrr_seq.values))
            hrr_curr = hrr_seq
            np_curr = np_seq
            for i in seq_order:
                # decode the Seq reps per item, checking outputs are equal
                # retrieve strongest component, using angle between vectors
                # as the similarity f'n since these vectors are complex
                def cosine_theta(x, y):
                    return spatial.distance.cosine(x, y)

                def cosine_theta_hrr(x, y):
                    return spatial.distance.cosine(np.array(x.values),
                                                   np.array(y.values))
                # assert np.linalg.norm(np_curr) ==
                # np.linalg.norm(np.array(hrr_curr.values))
                hrr_item = list(hrr.getClosest(
                    hrr_curr, hrr_memory,
                    howMany=1, similarityFn=cosine_theta_hrr))[0]
                np_item = _np_getClosest(
                    np_curr, np_memory,
                    howMany=1, similarityFn=cosine_theta)[0]
                # assert np_item == hrr_item
                np.testing.assert_allclose(
                    np.array(hrr_memory[hrr_item].values), np_memory[np_item])
                # "correlate out" the cleaned up component from the seq rep
                # and test that  the resulting traces are equal
                hrr_curr = hrr_curr.decode(hrr_memory[hrr_item])
                np_curr = _np_circ_decode(np_curr, np_memory[np_item])
                np.testing.assert_allclose(np.array(hrr_curr.values), np_curr)
        ab()
        tri()
        positional()

    def test_HRR_stack(self):
        seq_order = ['a', 'b', 'c']
        push_item = 'd'
        np_memory = {i: np.random.normal(0, np.sqrt(1 / 512), 512)
                     for i in seq_order + [push_item]}
        hrr_memory = {i: hrr.HRR(np_memory[i])
                      for i in seq_order + [push_item]}
        p_np = np.random.normal(0, np.sqrt(1 / 512), 512)
        p_hrr = hrr.HRR(p_np)
        np_seq = _np_make_Sequence_positional(
            [np_memory[i] for i in seq_order], p=p_np)
        hrr_seq = hrr.makeSequence(
            [hrr_memory[i] for i in seq_order], encoding='positional', p=p_hrr)
        np.testing.assert_allclose(np_seq, np.array(hrr_seq.values))

        def cosine(x, y):
            return spatial.distance.cosine(x, y)

        def test_top():
            # compare scores of top?
            assert _np_stackTop(np_seq, np_memory, lambda x, y: cosine(x, y)) \
                == list(hrr.stackTop(hrr_seq, hrr_memory, lambda x, y:
                                cosine(np.array(x.values), np.array(y.values))).keys())[0]

        def test_push():
            hrr.stackPush(hrr_seq, hrr_memory[push_item], p_hrr)
            _np_push(np_seq, np_memory[push_item], p_np)
            np.testing.assert_allclose(np.array(hrr_seq.values), np_seq)

        def test_pop():
            for i in seq_order + [push_item]:
                np.testing.assert_allclose(np.array(hrr.stackPop(hrr_seq, hrr_memory, p_hrr, lambda x, y: cosine(np.array(x.values), np.array(y.values))).values),
                                           _np_stackPop(np_seq, np_memory, p_np, lambda x, y: cosine(x, y)))
                np.testing.assert_allclose(np.array(hrr_seq.values), np_seq)
        test_top()
        test_push()
        test_pop()

    def test_HRR_variable(self):
        def subsets(inp):  # return all subsets
            return reduce(lambda x, y: x + y,
                          (list(combinations(inp, r + 1))
                           for r in range(len(inp))))
        Vars = ['x', 'y', 'z']
        Vals = [1, 3, 7]  # binding x=1, y=3
        np_M = {i: np.random.normal(0, np.sqrt(1 / 512), 512) for i in Vars + Vals}
        hrr_M = {i: hrr.HRR(np_M[i]) for i in Vars + Vals}
        hrr_term = reduce(lambda x, y: x + y,
                          (hrr.bindVariable(hrr_M[Vars[i]], hrr_M[Vals[i]])
                           for i in range(len(Vars))))
        np_term = reduce(np.add, (_np_circ_convolve(np_M[Vars[i]],
                                                    np_M[Vals[i]])
                                  for i in range(len(Vars))))
        np.testing.assert_allclose(np.array(hrr_term.values), np_term)
        for v in Vars:  # are the value reps for a query variable the same?
            h_val = hrr_term.decode(hrr_M[v])
            np_val = _np_circ_decode(np_term, np_M[v])
            np.testing.assert_allclose(np.array(h_val.values), np_val)
        for i in subsets(Vars):  # can we recover the trace post-unbinding?
            h_curr = reduce(lambda x, y: hrr.unbindVariable(x, y),
                            [hrr_term] + [hrr_M[x] for x in i])
            np_curr = reduce(lambda x, y: _np_circ_decode(x, y),
                             [np_term] + [np_M[x] for x in i])
            np.testing.assert_allclose(np.array(h_curr.values), np_curr)

    def test_frame(self):
        # a simple frame (no recursion)
        frame_elems = ['label', 'slot1', 'filler1', 'slot2', 'filler2']
        Mnp = {i: np.random.normal(0, np.sqrt(1 / 512), 512) for i in frame_elems}
        Mhrr = {i: hrr.HRR(Mnp[i]) for i in frame_elems}
        npFrame = _np_makeFrame(Mnp['label'],
                                (Mnp['slot1'], Mnp['filler1']),
                                (Mnp['slot2'], Mnp['filler2']))
        hrrFrame = hrr.makeFrame(Mhrr['label'],
                                 (Mhrr['slot1'], Mhrr['filler1']),
                                 (Mhrr['slot2'], Mhrr['filler2']))
        np.testing.assert_allclose(npFrame, np.array(hrrFrame.values))
        # a recursive frame
        frame_elems += ['slot3', ('sublabel', 'subslot1', 'subfiller1')]
        for s in ('slot3', 'sublabel', 'subslot1', 'subfiller1'):
            Mnp[s] = np.random.normal(0, np.sqrt(1 / 512), 512)
            Mhrr[s] = hrr.HRR(Mnp[s])
        npFrame = _np_makeFrame(Mnp['label'],
                                (Mnp['slot1'], Mnp['filler1']),
                                (Mnp['slot2'], Mnp['filler2']),
                                (Mnp['slot3'], (Mnp['sublabel'],
                                                (Mnp['subslot1'],
                                                 Mnp['subfiller1']))))
        hrrFrame = hrr.makeFrame(Mhrr['label'],
                                 (Mhrr['slot1'], Mhrr['filler1']),
                                 (Mhrr['slot2'], Mhrr['filler2']),
                                 (Mhrr['slot3'], (Mhrr['sublabel'],
                                                  (Mhrr['subslot1'],
                                                   Mhrr['subfiller1']))))
        np.testing.assert_allclose(npFrame, np.array(hrrFrame.values))


if __name__ == '__main__':
    unittest.main()
