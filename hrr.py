"""
A module for implementing holographic reduced representations, other
convolution distributed representations for comparison, and functions to
construct a few basic data structures (Sequence, Stack, bound variables) using
HRRs.
"""

import math
import random
from functools import reduce
from itertools import chain, islice


class Vector:
    """
    List of floats to hold some generic vector methods
    relevant for distributed representations.

    ***Not a general-purpose vector!***
    """

    def __init__(self, values):
        # not robust
        self.values = list(values)
        if all(type(v) != complex for v in self.values):
            self.values = list(map(lambda x: float(x), self.values))
        else:
            self.values = list(map(lambda x: complex(x), self.values))
        self.index = -1

    def __len__(self):
        return len(self.values)

    def __str__(self):
        if len(self) <= 10:
            return str(self.values)
        else:
            return '[' + ','.join(map(str, self[0:3])) + \
                ' ... ' + ','.join(map(str, self[-3:])) + ']'

    def __repr__(self):
        return str(self.values)

    def __add__(self, other):
        if issubclass(type(other), Vector):
            if len(self) == len(other):
                return type(self)([
                    self.values[i] + other.values[i]
                    for i in range(len(self))
                ])
            else:
                ValueError('Added vector-likes not same size')
        else:
            raise TypeError('Need two vector-likes for add')

    __radd__ = __add__

    def __mul__(self, other):
        """dot product"""
        if issubclass(type(other), Vector) and len(self) == len(other):
            return sum((self.values[i] * other.values[i]
                        for i in range(len(self))))
        elif isinstance(other, (int, float)):
            return type(self)(map(lambda x: x * other, self.values))
        else:
            raise TypeError('Expect (vector-like | int | float) for mul')

    # __rmul__ = __mul__

    def __pow__(self, other):
        """Raise Vector (or derived class) to an integer power.

        Fourier transform -> element-wise exp -> inverse Fourier transform"""
        if not isinstance(other, int):
            raise TypeError('can only ** a Vector by an int')
        # discrete FT to translate to frequency space
        n = len(self)
        freq = (sum((self[k] *
                     (math.e ** ((-1 * 1j * 2 * math.pi * j * k) / n))
                     for k in range(0, n))) for j in range(0, n))
        exp = list(map(lambda x: x**other, freq))
        out = ((1 / n) * sum((exp[k] *
                              (math.e ** ((1j * 2 * math.pi * j * k) / n))
                              for k in range(0, n))) for j in range(0, n))
        return type(self)(list(out))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(list(map(lambda x: x / other, self.values)))
        else:
            raise TypeError  # TODO: what kind of exception

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(list(map(lambda x: x // other, self.values)))
            # self.values = list(map(lambda x: x // other, self.values))
        else:
            raise TypeError  # TODO: what kind of exception

    def __next__(self):
        if self.index == len(self) - 1:
            raise StopIteration
        self.index += 1
        return self.values[self.index]

    def __iter__(self):
        return self  # TODO: return iterator object that refers to self.values

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def length(self):
        """euclidean length"""
        return math.sqrt(sum([x**2 for x in self.values]))

    def outer_product(self, other):
        if issubclass(type(other), Vector) and len(self) == len(other):
            return SquareMatrix([
                [i * j for j in other.values]
                for i in self.values])
        else:
            raise TypeError  # TODO: what kind of exception
    dot = __mul__


class SquareMatrix:
    """A square matrix (n x n) for representing matrix memories
     and (TODO:) convolution operations (e.g. a "circulant matrix").

    Initialized by an n-element list of n-element lists
    representing the rows of the matrix.

    *** Not a general-purpose matrix! ***
    """

    def __init__(self, rowValues):
        if all([(len(rowValues) == len(i)) for i in rowValues]):
            self.values = [list(r) for r in rowValues]
        else:
            raise TypeError  # TODO: what kind of exception

    def _check(self, other):
        """check if self and other are SquareMatrix of the same size"""
        if issubclass(other, SquareMatrix) and len(self) == len(other):
            return True
        else:
            return False

    def __add__(self, other):
        if self._check(self, other):
            return SquareMatrix([[sum(y) for y in zip(x[0], x[1])]
                                 for x in zip(self.values, other.values)])

    def __mul__(self, other):
        """element-wise multiplication"""
        if self._check(self, other):
            return SquareMatrix([[y[0] * y[1] for y in zip(x[0], x[1])]
                                 for x in zip(self.values, other.values)])

    def __getitem__(self, index):
        return self.values[index]

    def __str__(self):
        return ',\n'.join([str(row) for row in self.values])

    def __repr__(self):
        # not pythonic -- repr for unambiguity
        return ',\n'.join([str(row) for row in self.values])

    def __len__(self):
        assert all([(len(self.values) == len(i)) for i in self.values])
        return len(self.values)

# TODO: Tensor representations generally (Smolensky -- variable binding, etc.)

# A BASIC ADDITION MEMORY #
# TODO: build from:
#   structure : 1. single bit rep's 2. multiple bit rep's 3. bit->digit->real?
#   functions : 1. addition 2. Binary-OR 3. ...
# TODO: a binary addition memory


class AdditionMemory(Vector):
    def __init__(self, values=None, n_dims=512):
        if values:
            Vector.__init__(self, values)
        elif n_dims:
            # generate n_dims of floats from N(mean=0, var=1/n_dims)
            rand_vals = [random.gauss(0.0, math.sqrt(1 / n_dims))
                         for n in range(n_dims)]
            Vector.__init__(self, rand_vals)
            # self.n_dims = n_dims
            # self.distribution = 'Normal'
            # self.mean = 0.0
            # self.variance = 1/n_dims
        else:
            raise TypeError  # TODO: what kind of exception

    # define encode since we can't alias the inherited __add__ method
    def encode(self, other):
        return self + other
    compose = encode  # compose does not really exist
    # decode = __sub__


# CONVOLUTION-CORRELATION (i.e. HOLOGRAPHIC-LIKE) MEMORIES  #
class HRR(Vector):
    """Vector-like distributed representation with circular convolution
    encoding.

    Instantiating fills out n_dims of floating point elements drawn from
    Normal(0, 1/n_dims). HRRs can be convolved(encoded) to form pair
    associations, composed to store multiple pair associations,
    and correlated (decoded) to recover noisy representations
    of an HRR's partner in a pair if the pair has been encoded.

    Default 512 elements.
    """

    def __init__(self, values=None, n_dims: int = 512):
        # TODO: add seed for random
        if values is not None:
            Vector.__init__(self, values)
        elif n_dims:
            # generate n_dims of normally distributed (mean=0, var=1/n_dims)
            # floats as elements of the Vector
            rand_vals = [random.gauss(0.0, math.sqrt(1 / n_dims))
                         for n in range(n_dims)]
            Vector.__init__(self, rand_vals)
            # self.n_dims = n_dims
            # self.distribution = 'Normal'
            # self.mean = 0.0
            # self.variance = 1/n_dims
        else:
            raise TypeError     # TODO: what kind of exception

    def __sub__(self, other: 'HRR') -> 'HRR':
        if type(other) == HRR and len(self) == len(other):
            return HRR([
                self.values[i] - other.values[i]
                for i in range(len(self))
            ])
        else:
            raise TypeError  # TODO: what kind of exception

    def approxInverse(self):
        """approximate inverse: simply reverses values"""
        out = self.values.copy()
        out.reverse()
        return HRR(out)

    def convolve(self, Item: 'HRR') -> 'HRR':
        """circular convolution

        warning: convolution is commutative, so the order of items convolved
        is *not* stored.
        """
        if len(self) != len(Item):
            raise TypeError  # TODO: which exception
        n = len(self)
        J = range(0, n)
        K = range(0, n)
        terms = [sum([Item[k % n] * self[(j - k) % n] for k in K]) for j in J]
        return HRR(terms)

    def correlate(self, Item: 'HRR') -> 'HRR':
        """circular correlation, self.correlate(Item)

        warning: correlation is not commutative, unlike circ. convolution.
        "A decode B" defaults to A.convolve(B.approxInverse) for this reason;
        since encoding is commutative and thus unordered, decoding should
        be as well.
        """
        if len(self) != len(Item):
            raise TypeError  # TODO: which exception
        n = len(self)
        J = range(0, n)
        K = range(0, n)
        terms = [sum([Item[k % n] * self[(j + k) % n] for k in K]) for j in J]
        return HRR(terms)

    def decode(self, Item: 'HRR') -> 'HRR':
        """Decode by convolving with approximate inverse"""
        if len(self) != len(Item):
            raise TypeError  # TODO: which exception
        return self.encode(Item.approxInverse())

    def compose(self, Item: 'HRR') -> 'HRR':
        return self + Item

    # some local method aliases TODO: can you easily alias inherited methods?
    encode = convolve


class Aperiodic(Vector):
    """Aperiodic convolution as encoding operation.

    """

    def __init__(self, values=None, n_dims: int = 3):
        if values:
            Vector.__init__(self, values)
        elif n_dims:
            # generate n_dims NprmL(mean=0, var=) floats as vector elements
            rand_vals = [random.gauss(0.0, 1) for n in range(n_dims)]
            Vector.__init__(self, rand_vals)
            # self.n_dims = n_dims
            # self.distribution = 'Normal'
            # self.mean = 0.0
            # self.variance = 1/n_dims
        else:
            raise TypeError  # TODO: what kind of exception

    def encode(self, Item: 'Aperiodic') -> 'Aperiodic':
        """a very hack way of summing the diagonals of a outer product

        TODO: turn the diagonal sum for outer product
        into a function and use in a generator"""
        if len(self) != len(Item):
            raise TypeError  # TODO: which exception
        n = len(self)
        assert n % 2 == 1  # TODO: should handle even-int length vecs in future
        # c = int(n / 2)
        J = range(-(n - 1), n)
        K = range(int(-(n - 1) / 2), int((n - 1) / 2) + 1)
        vals = []
        for j in J:
            tmp_sum = 0
            for k in K:
                # print('j:',j,'k:',k)
                item_ind = k + int((n - 1) / 2)
                self_ind = j - k + int((n - 1) / 2)
                if self_ind >= 0 and self_ind < n\
                        and item_ind >= 0 and item_ind < n:
                    # print('ITEM:',item_ind)
                    # print('SELF:',self_ind)
                    tmp_sum += Item[item_ind] * self[self_ind]
            vals.append(tmp_sum)
        # terms = [sum((Item[]*self[] for k in K)) for j in J]
        return Aperiodic(vals)

    def decode(self, Item: 'Aperiodic') -> 'Aperiodic':
        # TODO: write this
        pass

    def compose(self, Item: 'Aperiodic') -> 'Aperiodic':
        return self + Item
    # some method aliases
    correlate = decode
    convolve = encode


class Truncated(Vector):
    """Metcalfe's truncated aperiodic convolution as encoding operation."""

    def __init__(self, values=None, n_dims: int = 3):  # add seed for random
        if values:
            Vector.__init__(self, values)
        elif n_dims:
            # generate n_dims normally distributed (mean=0, var=)
            # floats as elements of the Vector
            rand_vals = [random.gauss(0.0, 1) for n in range(n_dims)]
            Vector.__init__(self, rand_vals)
            # self.n_dims = n_dims
            # self.distribution = 'Normal'
            # self.mean = 0.0
            # self.variance = 1/n_dims
        else:
            raise TypeError  # TODO: what kind of exception

    def encode(self, Item: 'Truncated') -> 'Truncated':
        if len(self) != len(Item):
            raise TypeError  # TODO: which exception
        n = len(self)
        J = range(int(-(n - 1) / 2), int((n - 1) / 2) + 1)
        K = range(int(-(n - 1) / 2), int((n - 1) / 2) + 1)
        vals = []
        for j in J:
            tmp_sum = 0
            for k in K:
                # print('j:',j,'k:',k)
                item_ind = k + int((n - 1) / 2)
                self_ind = j - k + int((n - 1) / 2)
                if self_ind >= 0 and self_ind < n\
                        and item_ind >= 0 and item_ind < n:
                    # print('ITEM:',item_ind)
                    # print('SELF:',self_ind)
                    tmp_sum += Item[item_ind] * self[self_ind]
            vals.append(tmp_sum)
        return Truncated(vals)

    def decode(self, Item: 'Truncated') -> 'Truncated':
        # TODO: write this
        pass

    def compose(self, Item: 'Truncated') -> 'Truncated':
        return self + Item
    # method aliases
    correlate = decode
    convolve = encode


class Trace(Vector):
    def __init__(self, values=None, Item1=None, Item2=None):
        if values:
            Vector.__init__(self, values)
        elif Item1 and Item2:
            if type(Item1) == type(Item2):
                Vector.__init__(self, Item1 + Item2)
            else:
                raise TypeError  # TODO: what kind of exception
        else:
            raise TypeError  # TODO: what kind of exception
    # method aliases

    def compose(self, Item: 'Trace') -> 'Trace':
        return self + Item

# functions for cleaning up a noisy representation
# and representation of complex structure


def getClosest(item: HRR, memoryDict: dict,
               howMany: int = 1, similarityFn=lambda x, y: x * y) -> dict:
    """Returns stored representation R maximizing
    similarityFn(item, R), and the output of similarityFn(item, R).

    The similarityFn defaults to x * y
    (for most vector reps in hrr.py, this is the dot product).
    howMany determines the number of entries to return.
    If howMany >= length of M, will return scores for all items in M.

    # TODO: assume there are ties, and return multiple values
    # TODO: make efficient (generator?)
    if there are a lot of lookups or items to be used
    """
    # a brute sort because we don't have many memories
    dists = {key: similarityFn(item, value) for key, value in memoryDict.items()}
    sortedDists = sorted(dists.keys(),
                         key=(lambda key: dists[key]), reverse=True)
    return sortedDists[:howMany]
    # {k: round(dists[k], 5) for k in
    #         sortedDists[:min(howMany, len(memoryDict))]}


# some simple hrr-implemented structures and functions


def makeSequence(seq: list, encoding='ab', **kwargs) -> HRR:
    """Encodes a sequence of HRR items"""
    if type(seq) != list or any([type(i) != HRR for i in seq]):
        raise TypeError('the input sequence must be a list of HRRs')
    elif any(len(i) != len(seq[0]) for i in seq):
        raise ValueError('input HRRs are not all same length')
    # TODO: chunked sequence, a list of lists of ... of HRRs
    # now, encode according to scheme specified
    # print(seq)
    if encoding == 'ab':
        if len(seq) > len(set(seq)):  # hack for checking no repeated seq elems
            raise ValueError('alpha-beta encoding cannot accurately represent \
                    some sequences with repeated items, see Plate (1995),§V.A')
        # functions for alpha, beta value from sequence position
        # alpha = -(1/len(seq))*index + 1
        if kwargs and kwargs['alpha']:
            alpha = kwargs['alpha']
        else:
            alpha = [x / len(seq) for x in range(1, len(seq) + 1)][::-1]
        # beta = -(1/(len(seq)-1))*index + 1
        if kwargs and kwargs['beta']:
            beta = kwargs['beta']
        else:
            beta = [x / (len(seq) - 1) for x in range(1, len(seq))][::-1]
        # now, compute output
        # pairwise multiply seq and alpha (ie dot product)
        alpha_elems = [p[0] * p[1] for p in zip(seq, alpha)]
        beta_elems = [p[0] * p[1]
                      for p in zip((seq[i].encode(seq[i + 1])
                                    for i in range(len(seq) - 1)), beta)]
        # Sum all elements to construct the sequence
        # We have to use reduce(x+y) instead of sum
        # sum() sums across HRR.values, not across HRRs in list
        return reduce(lambda x, y: x + y, alpha_elems + beta_elems)
    elif encoding == 'triangle':
        return seq[0] + reduce(lambda a, b: a + b,
                               (reduce(lambda x, y: x.encode(y), seq[:e])
                                for e in range(2, len(seq) + 1)))
    elif encoding == 'positional':
        # need a position encoding vector
        if kwargs and kwargs['p']:
            p = kwargs['p']
        else:
            p = HRR(n_dims=len(seq[0]))  # our position encoding vector
        # return sum of sequence elements each encoded by
        # p to the power of the element's position in the sequence
        return reduce(lambda x, y: x + y,  # sum over seq items of
                      ((p ** (i + 1)).encode(seq[i])  # p^(index of seq item+1)
                       for i in range(0, len(seq))))  # all seq items


def chunkSequence(seq: list,
                  encoding='ab', chunks=[], **kwargs) -> list:
    """
    Creates a sequence's chunks as indicated by any
    nested structure in the input list.
    """
    if type(seq) != list or any(type(i) not in [HRR, list] for i in seq):
        raise TypeError('the input sequence must be a list of\
                        lists and/or HRRs')
    elif all(type(i) == HRR for i in seq):
        base_seq = makeSequence(seq, encoding, kwargs)
        chunks.append(base_seq)
        return base_seq
    else:
        next_list = next(x for x in seq if type(x) == list)
        return chunkSequence(next_list, encoding, chunks, kwargs)


# stack methods
def makeStack(seq: list, **kwargs) -> (HRR, HRR):
    """Encodes a stack from a list of HRRs"""
    if type(seq) != list or any(type(i) != HRR for i in seq):
        raise TypeError('the input sequence must be a list of HRRs')
    # use any user-passed positional vector
    if 'p' in kwargs:
        p = kwargs['p']
    else:
        p = HRR(n_dims=len(seq[0]))
    # let's encode!
    return (makeSequence(seq, encoding='positional', p=p), p)


def stackPush(stack: HRR, item: HRR, p: HRR) -> None:
    """Pushes item to top of stack. Mutates stack that is passed as argument.

    Adds item rep to (position rep convolved with stack rep)
    """
    if any(type(i) != HRR for i in [stack, item, p]):
        raise TypeError('Push requires a HRR for: stack, item, and position')
    stack = item + p.convolve(stack)


def stackTop(stack: HRR, memory: dict, similarityFn=lambda x, y: x * y) -> HRR:
    """Return the item in memory that is most like the stack.

    By default, item of highest dot product with the stack. This is the item
    most likely to be at the top of the stack.

    TODO: threshold for whether an item from the memory is at all in the stack
    ie is the value returned large e.approxInverse()nough to say that the item is at top? vs
    values for other items in memory?
    """
    return getClosest(stack, memory,  # islice "indexes" a generator
                             howMany=1, similarityFn=similarityFn)[0]


def stackPop(stack: HRR, memory: dict,
             p: HRR, similarityFn=lambda x, y: x * y) -> HRR:
    """Pop top item from stack. Update the stack so the item is removed.

    1. find top item in stack (stackTop) 2. subtract item rep from stack rep
    3. convolve new stack rep with inverse of p ("remove" a p from stack items)

    """
    out = memory[stackTop(stack, memory, similarityFn)]
    stack = (stack - out).decode(p)  # mutate the stack
    return out


# variable binding
def bindVariable(name_hrr: HRR, value_hrr: HRR) -> HRR:
    """Binds a variable (name_hrr) to a value (value_hrr).

    The variable and value id HRRs should be associated with their values
    elsewhere.
    """
    return name_hrr.convolve(value_hrr)


def unbindVariable(trace_hrr: HRR, name_hrr: HRR) -> HRR:
    """Return HRR with *all* instances of a variable unbound from a trace"""
    return trace_hrr.decode(name_hrr)


# frames -- slot/filler
def makeFrame(frame_label: HRR, *args) -> HRR:
    """
    Encodes a frame using HRRs according to Plate 1995.

    Frames are composed of a frame label (HRR), and a set of roles with
    fillers. This assumes each role is (in this order) a single HRR/tuple, and
    each filler is a single HRR/tuple. When all HRRs are passed as role/filler
    elements, makeFrame encodes a simple frame; when a tuple is passed,
    makeFrame is called on the tuple's elements to encode a recursive frame.
    """
    if all(all(type(j) == HRR) for j in i for i in args):
        return reduce(lambda x, y: x + y,
                      [frame_label] + [t[0].convolve(t[1]) for t in args])
    else:
        return makeFrame()


def decodeFrame(frame: HRR, item: HRR) -> HRR:
    """Decode corresponding slot (or filler) for an input filler (or slot)."""
    assert type(item) == HRR, 'decoding item must be HRR'
    return frame.decode(item)
