import math
import random
from functools import reduce
from itertools import chain


class Vector:
    """
    List of floats to hold some generic vector methods
    relevant for distributed representations.

    ***Not a general-purpose vector!***
    """

    def __init__(self, values):
        if all(type(v)!=complex for v in values):
            self.values = [float(v) for v in values]  # not robust
        else:
            self.values = [complex(v) for v in values]
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

    def __mul__(self, other):
        """dot product"""
        if issubclass(type(other), Vector) and len(self) == len(other):
            return sum((self.values[i] * other.values[i]
                        for i in range(len(self))))
        elif isinstance(other, (int, float)):
            return Vector(map(lambda x: x * other, self.values))
        else:
            raise TypeError  # TODO: what kind of exception

    def __pow__(self, other):
        """Raise Vector (or derived class) to an integer power.

        Fourier transform -> element-wise exp -> inverse Fourier transform"""
        if not isinstance(other, int):
            raise TypeError('can only ** a Vector by an int')
        # discrete FT to translate to frequency space
        n = len(self)
        freq = (sum((self[k] * (math.e ** ((-1 * 1j * 2 * math.pi * j * k) / n)) for k in range(0, n))) for j in range(0, n))
        exp = list(map(lambda x: x**other, freq))
        out = ((1 / n) * sum((exp[k] * (math.e ** ((1j * 2 * math.pi * j * k) / n)) for k in range(0, n))) for j in range(0, n))
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
    """A square matrix (N x N) for representing  matrix memories
     and TODO: convolution operations (e.g. a "circulant matrix").

    Initialized by an N-element list of N-element lists
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
        if _check(self, other):
            return SquareMatrix([[sum(y) for y in zip(x[0], x[1])]
                                 for x in zip(self.values, other.values)])

    def __mul__(self, other):
        """element-wise multiplication"""
        if _check(self, other):
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

    def __add__(self, other):
        if issubclass(type(other), Vector) and len(self) == len(other):
            return AdditionMemory([
                self.values[i] + other.values[i]
                for i in range(len(self))
            ])
        else:
            raise TypeError  # TODO: what kind of exception
    # decode = __mul__#dot product is decoding for add. mem's
    encode = __add__  # compose = encode


# CONVOLUTION-CORRELATION (i.e. HOLOGRAPHIC-LIKE) MEMORIES  #
class HRR(Vector):
    """Vector distributed representation with circular convolution.

    Instantiating fills out n_dims of floating point elements drawn from
    Normal(0, 1/n_dims). HRRs can be convolved(encoded) to form pair
    associations, composed to store multiple pair associations,
    and correlated (decoded) to recover noisy representations
    of an HRR's partner in a pair if the pair has been encoded.

    Default 512 elements.
    """

    def __init__(self, values=None, n_dims=512):  # TODO: add seed for random
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

    def __add__(self, other: 'HRR') -> 'HRR':
        if type(other) == HRR and len(self) == len(other):
            return HRR([
                self.values[i] + other.values[i]
                for i in range(len(self))
            ])
        else:
            raise TypeError  # TODO: what kind of exception

    def approxInverse(self):
        """approximate inverse: simply reverses values"""
        out = self.values.copy()
        out.reverse()
        return HRR(out)

    def encode(self, Item: 'HRR') -> 'HRR':
        """circular convolution

        TODO: FFT is O(n log n)
        """
        if len(self) != len(Item):
            raise TypeError  # TODO: which exception
        n = len(self)
        J = range(0, n)
        K = range(0, n)
        terms = [sum([Item[k % n] * self[(j - k) % n] for k in K]) for j in J]
        return HRR(terms)

    def decode(self, Item: 'HRR') -> 'HRR':
        """circular correlation"""
        if len(self) != len(Item):
            raise TypeError  # TODO: which exception
        n = len(self)
        J = range(0, n)
        K = range(0, n)
        terms = [sum([Item[k % n] * self[(j + k) % n] for k in K]) for j in J]
        return HRR(terms)
    # some method aliases
    correlate = decode
    convolve = encode
    compose = __add__


class Aperiodic(Vector):
    """Aperiodic convolution as encoding operation.

    """

    def __init__(self, values=None, n_dims=3):
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

    def __add__(self, other: 'Aperiodic') -> 'Aperiodic':
        if type(other) == Aperiodic and len(self) == len(other):
            return Aperiodic([
                self.values[i] + other.values[i]
                for i in range(len(self))
            ])
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
        c = int(n / 2)
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
    # some method aliases
    correlate = decode
    convolve = encode
    compose = __add__


class Truncated(Vector):
    """Metcalfe's truncated aperiodic convolution as encoding operation."""

    def __init__(self, values=None, n_dims=3):  # add seed for random
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

    def __add__(self, other: 'Truncated') -> 'Truncated':
        if type(other) == Truncated and len(self) == len(other):
            return Truncated([
                self.values[i] + other.values[i]
                for i in range(len(self))
            ])
        else:
            raise TypeError  # TODO: what kind of exception

    def encode(self, Item: 'Truncated') -> 'Truncated':
        if len(self) != len(Item):
            raise TypeError  # TODO: which exception
        n = len(self)
        J = range(int(-(n - 1) / 2), int((n - 1) / 2) + 1)
        K = range(int(-(n - 1) / 2), int((n - 1) / 2) + 1)
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
    # method aliases
    correlate = decode
    convolve = encode
    compose = __add__


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

    def __add__(self, other: 'Trace') -> 'Trace':
        if type(other) == Trace and len(self) == len(other):
            return Trace([
                self.values[i] + other.values[i]
                for i in range(len(self))
            ])
        else:
            raise TypeError  # TODO: what kind of exception
    # method aliases
    compose = __add__

# functions for cleaning up a noisy representation
# and representation of complex structure


def getClosest(item, memoryDict, howMany=3, likenessFn=lambda x, y: x * y):
    """Returns stored representation R maximizing
    likenessFn(item, R) and value of likenessFn(item, R).

    The likenessFn defaults to x * y
    (for most vector reps in hrr.py, this is the dot product).
    howMany determines the number of entries to return.
    If howMany >= length of M, will return scores for all items in M.

    # TODO: assume there are ties, and return multiple values
    # TODO: make efficient (generator?)
    if there are a lot of lookups or items to be used
    """
    # a brute sort because we don't have many memories
    dists = {key: likenessFn(item, value) for key, value in memoryDict.items()}
    sortedDists = sorted(dists.keys(),
                         key=(lambda key: dists[key]), reverse=True)
    return {k: round(dists[k], 5) for k in
            sortedDists[:min(howMany, len(memoryDict))]}


# some simple hrr-implemented structures and functions


def makeSequence(seq: 'list', encoding='ab') -> 'HRR':
    """Encodes a sequence of HRR items"""
    if type(seq) != list or any(type(i) != HRR for i in seq):
        raise TypeError('input sequence must be a list of HRRs')
    elif any(len(seq[i]) != len(seq[0]) for i in seq):
        raise ValueError('input HRRs are not all same length')
    # TODO: chunked sequence, a list of lists of ... of HRRs
    # now, encode according to scheme specified
    if encoding == 'ab':
        if any([seq[i] == seq[j]
                for j in range(len(seq) - 1) if j > i
                for i in range(len(seq) - 1)]):
            raise ValueError('alpha-beta encoding cannot faithfully represent \
                    sequences with repeated items')
        # functions for alpha, beta value from sequence position
        # alpha = -(1/len(seq))*index + 1
        alpha = [x / len(seq) for x in range(1, len(seq) + 1)][::-1]
        # beta = -(1/(len(seq)-1))*index + 1
        beta = [x / (len(seq) - 1) for x in range(1, len(seq))][::-1]
        # now, compute output
        # pairwise multiply seq and alpha (ie dot product)
        alpha_elems = (p[0] * p[1] for p in zip(alpha, seq))
        beta_elems = (p[0] * p[1]
                      for p in zip(beta, (seq[i] * seq[i + 1]
                                   for i in range(len(seq) - 1))))
        return sum(chain(alpha_elems, beta_elems))
    elif encoding == 'triangle':
        return seq[0] +\
            sum((reduce(lambda x, y: x.encode(y), seq[:e])
                for e in range(2, len(seq) + 1)))
    elif encoding == 'positional':
        # need a position vector
        p = HRR(n_dims=len(seq[0]))


def makeStack(seq: 'list'):
    """Encodes a stack from a HRR sequence"""
    pass


def stackPop():
    pass


def stackPush():
    pass


def stackTop():
    pass


def bindVariable(name_hrr: 'HRR', value_hrr: 'HRR') -> 'HRR':
    """Binds a variable (w/ id name_hrr) to a value (w/ id value_hrr)"""
    pass

# generalize such that structures can be instantiated without HRRs


# class Stack(HRR):
