"""
LSH Locality Sensitive Hashing
- indexing for nearest neighbour searches in sublinear time

simple tutorial implementation based on
A. Andoni and P. Indyk, "Near-optimal hashing algorithms for approximate
nearest neighbor in high dimensions"

http://people.csail.mit.edu/indyk/p117-andoni.pdf
"""


from abc import abstractmethod
from itertools import izip, imap, chain
from math import sqrt
from random import gauss as random_gauss, \
    uniform as random_uniform
from collections import defaultdict
from operator import itemgetter

from util import gapply, lapply, dot


''' Metrics
'''


def L1_norm(u, v):
    """L1 hash metric - Rectilinear (Manhattan) distance"""
    return sum(abs(ui - vi) for ui, vi in izip(u, v))


def L2_norm(u, v):
    """L2 hash metric - Euclidean distance"""
    return sqrt(sum((ui - vi) ** 2.0 for ui, vi in izip(u, v)))


def Cosine_norm(u, v):
    """Cosine hash metric - Angular distance"""
    return 1.0 - dot(u, v) / sqrt(dot(u, u) * dot(v, v))

''' Hashes
'''


class Hash:

    def __init__(self, r):
        """Initialize
        :param r: a random vector
        """
        self.r = list(r)

    @abstractmethod
    def hash(self, vec):
        """Hash a vector of integers"""
        pass


class L1Hash(Hash):

    def __init__(self, r, w):
        """
        Initialize
        :param r: a random vector
        :param w: width of the quantization bin
        """
        Hash.__init__(self, r)
        self.w = w

    def hash(self, vec):
        float_gen = ((vec[idx] - s) / self.w
                     for idx, s in enumerate(self.r))
        return hash(tuple(imap(int, float_gen)))


class L2Hash(Hash):

    def __init__(self, r, b, w):
        """
        Initialize
        :param r: a random vector
        :param b: a random variable uniformly distributed between 0 and w
        :param w: width of the quantization bin
        """
        Hash.__init__(self, r)
        self.b = b
        self.w = w

    def hash(self, vec):
        val = (dot(vec, self.r) + self.b) / self.w
        return int(val)


class CosineHash(Hash):

    def hash(self, vec):
        return int(dot(vec, self.r) > 0)

''' Hash families
'''


class HashFamily:

    def __init__(self, size):
        self.size = int(size)

    @abstractmethod
    def get_hash_func(self):
        pass

    @abstractmethod
    def get_projection(self):
        pass

    def combine(self, hashes):
        """ combine hash values

        :param hashes: an iterable representing a vector of hashes
        """
        return hash(tuple(hashes))


class L1HashFamily(HashFamily):

    def __init__(self, size, w):
        """
        Initialize
        :param size: size of hash family
        :param w: width of the quantization bin
        """
        HashFamily.__init__(self, size)
        self.w = float(w)

    def get_hash_func(self):
        """ initialize each L1Hash with a different random
        partition vector"""
        return L1Hash(self.get_projection(), self.w)

    def get_projection(self):
        """Return a vector of size d drawn from a uniform
        distribution from 0 to w"""
        return gapply(self.size, random_uniform, 0, self.w)


class L2HashFamily(HashFamily):

    def __init__(self, size, w):
        """
        Initialize
        :param size: size of hash family
        :param w: width of the quantization bin
        """
        HashFamily.__init__(self, size)
        self.w = float(w)

    def get_hash_func(self):
        """initialize each L2Hash with a different random projection vector
        and offset
        """
        return L2Hash(self.get_projection(), random_uniform(0, self.w), self.w)

    def get_projection(self):
        """Return a vector of size d drawn from a Gaussian
        distribution with mean 0 and sigma 1
        """
        return gapply(self.size, random_gauss, 0, 1)


class CosineHashFamily(HashFamily):

    def get_hash_func(self):
        """initialize each CosineHash with a random projection vector"""
        return CosineHash(self.get_projection())

    def get_projection(self):
        """Random projection vector"""
        return gapply(self.size, random_gauss, 0, 1)

    def combine(self, hashes):
        """ combine by treating as a bit-vector """
        return sum(1 << idx if h > 0 else 0
                   for idx, h in enumerate(hashes))

''' Index
'''


class LSHIndex:

    tot_touched = 0
    num_queries = 0
    points = []

    def __init__(self, hash_family, k, L):
        """Initialize

        :param hash_family: HashFamily instance
        :param k: hash table size (increases selectivity)
        :param L: number of hash tables (increases recall)
        """
        self.hash_family = hash_family
        self.k = k
        self.L = 0
        self.hash_tables = []
        self.resize(L)

    def resize(self, L):
        """ update the number of hash tables to be used
        :param L: number of hash tables (increases recall)
        """
        if L < self.L:
            self.hash_tables = self.hash_tables[:L]
        else:
            # initialise a new hash table for each hash function
            hash_funcs = lapply(L - self.L,      # rows
                                lapply, self.k,  # columns
                                self.hash_family.get_hash_func)
            self.hash_tables.extend((g, defaultdict(list))
                                    for g in hash_funcs)

    def hash(self, g, point):
        """

        :param g:     A vector of Hash instances
        :param point: A point vector
        :return:      A combined hash digest
        """
        return self.hash_family.combine(h.hash(point) for h in g)

    def index(self, pts):
        """ index the supplied points
        :param pts: A list of points (represented as vectors)
        """
        self.points = pts
        for g, table in self.hash_tables:
            for idx, p in enumerate(self.points):
                table_idx = self.hash(g, p)
                table[table_idx].append(idx)
        # reset stats
        self.tot_touched = 0
        self.num_queries = 0

    def query(self, query, metric, max_results):
        """ find the max_results closest indexed points to query according
        to the supplied metric
        :param query:       query vector
        :param metric:      distance metric to use
        :param max_results: maximum number of results to return
        :returns :          sorted list of tuples of form <index, distance>
        :rtype :            list
        """
        candidates = set(chain.from_iterable(
            table.get(self.hash(g, query), [])
            for g, table in self.hash_tables))

        # update stats
        self.tot_touched += len(candidates)
        self.num_queries += 1

        # re-rank candidates according to supplied metric
        return sorted(((idx, metric(query, self.points[idx]))
                       for idx in candidates),
                      key=itemgetter(1))[:max_results]

    def get_avg_touched(self):
        """ mean number of candidates inspected per query """
        return float(self.tot_touched) / float(self.num_queries)
