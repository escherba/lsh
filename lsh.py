"""
LSH Locality Sensitive Hashing
- indexing for nearest neighbour searches in sublinear time

simple tutorial implementation based on
A. Andoni and P. Indyk, "Near-optimal hashing algorithms for approximate
nearest neighbor in high dimensions"

http://people.csail.mit.edu/indyk/p117-andoni.pdf
"""

import time
from abc import abstractmethod
from itertools import izip, imap, chain
from functools import partial
from math import sqrt
from random import gauss as random_gauss, \
    uniform as random_uniform, randint as random_int
from collections import defaultdict
from operator import itemgetter

''' Helper functions
'''


def first(it):
    """return first element of an iterable"""
    return it[0]


def second(it):
    """return second element of an iterable"""
    return it[1]


def last(it):
    """return last element of an iterable"""
    return it[-1]


def gapply(n, func, *args, **kwargs):
    """Apply a generating function n times to the argument list"""
    for _ in xrange(n):
        yield func(*args, **kwargs)


def lapply(*args, **kwargs):
    """Same as gapply except treturn a list"""
    return list(gapply(*args, **kwargs))


def dot(u, v):
    """Return dot product of two vectors"""
    return sum(ui * vi for ui, vi in izip(u, v))


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
        gen = imap(int, float_gen)
        return hash(tuple(gen))


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
        return int((dot(vec, self.r) + self.b) / self.w)


class CosineHash(Hash):

    def hash(self, vec):
        return int(dot(vec, self.r) > 0)

''' Hash families
'''


class HashFamily:

    def __init__(self, size):
        self.size = size

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
        self.w = w

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
        self.w = w

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


class LSHTester:
    """
    grid search over LSH parameters, evaluating by finding the specified
    number of nearest neighbours for the supplied queries from the supplied
    points
    """

    def __init__(self, points, queries, num_neighbours):
        self.points = points
        self.queries = queries
        self.num_neighbours = num_neighbours

    def linear(self, query, metric, max_results):
        """ perform brute-force search by linear scan

        :param query:       Query vector
        :param metric:      Distance metric to use
        :param max_results: Maximum number of results to return
        :returns :          sorted list of tuples of form <index, distance>
        :rtype :            list
        """
        return sorted(enumerate(map(partial(metric, query), self.points)),
                      key=itemgetter(1))[:max_results]

    def run(self, name, metric, hash_family, k_vals, L_vals):
        """

        :param name:        name of test
        :param metric:      distance metric for nearest neighbour computation
        :param hash_family: hash family for LSH
        :param k_vals:      numbers of hashes to concatenate in each hash
                            function to try in grid search
        :param L_vals:      numbers of hash functions/tables to try in grid
                            search
        """

        exact_hits = list(self.linear(query, metric, self.num_neighbours + 1)
                          for query in self.queries)
        exact_indexes = map(partial(map, first), exact_hits)
        nearest_radii = map(second, map(last, exact_hits))

        print name, len(self.queries), "queries"
        print "{0:>4} {1:>4} {2:>8} {3:>8} {4:>8} {5:>8} {6:>8}"\
            .format('L', 'k', 'recall', 'touch', 'q-size', 'd-ratio', 'ms')

        for k in k_vals:
            # concatenating more hash functions increases selectivity
            lsh = LSHIndex(hash_family, k, 0)
            for L in L_vals:
                # using more hash tables increases recall
                lsh.resize(L)
                lsh.index(self.points)

                correct = 0
                total_dist_ratio = 0.0
                total_search_time = 0.0
                total_result_size = 0.0

                for query, hits, nearest_radius in izip(self.queries, exact_indexes, nearest_radii):
                    search_time_start = time.time()
                    lsh_query = lsh.query(query, metric, self.num_neighbours + 1)
                    search_time = time.time() - search_time_start
                    total_search_time += search_time
                    total_result_size += len(lsh_query)
                    for idx, dist in lsh_query:
                        if dist > nearest_radius:
                            total_dist_ratio += (dist - nearest_radius) / nearest_radius
                    if hits == map(first, lsh_query):
                        correct += 1

                avg_dist_ratio = total_dist_ratio / float(lsh.num_queries)
                avg_search_time = 1000.0 * total_search_time / float(lsh.num_queries)
                avg_query_size = total_result_size / float(lsh.num_queries)
                recall = float(correct) / float(lsh.num_queries)
                touch = float(lsh.get_avg_touched()) / float(len(self.points))
                print "{0:>4} {1:>4} {2:>8.0%} {3:>8.2%} {4:>8.2} {5:>8.2} {6:>8.2}"\
                    .format(L, k, recall, touch, avg_query_size, avg_dist_ratio, avg_search_time)


if __name__ == "__main__":
    # create a test dataset of vectors of non-negative integers
    d = 5
    xmax = 20
    num_points = 1000
    points = lapply(num_points,  # rows (number of vectors)
                    lapply, d,   # columns (vector cardinality)
                    random_int, 0, xmax)

    # seed the dataset with a fixed number of nearest neighbours
    # within a given small "radius"
    num_neighbours = 2
    radius = 1.0
    for point in points[:num_points]:
        for i in xrange(num_neighbours):
            points.append([x + random_uniform(-radius, radius)
                           for x in point])

    import json
    with open('data.txt', 'w') as outfile:
        json.dump(points, outfile)

    # test lsh versus brute force comparison by running a grid
    # search for the best lsh parameter values for each family
    tester = LSHTester(points, points[:int(num_points / 10)], num_neighbours)

    args = {'name': 'L1 (Rectilinear)',
            'metric': L1_norm,
            'hash_family': L1HashFamily(d, 10 * radius),
            'k_vals': [2, 4, 8],
            'L_vals': [2, 4, 8, 16]}
    tester.run(**args)

    args = {'name': 'L2 (Euclidean)',
            'metric': L2_norm,
            'hash_family': L2HashFamily(d, 10 * radius),
            'k_vals': [2, 4, 8],
            'L_vals': [2, 4, 8, 16]}
    tester.run(**args)

    args = {'name': 'Cosine',
            'metric': Cosine_norm,
            'hash_family': CosineHashFamily(d),
            'k_vals': [32, 64, 128],
            'L_vals': [1, 2, 4, 8]}
    tester.run(**args)
