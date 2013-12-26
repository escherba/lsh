__author__ = 'escherba'

import unittest
import time
from functools import partial
from random import randint as random_int
from operator import itemgetter
from itertools import izip

import lsh
from util import first, second, last


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
            lsh_index = lsh.LSHIndex(hash_family, k, 0)
            for L in L_vals:
                # using more hash tables increases recall
                lsh_index.resize(L)
                lsh_index.index(self.points)

                correct = 0
                total_dist_ratio = 0.0
                total_search_time = 0.0
                total_result_size = 0.0

                for query, hits, nearest_radius in izip(self.queries, exact_indexes, nearest_radii):
                    search_time_start = time.time()
                    lsh_query = lsh_index.query(query, metric, self.num_neighbours + 1)
                    search_time = time.time() - search_time_start
                    total_search_time += search_time
                    total_result_size += len(lsh_query)
                    for idx, dist in lsh_query:
                        if dist > nearest_radius:
                            total_dist_ratio += (dist - nearest_radius) / nearest_radius
                    if hits == map(first, lsh_query):
                        correct += 1

                avg_dist_ratio = total_dist_ratio / float(lsh_index.num_queries)
                avg_search_time = 1000.0 * total_search_time / float(lsh_index.num_queries)
                avg_query_size = total_result_size / float(lsh_index.num_queries)
                recall = float(correct) / float(lsh_index.num_queries)
                touch = float(lsh_index.get_avg_touched()) / float(len(self.points))
                print "{0:>4} {1:>4} {2:>8.0%} {3:>8.2%} {4:>8.2} {5:>8.2} {6:>8.2}"\
                    .format(L, k, recall, touch, avg_query_size, avg_dist_ratio, avg_search_time)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_stats(self):
        # create a test dataset of vectors of non-negative integers
        d = 5
        xmax = 20
        num_points = 1000
        points = lsh.lapply(num_points,  # rows (number of vectors)
                            lsh.lapply, d,   # columns (vector cardinality)
                            random_int, 0, xmax)

        # seed the dataset with a fixed number of nearest neighbours
        # within a given small "radius"
        num_neighbours = 2
        radius = 1.0
        for point in points[:num_points]:
            for i in xrange(num_neighbours):
                points.append([x + lsh.random_gauss(0, radius)
                              for x in point])

        #import json
        #with open('tests/data.txt', 'w') as outfile:
        #    json.dump(points, outfile)

        # test lsh versus brute force comparison by running a grid
        # search for the best lsh parameter values for each family
        tester = LSHTester(points, points[:int(num_points / 10)], num_neighbours)

        args = {'name': 'L1 (Rectilinear)',
                'metric': lsh.L1_norm,
                'hash_family': lsh.L1HashFamily(d, 10 * radius),
                'k_vals': [2, 4, 8],
                'L_vals': [2, 4, 8, 16]}
        tester.run(**args)

        args = {'name': 'L2 (Euclidean)',
                'metric': lsh.L2_norm,
                'hash_family': lsh.L2HashFamily(d, 10 * radius),
                'k_vals': [2, 4, 8],
                'L_vals': [2, 4, 8, 16]}
        tester.run(**args)

        args = {'name': 'Cosine',
                'metric': lsh.Cosine_norm,
                'hash_family': lsh.CosineHashFamily(d),
                'k_vals': [16, 32, 64],
                'L_vals': [1, 2, 4, 8]}
        tester.run(**args)


if __name__ == '__main__':
    unittest.main()
