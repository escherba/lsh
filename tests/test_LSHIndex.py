__author__ = 'escherba'

import random
import unittest
from ann import CosineHashFamily, LSHIndex


class MyTestCase(unittest.TestCase):
    def test_resize(self):
        lsh = LSHIndex(CosineHashFamily(5), 10, 1)
        lsh.resize(2)
        self.assertEqual(len(lsh.hash_tables), 3)
        lsh.resize(4)
        self.assertEqual(len(lsh.hash_tables), 7)

    def test_index(self):
        lsh = LSHIndex(CosineHashFamily(5), 10, 5)

        d = 5
        xmax = 20
        num_points = 10
        points = [[random.randint(0, xmax) for i in xrange(d)] for j in xrange(num_points)]

        # seed the dataset with a fixed number of nearest neighbours
        # within a given small "radius"
        num_neighbours = 2
        radius = 0.1
        for point in points[:num_points]:
            for i in xrange(num_neighbours):
                points.append([x + random.uniform(-radius, radius) for x in point])

        lsh.index(points)
        self.assertEqual(lsh.tot_touched, 0, "did not reset tot_touched")
        self.assertEqual(lsh.num_queries, 0, "did not reset tot_touched")


if __name__ == '__main__':
    unittest.main()
