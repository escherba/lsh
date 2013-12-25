__author__ = 'escherba'

import unittest
from lsh import L1HashFamily, L2HashFamily, CosineHashFamily


class MyTestCase(unittest.TestCase):
    def test_L1HashFamily(self):
        hf_d = 5
        hf_w = 100
        hf = L1HashFamily(hf_d, hf_w)
        self.assertEqual(hf.d, hf_d)
        self.assertEqual(hf.w, hf_w)

    def test_L2HashFamily(self):
        hf_d = 5
        hf_w = 100
        hf = L2HashFamily(hf_d, hf_w)
        self.assertEqual(hf.d, hf_d)
        self.assertEqual(hf.w, hf_w)

    def test_CosineHashFamily(self):
        hf_d = 5
        hf = CosineHashFamily(hf_d)
        self.assertEqual(hf.d, hf_d)


if __name__ == '__main__':
    unittest.main()
