# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:25:59 2019

@author: isancmen
"""

import unittest
import numpy as np
from feature_extract.arq_features import log_detector

class TestFeatureExtraction(unittest.TestCase):

    def test_len(self):
        L = np.arange(1,11)
        self.assertEqual(L.all(), np.arange(1,11).all())

    def test_lenn(self):
        L = np.arange(1,11)
        self.assertTrue(len(L)==10)
        self.assertFalse(len(L)>11)
        
    def test_logdetector(self):
        L = np.arange(2,12)
        # 6.3387402669680695
        self.assertTrue(log_detector(L)>6.)
        self.assertFalse(log_detector(L)>7.)

if __name__ == '__main__':
    unittest.main()