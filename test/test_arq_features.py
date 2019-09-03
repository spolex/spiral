# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:25:59 2019

@author: isancmen
"""

from preprocess.features import log_detector
import numpy as np
import unittest


class TestFeatureExtraction(unittest.TestCase):

    def test_len(self):
        l = np.arange(1, 11)
        self.assertTrue(l.all() == np.arange(1, 11).all())

    def test_lenn(self):
        l = np.arange(1, 11)
        self.assertTrue(len(l) == 10)
        self.assertFalse(len(l) > 11)

    def test_logdetector(self):
        l = np.arange(2, 12)
        # 6.3387402669680695
        self.assertTrue(log_detector(l) > 6.)
        self.assertFalse(log_detector(l) > 7.)
        l[-1] = -11
        # 20.763409211116233
        self.assertTrue(log_detector(l) > 20.)
        self.assertFalse(log_detector(l) > 21.)