import unittest
import os
from os.path import join as opj
import numpy as np
from clut import CLUT


class TestCLUT(unittest.TestCase):
    def test_init(self):
        clut = CLUT(2)
        clut = CLUT('resources/hald.png')

        points = np.linspace(0, 255, 4**2)
        b,g,r = np.meshgrid(*3*[points], indexing='ij')
        tab = np.stack([r,g,b]).T.astype(np.uint8)
        clut = CLUT(tab)

    def test_exceptions(self):
        with self.assertRaises(AssertionError):
            clut = CLUT(1) # step too small
        with self.assertRaises(AssertionError):
            clut = CLUT(17) # step too large
        with self.assertRaises(ValueError):
            clut = CLUT(3.) # float not understood
        tab = np.zeros(shape=(2,2))
        with self.assertRaises(AssertionError):
            clut = CLUT(tab) # ndim != 4
        tab = np.zeros(shape=(2,2,3,3))
        with self.assertRaises(AssertionError):
            clut = CLUT(tab) # matrix not square
        tab = np.zeros(shape=(2,2,3,2))
        with self.assertRaises(AssertionError):
            clut = CLUT(tab) # last dim not 3
        tab = np.zeros(shape=(2,2,2,3))
        with self.assertRaises(AssertionError):
            clut = CLUT(tab) # grid too small
        tab = np.zeros(shape=(4,4,4,3))
        tab[0,0,0,0] = -1
        with self.assertRaises(AssertionError):
            clut = CLUT(tab) # invalid range
        tab = np.zeros(shape=(4,4,4,3))
        tab[0,0,0,0] = 256
        with self.assertRaises(AssertionError):
            clut = CLUT(tab) # invalid range
        with self.assertRaises(AssertionError):
            clut = CLUT(opj('resources','bad_hald.png')) # invalid image
