import unittest
import os
from os.path import join as opj
import numpy as np
from clut import CLUT


class TestCLUT(unittest.TestCase):
    def test_init(self):
        clut = CLUT()
        self.AssertEqual(clut._dtype, np.uint8)

        points = np.linspace(0, 255, 4**2)
        b,g,r = np.meshgrid(*3*[points], indexing='ij')
        tab = np.stack([r,g,b]).T
        clut = CLUT(tab)


    def test_saveload(self):
        clut1 = CLUT()
        clut1.save('testclut.png')
        clut2 = CLUT('testclut.png')
        relerr = np.abs(clut1.clut - clut2.clut).max() # account for interpolation and compression errors
        self.LessEqual(relerr, 1)
        os.remove('testclut.png')


    def test_exceptions(self):
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
        with self.assertRaises(AssertionError):
            clut = CLUT(depth=10) # depth not supported
        with self.assertRaises(AssertionError):
            clut = CLUT()
            clut.save('path', size=1) # size < 2


    @classmethod
    def tearDownClass(self):
        for i in ['testclut.png']:
            if os.path.exists(i):
                os.remove(i)
