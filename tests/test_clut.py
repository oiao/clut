import unittest
import os
from os.path import join as opj
import numpy as np
from clut import CLUT
from PIL import Image


class TestCLUT(unittest.TestCase):
    def test_init(self):
        clut = CLUT()
        self.assertEqual(clut._dtype, np.uint8)

        points = np.linspace(0, 255, 4**2)
        b,g,r = np.meshgrid(*3*[points], indexing='ij')
        tab = np.stack([r,g,b]).T
        clut = CLUT(tab)


    def test_saveload(self):
        clut1 = CLUT()
        clut1.save('testclut.png')
        clut1.npsave('testclut')
        self.assertTrue(os.path.isfile('testclut.npy'))

        clut2 = CLUT('testclut.png')
        relerr = np.abs(clut1.clut - clut2.clut).max() # account for interpolation and compression errors
        self.assertLessEqual(relerr, 1)
        os.remove('testclut.png')

        clut2 = CLUT('testclut.npy')
        self.assertTrue( (clut1==clut2).all() )
        os.remove('testclut.npy')


    def test_call(self):
        clut = CLUT()

        with self.assertRaises(ValueError):
            clut(10) # wrong argument type
        with self.assertRaises(AssertionError):
            clut(np.empty(shape=(2,2))) # wrong shape
        with self.assertRaises(AssertionError):
            clut(np.empty(shape=(2,2,2))) # wrong number of colors
        with self.assertRaises(AssertionError):
            clut(300*np.ones(shape=(3,3,3))) # wrong number bit depth
        with self.assertRaises(AssertionError):
            clut(-300*np.ones(shape=(3,3,3))) # wrong number bit depth

        im = np.array(Image.open( opj('resources', 'bad_hald.png') ))
        self.assertTrue((im == clut(opj('resources', 'bad_hald.png'))).all())


    def test_methods(self):
        clut = CLUT()
        self.assertTrue((clut == CLUT()).all())
        self.assertTrue((CLUT() == clut.clut).all())
        self.assertFalse((clut == 1).all())

        clut[1,2,3] = 1
        self.assertTrue((clut[1,2,3] == [1,1,1]).all())

        clut.randomize()
        self.assertEqual(clut.shape, (256,256,256,3))

        clut.denoise(eps=.1)
        self.assertEqual(clut.shape, (256,256,256,3))

        self.assertEqual(clut.flat(4).shape, (64, 64, 3))
        with self.assertRaises(AssertionError):
            clut.flat(1)


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
        for i in ['testclut.png', 'testclut.png']:
            if os.path.exists(i):
                os.remove(i)
