from typing import *
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

__all__ = ['CLUT']

class CLUT:
    """
    3D ColorLookUpTable Class

    Used for mapping color spaces such that

    >>> clut[r_in, g_in, b_in] = [r_out, g_out, b_out]

    where r,g,b are the red green and blue channels of an image
    in the range of `[0, 2**depth)`.

    Attributes
    ----------
    clut : ndarray
        The array representing this instances CLUT
    shape : tuple
        The shape of `self.clut`
    """

    def __init__(self, path_or_array=None, depth=8, _nointerpol=False):
        """
        Initialize a new CLUT instance.

        Parameters
        ----------
        path_or_array : None, str, ndarray
            The type of the variable decides how to generate the new CLUT:

            * _if None_, will generate an full-size identity CLUT
            * _if str_, will try and load a [HaldCLUT](http://www.quelsolaar.com/technology/clut.html)
              image from the path and generate a respective 3D CLUT
            * _if ndarray_, will assume direct input of a 3D CLUT
        depth : int
            Number of bits/channel. Currently, only 8 (256 colors) is supported.
        """
        assert depth == 8, 'Currently, only a 8bit color depth is supported'
        self._depth  = int(depth)
        self._colors = 2**depth
        if depth <= 8:
            self._dtype = np.uint8
        elif dept <= 16:
            self._dtype = np.uint16
        else:
            self._dtype = np.uint32

        i = path_or_array

        if i is None:
            points = np.arange(self._colors)
            b,g,r  = np.meshgrid(*3*[points], indexing='ij')
            clut   = np.stack([r,g,b]).T

        elif isinstance(i, str):
            if i.endswith('.npy'):
                clut = np.load(i)
            else:
                clut = self.load(i, self._colors)

        elif isinstance(i, np.ndarray):
            assert i.ndim == 4, "Table must be 4-dimensional"
            assert [i.shape[0]  == i.shape[j] for j in range(2)], "Table is not square"
            assert i.shape[-1]  == 3,   "Last axis must contain three elements (r,g,b)"
            assert i.shape[0]   >= 4,   "Matrix has to consist of at least 3 points in each dimension"
            assert 0 <= i.min() <= self._colors-1, f"RGB Values must be in the range of [0, {self._colors-1}]"
            assert 0 <= i.max() <= self._colors-1, f"RGB Values must be in the range of [0, {self._colors-1}]"
            clut = i

        else:
            raise ValueError("Argument must either be an int (identity CLUT size), string (path to image file to be loaded) or ndarray (direct RGB CLUT).")

        if _nointerpol:
            self.clut = clut
        else:
            self._interpolate_to_full(clut)


    def __call__(self, image) -> np.ndarray:
        """ Apply CLUT to `image`, return ndarray """
        if isinstance(image, Image.Image):
            image.convert(colors=self._depth)
            image = np.array(image)
        elif isinstance(image, str):
            image = np.array(Image.open(image).convert('RGB'))
        elif not isinstance(image, np.ndarray):
            raise ValueError(f"Image must either be a string (fpath), a PIL.Image object or ndarray. Got {type(image)} instead.")
        assert image.ndim      == 3, f"Not a valid image: array must be 3-dimensional, is {image.ndim}d instead."
        assert image.shape[-1] == 3, f"Not a valid image: array, inner element must be 3-dimensional (r,g,b), is {image.shape[-1]} instead."
        assert 0 <= image.min() <= self._colors-1, f"Image appears to be of a different color depth than this CLUT instance (which is {self._depth})."
        assert 0 <= image.max() <= self._colors-1, f"Image appears to be of a different color depth than this CLUT instance (which is {self._depth})."

        shape  = image.shape
        im     = np.array(image).reshape(-1,3)
        r,g,b  = im[:,0], im[:,1], im[:,2]
        im_out = self.clut[r,g,b].reshape(shape)

        return im_out


    def flat(self, size=None, swapaxes=False):
        """
        Generate a 2d mapping from the 3D CLUT by shaping into a matrix
        of `(size**3, size**3)`.
        If `None`: ``size = sqrt(clut.shape[0])``
        To comply with the HaldCLUT format, use `swapaxes=True`, which swaps the
        red (x) and blue (z) channels.
        """
        shape = self.shape[0]
        if size is None or size**2 >= shape:
            clut = self.clut
        else:
            assert size >= 2, "Minimum compressed size is 2"
            points = np.linspace(0, self._colors-1, size**2).astype(self._dtype)
            rgb    = self.clut[points,points,points]
            b,g,r  = np.meshgrid(rgb[:,0], rgb[:,1], rgb[:,2], indexing='ij')
            clut   = np.stack([r,g,b]).T

        rs = int(np.sqrt(clut.shape[0])**3)
        if swapaxes:
            # When saving, red and blue channels are swapped
            clut = np.swapaxes(clut, 0, 2)
        return clut.reshape(rs, rs, -1)


    def __getitem__(self, elements):
        return self.clut[elements]

    def __setitem__(self,elements,value):
        value = np.array(value).astype(self._dtype)
        self.clut[elements] = value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.clut == other.clut
        elif isinstance(other, np.ndarray):
            return self.clut == other
        else:
            return np.zeros(shape=self.shape).astype(bool)

    @property
    def shape(self):
        return self.clut.shape

    def save(self, path:str, size:int=8, format=None):
        """
        Saves the CLUT table as an HaldCLUT image to disk
        reducing the resulting image to `(size**3, size**3)` pixels.
        """
        im = Image.fromarray(self.flat(size=size, swapaxes=True))
        im.save(path, format=format)

    def npsave(self, path):
        """
        Saves the CLUT in the numpy binary format.
        Loading from binary significantly speeds up re-initialization.
        """
        np.save(path, self.clut)

    @staticmethod
    def load(fpath, colors=256):
        """
        Loads a HaldCLUT from `fpath`, returns the associated 3D CLUT
        """
        im = Image.open(fpath)
        assert im.size[0] == im.size[1], "Image must be square."

        for i in range(2, int(np.sqrt(colors))):
            if i**3 == im.size[0]:
                cubesize = i**2
                break
        else:
            raise ValueError(f"Could not determine CLUT size. Should be between 8 and {int(np.sqrt(colors))} px")

        clut = np.array(im).reshape((cubesize,cubesize,cubesize,3))
        clut = np.swapaxes(clut, 0, 2) # When saved, red and blue channels are swapped
        return clut

    def randomize(self, mu=1, sigma=0.1):
        """
        Multiply the current CLUT by a random Gaussian distribution
        drawn from `~| N(mu, sigma**2) |`.
        """
        rand = np.abs( sigma*np.random.randn(*self.shape) + mu )
        clut = rand * self.clut.astype(float)
        clut[clut < 0] = 0
        clut[clut > self._colors-1] = self._colors-1
        self.clut = clut.astype(self._dtype)

    def gaussianfilter(self, sigma, **kwargs):
        """
        Apply a
        `scipy.gaussian_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html>`_
        to the CLUT instance.
        """
        self.clut = gaussian_filter(self.clut, sigma, **kwargs)

    def _interpolate_to_full(self,clut):
        # Use the full CLUT after init
        size = clut.shape[0]
        if size < self._colors:
            grid_in  = np.linspace(0, self._colors-1, size)
            interpol = RegularGridInterpolator(3*[grid_in], clut)
            points   = np.arange(self._colors)
            b,g,r    = np.meshgrid(*3*[points], indexing='ij')
            fullclut = np.stack([r,g,b]).T
            clut     = interpol(fullclut)
        self.clut = clut.astype(self._dtype)
