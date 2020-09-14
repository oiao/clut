import numpy as np
from PIL import Image
from scipy.interpolate  import RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor

__all__ = ['CLUT']

class CLUT:
    """
    3D ColorLookUpTable Class

    Used for mapping color spaces such that

    >>> clut[r_in, g_in, b_in] = [r_out, g_out, b_out]

    where r,g,b are the red green and blue channels of an image
    in the range of [0, 255].

    Attributes
    ----------
    clut : ndarray
        The array representing this instances CLUT
    shape : tuple
        The shpe of `self.clut`
    size : int
        The size of this CLUT instance, traslates to a 3d grid of `size**2`
        points along each axis
    """

    def __init__(self, size_path_array, depth=8):
        """
        Initialize a new CLUT instance.

        Parameters
        ----------
        size_path_array : int or str or ndarray
            The type of the variable decides how to generate the new CLUT:

            * _if int_, will generate an identity CLUT such that the resulting
              cube will be defined by `i**2` points per channel.
            * _if str_, will try and load a [HaldCLUT](http://www.quelsolaar.com/technology/clut.html)
              image from the path and generate a respective 3D CLUT
            * _if ndarray_, will assume direct input of a 3D CLUT
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
        self._hashtable = {}

        i = size_path_array

        if isinstance(i, int):
            maxi = int(np.sqrt(self._colors))
            assert 2 <= i <= maxi, f"For {self._depth}bit, size must be between 2 and {maxi}"

            points = np.linspace(0, self._colors-1, i**2)
            b,g,r  = np.meshgrid(*3*[points], indexing='ij')
            self.clut = np.stack([r,g,b]).T.astype(self._dtype)
            self.size = i

        elif isinstance(i, str):
            self.clut = self.load(i).astype(self._dtype)
            self.size = int(np.sqrt(self.clut.shape[0]))

        elif isinstance(i, np.ndarray):
            assert i.ndim == 4, "Table must be 4-dimensional"
            assert [i.shape[0]  == i.shape[j] for j in range(2)], "Table is not square"
            assert i.shape[-1]  == 3,   "Last axis must contain three elements (r,g,b)"
            assert i.shape[0]   >= 4,   "Matrix has to consist of at least 3 points in each dimension"
            assert 0 <= i.min() <= self._colors-1, f"RGB Values must be in the range of [0, {self._colors-1}]"
            assert 0 <= i.max() <= self._colors-1, f"RGB Values must be in the range of [0, {self._colors-1}]"
            self.size = int(np.sqrt(i.shape[0]))
            self.clut = i.astype(self._dtype)

        else:
            raise ValueError("Argument must either be an int (identity CLUT size), string (path to image file to be loaded) or ndarray (direct RGB CLUT).")


    def __call__(self, image, workers=None):
        """ Apply CLUT to `image`, return ndarray """
        if isinstance(image, Image.Image):
            image.convert(colors=self._depth)
            image = np.array(image)
        elif isinstance(image, str):
            image = np.array(Image.open(image))
        elif not isinstance(image, np.ndarray):
            raise ValueError(f"Image must either be a string (fpath), a PIL.Image object or ndarray. Got {type(image)} instead.")
        assert image.ndim      == 3, f"Not a valid image: array must be 3-dimensional, is {image.ndim}d instead."
        assert image.shape[-1] == 3, f"Not a valid image: array, inner element must be 3-dimensional (r,g,b), is {image.shape[-1]} instead."
        assert 0 <= image.min() <= self._colors-1, f"Image appears to be of a different color depth than this CLUT instance (which is {self._depth})."
        assert 0 <= image.max() <= self._colors-1, f"Image appears to be of a different color depth than this CLUT instance (which is {self._depth})."

        image = image.reshape((image.shape[0]*image.shape[1], 3))


        size = self.shape[0]
        if size < self._colors:
            grid     = 3*[np.linspace(0, self._colors, size)]
            interpol = RegularGridInterpolator(grid, self.clut)

            def atom(i):
                return interpol(i).astype(self._dtype)

            with ProcessPoolExecutor(max_workers=workers) as e:
                out = [i for i in e.map(atom, image)]
            out = np.squeeze(out).astype(self._dtype)

        return out


    def flat(self, size=None, swapaxes=False):
        """
        Generate a 2d mapping from the 3D CLUT by shaping into a matrix
        of `(step**3, step**3)`, where ``step = sqrt(clut.shape[0])``.
        To comply with the HaldCLUT format, use `swapaxes=True`, which swaps the
        red (x) and blue (z) channels.
        """
        rs = self.size**3
        t  = self.clut
        if swapaxes:
            # When saving, red and blue channels are swapped
            t = np.swapaxes(t, 0, 2)
        return t.reshape(rs, rs, -1)


    def __getitem__(self, elements):
        return self.clut[elements]


    @property
    def shape(self):
        return self.clut.shape


    def save(self, path, format=None):
        """ Saves the CLUT table as an HaldCLUT image to disk. """
        im = Image.fromarray(self.flat(swapaxes=True))
        im.save(path, format=format)


    @staticmethod
    def load(fpath):
        """ Loads a HaldCLUT from file, returns the associated 3D CLUT """
        im = Image.open(fpath)
        assert im.size[0] == im.size[1], "Image must be square."

        for i in range(2,65):
            if i**3 == im.size[0]:
                cubesize = i**2
                break
        else:
            raise ValueError("Could not determine CLUT size. Should be between 8 and 4096 px")

        clut = np.array(im).reshape((cubesize,cubesize,cubesize,3))
        clut = np.swapaxes(clut, 0, 2) # When saved, red and blue channels are swapped
        return clut






# Testing
# clut = CLUT(4)
# im = Image.open('IMG_3364.jpg')
# clut(im, workers=36)
