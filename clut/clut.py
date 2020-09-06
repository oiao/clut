import numpy as np
from PIL import Image

class CLUT:
    """
    3D ColorLookUpTable Class

    Used for mapping color spaces such that

    >>> clut[r_in, g_in, b_in] = (r_out, g_out, b_out)

    where r,g,b are the red green and blue channels of an image
    in the range of [0, 255].
    """

    def __init__(self, size_or_imagepath_or_array):
        """
        Initialize a new CLUT instance.

        Parameters
        ----------
        size_or_imagepath_or_array : int or str or ndarray
            The type of the variable decides how to generate the new CLUT:

            * _if int_, will generate an identity CLUT such that the resulting
              cube will be defined by `i**2` points
            * _if str_, will try and load a [HaldCLUT](http://www.quelsolaar.com/technology/clut.html)
              image from the path and generate a respective 3D CLUT
            * _if ndarray_, will assume direct input of a 3D CLUT
        """
        i = size_or_imagepath_or_array

        if isinstance(i, int):
            assert 2 <= i <= 16, "Size must be between 2 and 16"
            self.size = i
            points = np.linspace(0, 255, i**2)
            b,g,r = np.meshgrid(*3*[points], indexing='ij').astype(np.uint8)
            self.clut = np.stack([r,g,b]).T

        elif isinstance(i, str):
            self.clut = self.load(i)
            self.size  = int(np.sqrt(self.clut.shape[0]))

        elif isinstance(i, np.ndarray):
            assert i.ndim == 4, "Table must be 4-dimensional"
            assert [i.shape[0]  == i.shape[i] for i in range(2)], "Table is not square"
            assert i.shape[-1]  == 3, "Last axis must contain three elements (r,g,b)"
            assert 0 <= i.min() <= 255, "RGB Values must be in the range of [0, 255]"
            assert 0 <= i.max() <= 255, "RGB Values must be in the range of [0, 255]"
            self.size = int(np.sqrt(i.shape[0]))
            self.clut = i.astype(np.uint8)

        else:
            raise ValueError("Argument must either be an int (size of identity CLUT), string (path to image file to be loaded) or ndarray (direct RGB CLUT).")


    def flat(self, swapaxes=False):
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

    def __getitem__(self, id):
        return self.clut[id]

    @property
    def shape(self):
        return self.clut.shape

    def save(self, path, format=None):
        """ Saves the CLUT table as an HaldCLUT image to disk. """
        im = Image.fromarray(self.flat(swapaxes=True))
        im.save(path, foramt=format)

    @staticmethod
    def load(fpath):
        im = Image.open(fpath)
        assert im.size[0] == im.size[1], "Image must be square."

        for i in range(2,17):
            if i**3 == im.size[0]:
                cubesize = i**2
                break
        else:
            raise ValueError("Could not determine CLUT size. Should be between 8 and 4096 px")

        table = np.array(im).reshape((cubesize,cubesize,cubesize,3)).astype(np.uint8)
        table = np.swapaxes(table, 0, 2) # When saved, red and blue channels are swapped
        return table
