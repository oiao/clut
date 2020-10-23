import numpy as np
from typing import *
from PIL import Image, ImageFilter
from .clut import CLUT


__all__ = ['clutfit']

def clutfit(*images : Sequence[Tuple[str, str]], scale:float=0.5, shuffle=True, printstats=False, denoise=0) -> CLUT:
    """
    Fit a corresponding CLUT, given a series of (unfiltered, filtered) image tuples.

    Properties
    ----------
    images : sequence of tuples
        A sequence of tuples where each tuple contains the
        unedited input image in the first position and the same image with an applied
        CLUT filter in the second position.
        Elements can either be strings of file paths or `PIL.Image` objects.
    scale : 0 > float >= 1
        Apply a scaling to each image before processing.
        Setting this value to < 1 helps to reduce artifacts
        in the resulting CLUT
    shuffle : bool
        Shuffle the input and output colors before generating the CLUT.
        Setting this value to `True` helps to reduce artifacts in the resulting
        CLUT
    printstats : bool
        If `True`, will print the number of unique colors covered by all
        input images provided.
        Disabling this setting speeds up the fitting process
    denoise : float >= 0
        If greater than 0, will apply `CLUT.denoise(eps=denoise)` to the
        resulting CLUT. Doing so can greatly reduce artifacts.
        Typical values are 1e-4 (very good results, slow) to 0.1 (at least you tried, fast).

    Returns
    -------
    clut : CLUT
         A CLUT instance that re-creates the input-output mapping
         given in `images`
    """
    # clut = CLUT()
    RGB_IN  = []
    RGB_OUT = []

    # Convert to arrays
    for ims in images:
        im1 = _getim(ims[0])
        im2 = _getim(ims[1])
        assert im1.size == im2.size, 'Image sizes do not match'

        if 0 < scale < 1:
            resize = [int(scale*i) for i in im1.size]
            im1 = im1.resize(resize)
            im2 = im2.resize(resize)

        im1 = np.array(im1).reshape((-1,3))
        im2 = np.array(im2).reshape((-1,3))
        RGB_IN .append(im1)
        RGB_OUT.append(im2)

    RGB_IN  = np.concatenate(RGB_IN)
    RGB_OUT = np.concatenate(RGB_OUT)

    if shuffle:
        order   = np.arange(len(RGB_IN))
        np.random.shuffle(order)
        RGB_IN  = RGB_IN[order]
        RGB_OUT = RGB_OUT[order]

    # Remove duplicate colors
    if printstats:
        hashtable = set()
        mask = []
        for rgbin in RGB_IN:
            b = rgbin.tobytes()
            if b in hashtable:
                mask.append(False)
            else:
                hashtable.add(b)
                mask.append(True)
        RGB_IN, RGB_OUT = RGB_IN[mask], RGB_OUT[mask]

        oldlen = len(mask)
        newlen = len(RGB_IN)
        print(f"Unique colors: {newlen}. Duplicate colors: {oldlen-newlen}")
        print(f"This covers {100 * (newlen/(256**3)):.2f}% of the complete color space.")

    clut = CLUT()
    r,g,b   = RGB_IN[:,0], RGB_IN[:,1], RGB_IN[:,2]
    clut[r,g,b] = RGB_OUT
    if denoise > 0:
        if denoise < 1e-3:
            print(f"Denoising, this can usually take up to a couple of minutes ...")
        clut.denoise(eps=denoise)
    return clut


def _getim(im:Union[str, Image.Image]) -> Image.Image:
    if isinstance(im, Image.Image):
        return im
    elif isinstance(im, str):
        return Image.open(im).convert('RGB')
    else:
        raise ValueError(f"Provided image `{im}` is neither a filepath nor a `PIL.Image`.")
