# Python Color Lookup Tables

<p align="center">
:exclamation: work in progress :exclamation:<br>
not all features are implemented yet<br>
---
</p>

This package implements 3d [color lookup tables](https://en.wikipedia.org/wiki/3D_lookup_table)
(CLUT) for image manipulation along with a few conveniences such as interpolation (TODO) and
saving/loading from and to the [HaldCLUT](http://www.quelsolaar.com/technology/clut.html) format.

It is a Python-only implementation in the scope of CLUT generation and application
to 3d arrays.
All IO operations are still handled by [Pillow](https://github.com/python-pillow/Pillow).

If you can live with applying the CLUT filter through a C backend, there are
[other packages](https://github.com/homm/pillow-lut-tools) available.  


## Installation
In PIP editable mode:
* `git clone git@github.com:oiao/clut.git`
* `pip -e clut install`


## Example usage

``` python
import numpy as np
from PIL import Image
from clut import CLUT

# Initialize either an identity CLUT
clut = CLUT()
# Or load a HaldCLUT image from file
clut = CLUT('path/to/haldclut.png')

# Apply CLUT
im_out = clut('target/img.png') # directly to image files
im_out = clut(np.array(Image.open('target/img.png'))) # or to numpy arrays

# Save the modified image
im_out = Image.fromarray(im_out)
im_out.save('modified/img_out.png')

# Modify the CLUT by accessing the r,g,b channels through indexing
clut[120,0,255] = [0, 0, 0] # map rgb[120,0,255] to black

# Save the CLUT
clut.save('haldclut.png', size=8)
```
