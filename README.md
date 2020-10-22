# Python Color Lookup Tables
This package implements 3d [color lookup tables](https://en.wikipedia.org/wiki/3D_lookup_table)
(CLUT) for image manipulation and saving/loading from and to the
[HaldCLUT](http://www.quelsolaar.com/technology/clut.html) format.

The package was originally created for the purpose of re-creating CLUT filters from a set of edited and un-edited
images.
When is this necessary? Usually, if one wanted to 'rip' a filter from your favourite app,
all that needs to be done is loading an identity HaldCLUT image and applying the desired filter to it.
This is not easily possible if the application of a filter happens on a camera instead of an app,
since the raw image data is often modified directly.
The Ricoh GR for example has a much-loved positive film effect,
but I could not apply it to arbitrary images on my computer... until now.



## Installation
Dependencies:
[Pillow](https://github.com/python-pillow/Pillow),
[numpy](https://numpy.org/),
[scipy](https://www.scipy.org/).

Install in PIP editable mode:
* `git clone git@github.com:oiao/clut.git`
* `pip install -e clut`

For verification run the unit tests with `tests/run_tests.sh`.



## Usage: Basics
``` python
import numpy as np
from PIL import Image
from clut import CLUT
```

#### Initialize
``` python
# Either an identity CLUT
clut = CLUT()
# Or load a HaldCLUT image from file
clut = CLUT('path/to/haldclut.png')
```

#### Apply CLUT
``` python
im_out = clut('target/img.png') # directly to image files
im_out = clut(np.array(Image.open('target/img.png'))) # or to numpy arrays
```

#### Save the modified image
``` python
im_out = Image.fromarray(im_out)
im_out.save('modified/img_out.png')
```

#### Modify the CLUT
by accessing the r,g,b channels through indexing
``` python
clut[120,0,255] = [0, 0, 0] # map rgb[120,0,255] to black
```

#### Save the CLUT
``` python
clut.save('haldclut.png', size=8)
```

## Usage: CLUT Fitting
Too tired, will finish later
