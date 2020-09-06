from setuptools import setup, find_packages
from os.path import join as opj

NAME = 'clut'
DESCR = 'Python Color Lookup Tables'
packages = [NAME]+[f'{NAME}.'+i for i in find_packages(NAME)]

with open(opj(NAME, '_version.py')) as f:
    exec(f.read())

setup(
    name             = NAME,
    version          = __version__,
    author           = 'Leo Komissarov',
    url              = f'https://github.com/oiao/{NAME}',
    download_url     = f'https://github.com/oiao/{NAME}/archive/master.zip',
    license          = 'GPLv3+',
    description      = DESCR,
    classifiers      = [
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python :: 3.6',
    ],
    keywords         = ['image editing', 'CLUT', '3DCLUT', 'HaldCLUT', 'color lookup table', 'image filtering'],
    python_requires  = '>=3.6',
    install_requires = ['numpy', 'Pillow'],
    packages         = packages,
    package_dir      = {NAME : NAME},
    package_data     = {NAME : ['tests/*']},
    # scripts          = [opj('scripts', NAME)],
)
