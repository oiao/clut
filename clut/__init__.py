def __autoimport(use_only=None):
    """
    Import all submodules with a defined `__all__` attribute from this directory level.
    An explicit sequence of modules can be provided in the `use_only` parameter
    to limit the import to these files.
    """
    import os
    from os.path import join as opj, dirname as opd
    ismodule = lambda x: x.endswith('.py') and not x.startswith('__init__') and not x.startswith('setup')
    here     = opd(__file__)

    ret     = []
    modules = [os.path.splitext(i)[0] for i in os.listdir(here) if ismodule(i)]
    if use_only:
        modules = [i for i in modules if i in use_only]

    for module in modules:
        tmp = __import__(module, globals=globals(), fromlist=['*'], level=1)
        if hasattr(tmp, '__all__'):
            ret += tmp.__all__
            for name in tmp.__all__:
                globals()[name] = vars(tmp)[name]
    return ret


from ._version import *

__all__  = ['__version__']
__all__ += __autoimport(['clut'])
