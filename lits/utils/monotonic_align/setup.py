from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

setup(
    name="monotonic_align",
    ext_modules=cythonize(
        [Extension("core", ["core.pyx"], include_dirs=[numpy.get_include()])],
    ),
)
