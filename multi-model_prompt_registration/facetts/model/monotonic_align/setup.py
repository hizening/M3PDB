""" from https://github.com/jaywalnut310/glow-tts """

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'monotonic_align',
    ext_modules = cythonize("/gemini/space/zby/facetts/model/monotonic_align/core.pyx"),
    include_dirs=[numpy.get_include()]
)
