from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy, sys

args = ["/openmp"] if sys.platform.startswith("win") else ["-fopenmp"]
ext = [Extension("svcj_wrapper", ["svcj_wrapper.pyx", "svcj.c"], include_dirs=[numpy.get_include(), "."], extra_compile_args=args, extra_link_args=args)]

setup(name="SVCJ_Factor_Engine", ext_modules=cythonize(ext), zip_safe=False)