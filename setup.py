from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

# Platform specific OpenMP flags for C++
if sys.platform.startswith("win"):
    compile_args = ["/openmp"]
    link_args = []
else:
    # Use C++11 standard for modern features
    compile_args = ["-fopenmp", "-std=c++11"]
    link_args = ["-fopenmp"]

extensions = [
    Extension(
        "svcj_wrapper",
        sources=["svcj_wrapper.pyx", "svcj.cpp"], # Note .cpp
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++"  # CRITICAL: Compile as C++
    )
]

setup(
    name="SVCJ_Factor_Engine",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    zip_safe=False,
)