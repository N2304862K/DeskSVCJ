from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy, sys

# Ensure numpy is installed
setup_requires = ['numpy']

if sys.platform.startswith("win"):
    args = ["/openmp"]
else:
    args = ["-fopenmp"]

ext = [Extension("svcj_wrapper", ["svcj_wrapper.pyx", "svcj.c"], include_dirs=[numpy.get_include(), "."], extra_compile_args=args, extra_link_args=args)]

setup(
    name="SVCJ_Factor_Engine", 
    ext_modules=cythonize(ext), 
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=['numpy>=1.20.0', 'pandas>=1.3.0']
)