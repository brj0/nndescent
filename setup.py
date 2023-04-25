# From distutils.core import setup, Extension
from setuptools import dist, Extension, setup

# Hack to install numpy first
# dist.Distribution().fetch_build_eggs(["Cython", "numpy"])
import numpy as np
# import pybind11

__version__ = "0.0.0"

include_dirs = [
    # pybind11.get_include(),
    np.get_include(),
]

module = Extension(
    name="nndescent",
    sources=["utils.cpp", "dtypes.cpp", "nnd.cpp", "pybindings.cpp"],
    # sources=["utils.cpp", "dtypes.cpp", "nnd.cpp", "pybind11ings.cpp"],
    include_dirs=include_dirs,
    extra_compile_args =["-O3", "-march=native"],
    language="c++",
)

setup(
    name="nnd",
    version=__version__,
    description="C++ extension implementing nearest neighbour descent",
    install_requires=[
        "numpy",
        "h5py",
        "pynndescent",
    ],
    ext_modules=[module],
)
