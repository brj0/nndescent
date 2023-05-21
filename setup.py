from setuptools import dist, Extension, setup
import numpy as np
import pybind11

__version__ = "0.0.0"

include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]

module = Extension(
    name="nndescent",
    sources=[
        "src/utils.cpp",
        "src/dtypes.cpp",
        "src/rp_trees.cpp",
        "src/nnd.cpp",
        "pybindings/pybind11ings.cpp",
    ],
    include_dirs=include_dirs,
    extra_compile_args=[
        "-Ofast",
        "-march=native",
        "-fopenmp",
        "-flto",
        "-fno-math-errno",
        "-g",
    ],
    extra_link_args=["-fopenmp"],
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
