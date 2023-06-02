from setuptools import Extension, setup
import numpy as np
import re
import pybind11


def get_version_string():
    """Extracts version number from source code."""
    pattern = 'const std::string PROJECT_VERSION = "(.*)";'
    with open("src/nnd.h", "r") as f:
        code = f.read()
        result = re.search(pattern, code).group(1)
    return result


include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]

module = Extension(
    name="nndescent",
    sources=[
        "pybindings/pybind11ings.cpp",
        "src/dtypes.cpp",
        "src/nnd.cpp",
        "src/rp_trees.cpp",
        "src/utils.cpp",
    ],
    include_dirs=include_dirs,
    extra_compile_args=[
        "-Ofast",
        "-flto",
        "-fno-math-errno",
        "-fopenmp",
        "-g",
        "-march=native",
    ],
    # CXXFLAGS for debugging
    # extra_compile_args=[
    # "-O0",
    # "-Wall",
    # "-Wextra",
    # "-fno-inline-functions,"
    # "-fno-stack-protector",
    # "-g",
    # "-pg",
    # ],
    extra_link_args=["-fopenmp"],
    language="c++",
)

setup(
    name="nndescent",
    version=get_version_string(),
    description="C++ extension implementing nearest neighbour descent",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "full": ["h5py", "matplotlib", "pynndescent", "seaborn", "sklearn"],
    },
    ext_modules=[module],
)
