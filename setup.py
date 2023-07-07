from setuptools import Extension, setup
import glob
import numpy as np
import os
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

# Get the value of the NND_DEBUG environment variable
debug = os.getenv("NND_DEBUG") == "1"

# Set the compiler flags based on the debug flag
if debug:
    print("\n*** Debugging compiler flags are used. ***\n")
    compile_args = [
        "-O0",
        "-Wall",
        "-Wextra",
        "-fno-stack-protector",
        "-g",
        "-pg",
    ]
else:
    compile_args = [
        "-DALL_METRICS",
        "-Ofast",
        "-flto",
        "-fno-math-errno",
        "-fopenmp",
        "-g",
        "-march=native",
    ]

module = Extension(
    name="nndescent",
    sources=glob.glob("pybindings/*.cpp") + glob.glob("src/*.cpp"),
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=["-fopenmp"],
    language="c++",
)

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="nndescent",
    version=get_version_string(),
    description="C++ extension implementing nearest neighbour descent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "full": [
            "h5py",
            "matplotlib",
            "pynndescent",
            "scipy",
            "seaborn",
            "sklearn",
        ],
    },
    ext_modules=[module],
    keywords="nearest neighbor, knn, ANN",
    license="BSD 2-Clause License",
    author="Jon Brugger",
    url="https://github.com/brj0/nndescent",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
