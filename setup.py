import os
from typing import Dict

from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

about = dict()

with open(os.path.join(HERE, "pyaez", "__version__.py"), "r") as f:
    exec(f.read(), about)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PyAEZ",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__email__"],
    description="PyAEZ is a python package consisted of many algorithms related to Agro-ecalogical zoning (AEZ) framework.",
    py_modules=["pyaez"],
    # package_dir={'':'src'},
    license="MIT License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gicait/PyAEZ",
    packages=find_packages(),
    keywords=[
        "AEZ",
        "Python",
        "Agro-ecological Zoning",
        "Climate Regime",
        "Crop Simulations",
        "Climate Constraints",
        "Soil Constraints",
        "Terrain Constraints",
        "Economic Suitability Analysis",
        "Utility Calculation",
        "Biomass Calculations",
        "Evapotranspiration Calculation",
        "CropWat calculations",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'gdal',
        'numba',
    ],
    extras_require={"dev": [
        "pytest",
        "black",
        "flake8",
        "sphinx>=1.7",
        "pre-commit"
    ]},
    python_requires=">=3.6",
)
