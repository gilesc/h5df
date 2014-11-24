import os
from setuptools import setup

setup(
    name = "h5df",
    version = "0.0.1",
    author = "Cory Giles",
    author_email = "mail@corygil.es",
    description = "Library and CLI for storing numeric data frames in HDF5",
    license = "GPLv3",
    url = "http://github.com/gilesc/h5df",
    modules="h5df",
    entry_points={
        "console_scripts": ["h5df = h5df.main" ]
    }
)
