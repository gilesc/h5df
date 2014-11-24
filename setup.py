import os
from setuptools import setup

from h5df import __version__

setup(
    name = "h5df",
    version = __version__,
    author = "Cory Giles",
    author_email = "mail@corygil.es",
    description = "Library and CLI for storing numeric data frames in HDF5",
    license = "GPLv3",
    url = "https://github.com/gilesc/h5df",
    download_url="https://github.com/gilesc/h5df/tarball/"+ \
            __version__,
    modules="h5df",
    entry_points={
        "console_scripts": ["h5df = h5df.main" ]
    }
)
