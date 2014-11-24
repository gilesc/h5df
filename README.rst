====
h5df
====

Python library and CLI for storing numeric data frames in HDF5.

Rationale
=========

Pandas has utilities for storing data frames in HDF5, but it uses
PyTables under the hood, which means it is limited to frames with a
relatively low number of columns (low 1000s).

This library is intended for storing and querying arbitrarily large
numeric matrices which have row and column names. It has a CLI
which can export/import to/from delimited text, or it can be used
from within Python with tight integration with Pandas.

This library stores only *numeric* matrices, so it cannot handle
data frames with mixed types (e.g., some strings and some numbers).

Installation
============

From PyPI:

.. code-block:: bash

    pip install h5df

Latest version:

.. code-block:: bash

    git clone https://github.com/gilesc/h5df.git
    cd h5df
    python setup.py install --user

This installs the CLI script "h5df", and a Python module with the
same name.

Usage
=====

.. code-block:: bash

    $ cat in.txt

        A   B   C
    X   1   2   3
    Y   4   5   5

    $ h5df load foo.h5 /my/path < in.txt
    $ h5df dump foo.h5 /my/path

        A   B   C
    ...

(more examples to come)

License
=======

GPLv3
