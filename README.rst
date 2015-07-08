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

To select an individual row or column, use "h5py row|column":

.. code-block:: bash

    $ h5df row foo.h5 X
    

CLI flags
=========

Use ``h5df <cmd> --help`` for a full listing of options, but a few useful ones:

- ``h5df load -v`` : will output progress as a matrix is loaded (every 100 rows)
- ``h5df <any output command> -p N`` will output values with decimal precision N

API
===

The two main classes are ``h5df.Store`` and ``h5df.Frame``, representing a HDF5
file and individual data frame, respectively. Here is some example usage:

.. code-block:: python

    >> from h5df import Store
    >> import pandas as pd
    >> import numpy as np
    >> np.random.seed(0)

    # Create a Store object; the default mode is read-only. 
    # See http://docs.h5py.org/en/latest/high/file.html for available modes
    >> store = Store("test.h5df", mode="a")
    >> index = ["A","B","C"]
    >> columns = ["V","W","X","Y","Z"]
    >> mkdf = lambda: pd.DataFrame(np.random.random((3,5)), index=index, columns=columns)
    >> store.put("/frames/1", mkdf())
    >> store.put("/frames/2", mkdf())

    # Iterate through HDF5 paths corresponding to Frame objects
    >> for key in store: print(key)

    >> df1 = store["/frames/1"]

    # Various selection options

    # returns pandas.Series
    >> df1.column("W") 
    >> df1.row("A")

    # returns a pandas.DataFrame
    >> df1.rows(["A","C"]) 
    >> df1.cols(["W","Y"])

    # Returns the whole Frame as a pandas.DataFrame
    >> df1.to_frame()

The full list of methods supported by ``h5df.Frame`` is:

- ``Frame.row(key)`` and ``Frame.col(key)`` - return a ``pandas.Series``
  corresponding to the row/column

- ``Frame.rows(keys)`` and ``Frame.cols(keys)`` - given a list of row/column
  index names, return an in-memory ``pandas.DataFrame`` corresponding to the
  subset of the overall ``Frame`` containing the desired rows or columns

- ``Frame.shape`` - returns a tuple of (# rows, # columns)

- ``Frame.to_frame()`` - return the entire ``Frame`` as an in-memory
  ``pandas.DataFrame``. Make sure you have enough memory!

- ``Frame.add(key, data)`` - add a new row to the matrix with the given unique key. Due to the way of

Storage format
==============

Each ``h5df.Frame`` is stored as an HDF5 Group containing 3 Datasets: ``index``
and ``columns`` (both are 1D arrays of 8-byte integers or UTF-8 encoded binary
strings), and ``data`` (a 2D double array). 

The Group also contains a few HDF5 attributes:
- ``h5df.index_type`` and ``h5df.columns_type`` : a string, either "str" or "int", 
  marking the data type of each of the corresponding indices
- ``h5df.is_frame`` : a boolean, always set to true, which indicates that
  this Group contains valid ``Frame`` data

Because of this design, it is possible to store a ``Frame`` "inside" the Group
containing another ``Frame``, but is not recommended in case of future format
changes (and because it is confusing).

Performance notes
=================

Data is indexed row-major. Thus row-based queries will be much faster.
Generally you should pre-transpose your matrix before putting it into the
``Store`` to ensure that the most frequently queried axis will be on the rows.

The ``h5df.Store()`` constructor takes a keyword argument, "driver". The full
description of available drivers is at
http://docs.h5py.org/en/latest/high/file.html . For Linux systems, the default
stdio-based driver is "sec2", whereas "core" will memory-map the whole HDF5
file. If your system supports it and the file is frequently used (and therefore
will be in your OS page cache), "core" may be faster, especially for reads.

Limitations
===========

Currently there is no way to select rows by numeric index location (i.e., the
equivalent to ``pandas.DataFrame.iloc``).

Encoding and decoding indices (from unicode to binary) is a little slow,
meaning that quick queries are slower than they could be.

Iterating through the frames in a HDF5 file, ``Store.__iter__`` is quite
inefficient if the file contains large numbers of frames.

For ``Frame.dump()``, output formatting is not vectorized (slower than
necessary).

String indices are stored as ``np.dtype("|S100")`` encoded as ``"utf-8"``.
This has several practical consequences: 

1. index and column names are currently limited to 100 UTF-8 characters
2. UTF-8 encoding is hardcoded and other encodings are not supported 
   (thus, characters from other encodings that will fail 
   ``str.encode("utf-8")`` will cause an error.

There are plans to fix these limitations in future versions.

Potential gotchas
=================

When matrices are renamed or deleted using ``Store.rename()`` or
``Store.delete()``, existing ``Frame`` objects based on this data will not be
notified of this change and their behavior is undefined. Most likely an attempt
to use dangling ``Frame`` objects will result in an error, but may return
erroneous results for some methods.

It is up to the user to avoid this situation. When renaming a ``Frame``, the
user should subsequently get a new ``Frame`` object from the destination path
using ``Store.__getitem__`` if they plan to continue to use the data.

License
=======

AGPLv3+
