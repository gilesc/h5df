"""
Library and CLI for storing and querying numeric labeled matrices in HDF5.
"""

__all__ = ["Store", "Frame"]

import itertools
import sys

import numpy as np
import pandas as pd
import click
import h5py
from pydoc import locate

###################
# Utility functions
###################

def index_positions(xs):
    return dict(map(reversed, enumerate(xs)))

def as_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

def is_df_group(g):
    return ("data" in g) and ("index" in g) and ("columns" in g)

def get_encoder_for_type(t, encoding="utf-8"):
    if t is str:
        def encoder(xs):
            if not isinstance(xs, np.ndarray):
                xs = np.array(xs)
            xs = xs.astype(np.unicode_)
            return np.char.encode(xs, encoding).astype("|S100")
    elif t is int:
        def encoder(xs):
            assert isinstance(xs[0], int)
            if not isinstance(xs, np.ndarray):
                xs = np.array(xs)
            return xs
    else:
        raise TypeError("Indexes must be str or int")
    return encoder

def get_encoder(index, encoding="utf-8"):
    assert len(set(map(type, index))) == 1
    xt = type(index[0])
    return get_encoder_for_type(xt)

def get_decoder(index_type_str, encoding="utf-8"):
    assert isinstance(index_type_str, str)
    t = locate(index_type_str)
    if t is str:
        return lambda xs: np.char.decode(np.array(xs), encoding)
    elif t is int:
        return lambda xs: np.array(xs)

#####
# API
#####

class Store(object):
    """
    A thin wrapper over h5py.File allowing storage
    and retrieval of numeric, labeled matrices from 
    HDF5 files.
    """
    def __init__(self, path, driver=None, mode="r"):
        self.path = path
        self.mode = mode
        if driver is None:
            driver = "core" if mode == "r" else "sec2"
        self.handle = h5py.File(path, mode=mode, driver=driver)

    def create(self, path, columns, index_type=str):
        assert index_type in (str, int)

        nc = len(columns)
        group = self.handle.create_group(path)
        group.attrs["index_type"] = index_type.__name__
        group.attrs["columns_type"] = type(columns[0]).__name__
        columns_encoder = get_encoder(columns)
        index_encoder = get_encoder([index_type()])

        _columns = group.create_dataset("columns",
                data=columns_encoder(columns))
        _data = group.create_dataset("data", 
                maxshape=(None, nc), shape=(0, nc))
        _index = group.create_dataset("index",
                maxshape=(None,), 
                data=index_encoder([]))
        return Frame(group)

    def load(self, handle, path, verbose=False, delimiter="\t"):
        """
        Load delimited text from the provided file handle 
        into the Store at the given path, creating and returning 
        a new Frame.
        """

        chunk_size = 100

        chunks = iter(pd.read_csv(handle, sep=delimiter, 
                chunksize=chunk_size,
                header=0, engine="c", 
                quoting=3, index_col=0, 
                na_values=["nan"]))
        df = next(chunks)
        columns = list(map(str, df.columns))
        frame = self.create(path, columns, index_type=type(df.index[0]))

        chunks = itertools.chain([df], chunks)
        for i,chunk in enumerate(chunks):
            frame.append(chunk)
            if verbose:
                msg = "* {} : row {}".format(path, (i+1) * chunk_size)
                print(msg, file=sys.stderr)
        return frame

    def put(self, path, df):
        """
        Add a pandas DataFrame to the Store, returning a new Frame.
        """
        columns = list(map(str, df.columns))
        frame = self.create(path, columns, index_type=type(df.index[0]))
        frame.append(df)
        return frame

    def __getitem__(self, group):
        """
        Retrieve a Frame from this Store.
        """
        if not group in self.handle:
            raise KeyError
        return Frame(self.handle[group])

    def __iter__(self):
        o = []
        def visitor(key):
            if is_df_group(self.handle[key]):
                o.append(key)
        self.handle.visit(visitor)
        return iter(o)

class Frame(object):
    """
    A single numeric matrix dataset.
    """
    def __init__(self, group):
        self._group = group
        self._columns = group["columns"]
        self._data = group["data"]
        self._index = group["index"]

        index_t = self._group.attrs["index_type"]
        columns_t = self._group.attrs["columns_type"]
        self._index_decoder = get_decoder(index_t)
        self._index_encoder = get_encoder_for_type(locate(index_t))
        self._columns_decoder = get_decoder(columns_t)

        self._reindex()

    def _reindex(self):
        self._index_ix = index_positions(self.index)
        self._columns_ix = index_positions(self.columns)

    ####################
    # Generic properties
    ####################

    @property
    def index(self):
        return self._index_decoder(self._index)

    @property
    def columns(self):
        return self._columns_decoder(self._columns)

    @property
    def nc(self):
        return self._data.shape[1]

    @property
    def nr(self):
        return self._data.shape[0]

    @property
    def shape(self):
        """
        Return the dimensions of the Frame.
        """
        return self._data.shape

    #############
    # Adding data
    #############

    def add(self, key, row):
        """
        Add a single row to the Frame with the given row name (key).
        """
        assert len(row) == len(self._columns)
        nr = self.nr
        self._data.resize((nr+1, self.nc))
        self._data[i,:] = row
        self._index.resize((nr+1,))
        self._index[-1] = self._index_encoder([key])

    def append(self, df):
        """
        Append a DataFrame (with the same columns) to the Frame.
        """
        assert df.shape[1] == len(self._columns)
        nr = self.nr
        nnr = nr + df.shape[0]
        self._data.resize((nnr, self.nc))
        data = df.as_matrix()
        self._data[nr:,:] = data
        self._index.resize((nnr,))
        self._index[nr:] = self._index_encoder(df.index)

    #############
    # Bulk output
    #############

    def dump(self, handle=sys.stdout, precision=3, 
            delimiter="\t"):
        """
        Export the Frame to delimited text on the provided
        file handle.
        """
        assert isinstance(precision, int) and precision >= 0
        number_format="%0.{}f".format(precision)

        print("", *self.columns, 
                sep=delimiter, file=handle)
        for i,ix in zip(range(self.shape[0]), self.index):
            print(ix, *[(number_format % x).rstrip("0").rstrip(".")
                for x in self._data[i,:].round(precision)], 
                sep=delimiter, file=handle)

    def to_frame(self):
        """
        Realize the entire Frame in memory and return as a pandas
        DataFrame.
        """
        df = pd.DataFrame(np.array(self._data), 
                index=self.index,
                columns=self.columns)
        df.name = self._group.name
        return df

    ################
    # Subset queries
    ################

    def row(self, name):
        """
        Return the row with the given name as a pandas Series.
        """
        i = self._index_ix[name]
        s = pd.Series(self._data[i,:], index=self.columns)
        s.name = name
        return s

    def col(self, name):
        """
        Return the column with the given name as a pandas Series.
        """
        j = self._columns_ix[name]
        s = pd.Series(self._data[:,j], index=self.index)
        s.name = name
        return s

    def rows(self, names):
        """
        Return the subset of rows indexed by the given
        names as a pandas DataFrame.
        """
        ixs = [self._index_ix[n] for n in names]
        return pd.DataFrame(self._data[ixs,:], 
                index=self.index[ixs],
                columns=self.columns)

    def cols(self, names):
        """
        Return the subset of columns indexed by the given
        names as a pandas DataFrame.
        """
        ixs = [self._columns_ix[n] for n in names]
        return pd.DataFrame(self._data[:,ixs],
                index=self.index,
                columns=self.columns[ixs])

########################
# Command-line interface
########################

@click.group()
def cli():
    pass

@cli.command(help="Import from delimited text.")
@click.argument("h5file")
@click.argument("path")
@click.option("--delimiter", "-d", default="\t",
        help="Column delimiter (default tab).")
@click.option("--verbose", "-v", 
        help="Output progress on stderr",
        default=False, is_flag=True)
def load(h5file, path, delimiter, verbose):
    store = Store(h5file, mode="a")
    frame = store.load(sys.stdin, path, verbose=verbose, delimiter=delimiter)

@cli.command(help="Export to delimited text.")
@click.argument("h5file")
@click.argument("path")
@click.option("--precision", "-p", type=int, default=3,
        help="Number of digits after the decimal to print.")
@click.option("--delimiter", "-d", default="\t",
        help="Column delimiter (default tab).")
def dump(h5file, path, precision, delimiter):
    assert precision >= 0

    store = Store(h5file, mode="r")
    frame = store[path]
    frame.dump(precision=precision, delimiter=delimiter)

@cli.command(help="Retrieve data for a single row.")
@click.argument("h5file")
@click.argument("path")
@click.argument("key")
def row(h5file, path, key):
    store = Store(h5file, mode="r")
    frame = store[path]
    row = frame.row(key)
    row.to_frame().to_csv(sys.stdout, sep="\t")

@cli.command(help="Retrieve data for a single column.")
@click.argument("h5file")
@click.argument("path")
@click.argument("key")
def col(h5file, path, key):
    store = Store(h5file, mode="r")
    frame = store[path]
    column = frame.col(key)
    column.to_frame().to_csv(sys.stdout, sep="\t")

@cli.command(help="Select subframe containing row names provided on stdin")
@click.argument("h5file")
@click.argument("path")
def rows(h5file, path):
    keys = [line.strip("\n") for line in sys.stdin]
    store = Store(h5file, mode="r")
    frame = store[path]
    frame.rows(keys).to_csv(sys.stdout, sep="\t")
 
@cli.command(help="Select subframe containing column names provided on stdin")
@click.argument("h5file")
@click.argument("path")
def cols(h5file, path):
    keys = [line.strip("\n") for line in sys.stdin]
    store = Store(h5file, mode="r")
    frame = store[path]
    frame.cols(keys).to_csv(sys.stdout, sep="\t")
 
def main():
    # Don't display BrokenPipeError if output commands are truncated by 
    # an external program
    from signal import signal, SIGPIPE, SIG_DFL
    signal(SIGPIPE, SIG_DFL)

    cli()

if __name__ == "__main__":
    main()
