__version__ = "0.0.2"
__all__ = ["Store", "Frame"]

import sys

import numpy as np
import pandas as pd
import click
import h5py

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

def encode_index_item(x):
    if isinstance(x, int):
        x = str(x)
    if isinstance(x, str):
        return x.encode("utf-8")
    else:
        raise ValueError("Invalid index type")

def decode_index_item(x):
    return x.decode("utf-8")

def encode_index(xs):
    return list(map(encode_index_item, xs))

def decode_index(xs):
    return list(map(decode_index_item, xs))

#####
# API
#####

class Store(object):
    """
    A thin wrapper over h5py.File allowing storage
    and retrieval of numeric matrices from HDF5 files.
    """
    def __init__(self, path, mode="r"):
        self.path = path
        self.mode = mode
        self.handle = h5py.File(path, mode)

    def create(self, path, columns):
        nc = len(columns)
        group = self.handle.create_group(path)
        _columns = group.create_dataset("columns",
                data=encode_index(columns), dtype="|S100")
        _data = group.create_dataset("data", 
                maxshape=(None, nc), shape=(0, nc))
        _index = group.create_dataset("index",
                maxshape=(None,), shape=(0,), dtype="|S100")
        return Frame(group)

    def load(self, handle, path, delimiter="\t"):
        """
        Load delimited text from the provided file handle 
        into the Store at the given path.
        """
        columns = next(handle).strip("\n").split(delimiter)[1:]
        frame = self.create(path, columns)

        for i,line in enumerate(handle):
            key, *rest = line.strip("\n").split(delimiter)
            rest = list(map(as_float, rest))
            frame.add(key, rest)
        return frame

    def put(self, df, path):
        """
        Add a pandas DataFrame to the Store.
        """
        raise NotImplementedError

    def __getitem__(self, group):
        """
        Retrieve a Frame from this Store.
        """
        if not group in self.handle:
            raise KeyError
        return Frame(self.handle[group])

class Frame(object):
    """
    A single numeric matrix dataset.
    """
    def __init__(self, group):
        self.group = group
        self.columns = group["columns"]
        self.data = group["data"]
        self.index = group["index"]
        self._reindex()

    def _reindex(self):
        self.index_ix = \
                index_positions(decode_index(self.index))
        self.columns_ix = \
                index_positions(decode_index(self.columns))

    # I/O and adding data

    def add(self, key, row):
        """
        Add a single row to the Frame with the given row name (key).
        """
        assert len(row) == len(self.columns)
        i = self.data.shape[0]
        nc = len(self.columns)
        self.data.resize((i+1, nc))
        self.data[i,:] = row
        self.index.resize((i+1,))
        self.index[-1] = encode_index_item(key)

    def dump(self, handle=sys.stdout, float_format="%0.3f",
            delimiter="\t"):
        """
        Export the Frame to delimited text on the provided
        file handle.
        """
        print("", *decode_index(self.columns), 
                sep=delimiter, file=handle)
        for i in range(self.data.shape[0]):
            print(decode_index_item(self.index[i]), 
                    *[float_format % x for x in self.data[i,:]], 
                    sep=delimiter, file=handle)

    def to_frame(self):
        """
        Realize the entire Frame in memory and return as a pandas
        DataFrame.
        """
        df = pd.DataFrame(np.array(self.data), 
                index=decode_index(self.index), 
                columns=decode_index(self.columns))
        df.name = self.group.name
        return df

    # Query API

    def shape(self):
        """
        Return the dimensions of the Frame.
        """
        return self.data.shape

    def row(self, name):
        """
        Return the row with the given name as a pandas Series.
        """
        i = self.index_ix[name]
        s = pd.Series(self.data[i,:], 
                index=decode_index(self.columns))
        s.name = name
        return s

    def column(self, name):
        """
        Return the column with the given name as a pandas Series.
        """
        j = self.columns_ix[name]
        s = pd.Series(self.data[:,j], 
                index=decode_index(self.index))
        s.name = name
        return s

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
def load(h5file, path, delimiter):
    store = Store(h5file, mode="w")
    frame = store.load(sys.stdin, path, delimiter=delimiter)

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
    fmt = "%0." + str(precision) + "f"
    frame.dump(float_format=fmt, delimiter=delimiter)

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
def column(h5file, path, key):
    store = Store(h5file, mode="r")
    frame = store[path]
    column = frame.row(key)
    column.to_frame().to_csv(sys.stdout, sep="\t")

def main():
    cli()

if __name__ == "__main__":
    main()
