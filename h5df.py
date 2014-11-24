#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import click
import h5py

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

class Store(object):
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
        columns = next(handle).strip("\n").split(delimiter)[1:]
        frame = self.create(path, columns)

        for i,line in enumerate(handle):
            key, *rest = line.strip("\n").split(delimiter)
            rest = list(map(as_float, rest))
            frame.add(key, rest)
        return frame

    def put(self, df, path):
        raise NotImplementedError

    def __getitem__(self, group):
        return Frame(self.handle[group])

class Frame(object):
    def __init__(self, group):
        self.group = group
        self.columns = group["columns"]
        self.data = group["data"]
        self.index = group["index"]

    def add(self, key, row):
        assert len(row) == len(self.columns)
        i = self.data.shape[0]
        nc = len(self.columns)
        self.data.resize((i+1, nc))
        self.data[i,:] = row
        self.index.resize((i+1,))
        self.index[-1] = encode_index_item(key)

    def dump(self, handle=sys.stdout, float_format="%0.3f",
            delimiter="\t"):
        print("", *decode_index(self.columns), 
                sep=delimiter, file=handle)
        for i in range(self.data.shape[0]):
            print(decode_index_item(self.index[i]), 
                    *[float_format % x for x in self.data[i,:]], 
                    sep=delimiter, file=handle)

    def shape(self):
        return self.data.shape

@click.group()
def cli():
    pass

@cli.command()
@click.argument("h5path")
@click.argument("group")
def load(h5path, group):
    store = Store(h5path, mode="w")
    frame = store.load(sys.stdin, group)

@cli.command()
@click.argument("h5path")
@click.argument("group")
def dump(h5path, group):
    store = Store(h5path, mode="r")
    frame = store[group]
    frame.dump()

if __name__ == "__main__":
    cli()
