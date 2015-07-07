import tempfile

from h5df import Store
import pandas as pd
import numpy as np

np.random.seed(0)

with tempfile.NamedTemporaryFile() as tf:
    path = tf.name
    # Create a Store object; the default mode is read-only. 
    # See http://docs.h5py.org/en/latest/high/file.html for available modes
    store = Store(path, mode="a")
    index = ["A","B","C"]
    columns = ["V","W","X","Y","Z"]
    mkdf = lambda: pd.DataFrame(np.random.random((3,5)), index=index, columns=columns)
    df1 = mkdf()
    df2 = mkdf()
    store.put("/frames/1", df1)
    store.put("/frames/2", df2)

    for key in store: print(key)

    df1 = store["/frames/1"]
    print(df1.column("W"))
    # To get a normal (in-memory) pandas.DataFrame
    print(df1.to_frame())
