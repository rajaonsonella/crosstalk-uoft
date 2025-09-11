"""Module for handling datasets."""

import enum
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import scipy


FINGERPRINT_TYPES = [
    "ATOMPAIR",
    "MACCS",
    "ECFP6",
    "ECFP4",
    "FCFP4",
    "FCFP6",
    "TOPTOR",
    "RDK",
    "AVALON",
]

def basic_dataloader(
    filepath, x_col, y_col='DELLabel', max_to_load=1000, chunk_size=5000
):
    """
    Loads data from a Parquet file into memory, optionally as a sparse matrix.

    Args:
        filepath (str): Path to the Parquet file. This is the location of your data file on disk.
        x_col (str): Name of the feature column. This column should contain your input features as comma-separated strings.
        y_col (str, optional): Name of the label column. If None, only features are loaded. Defaults to 'DELLabel'. This column contains the target values (labels) for supervised learning.
        max_to_load (int, optional): Number of rows to load. If None, loads all rows. Defaults to 1000. Use this to work with a smaller sample of your data.
        chunk_size (int, optional): Number of rows to read at a time from disk. Defaults to 1000. This controls memory usage when loading large files.
        sparse (bool, optional): If True, returns a scipy sparse matrix for X. Defaults to False.

    Returns:
        X (np.ndarray or scipy.sparse.csr_matrix): Feature matrix.
        y (np.ndarray or None): Label array if y_col is provided, else None.
    """

    pf = pq.ParquetFile(filepath)
    columns = [x_col] + ([y_col] if y_col is not None else [])
    if max_to_load is None:
        max_to_load = pf.metadata.num_rows
    mats = []
    y_list = []
    loaded = 0

    n_chunks = int(np.ceil(max_to_load / chunk_size))
    pbar = tqdm(total=n_chunks, desc='Loading chunks')
    for batch in pf.iter_batches(columns=columns, batch_size=min(chunk_size, max_to_load)):
        batch_df = pa.Table.from_batches([batch]).to_pandas()
        remaining = max_to_load - loaded
        if len(batch_df) > remaining:
            batch_df = batch_df.iloc[:remaining]
        # Convert feature column to matrix
        exploded = batch_df[x_col].str.split(',', expand=True).astype(float, copy=False)
        mats.append(scipy.sparse.csr_matrix(exploded))
        if y_col is not None:
            y_list.append(batch_df[y_col].values)
        loaded += len(batch_df)
        del batch_df, exploded
        pbar.update(1)
        if loaded >= max_to_load:
            break
    pbar.n = pbar.total  # force bar to 100%
    pbar.refresh()
    pbar.close()

    X = scipy.sparse.vstack(mats)
    if y_col is not None and y_list:
        y = np.concatenate(y_list)
        return X, y
    else:
        return X

def parquet_split_dataloader(filename, x_col, y_col=None, chunk_size=10000, test_size=0.2, random_state=42, max_batches=None, max_to_load=None):
    """
    Loads data from a Parquet file in batches, splits each batch into train/test using sklearn's train_test_split,
    and optionally collects all test data. Allows stopping after a specified number of batches.

    Args:
        filename (str): Path to the Parquet file.
        x_col (str): Name of the feature column.
        y_col (str, optional): Name of the label column. Defaults to None.
        batch_size (int, optional): Number of rows per batch. Defaults to 1000.
        test_size (float, optional): Proportion of test data. Defaults to 0.2.
        random_state (int, optional): Random seed. Defaults to 42.
        max_batches (int, optional): Maximum number of batches to process. Defaults to None (all batches).
    Yields:
        (X_train, y_train), (X_test, y_test): Train and test splits for each batch.
    """
    pf = pa.parquet.ParquetFile(filename)
    columns = [x_col] + ([y_col] if y_col is not None else [])
    batch_iter = pf.iter_batches(columns=columns, batch_size=chunk_size)
    test_X_list, test_y_list = [], []
    for i, batch in enumerate(batch_iter):
        if max_batches is not None and i >= max_batches:
            break
        df = pa.Table.from_batches([batch]).to_pandas()
        X = df[x_col].str.split(',', expand=True).astype(float, copy=False).values
        y = df[y_col].values if y_col is not None else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        yield (X_train, y_train), (X_test, y_test)




@dataclass
class Dataset:
    """Basic dataset class holding a dataset."""

    x_col: str
    filename: str
    y_col: str = "DELLabel"
    test: bool = False
    X: np.ndarray = None
    y: np.ndarray = None

    def __post_init__(self):

        if self.x_col not in FINGERPRINT_TYPES:
            raise ValueError("Invalid fingerprint type")

        if self.test:
            df = pd.read_parquet(self.filename, columns=[self.x_col])
            self.y = None
        else:
            df = pd.read_parquet(self.filename, columns=[self.x_col, self.y_col])
            self.y = df[self.y_col].values
            df = df.drop(columns=[self.y_col])
            if not np.all(np.isin(self.y, [0, 1])):
                raise ValueError("y must contain only binary labels (0 or 1)")

        first_row = np.fromstring(df[self.x_col].iloc[0], sep=",", dtype=np.float32)
        self.X = np.empty((len(df), len(first_row)), dtype=np.float32)
        for i, x in enumerate(df[self.x_col].values):
            self.X[i, :] = np.fromstring(x, sep=",", dtype=np.float32)

        invalid_mask = np.isnan(self.X).any(axis=1)
        invalid_rows = np.where(invalid_mask)[0]
        if len(invalid_rows) > 0:
            print(f"Warning: Found {len(invalid_rows)} invalid rows in dataset")

        del df


def calculate_np_memory(shape, dtype=np.float32):
    """
    Calculates the memory in GB for the data buffer of a NumPy array
    given its shape and dtype.
    """
    num_elements = np.prod(shape)
    item_size = np.dtype(dtype).itemsize
    memory_in_bytes = num_elements * item_size
    return memory_in_bytes / (1024**3)


def calculate_feature_dims(features, dims):
    """
    Calculates the start and end indices for each feature in a stacked array.
    """
    stacked_map = {}
    offset = 0
    for c, dim in zip(features, dims):
        start_index = offset
        end_index = offset + dim
        stacked_map[c] = (start_index, end_index)
        offset += dim
    return stacked_map


def get_feature_dims(parquet_file, features: list[str]) -> list[int]:
    """Return the dimensions of the features."""
    first_row = next(parquet_file.iter_batches(batch_size=1)).to_pandas().iloc[0]
    dims = []
    for c in features:
        dims.append(len(np.array(first_row[c].split(","), dtype=np.float32)))
    return dims


def parse_pyarrow_string_array(str_arr):
    """
    Parses a PyArrow StringArray into a 2D NumPy array."""
    list_str_arr = pc.split_pattern(str_arr, pattern=",")
    flat_str_values = list_str_arr.values
    flat_float_values = pc.cast(flat_str_values, pa.float32(), safe=False)
    list_float_arr = pa.ListArray.from_arrays(list_str_arr.offsets, flat_float_values)
    arr = list_float_arr.to_numpy(zero_copy_only=False)
    # n_rows = len(list_float_arr)
    # dim = len(flat_float_values) // n_rows
    return np.vstack(arr)


def load_y(filename: str, y_col: str = "DELLabel", batch_size=1000) -> np.ndarray:
    """Loads label features from a Parquet file."""
    parquet_file = pq.ParquetFile(filename)
    pq_iter = parquet_file.iter_batches(batch_size=batch_size)
    values = []
    for record_batch in pq_iter:
        values.append(record_batch.column(y_col).to_numpy(np.float32))
    return np.hstack(values).reshape(-1, 1)


def load_x(
    filename: str, x_cols: list[str], y_col: str = "DELLabel", batch_size=1000
) -> np.ndarray:
    """Loads input features from a Parquet file."""
    parquet_file = pq.ParquetFile(filename)
    n_rows = parquet_file.metadata.num_rows
    print(f"Total rows: {n_rows}")
    feat_dims = get_feature_dims(parquet_file, x_cols)
    n_dim = sum(feat_dims)
    n_chunks = n_rows // batch_size
    print(f"Expected Memory for inputs: {calculate_np_memory((n_rows, n_dim)):.2f} GBs")
    feature_dims = calculate_feature_dims(x_cols, feat_dims)
    pq_iter = parquet_file.iter_batches(batch_size=batch_size, columns=None)
    x = np.zeros((n_rows, n_dim), dtype=np.float32)
    for i, record_batch in tqdm(enumerate(pq_iter), total=n_chunks):
        x_start = i * batch_size
        x_end = min((i + 1) * batch_size, n_rows)
        x_slice = slice(x_start, x_end)
        for c in x_cols:
            f_slice = slice(*feature_dims[c])
            chunked_arr = record_batch.column(c)
            values = parse_pyarrow_string_array(chunked_arr)
            x[x_slice, f_slice] = values
    return x
