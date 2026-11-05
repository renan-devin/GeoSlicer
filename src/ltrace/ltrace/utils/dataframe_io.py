"""Pickle-free DataFrame persistence for job result files.

Result files are exchanged between the cluster and the workstation through shared folders, so they are stored as
Parquet, which (unlike pickle) cannot execute code when read.
"""

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _to_json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _is_nested(value) -> bool:
    return isinstance(value, (dict, list, tuple, set, np.ndarray))


def write_dataframe(df: pd.DataFrame, path: PathLike) -> None:
    """Write ``df`` to ``path`` as Parquet. Nested values (dicts, lists, arrays) are stored as JSON strings."""
    df = df.copy()
    df.columns = [str(column) for column in df.columns]
    for column in df.columns:
        if df[column].dtype == object and df[column].map(_is_nested).any():
            df[column] = df[column].map(
                lambda value: json.dumps(value, default=_to_json_compatible) if _is_nested(value) else value
            )
    df.to_parquet(str(path), engine="pyarrow")


def read_dataframe(path: PathLike) -> pd.DataFrame:
    """Read a DataFrame written by :func:`write_dataframe`."""
    return pd.read_parquet(str(path), engine="pyarrow")
