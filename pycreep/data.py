"""Helper functions for loading data from file"""

import os.path

import pandas as pd


def load_data_from_file(fname, file_type=None):
    """
    Load data from file into a pandas data frame.

    Args:
        fname (str):        filename to read from

    Keyword Args:
        file_type (str):    test file type, options are "csv"
                            if None (default) then infer by
                            extension
    """
    if file_type is None:
        _, ext = os.path.splitext(fname)

        if ext in [".csv"]:
            file_type = "csv"
        else:
            raise ValueError(f"Could not infer file type from extension {ext}!")

    if file_type == "csv":
        return pd.read_csv(fname)
    raise ValueError(f"Invalid file type {ext}!")
