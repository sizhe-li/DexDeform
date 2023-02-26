import os
import cv2
import numpy as np
import torch
import copy
import pickle
from typing import Optional, TypeVar, Type, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def match(args, cols):
    out = []
    for i in args:
        if i is not None:
            cols.remove(i)
    for i in args:
        if i is None:
            i = cols[0]
            cols = cols[1:]
        out.append(i)
    return out

def match_table(table: pd.DataFrame, y=None, x=None, hue=None, y_level=None):
    """
    Basically, given a panda series or a DataFrame
    Reorganize it into a flat table for further analysis
    """
    if y is not None:
        if y_level is None:
            if isinstance(table, pd.DataFrame) and y in table.columns:
                pass
            else:
                for level in table.index.names:
                    if y in table.index.unique(level):
                        y_level = level
        assert y_level is not None, f"Index: {table.index}"

    if y_level is not None:
        table = table.unstack(level=y_level)

    table: pd.DataFrame = table.reset_index()
    x, hue, y = match([x, hue, y], list(table.columns))
    return table, (x, hue, y)
