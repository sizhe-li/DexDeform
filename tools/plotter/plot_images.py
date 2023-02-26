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
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from .utils import match_table

fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')

def plot_image(table: pd.DataFrame, y=None, x=None, hue=None, y_level=None):
    table, (x, hue, y) = match_table(table, y, x, hue, y_level)
    nx = list(table[x].unique())
    ny = list(table[hue].unique())
    images: Optional[np.ndarray] = None
    shape: Optional[tuple] = None
    def find(a, b):
        for idx, c in enumerate(a):
            if c == b:
                return idx
        return -1
    for X, Hue, path in zip(table[x], table[hue], table[y]):
        img: Image = Image.open(path)
        if shape is not None:
            img = img.resize((shape[1], shape[0]))

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf", size=40)
        name = f"{X}:{Hue}"
        draw.text((0, 0), name, (255, 255, 255), font=font)

        img = np.array(img)
        if images is None:
            shape = np.array(img).shape
            images = np.zeros((shape[0] * len(nx), shape[1] * len(ny)) + shape[2:], dtype=img.dtype)
        idx = find(nx, X)
        idy = find(ny, Hue)
        images[idx * shape[0]:(idx+1)*shape[0], idy*shape[1]:(idy+1)*shape[1]] = img
    return images
