import os
import cv2
import numpy as np
import torch
import copy
from typing import Optional, TypeVar, Type, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import collections
import statistics

DataPoint = collections.namedtuple('DataPoint', ('x', 'y', 'std'))
plt.style.use('seaborn')


def merge_curves(xy_list, bin_width=1000, max_x=None):
    def add_one_bin(x, y):
        nonlocal bin_list
        if len(x) == 0:
            return
        elif len(x) == 1:
            bin_list.append(DataPoint(statistics.mean(x), statistics.mean(y), 0.0))
        else:
            bin_list.append(DataPoint(statistics.mean(x), statistics.mean(y), statistics.stdev(y)))

    dp_list = []

    for (x, y) in xy_list:
        for i in range(len(x)):
            if max_x and x[i] > max_x:
                continue
            dp_list.append(DataPoint(x[i], y[i], 0.0))
    dp_list = sorted(dp_list, key=lambda dp: dp.x)
    bin_list = []
    x, y = [], []
    l = - int(bin_width / 2)
    for dp in dp_list:
        if dp.x > l + bin_width:
            add_one_bin(x, y)
            x, y = [], []
            while dp.x > l + bin_width:
                l += bin_width
        x.append(dp.x)
        y.append(dp.y)

    add_one_bin(x, y)
    return bin_list


def smooth(y, smoothingWeight=0.95):
    y_smooth = []
    last = y[0]
    for i in range(len(y)):
        y_smooth.append(last * smoothingWeight + (1 - smoothingWeight) * y[i])
        last = y_smooth[-1]
    return y_smooth


def plot_curve_with_shade(ax, dp_list, label, color='green', smoothingWeight=0.):
    big_dp = DataPoint(*zip(*dp_list))
    y_smooth = smooth(big_dp.y, smoothingWeight)
    std_smooth = smooth(big_dp.std, smoothingWeight)
    y_up_smooth = [t[0] + t[1] for t in zip(y_smooth, std_smooth)]
    y_down_smooth = [t[0] - t[1] for t in zip(y_smooth, std_smooth)]

    # y_up_smooth = smooth([t[0] + t[1] for t in zip(y_smooth, big_dp.std)])
    # y_down_smooth = smooth([t[0] - t[1] for t in zip(y_smooth, big_dp.std)])

    x = [i for i in big_dp.x]
    ax.fill_between(x, y_up_smooth, y_down_smooth, facecolor=color, alpha=0.2)
    ax.plot(x, y_smooth, color=color, label=label, linewidth=3)


def plot_curves(data, y=None, y_level=None, width=5, bin_width=200, max_x=np.inf, mode='fig'):
    # TODO: support plotting multiple curve and calculate the std..
    y_label = y
    if y_label is not None:
        assert y_label is not None
        if y_level is None:
            for i in range(len(data.columns.names) - 1, -1, -1):
                if y_label in data.columns.unique(i):
                    y_level = i
                    break
        assert y_level is not None, "must provide a y_level or it's a value in one level of columns"
        data = data.xs(y_label, level=y_level, axis=1)

    if y_label is None:
        y_label = "Value"
    if len(data.columns.names) == 1:
        # https://stackoverflow.com/questions/14744068/prepend-a-level-to-a-pandas-multiindex
        data = pd.concat({"Figure": data}, axis=1, names=[None])
    envs = data.columns.unique(0)
    labels = data.columns.unique(1)

    if mode == 'fig':
        width = min(width, len(envs))
        A, B = (len(envs) + width - 1)//width, width
        fig, subfigures = plt.subplots((len(envs) + width-1) // width, width, figsize=(35, 15))
        if A == 1:
            subfigures = [subfigures]
        if B == 1:
            subfigures = [subfigures]
        tmp = []
        for i in subfigures:
            for j in i:
                tmp.append(j)
        subfigures = tmp

        title_font = {'fontname': 'Arial', 'size': '44', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
        axis_font = {'fontname': 'Arial', 'size': '24'}
        color_list = [
            '#CB4042',
            'green',
            '#005CAF',
            '#F05E1C',
            '#66327C',
            'slategray',
            '#656765',
            "yellow",
            "black"
        ]

        for name, ax in zip(envs[::-1], subfigures):
            ax.tick_params(axis='both', labelsize=20)

            for i, l in enumerate(labels):
                if not (name, l) in data.columns:
                    continue
                xy = data.loc[:, (name, l)].dropna()
                if isinstance(xy, pd.DataFrame) and len(xy.columns)>1:
                    alls = []
                    for key in xy.columns:
                        alls.append(xy[key].dropna())
                    xy = pd.concat(alls)

                xy = (xy.index.to_numpy(), xy.to_numpy().reshape(-1))
                if len(xy[0]) == 0:
                    continue
                dp_list = merge_curves([xy], bin_width=bin_width, max_x=max_x)
                plot_curve_with_shade(ax, dp_list, label=l, color=color_list[i], smoothingWeight=0.3)

            ax.set_xlabel(data.index.name, **axis_font)
            ax.set_ylabel(y_label, **axis_font)

            ax.set_facecolor((1.0, 1., 1.))
            ax.set_title(name, **title_font)
        handles, _labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, \
                   prop={'size': 40}, bbox_to_anchor=(0.45, 0.12))
        fig.subplots_adjust(top=0.95, left=0.155, right=0.99, bottom=0.2)
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        return fig
    elif mode.startswith('tb'):
        key = pd.util.hash_pandas_object(data, index=False).sum()
        from ...logger import getLogger
        if os.path.exists(f"/tmp/{key}"):
            os.system(f"rm -rf /tmp/{key}")
        for l in labels:
            # create loggers for different methods..
            logger = getLogger('env', f"/tmp/{key}/{l}").add_tb()
            for name in envs:
                if (name, l) in data.columns:
                    xy = data.loc[:, (name, l)]
                    xy = (xy.index.to_numpy(), xy.to_numpy().reshape(-1))
                    for a, b in zip(*xy):
                        logger.tb({name: b}, step=a)
        sp = mode.split('-')
        port = '' if len(sp) < 3 else ' --port' + sp[-1]
        cmd = f"tensorboard --logdir /tmp/{key}"
        if len(sp) == 1:
            print(f"Curves saved to /tmp/{key}. Run\n"
                  f"    {cmd}\n"
                  f" for visualization")
        else:
            os.system(cmd + port)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented!")
