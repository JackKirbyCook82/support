# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 2026
@name:   Plotter Objects
@author: Jack Kirby Cook

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from types import NoneType
from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict as ODict
from mpl_toolkits.mplot3d import Axes3D

from support.decorators import Dispatchers

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Plotter", "Plot"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class PlotType(Enum):
    SCATTER, SURFACE = range(2)

@dataclass(frozen=True)
class Dataset: data: object; type: object; color: str

@dataclass(frozen=True)
class Dimensions:
    x: Optional[str] = None; y: Optional[str] = None; z: Optional[str] = None

    def __iter__(self): yield self.x; yield self.y; yield self.z


class Plot(object):
    def __init__(self, *args, title=None, labels=None, axes=None, **kwargs):
        assert isinstance(labels, (list, NoneType)) and isinstance(axes, (list, NoneType))
        super().__init__()
        self.__labels = Dimensions(*labels) if labels is not None else labels
        self.__axes = Dimensions(*axes) if axes is not None else axes
        self.__datasets = list()
        self.__title = title

    def draw(self, dataset, *args, plottype, color="blue", **kwargs):
        plottype = PlotType[plottype.upper()] if isinstance(plottype, str) else plottype
        assert isinstance(plottype, PlotType)
        dataset = Dataset(data=dataset, type=plottype, color=color)
        self.datasets.append(dataset)
        return self

    def render(self, ax, *args, **kwargs):
        ax.set_title(self.title)
        ax.set_xlabel(self.labels.x)
        ax.set_ylabel(self.labels.y)
        ax.set_zlabel(self.labels.z)
        for dataset in self.datasets:
            plottype = dataset.plottype
            self.create(ax, dataset, *args, plottype=plottype, **kwargs)

    @Dispatchers.Value(locator="plottype")
    def create(self, ax, dataset, *args, plottype, **kwargs): raise ValueError(plottype)

    @create.register(PlotType.SCATTER)
    def scatter(self, ax, dataset, *args, **kwargs):
        columns = list(self.axes)
        dataframe, color = dataset.data, dataset.color
        scatter = dataframe[columns].dropna(how="any", inplace=False)
        x, y, z = [scatter[column] for column in columns]
        ax.scatter(x, y, z, s=30, color=color)
        return ax

    @create.register(PlotType.SURFACE)
    def surface(self, ax, dataset, *args, gridsize, **kwargs):
        surface, color = dataset.data, dataset.color
        x = np.linspace(surface.domain.x.min(), surface.domain.x.max(), int(gridsize))
        y = np.linspace(surface.domain.y.min(), surface.domain.y.max(), int(gridsize))
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = surface(x, y)
        ax.plot_surface(xx, yy, zz, alpha=0.75, color=color)
        return ax

    @property
    def datasets(self): return self.__datasets
    @property
    def title(self): return self.__title
    @property
    def labels(self): return self.__labels
    @property
    def axes(self): return self.__axes


class Plotter(ODict):
    def __init__(self, *args, plotsize=4, gridsize=100, **kwargs):
        self.__plotsize = int(plotsize)
        self.__gridsize = int(gridsize)
        super().__init__()

    def display(self, *args, **kwargs):
        plots = list(self.values())
        if not plots: return None
        rows, cols = self.layout(len(plots))
        figsize = (cols * self.plotsize, rows * self.plotsize)
        figure = plt.figure(figsize=figsize)
        for index, plot in enumerate(plots, start=1):
            ax = figure.add_subplot(rows, cols, index, projection="3d")
            plot.render(ax, *args, gridsize=self.gridsize, **kwargs)
        plt.tight_layout()
        plt.show()
        return figure

    @staticmethod
    def layout(count):
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
        layout = (rows, cols)
        return layout

    @property
    def plotsize(self): return self.__plotsize
    @property
    def gridsize(self): return self.__gridsize

