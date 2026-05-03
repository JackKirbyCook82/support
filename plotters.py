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
from typing import Optional
from dataclasses import dataclass
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

from support.decorators import Dispatchers
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Plotter", "Plot"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass
class Labels: x: Optional[str] = None; y: Optional[str] = None; z: Optional[str] = None
class PlotType(Enum): SCATTER, SURFACE = range(2)
class Plot(dict):
    def __init__(self, *args, title=None, labels, **kwargs):
        assert isinstance(labels, (tuple, Labels))
        plots = {plot: kwargs.get(str(plot.name).lower(), []) for plot in iter(PlotType)}
        plots = {plot: (datasets if isinstance(datasets, list) else [datasets]) for plot, datasets in plots.items()}
        plots = {plot: [(dataset if isinstance(dataset, tuple) else (dataset, "blue")) for dataset in datasets] for plot, datasets in plots.items()}
        if not isinstance(labels, Labels): labels = Labels(**{axis: value for axis, value in zip(list("xyz"), labels)})
        super().__init__(plots)
        self.__labels = labels
        self.__title = title

    @property
    def labels(self): return self.__labels
    @property
    def title(self): return self.__title


class Plotter(Logging):
    def __init__(self, *args, plotsize=4, gridsize=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.__plotsize = int(plotsize)
        self.__gridsize = int(gridsize)

    def __call__(self, plots, *args, **kwargs):
        assert isinstance(plots, (list, Plot))
        if not isinstance(plots, list): plots = [plots]
        rows, cols = self.layout(len(plots))
        figsize = (cols * self.plotsize, rows * self.plotsize)
        figure = plt.figure(figsize=figsize)
        for index, plot in enumerate(plots, start=1):
            ax = figure.add_subplot(rows, cols, index, projection="3d")
            ax.set_title(plot.title)
            ax.set_xlabel(plot.labels.x)
            ax.set_ylabel(plot.labels.y)
            ax.set_zlabel(plot.labels.z)
            for plottype, datasets in plot.items():
                for (dataset, color) in datasets:
                    self.draw(ax, dataset, *args, plottype=plottype, color=color, **kwargs)
        plt.tight_layout()
        plt.show()

    @Dispatchers.Value(locator="plottype")
    def draw(self, ax, dataset, *args, plottype, **kwargs): raise ValueError(plottype)

    @draw.register(PlotType.SCATTER)
    def scatter(self, ax, scatter, *args, color="red", **kwargs):
        scatter = scatter[list("xyz")].dropna(how="any", inplace=False)
        x, y, z = [scatter[axis] for axis in list("xyz")]
        ax.scatter(x, y, z, s=30, color=color)

    @draw.register(PlotType.SURFACE)
    def surface(self, ax, surface, *args, color="blue", **kwargs):
        x = np.linspace(surface.domain.x.min(), surface.domain.x.max(), self.gridsize)
        y = np.linspace(surface.domain.y.min(), surface.domain.y.max(), self.gridsize)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = surface(x, y)
        ax.plot_surface(xx, yy, zz, alpha=0.75, color=color)

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




