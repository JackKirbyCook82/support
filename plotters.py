# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 2026
@name:   Plotter Objects
@author: Jack Kirby Cook

"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from types import NoneType, SimpleNamespace
from collections import OrderedDict as ODict
from mpl_toolkits.mplot3d import Axes3D

from support.meta import AttributeMeta
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Plotter", "Plot", "Artist"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Axes:
    x: Optional[str] = None; y: Optional[str] = None; z: Optional[str] = None

    def __iter__(self): yield self.x; yield self.y; yield self.z


class Artist(ABC, metaclass=AttributeMeta):
    def __init__(self, source, *args, color, **kwargs):
        self.__source = source
        self.__color = color

    @abstractmethod
    def render(self, ax, *args, **kwargs): pass

    @property
    def source(self): return self.__source
    @property
    def color(self): return self.__color


class Surface(Artist, attribute="Surface"):
    def __init__(self, *args, gridsize=100, transparency=0.75, **kwargs):
        assert isinstance(gridsize, int)
        super().__init__(*args, **kwargs)
        self.__transparency = transparency
        self.__gridsize = gridsize

    def render(self, ax, *args, **kwargs):
        x = np.linspace(self.source.domain.x.min(), self.source.domain.x.max(), int(self.gridsize))
        y = np.linspace(self.source.domain.y.min(), self.source.domain.y.max(), int(self.gridsize))
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = self.source(x, y)
        coordinates = SimpleNamespace(xx=xx, yy=yy, zz=zz)
        ax.plot_surface(coordinates.xx, coordinates.yy, coordinates.zz, alpha=self.transparency, color=self.color)
        return ax

    @property
    def transparency(self): return self.__transparency
    @property
    def gridsize(self): return self.__gridsize


class Dataset(Artist, ABC):
    def __init__(self, *args, columns, thickness, **kwargs):
        assert isinstance(thickness, int)
        super().__init__(*args, **kwargs)
        self.__columns = Axes(*columns) if columns is not None else columns
        self.__thickness = thickness

    @singledispatchmethod
    def axes(self, source): pass

    @axes.register(pd.DataFrame)
    def dataframe(self, dataframe):
        coordinates = zip(list("xyz"), self.columns)
        coordinates = {coordinate: dataframe[axis].to_numpy() if axis in dataframe.columns else None for coordinate, axis in coordinates}
        return Axes(**coordinates)

    @axes.register(dict)
    def mapping(self, mapping):
        coordinates = zip(list("xyz"), self.columns)
        coordinates = {coordinate: mapping.get(axis, None) for coordinate, axis in coordinates}
        return Axes(**coordinates)

    @axes.register(tuple)
    @axes.register(list)
    def collection(self, collection):
        assert 0 < len(collection) <= 3
        collection = tuple(collection[:3]) + tuple([None]) * max(0, 3 - len(collection))
        coordinates = {coordinate: value for coordinate, value in zip(list("xyz"), collection)}
        return Axes(**coordinates)

    @property
    def thickness(self): return self.__thickness
    @property
    def columns(self): return self.__columns


class Scatter(Dataset, attribute="Scatter"):
    def render(self, ax, *args, **kwargs):
        columns = list(self.columns)
        source = self.source[columns].dropna(how="any", inplace=False)
        axes = self.axes(source)
        ax.scatter(axes.x, axes.y, axes.z, s=self.thickness, color=self.color)
        return ax


class Line(Scatter, attribute="Line"):
    def render(self, ax, *args, **kwargs):
        source = self.source
        assert len(source) == 2
        axes = self.axes(source)
        x = ax.get_xlim() if axes.x is None else [axes.x, axes.x]
        y = ax.get_ylim() if axes.y is None else [axes.y, axes.y]
        z = ax.get_zlim() if axes.z is None else [axes.z, axes.z]
        ax.plot(x, y, z, linewidth=self.thickness, color=self.color)
        return ax


class Point(Scatter, attribute="Point"):
    def render(self, ax, *args, **kwargs):
        source = self.source
        axes = self.axes(source)
        ax.scatter([axes.x], [axes.y], [axes.z], s=self.thickness, color=self.color)
        return ax


class Plot(object):
    def __init__(self, *args, title=None, labels=None, **kwargs):
        assert isinstance(labels, (list, NoneType))
        super().__init__()
        self.__labels = Axes(*labels) if labels is not None else labels
        self.__artists = list()
        self.__title = title

    def append(self, artist):
        assert isinstance(artist, Artist)
        self.artists.append(artist)

    def render(self, ax, *args, **kwargs):
        ax.set_title(self.title)
        ax.set_xlabel(self.labels.x)
        ax.set_ylabel(self.labels.y)
        ax.set_zlabel(self.labels.z)
        for artist in self.artists:
            artist.render(ax, *args, **kwargs)

    @property
    def artists(self): return self.__artists
    @property
    def title(self): return self.__title
    @property
    def labels(self): return self.__labels


class Plotter(Logging):
    def __init__(self, *args, plotsize=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.__plotsize = int(plotsize)
        self.__plots = ODict()

    def __getitem__(self, name): return self.plots[name]
    def __setitem__(self, name, plot):
        assert isinstance(plot, Plot)
        self.plots[name] = plot

    def display(self, *args, **kwargs):
        names = list(self.plots.keys())
        plots = list(self.plots.values())
        if not plots: return None
        rows, cols = self.layout(len(plots))
        figsize = (cols * self.plotsize, rows * self.plotsize)
        figure = plt.figure(figsize=figsize)
        for index, plot in enumerate(plots, start=1):
            ax = figure.add_subplot(rows, cols, index, projection="3d")
            plot.render(ax, *args, **kwargs)
        self.console("Plotted", f"Plots[{','.join(names)}]")
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
    def plots(self): return self.__plots



