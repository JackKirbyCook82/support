# -*- coding: utf-8 -*-
"""
Created on Weds Aug 17 2022
@name:   Visualization Objects
@author: Jack Kirby Cook

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3d
from abc import ABC, abstractmethod
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Figure", "Axes", "Plot"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


mpl.use("Qt5Agg")


class Figure(object):
    def __init__(self, *args, size=(8, 8), layout=(1, 1), name=None, **kwargs):
        self.__axes = ODict([(index, None) for index in range(1, np.prod(list(layout))+1)])
        self.__layout = layout
        self.__size = size
        self.__name = name

    def __getitem__(self, position): return self.axes[position]
    def __setitem__(self, position, axes):
        assert position in self.axes.keys()
        assert isinstance(axes, Axes)
        self.axes[position] = axes

    def __call__(self, *args, **kwargs):
        figure = plt.figure(figsize=self.size)
        figure.suptitle(self.name if self.name is not None else None)
        for position, axes in self.axes.items():
            ax = figure.add_subplot(*self.layout, position, projection=axes.projection)
            axes(ax, *args, **kwargs)
        plt.tight_layout()
        plt.show()

    @property
    def axes(self): return self.__axes
    @property
    def layout(self): return self.__layout
    @property
    def size(self): return self.__size
    @property
    def name(self): return self.__name


class Axes(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__projection__ = kwargs.get("projection", getattr(cls, "__projection__", None))
        setattr(Axes, cls.__name__, cls)

    def __init__(self, *args, name=None, **kwargs):
        self.__projection = self.__class__.__projection__
        self.__plots = ODict()
        self.__name = name

    def __getitem__(self, name): return self.plots[name]
    def __setitem__(self, name, plot):
        assert isinstance(plot, Plot)
        assert plot.projection == self.projection
        self.plots[name] = plot

    def __call__(self, ax, *args, **kwargs):
        ax.set_title(self.name if self.name is not None else None)
        for name, plot in self.plots.items():
            plot(ax, *args, **kwargs)
        self.execute(ax, *args, **kwargs)

    @staticmethod
    @abstractmethod
    def execute(ax, *args, **kwargs): pass

    @property
    def projection(self): return self.__projection
    @property
    def plots(self): return self.__plots
    @property
    def name(self): return self.__name


class Axes2D(Axes, projection=None):
    @staticmethod
    def execute(ax, *args, **kwargs):
        pass


class Axes3D(Axes, projection="3d"):
    @staticmethod
    def execute(ax, *args, **kwargs):
        pass


class Plot(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.__projection__ = kwargs.get("projection", getattr(cls, "__projection__", None))
        setattr(Plot, cls.__name__, cls)

    def __init__(self, *args, **kwargs):
        self.__projection = self.__class__.__projection__

    def __call__(self, ax, *args, **kwargs):
        self.execute(ax, *args, 8*kwargs)

    @staticmethod
    @abstractmethod
    def execute(ax, *args, **kwargs): pass

    @property
    def projection(self): return self.__projection


class Plot2D(Plot, ABC,  projection=None): pass
class Plot3D(Plot, ABC, projection="3d"): pass


class Line2D(Plot2D):
    @staticmethod
    def execute(ax, *args, **kwargs): pass


class Scatter2D(Plot2D):
    @staticmethod
    def execute(ax, *args, **kwargs): pass


class Line3D(Plot3D):
    @staticmethod
    def execute(ax, *args, **kwargs): pass


class Scatter3D(Plot3D):
    @staticmethod
    def execute(ax, *args, **kwargs): pass


class Surface3D(Plot3D):
    @staticmethod
    def execute(ax, *args, **kwargs): pass




