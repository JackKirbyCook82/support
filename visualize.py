# -*- coding: utf-8 -*-
"""
Created on Weds Aug 17 2022
@name:   Visualization Objects
@author: Jack Kirby Cook

"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Figure", "Axes", "Coordinate", "Plot"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


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


class AxesMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        cls = super(AxesMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, *args, projection=None, coordinates=[], **kwargs):
        assert isinstance(projection, (str, type(None))) and isinstance(coordinates, list)
        if not any([type(base) is AxesMeta for base in cls.__bases__]):
            return
        setattr(Axes, cls.__name__, cls)
        cls.__projection__ = projection
        cls.__coordinates__ = coordinates

    def __call__(cls, *args, projection=None, **kwargs):
        if not any([type(base) is AxesMeta for base in cls.__bases__]):
            subclasses = {subcls.__projection__: subcls for subcls in cls.__subclasses__()}
            subclass = subclasses[projection]
            return subclass(*args, **kwargs)
        parameters = dict(projection=cls.__projection__, coordinates=cls.__coordinates__)
        instance = super(AxesMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class Axes(object, metaclass=AxesMeta):
    def __init__(self, *args, projection, coordinates, name=None, **kwargs):
        self.__coords = ODict([(key, kwargs.get(key, None)) for key in coordinates])
        self.__plots = ODict.fromkeys([])
        self.__projection = projection
        self.__name = name

    def __getitem__(self, name): return self.plots[name]
    def __setitem__(self, name, plot):
        assert isinstance(plot, Plot)
        assert plot.projection == self.projection
        self.plots[name] = plot

    def __getattr__(self, variable):
        if variable in self.coords.keys():
            return self.coords[variable]
        return super().__getattr__(variable)

    def __call__(self, ax, *args, **kwargs):
        ax.set_title(self.name if self.name is not None else None)
        for variable, coordinate in self.coords.items():
            if coordinate is None:
                continue
            assert coordinate.variable == variable
            coordinate(ax, *args, **kwargs)
        for name, plot in self.plots.items():
            assert plot.name == name
            plot(ax, *args, **kwargs)

    @property
    def projection(self): return self.__projection
    @property
    def coords(self): return self.__coords
    @property
    def plots(self): return self.__plots
    @property
    def name(self): return self.__name


class Coordinate(ntuple("Coordinate", "variable name ticks labels rotation")):
    def __call__(self, ax, *args, **kwargs):
        getattr(ax, f"set_{self.variable}label")(self.name)
        getattr(ax, f"set_{self.variable}ticks")(self.ticks)
        getattr(ax, f"set_{self.variable}ticklabels")(self.labels)
        for label in getattr(ax, f"get_{self.variable}ticklabels")():
            label.set_rotation(self.rotation)


class PlotMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        cls = super(PlotMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, *args, projection=None, data=[], **kwargs):
        assert isinstance(projection, (str, type(None)))
        if not any([type(base) is PlotMeta for base in cls.__bases__]):
            return
        setattr(Plot, cls.__name__, cls)
        cls.__projection__ = projection
        cls.__data__ = data

    def __call__(cls, *args, projection=None, **kwargs):
        if not any([type(base) is PlotMeta for base in cls.__bases__]):
            subclasses = {subcls.__projection__: subcls for subcls in cls.__subclasses__()}
            subclass = subclasses[projection]
            return subclass(*args, **kwargs)
        parameters = dict(projection=cls.__projection__, data=cls.__data__)
        instance = super(PlotMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class Plot(ABC, metaclass=PlotMeta):
    def __init__(self, *args, data, projection, name=None, **kwargs):
        self.__data = ODict([(key, kwargs[key]) for key in data])
        self.__projection = projection
        self.__name = name

    def __getattr__(self, variable):
        if variable in self.data.keys():
            return self.data[variable]
        return super().__getattr__(variable)

    def __call__(self, ax, *args, **kwargs):
        self.execute(ax, *args, **kwargs)

    @abstractmethod
    def execute(self, ax, *args, **kwargs): pass

    @property
    def projection(self): return self.__projection
    @property
    def data(self): return self.__data
    @property
    def name(self): return self.__name


class Axes2D(Axes, projection=None, coordinates=["x", "y"]): pass
class Axes3D(Axes, projection="3d", coordinates=["x", "y", "z"]): pass
class AxesPolar(Axes, projection="polar", coordinates=["r", "θ"]): pass


class Hist2D(Plot, projection=None, data=["x", "y"]):
    def execute(self, ax, *args, **kwargs):
        ax.hist(self.y, bins=self.x, label=self.name)

class Line2D(Plot, projection=None, data=["x", "y"]):
    def execute(self, ax, *args, **kwargs):
        ax.plot(self.x, self.y, label=self.name)

class Scatter2D(Plot, projection=None, data=["x", "y", "s"]):
    def execute(self, ax, *args, **kwargs):
        ax.scatter(self.x, self.y, s=self.s, label=self.name)

class Line3D(Plot, projection="3d", data=["x", "y", "z"]):
    def execute(self, ax, *args, **kwargs):
        ax.plot(self.x, self.y, self.z, label=self.name)

class Scatter3D(Plot, projection="3d", data=["x", "y", "z", "s"]):
    def execute(self, ax, *args, **kwargs):
        ax.scatter(self.x, self.y, self.z, s=self.s, label=self.name)

class Surface3D(Plot, projection="3d", data=["xx", "yy", "zz"]):
    def execute(self, ax, *args, **kwargs):
        ax.plot_surface(self.xx, self.yy, self.zz, label=self.name)







