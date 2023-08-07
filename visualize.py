# -*- coding: utf-8 -*-
"""
Created on Weds Aug 17 2022
@name:   Visualization Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3d
from functools import reduce
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Figure", "Axes", "Plot"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


mpl.use("Qt5Agg")
BASE_COLORS = list(mpl.colors.BASE_COLORS.keys())
TAB_COLORS = list(mpl.colors.TABLEAU_COLORS.keys())
CSS_COLORS = list(mpl.colors.CSS4_COLORS.keys())


class VisualizeMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        cls = super(VisualizeMeta, mcs).__new__(mcs, name, bases, attrs)
        for base in bases:
            if issubclass(type(base), VisualizeMeta) or type(base) is VisualizeMeta:
                setattr(base, cls.__name__, cls)
        return cls

    def __init__(cls, *args, **kwargs):
        cls.__projection__ = kwargs.get("projection", getattr(cls, "__projection__", None))
        cls.__colors__ = kwargs.get("colors", getattr(cls, "__colors__", None))

    def __call__(cls, *args, **kwargs):
        instance = super(VisualizeMeta, cls).__call__(*args, projection=cls.__projection__, colors=cls.__colors__, **kwargs)
        return instance


class AxesMeta(VisualizeMeta): pass
class PlotMeta(VisualizeMeta): pass


class Figure(object):
    def __init__(self, *args, size=(8, 8), layout=(1, 1), name=None, **kwargs):
        self.__contents = {index: None for index in range(1, np.prod(list(layout))+1)}
        self.__size = size
        self.__layout = layout
        self.__name = name

    def __repr__(self): return "{}(size={}, layout={}, name={})".format(self.__class__.__name__, self.size, self.layout, self.name)
    def __len__(self): return len(self.__contents)

    def __getitem__(self, position):
        assert isinstance(position, int)
        assert position <= len(self)
        return self.__contents[position]

    def __setitem__(self, position, content):
        assert isinstance(position, int)
        assert position <= np.prod(list(self.layout))
        assert isinstance(content, Axes)
        self.__contents[position] = content

    def __call__(self):
        figure = plt.figure(figsize=self.size)
        figure.suptitle(str(self.name) if self.name is not None else None)
        for position, axes in self.items():
            ax = figure.add_subplot(*self.layout, position, projection=axes.projection)
            axes(ax)
        plt.tight_layout()
        plt.show()

    def keys(self): return tuple(self.__contents.keys())
    def values(self): return tuple(self.__contents.values())
    def items(self): return tuple(self.__contents.items())

    @property
    def size(self): return self.__size
    @property
    def layout(self): return self.__layout
    @property
    def name(self): return self.__name


class Span(ntuple("Span", "min max")):
    def __add__(self, other):
        if other is None:
            return self
        assert isinstance(other, type(self))
        minvalue = np.nanmin([self.min, other.min]) if not np.isnan(self.min) else other.min
        maxvalue = np.nanmax([self.max, other.max]) if not np.isnan(self.max) else other.max
        return self.__class__(minvalue, maxvalue)

    def __call__(self, length, digits=1):
        assert not np.isnan(self.min) and not np.isnan(self.max)
        assert self.min.dtype == self.max.dtype
        assert isinstance(length, int)
        if np.issubdtype(self.min.dtype, np.datetime64):
            array = pd.date_range(start=self.min, end=self.max, periods=length)
        else:
            array = np.linspace(self.min, self.max, length)
            array = np.around(array, decimals=digits)
        return array

    @classmethod
    def fromValues(cls, values): return cls(np.amin(values), np.amax(values))


class Axes(ABC, metaclass=AxesMeta):
    def __init__(self, *args, ticks=10, name=None, projection=None, colors, **kwargs):
        self.__projection = projection
        self.__colors = colors
        self.__ticks = ticks
        self.__name = name
        self.__plots = {}

    def __repr__(self): return "{}(ticks={}, name={})".format(self.__class__.__name__, self.ticks, self.name)
    def __len__(self): return len(self.__plots)

    def __getitem__(self, name):
        return self.__plots[name]

    def __setitem__(self, name, plot):
        assert isinstance(plot, Plot)
        assert plot.projection == self.projection
        plot.name = name
        self.__plots[name] = plot

    def __call__(self, ax):
        assert len(self) <= len(self.colors)
        ax.set_title(str(self.name) if self.name is not None else None)
        data, axis = ODict(), ODict()
        for index, (name, plot) in enumerate(self.items()):
            assert plot.name == name
            data[plot.key] = Span.fromValues(plot.data) + data.get(plot.key, None)
            for key, values in plot.axis.items():
                axis[key] = Span.fromValues(values) + axis.get(key, None)
            plot(ax, color=self.colors[index])
        datakeys = list(data.keys())
        axiskeys = list(axis.keys())
        data = reduce(lambda x, y: x + y, list(data.values()))(self.ticks, digits=1)
        axis = [values(self.ticks, digits=1) for values in axis.values()]
        self.execute(ax, data, *axis, datakeys=datakeys, axiskeys=axiskeys, ticks=self.ticks, colors=self.colors[:len(self)])

    def keys(self): return tuple(self.__plots.keys())
    def values(self): return tuple(self.__plots.values())
    def items(self): return tuple(self.__plots.items())

    @staticmethod
    @abstractmethod
    def execute(ax, *args, **kwargs): pass

    @property
    def projection(self): return self.__projection
    @property
    def colors(self): return self.__colors
    @property
    def ticks(self): return self.__ticks
    @property
    def name(self): return self.__name


class Axes2D(Axes, projection=None, colors=BASE_COLORS):
    @staticmethod
    def execute(ax, fx, x, *args, datakeys, axiskeys, **kwargs):
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelrotation=-25)
        ax.set_xlabel(axiskeys[0])
        ax.set_yticks(fx)
        ax.set_ylabel("\n".join(datakeys))
        ax.legend()


class Axes3D(Axes, projection="3d", colors=BASE_COLORS):
    @staticmethod
    def execute(ax, fxy, x, y, *args, datakeys, axiskeys, ticks, colors, **kwargs):
        assert len(datakeys) == len(colors)
        ax.set_xticks(np.linspace(*ax.xaxis.get_data_interval(), ticks))
        ax.set_xticklabels(x)
        ax.set_xlabel(axiskeys[0])
        ax.set_yticks(np.linspace(*ax.yaxis.get_data_interval(), ticks))
        ax.set_yticklabels(y)
        ax.set_ylabel(axiskeys[1])
        ax.set_zlabel("\n".join(datakeys))
        bars = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
        ax.legend(bars, datakeys)


class Plot(ABC, metaclass=PlotMeta):
    def __init__(self, data, *args, projection=None, **kwargs):
        self.__key = data.name
        self.__data = np.array(data.values)
        self.__axis = ODict([(axis, np.array(data.coords[axis].values)) for axis in data.dims])
        self.__projection = projection
        self.__name = None

    def __call__(self, ax, *args, **kwargs):
        self.execute(ax, self.data, *self.axis.values(), *args, key=self.key, axis=self.axis.keys(), **kwargs)

    @staticmethod
    @abstractmethod
    def execute(ax, *args, **kwargs): pass

    @property
    def key(self): return self.__key
    @property
    def data(self): return self.__data
    @property
    def axis(self): return self.__axis
    @property
    def projection(self): return self.__projection
    @property
    def name(self): return self.__name
    @name.setter
    def name(self, name): self.__name = name


class Plot2D(Plot, ABC,  projection=None): pass
class Plot3D(Plot, ABC, projection="3d"): pass


class Line(Plot2D, projection=None):
    @staticmethod
    def execute(ax, fx, x, *args, key, axis, color, **kwargs):
        mpl.axes.Axes.plot(ax, x, fx, label=key, color=color)


class Surface(Plot3D, projection="3d"):
    @staticmethod
    def execute(ax, fxy, x, y, *args, key, axis, color, **kwargs):
        i, j = np.arange(len(x)), np.arange(len(y))
        ii, jj = np.meshgrid(i, j)
        mpl3d.axes3d.Axes3D.plot_surface(ax, ii, jj, fxy, label=key, color=color)



