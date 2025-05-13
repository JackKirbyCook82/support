# -*- coding: utf-8 -*-
"""
Created on Thurs May 8 2025
@name:   Visualize Objects
@author: Jack Kirby Cook

"""

import numpy as np
import matplotlib.pyplot as plt
from operator import is_not
from functools import partial
from itertools import product
from abc import ABC, abstractmethod
from collections import OrderedDict as ODict

from support.meta import AttributeMeta
from support.mixins import Naming

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Figure", "Axes", "Coordinate", "Plot"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Figure(object):
    def __init__(self, *args, size=(8, 8), layout=(1, 1), name=None, **kwargs):
        generator = product(*[range(1, value+1) for value in layout])
        self.__axes = ODict.fromkeys(list(generator))
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

 #       for position, axes in self.axes.items():
 #           ax = figure.add_subplot(*self.layout, position, projection=axes.projection)
 #           axes(ax, *args, **kwargs)

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


class Coordinate(Naming, fields=["variable", "name", "values", "formatting", "rotation", "padding"], metaclass=AttributeMeta):
    def __new__(cls, variable, name, values, *args, formatting="{}", rotation=45, padding=0, **kwargs):
        parameters = dict(variable=variable, name=name, values=values, formatting=formatting, rotation=rotation, padding=padding)
        return super().__new__(cls, **parameters)

    def __iter__(self): return iter(self.values)
    def __len__(self): return len(self.values)

    def __call__(self, ax, *args, **kwargs):
        getattr(ax, f"set_{self.variable}label")(self.name, labelpad=self.padding)
        getattr(ax, f"set_{self.variable}ticks")(self.ticks)
        getattr(ax, f"set_{self.variable}ticklabels")(self.labels)
        for label in getattr(ax, f"get_{self.variable}ticklabels")():
            label.set_rotation(self.rotation)

    @property
    @abstractmethod
    def labels(self): pass
    @property
    @abstractmethod
    def ticks(self): pass

class Independent(Coordinate, attribute="Independent"):
    @property
    def labels(self): return list(map(self.formatting.format, self.values))
    @property
    def ticks(self): return np.arange(0, len(self.values))

class Dependent(Coordinate, attribute="Dependent"):
    @property
    def labels(self): return list(map(self.formatting.format, self.values))
    @property
    def ticks(self): return list(self.values)


class VariablesMeta(AttributeMeta):
    def __init__(cls, *args, **kwargs):
        virtual = not any([type(base) is VariablesMeta for base in cls.__bases__])
        attribute = str(cls.__name__) if not virtual else None
        super(VariablesMeta, cls).__init__(*args, attribute=attribute, **kwargs)
        cls.__projection__ = kwargs.get("projection", getattr(cls, "__projection__", None))
        cls.__variables__ = kwargs.get("variables", getattr(cls, "__variables__", []))


class Axes(object, metaclass=VariablesMeta):
    def __init__(self, *args, coords, name=None, **kwargs):
        assert isinstance(coords, dict)
        coords = ODict([(key, coords.get(key, None)) for key in self.variables])
        plots = kwargs.get("plots", []) + [kwargs.get("plot", None)]
        plots = list(filter(partial(is_not, None), plots))
        self.__coords = ODict(coords)
        self.__plots = list(plots)
        self.__name = name

    def __call__(self, ax, *args, **kwargs):
        ax.set_title(self.name if self.name is not None else None)
        for variable, coordinate in self.coords.items():
            if coordinate is None: continue
            coordinate(ax, *args, **kwargs)
        for plot in self.plots:
            plot(ax, *args, **kwargs)

    @property
    def projection(self): return type(self).__projection__
    @property
    def variables(self): return type(self).__variables__
    @property
    def coords(self): return self.__coords
    @property
    def plots(self): return self.__plots
    @property
    def name(self): return self.__name


class Axes2D(Axes, projection=None, variables=["x", "y"]): pass
class Axes3D(Axes, projection="3d", variables=["x", "y", "z"]): pass
class AxesPolar(Axes, projection="polar", variables=["r", "θ"]): pass


class Plot(ABC, metaclass=VariablesMeta):
    def __init__(self, *args, datasets, name=None, **kwargs):
        datasets = {variable: datasets[variable] for variable in list(self.variables)}
        self.__datasets = datasets
        self.__name = name

    def __call__(self, ax, *args, **kwargs):
        parameters = {key: value for key, value in self.datasets.items()}
        self.execute(ax, *args, **parameters, **kwargs)

    @abstractmethod
    def execute(self, ax, *args, **kwargs): pass

    @property
    def projection(self): return type(self).__projection__
    @property
    def variables(self): return type(self).__variables__
    @property
    def datasets(self): return self.__datasets
    @property
    def name(self): return self.__name


class Hist2D(Plot, projection=None, variables=["x", "y"]):
    def execute(self, ax, *args, x, y, **kwargs):
        ax.hist(y, bins=x)

class Line2D(Plot, projection=None, variables=["x", "y"]):
    def execute(self, ax, *args, x, y, **kwargs):
        ax.plot(x, y)

class Scatter2D(Plot, projection=None, variables=["x", "y", "s"]):
    def execute(self, ax, *args, x, y, s, **kwargs):
        ax.scatter(x, y, s=s)

class Line3D(Plot, projection="3d", variables=["x", "y", "z"]):
    def execute(self, ax, *args, x, y, z, **kwargs):
        ax.plot(x, y, z)

class Scatter3D(Plot, projection="3d", variables=["x", "y", "z", "s"]):
    def execute(self, ax, *args, x, y, z, s, **kwargs):
        ax.scatter(x, y, z, s=s)

class Surface3D(Plot, projection="3d", variables=["xx", "yy", "zz"]):
    def execute(self, ax, *args, xx, yy, zz, **kwargs):
        ax.plot_surface(xx, yy, zz)




