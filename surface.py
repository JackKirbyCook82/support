# -*- coding: utf-8 -*-
"""
Created on Tues Apr 21 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.interpolate import SmoothBivariateSpline, RectBivariateSpline, make_interp_spline, make_splrep

from support.meta import RegistryMeta
from support.concepts import NumRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceCreator", "Surface", "Curve"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass
class Axes:
    x: int | float | np.ndarray | NumRange | types.NoneType = None
    y: int | float | np.ndarray | NumRange | types.NoneType = None
    z: int | float | np.ndarray | NumRange | types.NoneType = None

    def __iter__(self):
        yield self.x; yield self.y; yield self.z


class Method(Enum): REGRESSION, INTERPOLATION = range(2)
class Curve(ABC, metaclass=RegistryMeta):
    def __call__(self, yaxis): return self.curve(yaxis)
    def __init__(self, scatter, /, **kwargs):
        scatter = scatter[list("yz")].dropna(how="any", inplace=False)
        yaxis, zaxis = [scatter[axis] for axis in list("yz")]
        order = np.argsort(yaxis)
        yaxis, zaxis = (yaxis[order], zaxis[order])
        assert not np.any(np.diff(yaxis) <= 0)
        curve = self.create(yaxis, zaxis, **kwargs)
        boundary = NumRange.create([yaxis.min(), yaxis.max()])
        self.__boundary = boundary
        self.__curve = curve

    @staticmethod
    @abstractmethod
    def create(yaxis, zaxis, /, smoothing=None, weights=None, **kwargs): pass

    @property
    def boundary(self): return self.__boundary
    @property
    def curve(self): return self.__curve


class RegressionCurve(Curve, register=Method.REGRESSION):
    @staticmethod
    def create(yaxis, zaxis, /, smoothing=None, weights=None, **kwargs): return make_splrep(yaxis, zaxis, k=3, s=smoothing, w=weights, bc_type=None)

class InterpolationCurve(Curve, register=Method.INTERPOLATION):
    @staticmethod
    def create(yaxis, zaxis, /, **kwargs): return make_interp_spline(yaxis, zaxis, k=3, bc_type="natural")


class Surface(ABC, metaclass=RegistryMeta):
    def __init__(self, scatter, /, **kwargs):
        scatter = scatter[list("xyz")].dropna(how="any", inplace=False)
        surface, domain = self.create(scatter, **kwargs)
        self.__surface = surface
        self.__domain = domain

    def __call__(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert not self.violation(x, y)
        return self.surface(x, y)

    def z(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert not self.violation(x, y)
        return self.surface.ev(x, y)

    def dzdx(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert not self.violation(x, y)
        return self.surface.ev(x, y, dx=1, dy=0)

    def dzdy(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert not self.violation(x, y)
        return self.surface.ev(x, y, dx=0, dy=1)

    def dz2dx2(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert not self.violation(x, y)
        return self.surface.ev(x, y, dx=2, dy=0)

    def dz2dy2(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert not self.violation(x, y)
        return self.surface.ev(x, y, dx=0, dy=2)

    def boundary(self):
        x = NumRange.create([self.domain.x.min(), self.domain.x.max()])
        y = NumRange.create([self.domain.y.min(), self.domain.y.max()])
        return Axes(x=x, y=y)

    def violation(self, x, y):
        boundary = self.boundary()
        x = np.any((x < boundary.x.minimum) | (x > boundary.x.maximum))
        y = np.any((y < boundary.y.minimum) | (y > boundary.y.maximum))
        return x | y

    @abstractmethod
    def create(self, scatter, /, smoothing, weights, **kwargs): pass

    @property
    def surface(self): return self.__surface
    @property
    def domain(self): return self.__domain


class RegressionSurface(Surface, register=Method.REGRESSION):
    def create(self, scatter, /, gridsize, smoothing=None, weights=None, **kwargs):
        xaxis, yaxis, zaxis = [scatter[axis] for axis in list("xyz")]
        surface = SmoothBivariateSpline(xaxis, yaxis, zaxis, kx=3, ky=3, s=smoothing, w=weights)
        xaxis = np.linspace(xaxis.min(), xaxis.max(), gridsize)
        yaxis = np.linspace(yaxis.min(), yaxis.max(), gridsize)
        domain = Axes(x=xaxis, y=yaxis)
        return surface, domain


class InterpolationSurface(Surface, register=Method.INTERPOLATION):
    def create(self, scatter, /, **kwargs):
        scatter = scatter.groupby(list("xy"), as_index=False)["z"].mean()
        axes = list(self.align(scatter, **kwargs))
        curves = [(xaxis, InterpolationCurve(pd.DataFrame({"y": yaxis, "z": zaxis}))) for (xaxis, yaxis, zaxis) in axes]
        surface = self.interpolate(curves, **kwargs)
        domain = self.grid(curves, **kwargs)
        return surface, domain

    @staticmethod
    def align(scatter, /, samplesize, **kwargs):
        for xaxis, dataframe in scatter.groupby("x", sort=True):
            dataframe = dataframe.sort_values("y")
            yaxis = dataframe["y"].to_numpy()
            zaxis = dataframe["z"].to_numpy()
            if len(yaxis) < samplesize: continue
            if np.any(np.diff(yaxis) <= 0): continue
            yield Axes(x=xaxis, y=yaxis, z=zaxis)

    @staticmethod
    def interpolate(curves, /, smoothing, gridsize, **kwargs):
        left = max(curve.boundary.minimum for (xaxis, curve) in curves)
        right = min(curve.boundary.maximum for (xaxis, curve) in curves)
        assert left < right
        xaxis = np.array([xaxis for (xaxis, curve) in curves], dtype=float)
        yaxis = np.linspace(left, right, gridsize)
        zaxis = np.array([curve(yaxis) for (xaxis, curve) in curves], dtype=float)
        return RectBivariateSpline(xaxis, yaxis, zaxis, kx=3, ky=3, s=smoothing)

    @staticmethod
    def grid(curves, /, gridsize, **kwargs):
        left = max(curve.boundary.minimum for (xaxis, curve) in curves)
        right = min(curve.boundary.maximum for (xaxis, curve) in curves)
        assert left < right
        xaxis = np.array([xaxis for (xaxis, curve) in curves], dtype=float)
        yaxis = np.linspace(left, right, gridsize)
        return Axes(x=xaxis, y=yaxis)


class SurfaceError(Exception): pass
class SurfaceQuantityError(SurfaceError): pass

class SurfaceCreator(Logging):
    def __init__(self, *args, quantity=35, gridsize=100, samplesize=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.__samplesize = samplesize
        self.__gridsize = gridsize
        self.__quantity = quantity

    def __call__(self, dataset, *args, method, smoothing=None, weights=None, **kwargs):
        method = Method[str(method).upper()] if isinstance(method, str) else method
        if len(dataset) < self.quantity: raise SurfaceQuantityError()
        parameters = dict(samplesize=self.samplesize, gridsize=self.gridsize)
        parameters = parameters | dict(method=method, smoothing=smoothing, weights=weights)
        surface = Surface[method](dataset.xyz, **parameters)
        return surface

    @property
    def samplesize(self): return self.__samplesize
    @property
    def gridsize(self): return self.__gridsize
    @property
    def quantity(self): return self.__quantity



