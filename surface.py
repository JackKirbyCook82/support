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
from operator import is_not
from functools import partial
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline, CubicSpline, PchipInterpolator, Akima1DInterpolator, RectBivariateSpline, SmoothBivariateSpline

from support.concepts import NumRange, Assembly
from support.meta import RegistryMeta
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceScreener", "SurfaceCreator", "Methods"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass
class Axes:
    x: int | np.ndarray | NumRange | types.NoneType = None
    y: int | np.ndarray | NumRange | types.NoneType = None
    z: int | np.ndarray | NumRange | types.NoneType = None


class CurveMethod(Enum): REGRESSION, SPLINE, SHAPE, VISUAL = range(4)
class SurfaceMethod(Enum): REGRESSION, INTERPOLATIVE = range(2)
class Methods(Assembly): Curve, Surface = CurveMethod, SurfaceMethod


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
    def create(yaxis, zaxis, /, smoothing, degree, **kwargs): pass

    @property
    def boundary(self): return self.__boundary
    @property
    def curve(self): return self.__curve


class RegressiveCurve(Curve, register=CurveMethod.REGRESSION):
    @staticmethod
    def create(yaxis, zaxis, /, smoothing, degree, **kwargs):
        return UnivariateSpline(yaxis, zaxis, w=None, s=smoothing, k=degree.y, ext=2)

class SplineCurve(Curve, register=CurveMethod.SPLINE):
    @staticmethod
    def create(yaxis, zaxis, /, **kwargs): return CubicSpline(yaxis, zaxis, bc_type="natural")

class ShapeInterpolativeCurve(Curve, register=CurveMethod.SHAPE):
    @staticmethod
    def create(yaxis, zaxis, /, **kwargs): return PchipInterpolator(yaxis, zaxis, extrapolate=False)

class VisualInterpolativeCurve(Curve, register=CurveMethod.VISUAL):
    @staticmethod
    def create(yaxis, zaxis, /, **kwargs): return Akima1DInterpolator(yaxis, zaxis)


class Surface(ABC, metaclass=RegistryMeta):
    def __init__(self, scatter, /, **kwargs):
        scatter = scatter[list("xyz")].dropna(how="any", inplace=False)
        surface, domain = self.create(scatter, weights=None, **kwargs)
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
    def create(self, scatter, /, degree, smoothing, weights, **kwargs): pass

    @property
    def surface(self): return self.__surface
    @property
    def domain(self): return self.__domain


class RegressiveSurface(Surface, register=SurfaceMethod.REGRESSION):
    def create(self, scatter, /, **kwargs):
        xaxis, yaxis, zaxis = [scatter[axis] for axis in list("xyz")]
        surface = self.regression(xaxis, yaxis, zaxis, **kwargs)
        domain = self.grid(xaxis, yaxis, **kwargs)
        return surface, domain

    @staticmethod
    def regression(xaxis, yaxis, zaxis, /, degree, smoothing, **kwargs):
        return SmoothBivariateSpline(xaxis, yaxis, zaxis, w=None, kx=degree.x, ky=degree.y, s=smoothing)

    @staticmethod
    def grid(xaxis, yaxis, /, gridsize, **kwargs):
        xaxis = np.linspace(xaxis.min(), xaxis.max(), gridsize)
        yaxis = np.linspace(yaxis.min(), yaxis.max(), gridsize)
        return Axes(x=xaxis, y=yaxis)


class InterpolativeSurface(Surface, register=SurfaceMethod.INTERPOLATIVE):
    def create(self, scatter, /, curve, **kwargs):
        scatter = scatter.groupby(list("xy"), as_index=False)["z"].mean()
        samples = self.samples(scatter, **kwargs)
        curves = [(xaxis, Curve[curve](pd.DataFrame({"y": yaxis, "z": zaxis}), **kwargs)) for (xaxis, yaxis, zaxis) in samples]
        surface = self.interpolation(curves, **kwargs)
        domain = self.grid(curves, **kwargs)
        return surface, domain

    @staticmethod
    def samples(scatter, /, samplesize, **kwargs):
        for xaxis, sample in scatter.groupby("x", sort=True):
            sample = sample.sort_values("y")
            yaxis = sample["y"].to_numpy()
            zaxis = sample["z"].to_numpy()
            if len(yaxis) < samplesize: continue
            if np.any(np.diff(yaxis) <= 0): continue
            sample = Axes(x=xaxis, y=yaxis, z=zaxis)
            yield sample

    @staticmethod
    def interpolation(curves, /, degree, smoothing, gridsize, **kwargs):
        left = max(curve.boundary.minimum for (xaxis, curve) in curves)
        right = min(curve.boundary.maximum for (xaxis, curve) in curves)
        assert left < right
        xaxis = np.array([xaxis for (xaxis, curve) in curves], dtype=float)
        yaxis = np.linspace(left, right, gridsize)
        zaxis = np.array([curve(yaxis) for (xaxis, curve) in curves], dtype=float)
        return RectBivariateSpline(xaxis, yaxis, zaxis, kx=degree.x, ky=degree.y, s=smoothing)

    @staticmethod
    def grid(curves, /, gridsize, **kwargs):
        left = max(curve.boundary.minimum for (xaxis, curve) in curves)
        right = min(curve.boundary.maximum for (xaxis, curve) in curves)
        assert left < right
        xaxis = np.array([xaxis for (xaxis, curve) in curves], dtype=float)
        yaxis = np.linspace(left, right, gridsize)
        return Axes(x=xaxis, y=yaxis)


class SurfaceCreator(Logging):
    def __init__(self, *args, smoothing=1e-4, degree=Axes(x=3, y=3), gridsize=100, samplesize=5, **kwargs):
        if isinstance(degree, tuple): degree = Axes(**{axis: value for axis, value in zip(list("xyz"), degree)})
        multiples = kwargs.get("surfaces", [])
        single = [kwargs.get("surface", None)]
        surfaces = list(filter(partial(is_not, None), multiples + single))
        surfaces = [surface if isinstance(surface, tuple) else (surface, None) for surface in surfaces]
        super().__init__(*args, **kwargs)
        self.__multiple = bool(multiples)
        self.__samplesize = samplesize
        self.__gridsize = gridsize
        self.__smoothing = smoothing
        self.__degree = degree
        self.__surfaces = surfaces

    def __call__(self, scatter, *args, **kwargs):
        parameters = dict(samplesize=self.samplesize, gridsize=self.gridsize, smoothing=self.smoothing, degree=self.degree)
        surfaces = [Surface[surface](scatter, *args, curve=curve, **parameters, **kwargs) for (surface, curve) in self.surfaces]
        if self.multiple: return surfaces
        assert len(surfaces) == 1
        return surfaces[0]

    @property
    def surfaces(self): return self.__surfaces
    @property
    def multiple(self): return self.__multiple
    @property
    def samplesize(self): return self.__samplesize
    @property
    def gridsize(self): return self.__gridsize
    @property
    def smoothing(self): return self.__smoothing
    @property
    def degree(self): return self.__degree


class SurfaceScreener(Logging):
    def __init__(self, *args, neighbors=12, threshold=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighbors = neighbors
        self.__threshold = threshold

    def __call__(self, scatter, *args, **kwargs):
        scatter = scatter[list("xyz")].dropna(how="any", inplace=False)
        if len(scatter) < max(3, self.neighbors + 1): return scatter
        xaxis = scatter["x"].to_numpy(dtype=float)
        yaxis = scatter["y"].to_numpy(dtype=float)
        zaxis = scatter["z"].to_numpy(dtype=float)
        residuals = self.residuals(xaxis, yaxis, zaxis)
        residuals = np.fromiter(residuals, dtype=float, count=len(scatter))
        mask = residuals > self.threshold
        return scatter.loc[~mask]

    def residuals(self, xaxis, yaxis, zaxis):
        xaxis = np.asarray(xaxis, dtype=float)
        yaxis = np.asarray(yaxis, dtype=float)
        zaxis = np.asarray(zaxis, dtype=float)
        x = (xaxis - np.median(xaxis)) / self.deviation(xaxis)
        y = (yaxis - np.median(yaxis)) / self.deviation(yaxis)
        xy = np.column_stack([x, y])
        tree = cKDTree(xy)
        _, ij = tree.query(xy, k=self.neighbors + 1)
        for index in range(len(zaxis)):
            nbr = zaxis[ij[index, 1:]]
            deviation = self.deviation(nbr)
            yield np.abs(zaxis[index] - np.median(nbr)) / deviation

    @staticmethod
    def deviation(axis):
        axis = np.asarray(axis, dtype=float)
        return 1.4826 * np.median(np.abs(axis - np.median(axis))) + 1e-12

    @property
    def neighbors(self): return self.__neighbors
    @property
    def threshold(self): return self.__threshold




