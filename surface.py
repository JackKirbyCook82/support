# -*- coding: utf-8 -*-
"""
Created on Tues Apr 21 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
from scipy.interpolate import UnivariateSpline, CubicSpline, PchipInterpolator, Akima1DInterpolator, RectBivariateSpline, SmoothBivariateSpline

from support.concepts import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


CurveAxes = ntuple("Axes", "x y")
SurfaceAxes = ntuple("Axes", "x y z")


class Curve(ABC):
    def __call__(self, yaxis): return self.curve(yaxis)
    def __init__(self, yaxis, zaxis, /, **kwargs):
        assert all([isinstance(axis, (pd.Series, np.ndarray)) for axis in (yaxis, zaxis)])
        assert len(yaxis) == len(zaxis)
        yaxis = np.asarray(yaxis, dtype=np.float32)
        zaxis = np.asarray(zaxis, dtype=np.float32)
        order = np.argsort(yaxis)
        yaxis, zaxis = (yaxis[order], zaxis[order])
        curve = self.create(yaxis, zaxis, **kwargs)
        boundary = NumRange.create([yaxis.min(), zaxis.max()])
        self.__boundary = boundary
        self.__curve = curve

    @staticmethod
    @abstractmethod
    def create(yaxis, zaxis, /, method, weights, smoothing, degree, **kwargs): pass

    @property
    def boundary(self): return self.__boundary
    @property
    def curve(self): return self.__curve


class RegressiveCurve(Curve):
    @staticmethod
    def create(yaxis, zaxis, /, weights, smoothing, degree, **kwargs):
        return UnivariateSpline(yaxis, zaxis, w=weights, s=smoothing, k=degree, ext=2)

class InterpolativeCurve(Curve):
    @staticmethod
    def create(yaxis, zaxis, /, method, **kwargs): return CubicSpline(yaxis, zaxis, bc_type=method)

class ShapeInterpolativeCurve(InterpolativeCurve):
    @staticmethod
    def create(yaxis, zaxis, /, **kwargs): return PchipInterpolator(yaxis, zaxis, extrapolate=False)

class VisualInterpolativeCurve(InterpolativeCurve):
    @staticmethod
    def create(yaxis, zaxis, /, **kwargs): return Akima1DInterpolator(yaxis, zaxis)


class Surface(ABC):
    def __init__(self, xaxis, yaxis, zaxis, /, weights=None, **kwargs):
        assert all([isinstance(axis, (pd.Series, np.ndarray)) for axis in (xaxis, yaxis, zaxis)])
        assert len(xaxis) == len(yaxis) == len(zaxis)
        mask = ~(np.isnan(xaxis) | np.isnan(yaxis) | np.isnan(zaxis))
        xaxis, yaxis, zaxis = (xaxis[mask], yaxis[mask], zaxis[mask])
        weights = weights[mask] if weights is not None else None
        surface, domain = self.create(xaxis, yaxis, zaxis, weights=weights, **kwargs)
        self.__surface = surface
        self.__domain = domain

    def __call__(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        assert not self.violation(x, y)
        return self.surface(x, x)

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
        return CurveAxes(x, y)

    def violation(self, x, y):
        boundary = self.boundary()
        x = np.any((x < boundary.x.minimum) | (y > boundary.t.maximum))
        y = np.any((y < boundary.y.minimum) | (y > boundary.y.maximum))
        return x | y

    @abstractmethod
    def create(self, xaxis, yaxis, zaxis, /, degree, smoothing, weights, **kwargs): pass

    @property
    def surface(self): return self.__surface
    @property
    def domain(self): return self.__domain


class RegressiveSurface(Surface):
    def create(self, xaxis, yaxis, zaxis, /, **kwargs):
        surface = self.regression(xaxis, yaxis, zaxis, **kwargs)
        domain = self.footprint(xaxis, yaxis, **kwargs)
        return surface, domain

    @staticmethod
    def regression(xaxis, yaxis, zaxis, /, degree, smoothing, weights=None, **kwargs):
        return SmoothBivariateSpline(xaxis, yaxis, zaxis, w=weights, kx=degree.x, ky=degree.y, s=smoothing)

    @staticmethod
    def footprint(xaxis, yaxis, /, grid, **kwargs):
        xaxis = np.linspace(xaxis.min(), xaxis.max(), grid)
        yaxis = np.linspace(yaxis.min(), yaxis.max(), grid)
        return CurveAxes(xaxis, yaxis)


class InterpolativeSurface(Surface):
    def create(self, xaxis, yaxis, zaxis, /, **kwargs):
        dataframe = pd.concat([xaxis, yaxis, zaxis], axis=1)
        dataframe = dataframe.groupby(["x", "y"], as_index=False)["z"].mean()
        samples = self.samples(dataframe, **kwargs)
        curves = [(xaxis, RegressiveCurve(yaxis, zaxis, **kwargs)) for (xaxis, yaxis, zaxis) in samples]
        surface = self.surface(curves, **kwargs)
        domain = self.domain(curves, **kwargs)
        return surface, domain

    @staticmethod
    def samples(dataframe, /, size, **kwargs):
        for xaxis, sample in dataframe.groupby("x", sort="x"):
            sample = sample.sort("x")
            yaxis = sample["y"].to_numpy()
            zaxis = sample["z"].to_numpy()
            if len(yaxis) < size: continue
            if np.any(np.diff(yaxis) > 0): continue
            order = np.argsort(yaxis)
            sample = SurfaceAxes(xaxis, yaxis[order], zaxis[order])
            yield sample

    @staticmethod
    def interpolation(curves, /, degree, smoothing, grid, **kwargs):
        left = max(curve.boundary.minimum for xaxis, curve in curves)
        right = min(curve.boundary.maximum for xaxis, curve in curves)
        assert left < right
        xaxis = np.array([curve.sample.t for curve in curves])
        yaxis = np.linspace(left, right, grid)
        zaxis = np.array([curve.spline(yaxis) for index, curve in enumerate(curves)])
        return RectBivariateSpline(xaxis, yaxis, zaxis, kx=degree.x, ky=degree.y, s=smoothing)

    @staticmethod
    def footprint(curves, /, grid, **kwargs):
        left = max(curve.boundary.minimum for xaxis, curve in curves)
        right = min(curve.boundary.maximum for xaxis, curve in curves)
        assert left < right
        xaxis = np.array([xaxis for (xaxis, curve) in curves])
        yaxis = np.linspace(left, right, grid)
        return CurveAxes(xaxis, yaxis)






