# -*- coding: utf-8 -*-
"""
Created on Tues Apr 21 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import math
import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline, CubicSpline, PchipInterpolator, Akima1DInterpolator, RectBivariateSpline, SmoothBivariateSpline
from mpl_toolkits.mplot3d import Axes3D

from support.concepts import NumRange, Assembly
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Surfaces", "Curves", "Screener", "Plotter", "Curvature", "Axes"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


CurveAxes = ntuple("Axes", "x y")
SurfaceAxes = ntuple("Axes", "x y z")


@dataclass
class Axes: x: Optional[int] = None; y: Optional[int] = None; z: Optional[int] = None
class Curvature(Enum): REGRESSIVE, INTERPOLATIVE, SHAPE, VISUAL = range(4)


class Curve(ABC):
    def __call__(self, yaxis): return self.curve(yaxis)
    def __init__(self, scatter, /, **kwargs):
        scatter = scatter["yz".split()].dropna(how="any", inplace=False)
        yaxis, zaxis = [scatter[axis] for axis in "yz".split()]
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


class RegressiveCurve(Curve):
    @staticmethod
    def create(yaxis, zaxis, /, smoothing, degree, **kwargs):
        return UnivariateSpline(yaxis, zaxis, w=None, s=smoothing, k=degree.y, ext=2)

class InterpolativeCurve(Curve):
    @staticmethod
    def create(yaxis, zaxis, /, **kwargs): return CubicSpline(yaxis, zaxis, bc_type="natural")

class ShapeInterpolativeCurve(InterpolativeCurve):
    @staticmethod
    def create(yaxis, zaxis, /, **kwargs): return PchipInterpolator(yaxis, zaxis, extrapolate=False)

class VisualInterpolativeCurve(InterpolativeCurve):
    @staticmethod
    def create(yaxis, zaxis, /, **kwargs): return Akima1DInterpolator(yaxis, zaxis)


class Surface(ABC):
    def __init__(self, scatter, /, **kwargs):
        scatter = scatter["xyz".split()].dropna(how="any", inplace=False)
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
        return CurveAxes(x, y)

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


class RegressiveSurface(Surface):
    def create(self, scatter, /, **kwargs):
        xaxis, yaxis, zaxis = [scatter[axis] for axis in "xyz".split()]
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
        return CurveAxes(xaxis, yaxis)


class InterpolativeSurface(Surface):
    def create(self, scatter, /, curvature, **kwargs):
        scatter = scatter.groupby(["x", "y"], as_index=False)["z"].mean()
        samples = self.samples(scatter, **kwargs)
        curves = {Curvature.REGRESSIVE: RegressiveCurve, Curvature.INTERPOLATIVE: InterpolativeCurve, Curvature.SHAPE: ShapeInterpolativeCurve, Curvature.VISUAL: VisualInterpolativeCurve}
        curves = [(xaxis, curves[curvature](pd.DataFrame({"y": yaxis.y, "z": zaxis.z}), **kwargs)) for (xaxis, yaxis, zaxis) in samples]
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
            sample = SurfaceAxes(xaxis, yaxis, zaxis)
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
        return CurveAxes(xaxis, yaxis)


class Screener(Logging):
    def __init__(self, *args, neighbors=12, threshold=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighbors = neighbors
        self.__threshold = threshold

    def __call__(self, scatter, *args, **kwargs):
        scatter = scatter["xyz".split()].dropna(how="any", inplace=False)
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


class Plotter(Logging):
    def __init__(self, *args, plotsize=4, gridsize=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.__plotsize = int(plotsize)
        self.__gridsize = int(gridsize)

    def __call__(self, scatter, *args, surfaces, **kwargs):
        scatter = scatter["xyz".split()].dropna(how="any", inplace=False)
        xaxis, yaxis, zaxis = [scatter[axis] for axis in "xyz".split()]
        figure = self.figure(len(surfaces))
        rows, cols = self.layout(len(surfaces))
        for index, surface in enumerate(surfaces, start=1):
            x = np.linspace(surface.domain.x.min(), surface.domain.x.max(), self.gridsize)
            y = np.linspace(surface.domain.y.min(), surface.domain.y.max(), self.gridsize)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            zz = surface(x, y)
            ax = figure.add_subplot(rows, cols, index, projection="3d")
            ax.set_xlabel("t"), ax.set_ylabel("k"), ax.set_zlabel("w")
            ax.plot_surface(xx, yy, zz, alpha=0.75, color="blue")
            ax.scatter(xaxis, yaxis, zaxis, s=30, color="red")
        plt.show()

    def figure(self, count):
        rows, cols = self.layout(count)
        figsize = (cols * self.plotsize, rows * self.plotsize)
        figure = plt.figure(figsize=figsize)
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


class Surfaces(Assembly): Regressive, Interpolative = RegressiveSurface, InterpolativeSurface
class Curves(Assembly):
    class Regressive(Assembly): Standard = RegressiveCurve
    class Interpolative(Assembly): Standard, Shape, Visual = InterpolativeCurve, ShapeInterpolativeCurve, VisualInterpolativeCurve



