# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Objects
@author: Jack Kirby Cook

"""

import os.path
import multiprocessing
import xarray as xr
import pandas as pd
import dask.dataframe as dk
from abc import ABC, abstractmethod

from support.dispatchers import kwargsdispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Saver", "Loader", "Reader", "Referer"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = ""


class File(ABC):
    __locks__ = {}

    def __call__(self, *args, file, **kwargs):
        locks = self.__class__.__locks__
        if file not in locks.keys():
            locks[file] = multiprocessing.Lock()
        with locks[file]:
            self.execute(*args, file=file, **kwargs)

    @abstractmethod
    def execute(self, content, *args, file, **kwargs):
        pass


class Saver(File):
    @kwargsdispatcher(key="file", func=lambda file: str(os.path.splitext(file)[-1]).strip("."))
    def execute(self, content, *args, file, **kwargs):
        raise ValueError(file)

    @execute.register.value("nc")
    def netcdf(self, content, *args, file, mode, **kwargs):
        assert isinstance(content, xr.Dataset)
        xr.Dataset.to_netcdf(content, file, mode=mode, compute=True)

    @execute.register.value("csv")
    def csv(self, content, *args, file, mode, **kwargs):
        assert isinstance(content, (pd.DataFrame, dk.DataFrame))
        parms = dict(index=False, header=True)
        if isinstance(content, dk.DataFrame):
            update = dict(compute=True, single_file=True, header_first_partition_only=True)
            parms.update(update)
        content.to_csv(file, mode=mode, **parms)

    @execute.register.value("hdf")
    def hdf5(self, content, *args, file, group=None, mode, **kwargs):
        assert isinstance(content, (pd.DataFrame, dk.DataFrame))
        parms = dict(format="fixed", append=False)
        content.to_hdf(file, group, mode=mode, **parms)


class Loader(File):
    @kwargsdispatcher(key="file", func=lambda file: str(os.path.splitext(file)[-1]).strip("."))
    def execute(self, *args, file, **kwargs):
        raise ValueError(file)

    @execute.register.value("nc")
    def netcdf(self, *args, file, **kwargs):
        return xr.open_dataset(file, chunks=None)

    @execute.register.value("csv")
    def csv(self, *args, file, datatypes={}, datetypes=[], **kwargs):
        parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
        return pd.read_csv(file, iterator=False, **parms)

    @execute.register.value("hdf")
    def hdf5(self, *args, file, group=None, **kwargs):
        return pd.read_hdf(file, key=group, iterator=False)


class Reader(File):
    @kwargsdispatcher(key="file", func=lambda file: str(os.path.splitext(file)[-1]).strip("."))
    def execute(self, *args, file, **kwargs):
        raise ValueError(file)

    @execute.register.value("csv")
    def csv(self, *args, file, rows, datatypes={}, datetypes=[], **kwargs):
        parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
        return pd.read_csv(file, chucksize=rows, iterator=True, **parms)

    @execute.register.value("hdf")
    def hdf5(self, *args, file, group=None, rows, **kwargs):
        return pd.read_csv(file, key=group, chunksize=rows, iterator=True)


class Referer(File):
    @kwargsdispatcher(key="file", func=lambda file: str(os.path.splitext(file)[-1]).strip("."))
    def execute(self, *args, file, **kwargs):
        raise ValueError(file)

    @execute.register.value("nc")
    def netcdf(self, *args, file, partitions={}, **kwargs):
        assert isinstance(partitions, dict)
        return xr.open_dataset(file, chunks=partitions)

    @execute.register.value("csv")
    def csv(self, *args, file, size, datatypes={}, datetypes=[], **kwargs):
        parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
        return dk.read_csv(file, blocksize=size, **parms)



