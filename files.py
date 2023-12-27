# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Functions/Objects
@author: Jack Kirby Cook

"""

import os.path
import multiprocessing
import xarray as xr
import pandas as pd
import dask.dataframe as dk
from functools import update_wrapper
from collections import OrderedDict as ODict

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["save", "load", "getFileLocker"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = ""


class FileLockerMeta(type):
    instances = {}

    def __call__(cls, name, *args, **kwargs):
        if name not in FileLockerMeta.instances.keys():
            FileLockerMeta.instances[name] = super(FileLockerMeta, cls).__call__(*args, **kwargs)
        return FileLockerMeta.instances[name]


class FileLocker(dict, metaclass=FileLockerMeta):
    def __getitem__(self, file):
        if file not in self.keys():
            self[file] = multiprocessing.Lock()
        return super().__getitem__(file)


def getFileLocker(name):
    return FileLocker(name)


def dispatcher(mainfunction):
    assert callable(mainfunction)
    __method__ = str(mainfunction.__name__)
    __registry__ = ODict()

    def retrieve(filetype, fileext): return __registry__[(filetype, fileext)]
    def update(filetype, fileext, function): __registry__[(filetype, fileext)] = function

    def register(filetype, fileext):
        def decorator(function):
            assert callable(function)
            update(filetype, fileext, function)
            return function
        return decorator

    def save_wrapper(content, *args, file, **kwargs):
        filetype = type(content)
        fileext = str(os.path.splitext(file)[-1]).strip(".")
        try:
            function = retrieve(filetype, fileext)
            return function(content, *args, file=file, **kwargs)
        except IndexError:
            return mainfunction(content, *args, file=file, **kwargs)

    def load_wrapper(*args, filetype, file, **kwargs):
        fileext = str(os.path.splitext(file)[-1]).strip(".")
        try:
            function = retrieve(filetype, fileext)
            return function(*args, file=file, **kwargs)
        except IndexError:
            return mainfunction(*args, file=file, **kwargs)

    wrappers = dict(save=save_wrapper, load=load_wrapper)
    wrapper = wrappers[mainfunction.__name__]
    wrapper.__method__ = __method__
    wrapper.__registry__ = __registry__
    wrapper.retrieve = retrieve
    wrapper.update = update
    wrapper.register = register
    update_wrapper(wrapper, mainfunction)
    return wrapper


@dispatcher
def save(content, *args, file, **kwargs):
    filetype = type(content)
    fileext = str(os.path.splitext(file)[-1]).strip(".")
    raise ValueError(filetype, fileext)

@save.register(xr.Dataset, "nc")
def save_netcdf(content, *args, file, mode, **kwargs):
    xr.Dataset.to_netcdf(content, file, mode=mode, compute=True)

@save.register(pd.DataFrame, "csv")
@save.register(dk.DataFrame, "csv")
def save_csv(content, *args, file, mode, **kwargs):
    header = not os.path.isfile(file) or mode == "w"
    parms = dict(index=False, header=header)
    if isinstance(content, dk.DataFrame):
        update = dict(compute=True, single_file=True, header_first_partition_only=True)
        parms.update(update)
    content.to_csv(file, mode=mode, **parms)

@save.register(pd.DataFrame, "hdf")
@save.register(dk.DataFrame, "hdf")
def save_hdf5(self, content, *args, file, group=None, mode, **kwargs):
    parms = dict(format="fixed", append=False)
    content.to_hdf(file, group, mode=mode, **parms)


@dispatcher
def load(*args, filetype, file, **kwargs):
    fileext = str(os.path.splitext(file)[-1]).strip(".")
    raise ValueError(filetype, fileext)

@load.register(xr.Dataset, "nc")
def load_netcdf(*args, file, partitions=None, **kwargs):
    return xr.open_dataset(file, chunks=partitions)

@load.register(dk.DataFrame, "csv")
def load_csv_delayed(*args, file, size, datatypes={}, datetypes=[], **kwargs):
    parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
    return dk.read_csv(file, blocksize=size, **parms)

@load.register(pd.DataFrame, "csv")
def load_csv_immediate(*args, file, datatypes={}, datetypes=[], **kwargs):
    parms = dict(index_col=None, header=0, dtype=datatypes, parse_dates=datetypes)
    return pd.read_csv(file,  iterator=False, **parms)

@load.register(pd.DataFrame, "hdf")
def load_hdf5(*args, file, group=None, **kwargs):
    return pd.read_csv(file, key=group, iterator=False)










