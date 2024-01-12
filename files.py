# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Functions/Objects
@author: Jack Kirby Cook

"""

import os.path
import xarray as xr
import pandas as pd
import dask.dataframe as dk
from abc import ABC, ABCMeta, abstractmethod
from functools import update_wrapper
from collections import OrderedDict as ODict

from support.locks import Locks

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DataframeFile"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = ""


class FileMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.__type__ = kwargs.get("type", getattr(cls, "__type__", None))

    def __call__(cls, contents, *args, capacity, **kwargs):
        filename = kwargs.get("name", cls.__name__)
        filetype = cls.__type__
        assert filetype is not None
        instance = super(cls, FileMeta).__call__(filename, filetype, *args, **kwargs)
        return instance


class File(ABC, metaclass=FileMeta):
    def __init__(self, filename, filetype, *args, repository, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        lockname = str(filename).replace("File", "Lock")
        self.__mutex = Locks(name=lockname, timeout=timeout)
        self.__repository = repository
        self.__type = filetype
        self.__name = filename

    def directory(self, *path): return (content for content in os.listdir(os.path.join(self.repository, *path)))
    def path(self, *path): return os.path.join(self.repository, *path)

    def read(self, *args, file, **kwargs):
        with self.mutex[str(file)]:
            content = load(*args, file=file, type=self.type, **kwargs)
            return content

    def write(self, content, *args, file, mode, **kwargs):
        with self.mutex[str(file)]:
            save(content, *args, file=file, mode=mode, **kwargs)

    @property
    def repository(self): return self.__repository
    @property
    def mutex(self): return self.__mutex
    @property
    def type(self): return self.__type
    @property
    def name(self): return self.__name


class DataframeFile(File, ABC, type=pd.DataFrame):
    def read(self, *args, **kwargs):
        datetypes = self.datetypes(*args, **kwargs)
        datatypes = self.datatypes(*args, **kwargs)
        parameters = dict(datetypes=datetypes, datatypes=datatypes)
        return super().read(*args, **parameters, **kwargs)

    def write(self, *args, **kwargs):
        dataheader = self.dataheader(*args, **kwargs)
        parameters = dict(dataheader=dataheader)
        super().write(*args, **parameters, **kwargs)

    @abstractmethod
    def dataheader(self, *args, **kwargs): pass
    @abstractmethod
    def datetypes(self, *args, **kwargs): pass
    @abstractmethod
    def datatypes(self, *args, **kwargs): pass


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
        fileext = str(os.path.splitext(file)[-1]).strip(".")
        filetype = type(content)
        try:
            function = retrieve(filetype, fileext)
            return function(content, *args, file=file, **kwargs)
        except IndexError:
            return mainfunction(content, *args, file=file, **kwargs)

    def load_wrapper(*args, file, **kwargs):
        fileext = str(os.path.splitext(file)[-1]).strip(".")
        filetype = kwargs["type"]
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
def save_csv(content, *args, file, mode, dataheader, **kwargs):
    header = not os.path.isfile(file) or mode == "w"
    parms = dict(index=False, header=header)
    if isinstance(content, dk.DataFrame):
        update = dict(compute=True, single_file=True, header_first_partition_only=True)
        parms.update(update)
    content[dataheader].to_csv(file, mode=mode, **parms)

@save.register(pd.DataFrame, "hdf")
@save.register(dk.DataFrame, "hdf")
def save_hdf5(self, content, *args, file, group=None, mode, dataheader, **kwargs):
    parms = dict(format="fixed", append=False)
    content[dataheader].to_hdf(file, group, mode=mode, **parms)


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
def load_hdf5(*args, file, group=None, datatypes={}, datetypes=[], **kwargs):
    return pd.read_csv(file, key=group, iterator=False)










