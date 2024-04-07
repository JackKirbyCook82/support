# -*- coding: utf-8 -*-
"""
Created on Sun 14 2023
@name:   File Objects
@author: Jack Kirby Cook

"""

import os
import logging
import multiprocessing
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dk
from enum import IntEnum
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher, kwargsdispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Archive", "DataframeFile", "DatasetFile", "FileTyping", "FileTiming"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


FileTyping = IntEnum("FileTyping", ["NC", "HDF", "CSV"], start=1)
FileTiming = IntEnum("FileTiming", ["EAGER", "LAZY"], start=1)
FileMethod = ntuple("FileMethod", "filetyping filetiming")
FileDataframe = ntuple("FileDataframe", "name index columns")
FileDataset = ntuple("FileDataset", "")

csv_eager = FileMethod(FileTyping.CSV, FileTiming.EAGER)
csv_lazy = FileMethod(FileTyping.CSV, FileTiming.LAZY)
hdf_eager = FileMethod(FileTyping.HDF, FileTiming.EAGER)
hdf_lazy = FileMethod(FileTyping.HDF, FileTiming.LAZY)
nc_eager = FileMethod(FileTyping.NC, FileTiming.EAGER)
nc_lazy = FileMethod(FileTyping.NC, FileTiming.LAZY)


class File(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.DataType = kwargs.get("type", getattr(cls, "DataType", None))

    def __repr__(self): return f"{str(self.name)}[{str(len(self))}]"
    def __init__(self, *args, variable, typing, timing, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__variable = variable
        self.__timing = timing
        self.__typing = typing

    def load(self, *args, folder, **kwargs):
        file = os.path.join(folder, self.filename)
        method = FileMethod(self.typing, self.timing)
        if not os.path.exists(file):
            return
        content = self.loader(*args, file=file, method=method, **kwargs)
        content = self.parser(content, *args, **kwargs)
        if self.empty(content):
            return
        return content

    def save(self, content, *args, folder, **kwargs):
        if self.empty(content):
            return
        file = os.path.join(folder, self.filename)
        method = FileMethod(self.typing, self.timing)
        content = self.formatter(content, *args, **kwargs)
        self.saver(content, *args, file=file, method=method, **kwargs)
        __logger__.info("Saved: {}".format(str(file)))

    @abstractmethod
    def loader(self, *args, file, **kwargs): pass
    @abstractmethod
    def saver(self, content, *args, file, **kwargs): pass
    @abstractmethod
    def parser(self, content, *args, **kwargs): pass
    @abstractmethod
    def formatter(self, content, *args, **kwargs): pass
    @abstractmethod
    def empty(self, content, *args, **kwargs): pass

    @property
    def filename(self): return ".".join([self.variable, str(self.typing.name).lower()])
    @property
    def variable(self): return self.__variable
    @property
    def timing(self): return self.__timing
    @property
    def typing(self): return self.__typing
    @property
    def name(self): return self.__name


class DataframeFile(File, type=pd.DataFrame):
    def __init_subclass__(cls, *args, variable, index, columns, **kwargs):
        assert isinstance(index, dict) and isinstance(columns, dict)
        cls.__parameters__ = FileDataframe(variable, index, columns)

    def __init__(self, *args, **kwargs):
        variable, index, columns = self.__class__.__parameters__
        super().__init__(*args, variable=variable, **kwargs)
        self.__types = {key: value for key, value in (index | columns).items() if not any([value is str, value is np.datetime64])}
        self.__dates = [key for key, value in (index | columns).items() if value is np.datetime64]
        self.__columns = list(columns.keys())
        self.__index = list(index.keys())

    @kwargsdispatcher("method")
    def loader(self, *args, file, mode, **kwargs): raise ValueError()
    @kwargsdispatcher("method")
    def saver(self, content, args, file, mode, **kwargs): raise ValueError()

    @loader.register.value(csv_eager)
    def loader_eager_csv(self, *args, file, mode, **kwargs):
        return pd.read_csv(file, iterator=False, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @loader.register.value(csv_lazy)
    def loader_lazy_csv(self, *args, file, mode, size, **kwargs):
        return dk.read_csv(file, blocksize=size, index_col=None, header=0, dtype=self.types, parse_dates=self.dates)

    @loader.register.value(hdf_eager)
    def loader_eager_hdf(self, *args, file, mode, **kwargs): pass
    @loader.register.value(hdf_lazy)
    def loader_lazy_hdf(self, *args, file, mode, **kwargs): pass

    @saver.register.value(csv_eager)
    def saver_eager_csv(self, content, *args, file, mode, **kwargs):
        content.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w")

    @saver.register.value(csv_lazy)
    def saver_lazy_csv(self, content, *args, file, mode, **kwargs):
        parameters = dict(compute=True, single_file=True, header_first_partition_only=True)
        content.to_csv(file, mode=mode, index=False, header=not os.path.isfile(file) or mode == "w", **parameters)

    @saver.register.value(hdf_eager)
    def saver_eager_hdf(self, content, *args, file, mode, **kwargs): pass
    @saver.register.value(hdf_lazy)
    def saver_lazy_hdf(self, content, *args, file, mode, **kwargs): pass

    def parser(self, content, *args, **kwargs):
        index = [column for column in self.index if column in content.columns]
        content = content.set_index(index, drop=True, inplace=False)
        columns = [column for column in self.columns if column in content.columns]
        return content[columns]

    def formatter(self, content, *args, **kwargs):
        content = content.reset_index(drop=False, inplace=False)
        columns = [column for column in self.index + self.columns if column in content.columns]
        return content[columns]

    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
    @empty.register(pd.DataFrame)
    def dataframe(self, content): return bool(content.empty)
    @empty.register(pd.Series)
    def dataframe(self, content): return bool(content.empty)
    @empty.register(type(None))
    def null(self, content): return content is None

    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index
    @property
    def types(self): return self.__types
    @property
    def dates(self): return self.__dates


class DatasetFile(File, type=xr.Dataset):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, **kwargs): pass

    @kwargsdispatcher("method")
    def loader(self, *args, file, mode, **kwargs): raise ValueError()
    @kwargsdispatcher("method")
    def saver(self, content, args, file, mode, **kwargs): raise ValueError()

    @loader.register.value(nc_eager)
    def loader_eager_nc(self, *args, file, mode, **kwargs): pass
    @loader.register.value(nc_lazy)
    def loader_lazy_nc(self, *args, file, mode, **kwargs): pass
    @saver.register.value(nc_eager)
    def saver_eager_nc(self, content, *args, file, mode, **kwargs): pass
    @saver.register.value(nc_lazy)
    def saver_lazy_nc(self, content, *args, file, mode, **kwargs): pass

    def parser(self, content, *args, **kwargs): pass
    def formatter(self, content, *args, **kwargs): pass

    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
    @empty.register(xr.Dataset)
    def dataframe(self, content): return all([self.empty(dataarray) for dataarray in content.data_vars.values()])
    @empty.register(xr.DataArray)
    def dataframe(self, content): return bool(np.count_nonzero(~np.isnan(content.values)))
    @empty.register(type(None))
    def null(self, content): return content is None


class ArchiveMeta(ABCMeta):
    locks = {}

    def __call__(cls, *args, repository, **kwargs):
        lock = ArchiveMeta.locks.get(str(repository),  multiprocessing.RLock())
        ArchiveMeta.locks[str(repository)] = lock
        instance = super(ArchiveMeta, cls).__call__(*args, repository=repository, lock=lock, **kwargs)
        return instance


class Archive(ABC, metaclass=ArchiveMeta):
    def __repr__(self): return f"{self.name}[{', '.join([name for name in self.files.keys()])}]"
    def __init__(self, *args, repository, loading=[], saving=[], lock, **kwargs):
        assert isinstance(loading, (File, list)) and isinstance(saving, (File, list))
        loading = {str(file.variable): file for file in ([loading] if isinstance(loading, File) else loading)}
        saving = {str(file.variable): file for file in ([saving] if isinstance(saving, File) else saving)}
        if not os.path.exists(repository):
            os.makedirs(repository)
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__repository = repository
        self.__loading = loading
        self.__saving = saving
        self.__mutex = lock

    def load(self, *args, folder, **kwargs):
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        if not os.path.exists(folder):
            return {}
        with self.mutex:
            contents = {name: file.load(*args, folder=folder, **kwargs) for name, file in self.loading.items()}
            contents = {name: content for name, content in contents.items() if content is not None}
            return contents

    def save(self, contents, *args, folder, **kwargs):
        folder = os.path.join(self.repository, folder) if folder is not None else self.repository
        if not os.path.exists(folder):
            os.makedirs(folder)
        with self.mutex:
            contents = {name: (file, contents.get(name, None)) for name, file in self.saving.items()}
            contents = {name: (file, content) for name, (file, content) in contents.items() if content is not None}
            for name, (file, content) in contents.items():
                file.save(content, *args, folder=folder, **kwargs)

    @property
    def directory(self): return os.listdir(self.repository)
    @property
    def empty(self): return not bool(os.listdir(self.repository))
    @property
    def size(self): return len(os.listdir(self.repository))
    @property
    def files(self): return self.loading | self.saving

    @property
    def repository(self): return self.__repository
    @property
    def loading(self): return self.__loading
    @property
    def saving(self): return self.__saving
    @property
    def mutex(self): return self.__mutex

