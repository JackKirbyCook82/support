# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher
from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variable", "Domain", "Equation", "Calculation", "Calculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Variable(object):
    pass


class Domain(object):
    def __init__(self, *args, **kwargs): pass
    def __call__(self):

class Equation(object):
    pass


class Fields(frozenset):
    def __getitem__(self, key): return self.todict()[key]
    def __new__(cls, mapping):
        assert isinstance(mapping, ODict)
        mapping = list(mapping.items())
        return super().__new__(cls, mapping)

    def todict(self): return ODict(list(self))
    def tolist(self): return list(self)
    def keys(self): return self.todict().keys()
    def values(self): return self.todict().values()
    def items(self): return self.todict().items()


class CalculationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs)
        if bool(cls):
            fields = Fields(cls.fields)
            cls.registry[fields] = cls
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        if not any([type(base) is CalculationMeta for base in bases]):
            return
        if not any([type(subbase) is CalculationMeta for base in bases for subbase in base]):
            assert all([attr not in kwargs["fields"] for attr in ("equation", "domain", "fields")])
            cls.__fields__ = ODict.fromkeys(list(set(kwargs["fields"])))
            cls.__registry__ = ODict()
        equation = kwargs.get("equation", getattr(cls, "__equation__", None))
        domain = kwargs.get("domain", getattr(cls, "__domain__", []))
        fields = ODict([(key, kwargs.get(key, value)) for key, value in getattr(cls, "__fields__", {}).items()])
        assert issubclass(equation, Equation) if equation is not None else True
        assert isinstance(domain, list) and all([isinstance(value, dict) for value in domain])
        cls.__equation__ = equation
        cls.__domain__ = domain
        cls.__fields__ = fields

    def __bool__(cls): return None not in cls.fields.values()
    def __iter__(cls): return iter(list(cls.registry.items()))

    def __call__(cls, *args, **kwargs):
        parameters = dict(equation=cls.__equation__, domain=cls.__domain)
        instance = super(CalculationMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def registry(cls): return cls.__registry__
    @property
    def fields(cls): return cls.__fields__


class Calculation(ABC, metaclass=CalculationMeta):
    def __init__(self, *args, equation, domain, **kwargs):
        self.__equation = equation(*args, **kwargs)
        self.__domain = domain(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        contents = list(generator)
        content = self.combine(contents[0], *contents[1:])
        return content

    @typedispatcher
    def combine(self, content, *contents): raise TypeError(type(content).__name__)
    @combine.register(xr.DataArray)
    def dataarray(self, content, *contents): return xr.merge([content] + list(contents))
    @combine.register(pd.Series)
    def series(self, content, *contents): return pd.concat([content] + list(contents), axis=1)

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def equation(self): return self.__equation
    @property
    def domain(self): return self.__domain


class Calculator(Processor, ABC, title="Calculated"):
    def __init_subclass__(cls, *args, calculation, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__calculation__ = calculation

    def __init__(self, *args, domain, equation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculation = self.__class__.__calculation__
        fields = ODict([(key, []) for key in list(calculation.fields.keys())])
        for field, calculation in iter(calculation):
            for key, value in field.items():
                fields[key].append(value)
        fields = ODict([(key, kwargs.get(key, values)) for key, values in field.items()])
        assert all([isinstance(values, list) for values in fields.values()])
        fields = [[(key, value) for value in values] for key, values in fields.items()]
        fields = [Fields(mapping) for mapping in product(*fields)]
        calculations = ODict([(field, calculation) for field, calculation in iter(calculation) if field in fields])
        self.__calculations = calculations

    @typedispatcher
    def empty(self, content): raise TypeError(type(content).__name__)
    @empty.register(xr.DataArray)
    def empty_dataarray(self, dataarray): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @empty.register(pd.DataFrame)
    def empty_dataframe(self, dataframe): return bool(dataframe.empty)
    @empty.register(pd.Series)
    def empty_series(self, series): return bool(series.empty)

    @property
    def calculations(self): return self.__calculations

