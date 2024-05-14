# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import types
import inspect
import numpy as np
import pandas as pd
import xarray as xr
from enum import Enum
from itertools import product
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict

from support.dispatchers import typedispatcher
from support.pipelines import Processor
from support.meta import SingletonMeta
from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variable", "Equation", "Domain", "Calculation", "Calculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Variable(Node, ABC):
    def __new__(cls, *args, function=None, **kwargs):
        if cls is Variable and bool(function):
            return Dependent(*args, function=function, **kwargs)
        elif cls is Variable and not bool(function):
            return Independent(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, varkey, varname, vartype, *args, **kwargs):
        super().__init__(*args, name=varname, **kwargs)
        self.__type = vartype
        self.__key = varkey

    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)

    def __str__(self): return "|".join([self.key, self.name])
    def __len__(self): return self.size

    @property
    def arguments(self): return tuple([self.key, self.name, self.type])
    @property
    def parameters(self): return dict(formatter=self.formatter, style=self.style)
    def copy(self): return type(self)(*self.arguments, **self.parameters)

    @property
    def type(self): return self.__type
    @property
    def key(self): return self.__key


class Independent(Variable):
    def __init__(self, *args, position, **kwargs):
        assert isinstance(position, str)
        super().__init__(*args, **kwargs)
        self.__position = position

    @property
    def parameters(self): return super().parameters | dict(position=self.position)
    @property
    def position(self): return self.__position


#    def __call__(self, *args, **kwargs):
#        return self.locate(*args, **kwargs)
#        self.__locator = str(locator.name).lower() if isinstance(locator, Enum) else locator

#    def locate(self, *args, **kwargs):
#        contents = args[self.position] if isinstance(self.position, int) else kwargs[self.position]
#        if isinstance(contents, Number):
#            return contents
#        assert isinstance(contents, (xr.Dataset, pd.DataFrame))
#        return contents[self.locator]

#    def execute(self, order):
#        wrapper = lambda *contents: contents[order.index(self)]
#        wrapper.__name__ = str(self.name).lower()
#        return wrapper


class Dependent(Variable):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(function, types.LambdaType)
        self.__function = function

    @property
    def parameters(self): return super().parameters | dict(function=self.function)
    @property
    def sources(self): return list(set(self.leafs))
    @property
    def domain(self): return list(self.children)
    @property
    def function(self): return self.__function


#    def __call__(self, *args, **kwargs):
#        mapping = ODict([(stage, stage.locate(*args, **kwargs)) for stage in self.sources])
#        order, contents = list(mapping.keys()), list(mapping.values())
#        astype = set([type(content) for content in contents if not isinstance(content, Number)])
#        assert len(astype) == 1
#        execute = self.execute(order=order)
#        content = self.calculate(astype=list(astype)[0], execute=execute, contents=contents)
#        content = content.rename(self.name)
#        return content

#    @kwargsdispatcher("astype")
#    def calculate(self, *args, astype, **kwargs): raise TypeError(astype.__name__)
#    @calculate.register.value(xr.DataArray)
#    def dataarray(self, *args, execute, contents, **kwargs): return xr.apply_ufunc(execute, *contents, output_dtypes=[self.type], vectorize=True)
#    @calculate.register.value(pd.Series)
#    def series(self, *args, execute, contents, **kwargs): return pd.concat(list(contents), axis=1).apply(execute, axis=1, raw=True)

#    def execute(self, order):
#        executes = [variable.execute(order) for variable in self.children]
#        wrapper = lambda *args, **kwargs: self.function(*[execute(*args, **kwargs) for execute in executes])
#        wrapper.__name__ = str(self.name).lower()
#        return wrapper


class AlgorithmMeta(SingletonMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, variable in attrs.items() if isinstance(variable, Variable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __iter__(cls): return list(cls.__variables__.items())
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        existing = {key: variable.copy() for key, variable in getattr(cls, "__variables__", {}).items()}
        updated = {variable.key: variable for variable in attrs.values() if isinstance(variable, Variable)}
        cls.__variables__ = existing | updated


class Algorithm(ABC, metaclass=AlgorithmMeta):
    def __iter__(self): return list(self.variables.items())
    def __init__(self, *args, variables, **kwargs):
        assert isinstance(variables, dict)
        assert all([isinstance(variable, Variable) for variable in variables.values()])
        self.__variables = variables

    @property
    def variables(self): return self.__variables


class DomainMeta(AlgorithmMeta): pass
class EquationMeta(AlgorithmMeta): pass


class Domain(Algorithm, metaclass=DomainMeta): pass
class Equation(Algorithm, metaclass=EquationMeta): pass


class Fields(frozenset):
    def __getitem__(self, key): return self.todict()[key]
    def __bool__(self): return None not in self.values()

    def __new__(cls, contents):
        assert isinstance(contents, (dict, list))
        contents = ODict.fromkeys(contents) if isinstance(contents, list) else contents
        contents = list(contents.items())
        return super().__new__(cls, contents)

    def __or__(self, contents):
        assert isinstance(contents, (dict, list))
        contents = ODict.fromkeys(contents) if isinstance(contents, list) else contents
        contents = self.todict() | contents
        return type(self)(contents)

    def todict(self): return ODict(list(self))
    def tolist(self): return list(self)
    def keys(self): return self.todict().keys()
    def values(self): return self.todict().values()
    def items(self): return self.todict().items()


class CalculationMeta(ABCMeta):
    def __iter__(cls): return iter(list(cls.registry.items()))
    def __init__(cls, *args, **kwargs):
        if not any([type(base) is CalculationMeta for base in cls.__bases__]):
            return

#        mapping = lambda contents: {index: value for index, value in enumerate(contents)}
#        equation = kwargs.get("equation", getattr(cls, "Equation", None))
#        domain = kwargs.get("domain", getattr(cls, "Domain", {}))
#        domain = [domain] if issubclass(domain, Domain) else domain
#        domain = mapping(domain) if isinstance(domain, list) else domain
#        cls.Equation, cls.Domain = equation, domain

        if not any([type(subbase) is CalculationMeta for base in cls.__bases__ for subbase in base.__bases__]):
            cls.__fields__ = Fields(list(set(kwargs.get("fields", []))))
            cls.__registry__ = ODict()
            return
        fields = cls.fields | {key: kwargs[key] for key in cls.fields.keys() if key in kwargs.keys()}
        if bool(fields):
            cls.registry[fields] = cls
        cls.__fields__ = fields

    def __call__(cls, *args, **kwargs):
         pass

#        domain = {key: value(*args, **kwargs) for key, value in cls.Domain(*args, **kwargs)}
#        equation = cls.Equation(*args, domain=domain.values(), **kwargs)
#        variables = ODict(list(iter(equation)))
#        for domain in list(cls.Domain.values()):
#            variables = variables | ODict(list(iter(domain)))
#        for variable in variables.values():
#            if isinstance(variable, Dependent):
#                for key in list(inspect.signature(variable.function).parameters.keys()):
#                    variable[key] = variables[key]

    @property
    def registry(cls): return cls.__registry__
    @property
    def fields(cls): return cls.__fields__


class Calculation(ABC, metaclass=CalculationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, **kwargs): pass
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


class Calculator(Processor, ABC, title="Calculated"):
    def __init_subclass__(cls, *args, calculation, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__calculation__ = calculation

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculation = self.__class__.__calculation__
        fields = ODict([(key, []) for key in list(calculation.fields.keys())])
        for field, calculation in iter(calculation):
            for key, value in field.items():
                fields[key].append(value)
        fields = ODict([(key, kwargs.get(key, values)) for key, values in fields.items()])
        assert all([isinstance(values, list) for values in fields.values()])
        fields = [[(key, value) for value in values] for key, values in fields.items()]
        fields = [Fields(ODict(mapping)) for mapping in product(*fields)]
        calculations = {field: calculation(*args, **kwargs) for field, calculation in iter(calculation) if field in fields}
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

