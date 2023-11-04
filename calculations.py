# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import types
import logging
from abc import ABC, ABCMeta
from collections import namedtuple as ntuple

from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "equation", "source"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
Parameter = ntuple("Parameter", "var name")
Style = ntuple("Style", "branch terminate run blank")
double = Style("╠══", "╚══", "║  ", "   ")
single = Style("├──", "└──", "│  ", "   ")
curved = Style("├──", "╰──", "│  ", "   ")


def renderer(node, layers=[], style=single):
    last = lambda i, x: i == x
    func = lambda i, x: "".join([pads(), pre(i, x)])
    pre = lambda i, x: style.terminate if last(i, x) else style.blank
    pads = lambda: "".join([style.blank if layer else style.run for layer in layers])
    if not layers:
        yield "", None, node
    children = iter(node.__children__.items())
    size = len(list(children))
    for index, (key, values) in enumerate(children):
        values = [values] if not isinstance(values, (list, tuple)) else list(values)
        for value in values:
            yield func(index, size - 1), key, value
            yield from renderer(value, layers=[*layers, last(index, size - 1)], style=style)

def source(dataVar, dataName, *args, **kwargs):
    for dataKey, dataValue in kwargs.get("vars", {}).items():
        position = Parameter(dataKey, dataValue)
        variable = Parameter(dataVar, dataName)
        cls = type(".".join(dataVar, dataKey), (Source,), {}, position=position, variable=variable)
        yield cls

def equation(dataVar, dataName, dataType, *args, domain, function, **kwargs):
    assert isinstance(domain, tuple) and callable(function)
    variable = Parameter(dataVar, dataName)
    cls = type(dataVar, (Equation,), {}, datatype=dataType, funciton=function, variable=variable)
    yield cls


class StageMeta(ABCMeta):
    def __repr__(cls): return str(cls.__name__)
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        if not any([type(base) is Stage for base in bases]) or ABC in bases:
            return super(StageMeta, mcs).__new__(mcs, name, bases, attrs)
        cls = super(StageMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, *args, **kwargs):
        cls.__style__ = kwargs.get("style", getattr(cls, "__style__", single))

    def __call__(cls, *args, **kwargs):
        pass

    @property
    def hierarchy(cls):
        generator = renderer(cls, style=cls.__style__)
        rows = [pre + repr(value) for pre, key, value in iter(generator)]
        return "\n".format(rows)


class Stage(Node, ABC, metaclass=StageMeta):
    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)
    def __repr__(self): return str(self.tree)
    def __len__(self): return self.size


class Source(Stage, ABC):
    def __init_subclass__(cls, *args, variable, position=None, **kwargs):
        cls.__position__ = position
        cls.__variable__ = variable


class Equation(Stage, ABC):
    def __init_subclass__(cls, *args, datatype, function, variable, **kwargs):
        cls.__variable__ = variable
        cls.__function__ = function
        cls.__type__ = datatype


class CalculationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        sources = [key for key, value in attrs.items() if isinstance(value, types.GeneratorType) and value.__name__ is "source"]
        equations = [key for key, value in attrs.items() if isinstance(value, types.GeneratorType) and value.__name__ is "equation"]
        exclude = sources + equations
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        try:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, variables={}, **kwargs):
        assert isinstance(variables, dict)
        create = lambda key, value: type(key, (Source,), {}, variable=Parameter(key, value))
        sources = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ is "source"]
        sources = [value for generator in iter(sources) for value in iter(generator)]
        sources = sources + [create(key, value) for key, value in variables.items()]
        sources = {value.__name__: value for value in sources}
        equations = [value for value in attrs.values() if isinstance(value, types.GeneratorType) and value.__name__ is "equation"]
        equations = [value for generator in iter(equations) for value in iter(generator)]
        equations = {value.__name: value for value in equations}
        cls.__sources__ = getattr(cls, "__sources__", {}) | sources
        cls.__equations = getattr(cls, "__equations__", {}) | equations

    def __call__(cls, *args, **kwargs):
        pass


class Calculation(ABC, metaclass=CalculationMeta):
    pass



