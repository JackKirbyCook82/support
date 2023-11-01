# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import inspect
import logging
from abc import ABC, ABCMeta
from collections import namedtuple as ntuple

from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "Equation", "Feed", "equation", "source"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


def equation(name, datatype, *args, domain, function, **kwargs):
    assert isinstance(domain, tuple) and callable(function)


def source(name, variables, *args, **kwargs):
    assert isinstance(variables, dict)


class Source(object):
    pass


class Stage(Node):
    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)
    def __repr__(self): return str(self.tree)
    def __len__(self): return self.size


class Feed(Stage):
    pass


class Equation(Stage):
    pass


class CalculationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        equations = [key for key, value in attrs.items() if inspect.isclass(value) and issubclass(value, Equation)]
        sources = [key for key, value in attrs.items() if isinstance(value, Source)]
        exclude = equations + sources
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        try:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, variables={}, **kwargs):
        assert isinstance(variables, dict)
        arguments = {index: value for index, value in getattr(cls, "__arguments__", {}).items()}
        parameters = {key: value for key, value in getattr(cls, "__parameters__", {}).items()}
        arguments[0] = {variable: name for variable, name in arguments.get(0, {}).items()}
        arguments[0].update(variables)
        sources = {key: value for key, value in attrs.items() if isinstance(value, Source)}
        for key, value in sources.items():
            parameters[key] = {variable: name for variable, name in parameters.get(key, {}).items()}
            parameters[key].update(value.variables)
        cls.__arguments__ = arguments
        cls.__parameters__ = parameters

    def __call__(cls, *args, **kwargs):
        pass


class Calculation(ABC, metaclass=CalculationMeta):
    pass



