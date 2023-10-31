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
__all__ = ["Calculation", "Equation", "Feed", "equation"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


def equation():
    pass


Locator = ntuple("Locator", "key name")
Position = ntuple("Position", "source variable")


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
        attrs = {key: value for key, value in attrs.items() if key not in equations}
        try:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError:
            cls = super(CalculationMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, *args, sources={}, variables={}, constants={}, **kwargs):
        pass

    def __call__(cls, *args, **kwargs):
        pass


class Calculation(ABC, metaclass=CalculationMeta):
    pass



