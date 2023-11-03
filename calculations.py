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
__all__ = ["Calculation", "equation", "source"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


def source(name, variables, *args, **kwargs):
    assert isinstance(variables, dict)


def equation(name, datatype, *args, domain, function, **kwargs):
    assert isinstance(domain, tuple) and callable(function)


class Stage(Node):
    def __setitem__(self, key, value): self.set(key, value)
    def __getitem__(self, key): return self.get(key)
    def __repr__(self): return str(self.tree)
    def __len__(self): return self.size


class CalculationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        pass

    def __init__(cls, name, bases, attrs, *args, variables={}, **kwargs):
        assert isinstance(variables, dict)

    def __call__(cls, *args, **kwargs):
        pass


class Calculation(ABC, metaclass=CalculationMeta):
    pass



