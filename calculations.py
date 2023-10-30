# -*- coding: utf-8 -*-
"""
Created on Weds Jul 12 2023
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import logging
from abc import ABC, ABCMeta

from support.mixins import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Calculation", "Equation", "Feed", "equation"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


def equation():
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
        pass

    def __init__(cls, *args, **kwargs):
        cls.__calculations__ = {key: value for key, value in getattr(cls, "__calculations__", {}).items()}
        cls.__calculations__.update({key: value for key, value in kwargs.get("calculations", {}).items()})
        cls.__variables__ = {key: value for key, value in getattr(cls, "__variables__", {}).items()}
        cls.__variables__.update({key: value for key, value in kwargs.get("variables", {}).items()})
        cls.__sources__ = {key: value for key, value in getattr(cls, "__sources__", {}).items()}
        cls.__sources__.update({key: value for key, value in kwargs.get("sources", {}).items()})


class Calculation(ABC, metaclass=CalculationMeta):
    pass



