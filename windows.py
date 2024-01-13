# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 2024
@name:   Windows Objects
@author: Jack Kirby Cook

"""

import PySimpleGUI as gui
from enum import IntEnum
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Window", "Table", "Column", "Justify"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


Justify = IntEnum("Justify", ["LEFT", "CENTER", "RIGHT"], start=1)


class Column(ntuple("Column", "name width justify parser")):
    pass


class Table(ABC):
    @property
    def layout(self): pass


class Window(ABC):
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        window = gui.Window(repr(self), self.layout, resizable=True, finalize=True)
        window.Maximize()
        while True:
            event, values = window.read()
            if event == gui.WINDOW_CLOSED:
                break
        window.close()

    @abstractmethod
    def layout(self, *args, **kwargs): pass
    @abstractmethod
    def process(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def name(self): return self.__name



