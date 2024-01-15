# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 2024
@name:   Windows Objects
@author: Jack Kirby Cook

"""

import PySimpleGUI as gui
from enum import IntEnum
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Window", "Textual", "Tabular", "Format", "Column", "Justify"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


Justify = IntEnum("Justify", ["LEFT", "CENTER", "RIGHT"], start=1)
Column = ntuple("Column", "name width parser")
Format = ntuple("Format", "name font parser")


class ElementMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        attrs = {key: value for key, value in attrs.items() if not isinstance(value, Column) and not isinstance(value, Format)}
        return super(ElementMeta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        columns = {key: value for key, value in attrs.items() if isinstance(value, Column)}
        formats = {key: value for key, value in attrs.items() if isinstance(value, Format)}
        cls.__columns__ = getattr(cls, "__columns__", {}) | columns
        cls.__formats__ = getattr(cls, "__formats__", {}) | formats

    def __call__(cls, *args, name, **kwargs):
        parameters = dict(name=name, formats=cls.__formats__, columns=cls.__columns__)
        instance = super(ElementMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class Element(ABC, metaclass=ElementMeta):
    def __repr__(self): return self.name
    def __str__(self): return f"--{str(self.name).lower()}--"
    def __init__(self, *args, name, **kwargs): self.__name = name

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    @abstractmethod
    def execute(self, *args, **kwargs): pass
    @property
    def name(self): return self.__name


class Textual(Element, ABC):
    def __init__(self, *args, formats, **kwargs):
        super().__init__(args, **kwargs)
        self.__formats = ODict([(value.name, value.parser) for value in formats.values()])

    @property
    def formats(self): return self.__formats


class Tabular(Element, ABC):
    def __init__(self, *args, columns, justify=Justify.RIGHT, height=None, events=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(height, (int, type(None))) and isinstance(events, bool)
        header = [str(value.name) for value in columns.values()]
        width = [int(value.width) for value in columns.values()]
        justify = str(justify.name).lower()
        formatting = dict(col_widths=width, row_height=height, auto_size_columns=False, justification=justify)
        parameters = dict(headings=header, enable_events=events) | formatting
        self.__columns = ODict([(value.name, value.parser) for value in columns.values()])
        self.__parameters = parameters

    @property
    def columns(self): return self.__columns
    @property
    def parameters(self): return self.__parameters


class Window(ABC):
    def __str__(self): return f"--{str(self.name).lower()}--"
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__window = None

    def __call__(self, *args, **kwargs):
        layout = self.layout(*args, **kwargs)
        self.window = gui.Window(self.name, layout, resizable=True, finalize=True)
        while True:
            self.refresh(*args, **kwargs)
            event, handles = self.window.read()
            handles = handles if isinstance(handles, dict) else {}
            self.execute(event, handles, *args, **kwargs)
            if event == gui.WINDOW_CLOSED:
                break
        self.window.close()
        self.window = None

    @abstractmethod
    def layout(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, event, handles, *args, **kwargs): pass

    @property
    def window(self): return self.__window
    @window.setter
    def window(self, window): self.__window = window
    @property
    def name(self): return self.__name



