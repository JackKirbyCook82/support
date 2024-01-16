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
__all__ = ["Driver", "Window", "Text", "Table", "Format", "Column", "Justify"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


Justify = IntEnum("Justify", ["LEFT", "CENTER", "RIGHT"], start=1)
Column = ntuple("Column", "name width parser")
Format = ntuple("Format", "name font parser")


class ElementMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        attrs = {key: value for key, value in attrs.items() if not isinstance(value, Format) or not isinstance(value, Column)}
        return super(ElementMeta, mcs).__new__(mcs, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        instance = super(ElementMeta, cls).__call__(*args, **cls.parameters, **kwargs)
        return instance

    @property
    def parameters(cls): return {}


class TextMeta(ElementMeta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        formats = {key: value for key, value in attrs.items() if isinstance(value, Format)}
        formats = getattr(cls, "__formats__", {}) | formats
        cls.__formats__ = formats

    @property
    def parameters(cls): return dict(formats=cls.__formats__)


class TableMeta(ElementMeta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        columns = {key: value for key, value in attrs.items() if isinstance(value, Column)}
        justify = kwargs.get("justify", getattr(cls, "__justify__", Justify.RIGHT))
        height = kwargs.get("height", getattr(cls, "__height__", None))
        events = kwargs.get("event", getattr(cls, "__events__", False))
        columns = getattr(cls, "__columns__", {}) | columns
        assert isinstance(height, (int, type(None))) and isinstance(events, bool)
        cls.__columns__ = columns
        cls.__justify__ = justify
        cls.__height__ = height
        cls.__events__ = events

    @property
    def parameters(cls):
        header = [str(value.name) for value in cls.__columns__.values()]
        width = [int(value.width) for value in cls.__columns__.values()]
        justify = str(cls.__justify__.name).lower()
        formatting = dict(col_widths=width, row_height=cls.__height__, auto_size_columns=False, justification=justify)
        return dict(columns=cls.__columns__, header=header, formatting=formatting, events=cls.__events__)


class Element(ABC, metaclass=ElementMeta):
    def __str__(self): return f"--{str(self.name).lower()}--"
    def __repr__(self): return self.name
    def __init__(self, name, element):
        self.__element = element
        self.__name = name

    @abstractmethod
    def layout(self, *args, **kwargs): pass
    @property
    def element(self): return self.__element
    @property
    def name(self): return self.__name


class Text(Element, ABC, metaclass=TextMeta):
    def __init__(self, *args, name, content, formats={}, **kwargs):
        key = f"--{str(name).lower()}--"
        formats = ODict([(value.name, value.parser) for value in formats.values()])
        layout = self.layout(*args, content=content, formats=formats, **kwargs)
        element = gui.Text(layout, key=key)
        super().__init__(name, element)


class Table(Element, ABC, metaclass=TableMeta):
    def __init__(self, *args, name, rows=[], columns={}, header, formatting, events, **kwargs):
        key = f"--{str(name).lower()}--"
        columns = ODict([(value.name, value.parser) for value in columns.values()])
        layout = self.layout(*args, rows=rows, columns=columns, **kwargs)
        element = gui.Table(layout, key=key, headings=header, enable_events=events, **formatting)
        super().__init__(name, element)

    @staticmethod
    def layout(*args, rows=[], columns={}, **kwargs):
        return [[parser(row) for name, parser in columns.items()] for row in rows]


class Window(Element, ABC):
    def __str__(self): return f"--{str(self.name).lower()}--"
    def __repr__(self): return self.name

    def __init__(self, *args, name, **kwargs):
        key = f"--{str(name).lower()}--"
        layout = self.layout(*args, **kwargs)
        window = gui.Window(name, layout, key=key, resizable=True, finalize=True)
        self.__window = window
        self.__name = name

    def __enter__(self): return self
    def __exit__(self, error_type, error_value, error_traceback):
        self.window.close()

#    def read(self):
#        return self.window.read()

    @abstractmethod
    def layout(self, *args, **kwargs): pass
    @property
    def window(self): return self.__window
    @property
    def name(self): return self.__name


class Driver(ABC):
    def __init_subclass__(cls, *args, window, **kwargs): cls.__window__ = window
    def __init__(self, *args, **kwargs):
        self.__window = kwargs.get("window", self.__class__.__window__)
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __repr__(self): return self.name
    def __call__(self, *args, **kwargs):
        title = repr(self)
        with self.window(*args, name=title, **kwargs) as window:
            assert isinstance(window, Window)
            while True:
                pass

#                event, handles = window.read()
#                if event == gui.WINDOW_CLOSED:
#                    break

    @property
    def window(self): return self.__window
    @property
    def name(self): return self.__name



