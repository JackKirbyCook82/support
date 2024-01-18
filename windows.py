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
__all__ = ["Windows", "Window", "Frame", "Table", "Text", "Column", "Justify"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


Justify = IntEnum("Justify", ["LEFT", "CENTER", "RIGHT"], start=1)
Column = ntuple("Column", "name width parser")
Text = ntuple("Format", "name font parser")


class ElementMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        attrs = {key: value for key, value in attrs.items() if not isinstance(value, Text) or not isinstance(value, Column)}
        return super(ElementMeta, mcs).__new__(mcs, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        instance = super(ElementMeta, cls).__call__(*args, **cls.parameters, **kwargs)
        return instance

    @property
    def parameters(cls): return {}


class FrameMeta(ElementMeta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        texts = {key: value for key, value in attrs.items() if isinstance(value, Text)}
        texts = getattr(cls, "__texts__", {}) | texts
        cls.__texts__ = texts

    @property
    def parameters(cls): return dict(texts=cls.__texts__)


class TableMeta(ElementMeta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        columns = {key: value for key, value in attrs.items() if isinstance(value, Column)}
        justify = kwargs.get("justify", getattr(cls, "__justify__", Justify.RIGHT))
        height = kwargs.get("height", getattr(cls, "__height__", None))
        events = kwargs.get("events", getattr(cls, "__events__", False))
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
    def __repr__(self): return self.name
    def __str__(self): return self.key

    def __init__(self, *args, name, key, element, **kwargs):
        self.__element = element
        self.__name = name
        self.__key = key

    @property
    def element(self): return self.__element
    @property
    def name(self): return self.__name
    @property
    def key(self): return self.__key


class Frame(Element, ABC, metaclass=FrameMeta):
    def __init__(self, *args, name, key, content, texts={}, **kwargs):
        create = lambda strings, font: [gui.Text(string, font=font) for string in strings] if isinstance(strings, list) else gui.Text(strings, font=font)
        texts = ODict([(value.name, create(value.parser(content), value.font)) for value in texts.values()])
        layout = self.layout(*args, **texts, **kwargs)
        element = gui.Frame("", layout, key=key)
        super().__init__(*args, name=name, key=key, element=element, **kwargs)

    @staticmethod
    @abstractmethod
    def layout(*args, **kwargs): pass


class Table(Element, ABC, metaclass=TableMeta):
    def __init__(self, *args, name, key, content=[], columns={}, header, formatting, events, **kwargs):
        columns = ODict([(value.name, value.parser) for value in columns.values()])
        layout = [[parser(row) for name, parser in columns.items()] for row in content]
        element = gui.Table(layout, key=key, headings=header, enable_events=events, **formatting)
        super().__init__(*args, name=name, key=key, element=element, **kwargs)
        self.__columns = columns

    def __call__(self, *args, content=[], **kwargs):
        layout = [[parser(row) for name, parser in self.columns.items()] for row in content]
        self.element.update(layout)

    @property
    def columns(self): return self.__columns


class WindowMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super(WindowMeta, cls).__call__(*args, **kwargs)
        instance.start()
        return instance


class Window(ABC, metaclass=WindowMeta):
    def __bool__(self): return self.started and not self.closed
    def __repr__(self): return self.name
    def __str__(self): return self.key

    def __init__(self, *args, name, key, **kwargs):
        layout = self.layout(*args, **kwargs)
        window = gui.Window(name, layout, resizable=True, finalize=False, metadata=key)
        self.__started = False
        self.__closed = False
        self.__window = window
        self.__name = name
        self.__key = key

    def start(self):
        self.window.finalize()
        self.closed = False
        self.started = True

    def stop(self):
        self.window.close()
        self.closed = True
        self.started = False

    @staticmethod
    @abstractmethod
    def layout(*args, **kwargs): pass

    @property
    def closed(self): return self.__closed
    @closed.setter
    def closed(self, closed): self.__closed = closed
    @property
    def started(self): return self.__started
    @started.setter
    def started(self, started): self.__started = started
    @property
    def window(self): return self.__window
    @property
    def name(self): return self.__name
    @property
    def key(self): return self.__key


class Windows(dict):
    def __bool__(self): return any([bool(window) for window in self.values()])

    def __enter__(self): return self
    def __exit__(self, error_type, error_value, error_traceback):
        for window in self.values():
            window.stop()








