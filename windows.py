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
__all__ = ["Terminal", "Window", "Frame", "Table", "Button", "Text", "Column", "Justify"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


Trinket = IntEnum("Trinket", ["FRAME", "BUTTON", "TABLE"], start=1)
Justify = IntEnum("Justify", ["LEFT", "CENTER", "RIGHT"], start=1)
Column = ntuple("Column", "name width parser")
Text = ntuple("Format", "name font parser")


class ElementMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        attrs = {key: value for key, value in attrs.items() if not isinstance(value, Text) or not isinstance(value, Column)}
        return super(ElementMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        columns = {key: value for key, value in attrs.items() if isinstance(value, Column)}
        texts = {key: value for key, value in attrs.items() if isinstance(value, Text)}
        trinket = kwargs.get("trinket", getattr(cls, "__trinket__", None))
        columns = getattr(cls, "__columns__", {}) | columns
        texts = getattr(cls, "__texts__", {}) | texts
        cls.__trinket__ = trinket
        cls.__columns__ = columns
        cls.__texts__ = texts

    def __call__(cls, *args, **kwargs):
        parameters = dict(trinket=cls.__trinket__, columns=cls.__columns__, texts=cls.__texts__)
        instance = super(ElementMeta, cls).__call__(*args, **parameters, **kwargs)
        return instance


class Element(ABC, metaclass=ElementMeta):
    def __str__(self): return self.key(name=self.name, tag=self.tag, trinket=self.trinket)
    def __repr__(self): return self.name

    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, name, tag, trinket, element, window, **kwargs):
        self.__window = window
        self.__element = element
        self.__trinket = trinket
        self.__name = name
        self.__tag = tag

    @staticmethod
    def key(*args, name, tag, trinket, **kwargs):
        return f"--{str(name).lower()}|{str(trinket.name).lower()}[{int(tag):.0f}]--"

    @property
    def window(self): return self.__window
    @property
    def element(self): return self.__element
    @property
    def trinket(self): return self.__trinket
    @property
    def name(self): return self.__name
    @property
    def tag(self): return self.__tag


class Button(Element, ABC, trinket=Trinket.BUTTON):
    def __init__(self, *args, name, tag, **kwargs):
        key = self.key(*args, name=name, tag=tag, **kwargs)
        element = gui.Button(name, key=key, metadata=self)
        super().__init__(*args, name=name, tag=tag, element=element, **kwargs)


class Frame(Element, ABC, trinket=Trinket.FRAME):
    def __init__(self, *args, name, tag, content, texts={}, **kwargs):
        create = lambda strings, font: [gui.Text(string, font=font) for string in strings] if isinstance(strings, list) else gui.Text(strings, font=font)
        texts = ODict([(value.name, create(value.parser(content), value.font)) for value in texts.values()])
        layout = self.layout(*args, **texts, **kwargs)
        element = gui.Frame("", layout)
        super().__init__(*args, name=name, tag=tag, element=element, **kwargs)

    @staticmethod
    @abstractmethod
    def layout(*args, **kwargs): pass


class Table(Element, ABC, trinket=Trinket.TABLE):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        justify = kwargs.get("justify", getattr(cls, "__justify__", Justify.RIGHT))
        height = kwargs.get("height", getattr(cls, "__height__", None))
        events = kwargs.get("events", getattr(cls, "__events__", False))
        assert isinstance(height, (int, type(None))) and isinstance(events, bool)
        cls.__justify__ = justify
        cls.__height__ = height
        cls.__events__ = events

    def __init__(self, *args, name, tag, contents=[], columns={}, **kwargs):
        assert isinstance(contents, list)
        header = [str(value.name) for value in columns.values()]
        width = [int(value.width) for value in columns.values()]
        justify = str(self.__class__.__justify__.name).lower()
        height = int(self.__class__.__height__)
        events = bool(self.__class__.__events__)
        columns = ODict([(value.name, value.parser) for value in columns.values()])
        formatting = dict(col_widths=width, row_height=height, auto_size_columns=False, justification=justify)
        parameters = dict(headings=header, enable_events=events)
        layout = [[parser(row) for name, parser in columns.items()] for row in contents]
        key = self.key(*args, name=name, tag=tag, **kwargs)
        element = gui.Table(layout, key=key, metadata=self, **formatting, **parameters)
        super().__init__(*args, name=name, tag=tag, element=element, **kwargs)
        self.__columns = columns

    def refresh(self, contents=[]):
        assert isinstance(contents, list)
        layout = [[parser(row) for name, parser in self.columns.items()] for row in contents]
        self.element.update(layout)

    @property
    def columns(self): return self.__columns


class WindowMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super(WindowMeta, cls).__call__(*args, **kwargs)
        instance.start()
        return instance


class Window(ABC, metaclass=WindowMeta):
    def __bool__(self): return self.opened and not self.closed
    def __repr__(self): return self.name

    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, name, parent=None, **kwargs):
        layout = self.layout(*args, **kwargs)
        element = gui.Window(name, layout, resizable=True, finalize=False, metadata=self)
        self.__opened = False
        self.__closed = False
        self.__parent = parent
        self.__element = element
        self.__name = name

    def start(self):
        self.element.finalize()
        self.closed = False
        self.opened = True

    def stop(self):
        self.element.close()
        self.closed = True
        self.opened = False

    @staticmethod
    @abstractmethod
    def layout(*args, **kwargs): pass

    @property
    def closed(self): return self.__closed
    @closed.setter
    def closed(self, closed): self.__closed = closed
    @property
    def opened(self): return self.__opened
    @opened.setter
    def opened(self, opened): self.__opened = opened
    @property
    def parent(self): return self.__parent
    @property
    def element(self): return self.__element
    @property
    def name(self): return self.__name


class Terminal(Window, ABC):
    def __call__(self, *args, **kwargs):
        while bool(self):
            instance, event, values = gui.read_all_windows(timeout=2000)
            if event == gui.TIMEOUT_EVENT:
                continue
            if event == gui.WINDOW_CLOSED:
                window = instance.metadata
                window.stop()
                continue

#    @abstractmethod
#    def execute(self, *args, **kwargs): pass


















