# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 18 2024
@name:   Window Objects
@author: Jack Kirby Cook

"""

import tkinter as tk
from tkinter import ttk
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple

from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Application", "Stencils"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Element(ABC): pass
class Action(ABC): pass


class Text(ntuple("Text", "text font justify"), Element):
    def __new__(cls, string, *args, font, justify, **kwargs):
        return super().__new__(cls, string, font, justify)


class Column(ntuple("Column", "heading width parser"), Element):
    def __new__(cls, heading, *args, width, parser, **kwargs):
        return super().__new__(cls, heading, width, parser)


class CollectionMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, value in attrs.items() if isinstance(value, Element)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(CollectionMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        elements = {key: value for key, value in attrs.items() if isinstance(value, Element)}
        cls.elements = getattr(cls, "elements", {}) | elements

    def __call__(cls, parent, *args, **kwargs):
        instance = super(CollectionMeta, cls).__call__(parent, *args, **kwargs)
        for key, element in cls.elements.items():
            content = element(instance, *args, **kwargs)
            instance[key] = content
        return instance


class Collection(tk.Frame, metaclass=CollectionMeta):
    def __setitem__(self, key, value): self.contents[key] = value
    def __getitem__(self, key): return self.contents[key]

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)
        self.__contents = dict()
        self.__parent = parent

    @property
    def contents(self): return self.__contents
    @property
    def parent(self): return self.__parent


class Table(Collection): pass
class Layout(Collection): pass


class Button(tk.Button, Action):
    @staticmethod
    @abstractmethod
    def click(*args, **kwargs): pass


class WindowMeta(ABCMeta):
    def __init__(cls, *args, **kwargs):
        cls.elements = getattr(cls, "elements", {}) | kwargs.get("elements", {})
        cls.title = kwargs.get("title", getattr(cls, "__title__", cls.__name__))

    def __call__(cls, parent, *args, **kwargs):
        instance = super(WindowMeta, cls).__call__(parent, *args, **kwargs)
        instance.title(cls.title)
        for key, element in cls.elements.items():
            content = element(instance, *args, **kwargs)
            instance[key] = content


class Window(tk.Frame):
    def __setitem__(self, key, value): self.contents[key] = value
    def __getitem__(self, key): return self.contents[key]

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)
        self.grid(row=0, column=0, sticky="nsew")
        self.__contents = dict()
        self.__parent = parent

    @property
    def contents(self): return self.__contents
    @property
    def parent(self): return self.__parent


class ApplicationMeta(SingletonMeta):
    def __init__(cls, *args, **kwargs):
        cls.window = kwargs.get("window", getattr(cls, "__window__", None))

    def __call__(cls, *args, **kwargs):
        instance = super(ApplicationMeta, cls).__call__(*args, **kwargs)
        root = tk.Frame(instance)
        root.pack(side="top", fill="both", expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        window = cls.window(root, *args, **kwargs)
        instance.window = window
        return instance


class Application(tk.Tk, metaclass=SingletonMeta):
    def __setitem__(self, key, value): self.windows[key] = value
    def __getitem__(self, key): return self.windows[key]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__windows = dict()
        self.__window = None

    def __call__(self, *args, **kwargs):
#        self.execute(*args, **kwargs)
        self.mainloop()

#    @abstractmethod
#    def execute(self, *args, **kwargs): pass

    @property
    def window(self): return self.__window
    @window.setter
    def window(self, window): self.__window = window


class Stencils:
    Window = Window
    Layout = Layout
    Table = Table
    Button = Button
    Column = Column
    Text = Text



