# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 18 2024
@name:   Window Objects
@author: Jack Kirby Cook

"""

import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod
from collections import namedtuple as ntuple

from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Stencils"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Text(ntuple("Text", "text font justify")):
    def __new__(cls, string, *args, font, justify, **kwargs):
        return super().__new__(cls, string, font, justify)


class Column(ntuple("Column", "heading width parser")):
    def __new__(cls, heading, *args, width, parser, **kwargs):
        return super().__new__(cls, heading, width, parser)


class Button(tk.Button, ABC):
    @staticmethod
    @abstractmethod
    def click(*args, **kwargs): pass


class TableMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        pass


class Table(ttk.Treeview, metaclass=TableMeta):
    pass


class Frame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)
        self.grid(row=0, column=0, sticky="nsew")


class Window(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)
        self.grid(row=0, column=0, sticky="nsew")


class ApplicationMeta(SingletonMeta):
    def __init__(cls, *args, **kwargs):
        cls.Window = kwargs.get("window", getattr(cls, "window", None))

    def __call__(cls, *args, **kwargs):
        instance = super(ApplicationMeta, cls).__call__(*args, window=cls.Window, **kwargs)
        instance.window.tkraise()
        return instance


class Application(tk.Tk, metaclass=SingletonMeta):
    def __init_subclass__(cls, *args, window, **kwargs):
        cls.Window = window

    def __init__(self, *args, window, **kwargs):
        parent = tk.Frame(self)
        parent.pack(side="top", fill="both", expand=True)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        window = window(parent, self, *args, **kwargs)
        self.__window = type(self).Window()
        self.__windows = dict()

    def __call__(self, *args, **kwargs):
        self.mainloop()

    @property
    def windows(self): return self.__windows
    @property
    def window(self): return self.__window


class Stencils:
    Application = Application
    Window = Window
    Frame = Frame
    Table = Table
    Button = Button
    Column = Column
    Text = Text



