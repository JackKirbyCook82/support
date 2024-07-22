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
    def click(application, window, *args, **kwargs): pass


class TableMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        pass


class Table(ttk.Treeview, metaclass=TableMeta):
    pass


class Frame(tk.Frame):
    pass


class Window(tk.Frame):
    pass


class Application(tk.Tk):
    def __init_subclass__(cls, *args, window, **kwargs):
        assert isinstance(window, Window)
        cls.Window = window

    def __init__(self, *args, **kwargs):
        super().__init__()
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        key = type(self).Window
        window = type(self).Window(container, self)
        self.__windows = {key: window}

    def __call__(self, *args, **kwargs):
        self.mainloop()

    @property
    def windows(self): return self.__windows


class Stencils:
    Application = Application
    Window = Window
    Frame = Frame
    Table = Table
    Button = Button
    Column = Column
    Text = Text



