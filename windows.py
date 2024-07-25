# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 18 2024
@name:   Window Objects
@author: Jack Kirby Cook

"""

import pandas as pd
import tkinter as tk
from tkinter import ttk
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Application", "Stencils", "Content", "Column"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Locator(ntuple("Locator", "row column")): pass
class Content(object):
    def __init__(self, element, locator=None, **parameters):
        locator = locator if locator is not None else (0, 0)
        assert isinstance(locator, tuple) and len(locator) == 2
        self.locator = Locator(*locator)
        self.parameters = parameters
        self.element = element

    def __call__(self, parent, *args, **kwargs):
        parameters = {key: value for key, value in self.parameters.items()}
        locator = dict(row=self.locator.row, column=self.locator.column)
        instance = self.element(parent, **parameters, **locator)
        return instance


class Element(tk.BaseWidget):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)


class Action(tk.BaseWidget):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)


class Label(tk.Label, Element):
    def __init__(self, *args, text, font, justify, row, column, **kwargs):
        super().__init__(*args, text=text, font=font, justify=justify, **kwargs)
        self.grid(row=row, column=column, sticky=tk.NW, padx=5, pady=5)


class Button(tk.Button, Action):
    def __init__(self, *args, text, font, justify, row, column, **kwargs):
        super().__init__(*args, text=text, font=font, justify=justify, **kwargs)
        self.grid(row=row, column=column, sticky=tk.SW, padx=10, pady=5)


class CollectionMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, value in attrs.items() if isinstance(value, tuple)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(TableMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        contents = [value[0] for value in attrs.values() if isinstance(value, tuple)]
        locators = [value[1] for value in attrs.values() if isinstance(value, tuple)]
        cls.title = kwargs.get("title", getattr(cls, "title", ""))
        cls.contents = getattr(cls, "contents", []) + list(contents)
        cls.locators = getattr(cls, "locator", []) + list(locators)

    def __call__(cls, *args, **kwargs):
        instance = super(CollectionMeta, cls).__call__(*args, **kwargs)
        for content, locator in zip(cls.contents, cls.locators):
            content(instance, *args, row=locator.row, column=locator.column, **kwargs)
        instance.title(cls.title)
        return instance


class Frame(tk.Frame, Element, metaclass=CollectionMeta):
    def __init__(self, *args, row, column, **kwargs):
        super().__init__(*args, borderwidth=5, **kwargs)
        self.grid(row=row, column=column, sticky=tk.NW, padx=10, pady=10)


class Window(tk.Frame, Element, metaclass=CollectionMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid(row=0, column=0, sticky=tk.NSEW)


class Column(ntuple("Column", "name width parser locator")):
    def __new__(cls, name, *args, width, parser, locator, **kwargs):
        assert callable(parser) and callable(locator)
        return super().__new__(cls, name, width, parser, locator)

    def __call__(self, row, *args, **kwargs):
        assert isinstance(row, pd.Series)
        value = self.locator(row)
        string = self.parser(value)
        return string


class TableMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, value in attrs.items() if isinstance(value, Column)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(TableMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        columns = ODict([(key, value) for key, value in attrs.items() if isinstance(value, Column)])
        cls.columns = getattr(cls, "columns", ODict()) | columns

    def __call__(cls, *args, **kwargs):
        instance = super(TableMeta, cls).__call__(*args, columns=cls.columns, **kwargs)
        for key, column in cls.columns.items():
            instance.column(key, width=int(column.width), anchor=tk.CENTER)
            instance.heading(key, text=str(column.name), anchor=tk.CENTER)
        return instance


class Table(ttk.Treeview, Element, metaclass=TableMeta):
    def __init__(self, *args, columns, **kwargs):
        super().__init__(*args, columns=list(columns.keys()), show="headings", **kwargs)
        self.columns = columns
        self.pack()

    def __call__(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        for index, series in dataframe.iterrows():
            row = [parser(series) for column, parser in self.columns.items()]
            self.insert("", tk.END, iid=None, values=tuple(row))


class Application(tk.Tk, metaclass=SingletonMeta):
    def __init__(self, *args, **kwargs):
        root = tk.Frame(self)
        root.pack(side="top", fill="both", expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        self.root = root

    def __call__(self, *args, **kwargs):
        self.mainloop()


class Stencils:
    Label = Label
    Button = Button
    Frame = Frame
    Window = Window
    Table = Table



