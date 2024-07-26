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
__all__ = ["Application", "Stencils", "Layouts"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Widget(object):
    def __init__(self, stencil, **parameters):
        Locator = ntuple("Locator", "row column")
        parameters = {key: value for key, value in parameters.items()}
        locator = parameters.get("locator", (0, 0))
        assert isinstance(locator, tuple) and len(locator) == 2
        parameters["locator"] = Locator(*locator)
        self.parameters = parameters
        self.stencil = stencil

    def __call__(self, parent, *args, **kwargs):
        fields = list(self.parameters.keys())
        values = list(self.parameters.values())
        Parameters = ntuple("Parameters", fields)
        parameters = Parameters(*values)
        instance = self.stencil(parent, parameters, *args, **kwargs)
        return instance


class ContainerMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, value in attrs.items() if isinstance(value, Widget)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(ContainerMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(ContainerMeta, cls).__init__(name, bases, attrs)
        widgets = {key: value for key, value in attrs.items() if isinstance(value, Widget)}
        cls.widgets = getattr(cls, "widgets", {}) | dict(widgets)

    def __call__(cls, *args, **kwargs):
        instance = super(ContainerMeta, cls).__call__(*args, **kwargs)
        for key, widget in cls.widgets.items():
            content = widget(instance, *args, **kwargs)
            instance[key] = content
        return instance


class Stencil(object):
    def __init__(self, *args, **kwargs): pass

class Element(Stencil): pass
class Action(Stencil): pass
class Container(Stencil, metaclass=ContainerMeta):
    def __setitem__(self, key, value): pass


class Window(Container, tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent)
        Container.__init__(self, *args, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)


class Notebook(Container, ttk.Notebook):
    def __init__(self, parent, *args, **kwargs):
        ttk.Notebook.__init__(self, parent)
        Container.__init__(self, *args, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)

    def __setitem__(self, key, value):
        self.add(value, text=str(key))


class Frame(Container, tk.Frame):
    def __init__(self, parent, parameters, *args, **kwargs):
        tk.Frame.__init__(self, parent, borderwidth=5)
        Container.__init__(self, *args, **kwargs)
        self.grid(row=parameters.locator.row, column=parameters.locator.column, padx=10, pady=10)


class Label(Element, tk.Label):
    def __init__(self, parent, parameters, *args, **kwargs):
        tk.Label.__init__(self, parent, text=parameters.text, font=parameters.font, justify=parameters.justify)
        Element.__init__(self, *args, **kwargs)
        self.grid(row=parameters.locator.row, column=parameters.locator.column, padx=5, pady=5)


class Button(Action, tk.Button):
    def __init__(self, parent, parameters, *args, **kwargs):
        tk.Button.__init__(self, parent, text=parameters.text, font=parameters.font, justify=parameters.justify)
        Action.__init__(self, *args, **kwargs)
        self.grid(row=parameters.locator.row, column=parameters.locator.column, padx=10, pady=5)


class Scroll(Action, tk.Scrollbar):
    def __init__(self, parent, parameters, *args, **kwargs):
        tk.Scrollbar.__init__(self, parent, orient=parameters.orientation)
        Action.__init__(self, *args, **kwargs)
        self.grid(row=parameters.locator.row, column=parameters.locator.column)


class Column(ntuple("Column", "text width parser locator")):
    def __new__(cls, *args, text, width, parser, locator, **kwargs):
        assert callable(parser) and callable(locator)
        return super().__new__(cls, text, width, parser, locator)

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
        super(TableMeta, cls).__init__(name, bases, attrs)
        columns = ODict([(key, value) for key, value in attrs.items() if isinstance(value, Column)])
        cls.columns = getattr(cls, "columns", ODict()) | columns

    def __call__(cls, *args, **kwargs):
        instance = super(TableMeta, cls).__call__(*args, columns=cls.columns, **kwargs)
        for key, column in cls.columns.items():
            instance.column(key, width=int(column.width), anchor=tk.CENTER)
            instance.heading(key, text=str(column.text), anchor=tk.CENTER)
        return instance


class Table(Action, ttk.Treeview, metaclass=TableMeta):
    def __init__(self, parent, parameters, *args, columns, **kwargs):
        ttk.Treeview.__init__(self, parent, columns=list(columns.keys()), show="headings")
        Action.__init__(self, *args, **kwargs)
        self.grid(row=parameters.locator.row, column=parameters.locator.column)
        self.columns = columns

#    def __call__(self, dataframe):
#        assert isinstance(dataframe, pd.DataFrame)
#        for index, series in dataframe.iterrows():
#            row = [parser(series) for column, parser in self.columns.items()]
#            self.insert("", tk.END, iid=None, values=tuple(row))


class ApplicationMeta(SingletonMeta):
    def __init__(cls, *args, **kwargs):
        super(ApplicationMeta, cls).__init__(*args, **kwargs)
        cls.Window = kwargs.get("window", getattr(cls, "Window", None))
        cls.Title = kwargs.get("title", getattr(cls, "title", None))

    def __call__(cls, *args, **kwargs):
        instance = super(ApplicationMeta, cls).__call__(*args, **kwargs)
        instance.title(cls.Title)
        return instance


class Application(tk.Tk, metaclass=ApplicationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, **kwargs):
        super().__init__()
        parent = tk.Frame(self)
        parent.pack(fill=tk.BOTH, expand=True)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        type(self).Window(parent, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.mainloop()


class Layouts:
    Widget = Widget
    Column = Column

class Stencils:
    Window = Window
    Notebook = Notebook
    Label = Label
    Button = Button
    Frame = Frame
    Window = Window
    Table = Table
    Scroll = Scroll



