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


class Containable(object): pass
class Widget(ntuple("Widget", "element styling"), Containable):
    def __new__(cls, element, *args, locator=(0, 0), **parameters):
        assert isinstance(locator, tuple) and len(locator) == 2
        Styling = ntuple("Styling", ["locator"] + list(parameters.keys()))
        Locator = ntuple("Locator", ["row", "column"])
        styling = Styling(Locator(*locator), *list(parameters.values()))
        return super().__new__(cls, element, styling)

class Column(ntuple("Column", "text width function"), Containable):
    def __new__(cls, *args, text, width, parser, locator, **kwargs):
        assert callable(parser) and callable(locator)
        function = lambda series: parser(locator(series))
        return super().__new__(cls, text, width, function)


class ContainerMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, value in attrs.items() if isinstance(value, Containable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(ContainerMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(ContainerMeta, cls).__init__(name, bases, attrs)
        contents = ODict([(key, value) for key, value in attrs.items() if isinstance(value, Containable)])
        cls.contents = getattr(cls, "contents", ODict()) | contents

    def __call__(cls, *args, **kwargs):
        instance = super(ContainerMeta, cls).__call__(*args, **kwargs)
        contents = instance.create(cls.contents, *args, **kwargs)
        assert isinstance(contents, dict)
        return instance


class Element(object):
    def __init__(self, *args, **kwargs): pass

class Container(Element, metaclass=ContainerMeta):
    def create(self, widgets, *args, **kwargs):
        function = lambda element, styling: element(self, styling, *args, **kwargs)
        contents = {key: function(widget.element, widget.styling) for key, widget in widgets.tiems()}
        return contents


class Label(Element, tk.Label):
    def __init__(self, parent, styling, *args, **kwargs):
        tk.Label.__init__(self, parent, text=styling.text, font=styling.font, justify=styling.justify)
        Element.__init__(self, *args, **kwargs)
        self.grid(row=styling.locator.row, column=styling.locator.column, padx=5, pady=5)


class Button(Element, tk.Button):
    def __init__(self, parent, styling, *args, **kwargs):
        tk.Button.__init__(self, parent, text=styling.text, font=styling.font, justify=styling.justify)
        Element.__init__(self, *args, **kwargs)
        self.grid(row=styling.locator.row, column=styling.locator.column, padx=10, pady=5)


class Scroll(Element, tk.Scrollbar):
    def __init__(self, parent, styling, *args, **kwargs):
        tk.Scrollbar.__init__(self, parent, orient=styling.orientation)
        Element.__init__(self, *args, **kwargs)
        self.grid(row=styling.locator.row, column=styling.locator.column)


class Window(Container, tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent)
        Element.__init__(self, *args, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)


class Frame(Container, tk.Frame):
    def __init__(self, parent, styling, *args, **kwargs):
        tk.Frame.__init__(self, parent, borderwidth=5)
        Element.__init__(self, *args, **kwargs)
        self.grid(row=styling.locator.row, column=styling.locator.column, padx=10, pady=10)


class Notebook(Container, ttk.Notebook):
    def __init__(self, parent, *args, **kwargs):
        ttk.Notebook.__init__(self, parent)
        Element.__init__(self, *args, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)

    def create(self, widgets, *args, **kwargs):
        contents = Container.create(self, widgets, *args, **kwargs)
        for key, content in contents.items():
            self.add(content, text=str(key))
        return contents


class Table(Container, ttk.Treeview):
    def __init__(self, parent, styling, *args, **kwargs):
        ttk.Treeview.__init__(self, parent, show="headings")
        Element.__init__(self, *args, **kwargs)
        self.grid(row=styling.locator.row, column=styling.locator.column)
        self.functions = ODict()

    def create(self, columns, *args, **kwargs):
        self["columns"] = list(columns.keys())
        for key, column in columns.items():
            self.column(key, width=int(column.width), anchor=tk.CENTER)
            self.heading(key, text=str(column.text), anchor=tk.CENTER)
            self.functions[key] = column.function
        return {}

    def draw(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        for row in self.get_children():
            self.delete(row)
        for index, series in dataframe.iterrows():
            row = [function(series) for key, function in self.functions.items()]
            self.insert("", tk.END, iid=index, values=tuple(row))


class ApplicationMeta(SingletonMeta):
    def __init__(cls, *args, **kwargs):
        super(ApplicationMeta, cls).__init__(*args, **kwargs)
        cls.heading = kwargs.get("heading", getattr(cls, "heading", None))
        cls.window = kwargs.get("window", getattr(cls, "window", None))

    def __call__(cls, *args, **kwargs):
        instance = super(ApplicationMeta, cls).__call__(*args, **kwargs)
        instance.title(cls.heading)
        cls.window(instance.parent, *args, **kwargs)
        return instance


class Application(tk.Tk, metaclass=ApplicationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, **kwargs):
        super().__init__()
        parent = tk.Frame(self)
        parent.pack(fill=tk.BOTH, expand=True)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self.parent = parent

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



