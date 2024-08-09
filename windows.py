# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 18 2024
@name:   Window Objects
@author: Jack Kirby Cook

"""

import multiprocessing
import pandas as pd
import tkinter as tk
from enum import StrEnum
from tkinter import ttk
from abc import ABC, abstractmethod
from functools import update_wrapper
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["MVC", "Application", "Stencils", "Widget", "Events"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


Status = StrEnum("Status", {"ACTIVE": "<Activate>", "INACTIVE": "<Deactivate>", "DESTROY": "<Destroy>", "VISIBLE": "<Visibility>"})
MouseClickSingle = StrEnum("MouseSingleClick", {"LEFT": "<Button-1>", "MIDDLE": "<Button-2>", "RIGHT": "<Button-3>"})
MouseClickDouble = StrEnum("MouseSingleClick", {"LEFT": "<Double-Button-1>", "MIDDLE": "<Double-Button-2>", "RIGHT": "<Double-Button-3>"})
MouseMovement = StrEnum("MouseMovement", {"ENTER": "<Enter>", "EXIT": "<Leave>"})
MouseWheel = StrEnum("MouseWheel", {"SCROLL": "<MouseWheel>"})
Keyboard = StrEnum("Keyboard", {"RETURN": "<Return>", "PRESS": "<Key>"})
Table = StrEnum("Table", {"SELECT": "<<TreeviewSelect>>"})


class Widget(ntuple("Widget", "element styling locator parser")):
    def __new__(cls, element=None, locator=(0, 0), parser=None, **parameters):
        assert isinstance(locator, tuple) and len(locator) == 2
        Styling = ntuple("Styling", list(parameters.keys()))
        styling = Styling(*list(parameters.values()))
        return super().__new__(cls, element, styling, locator, parser)

    def __call__(self, parent, identity):
        parameters = dict(styling=self.styling, locator=self.locator, parser=self.parser)
        instance = self.element(parent, identity=identity, **parameters)
        return instance


class ContainerMeta(type):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, value in attrs.items() if isinstance(value, Widget)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(ContainerMeta, mcs).__new__(mcs, name, bases, attrs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(ContainerMeta, cls).__init__(name, bases, attrs)
        widgets = ODict([(key, value) for key, value in attrs.items() if isinstance(value, type(cls).element)])
        cls.widgets = getattr(cls, "widgets", ODict()) | widgets

    def __call__(cls, *args, **kwargs):
        instance = super(ContainerMeta, cls).__call__(*args, **kwargs)
        elements = instance.create(cls.widgets)
        assert isinstance(elements, dict)
        for identity, element in elements.items():
            assert not hasattr(instance, identity)
            setattr(instance, identity)
        return instance


class Element(object):
    def __init__(self, parent, *args, identity, **kwargs):
        self.__mutex = multiprocessing.RLock()
        self.__identity = identity
        self.__parent = parent

    @property
    def identity(self): return self.__identity
    @property
    def parent(self): return self.__parent
    @property
    def mutex(self): return self.__mutex


class Container(Element, metaclass=ContainerMeta):
    def __iter__(self): return iter(list(self.elements.items()))
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements = {}

    def create(self, widgets):
        function = lambda widget, identity: widget(self, identity)
        elements = {identity: function(widget, identity) for identity, widget in widgets.items()}
        return elements


class Label(Element, tk.Label):
    def __init__(self, parent, *args, styling, locator, **kwargs):
        tk.Label.__init__(self, parent, text=styling.text, font=styling.font, justify=styling.justify)
        Element.__init__(self, parent, *args, **kwargs)
        self.grid(row=locator.row, column=locator.column, padx=5, pady=5)


class Variable(Element, tk.Label):
    def __init__(self, parent, *args, styling, locator, parser, **kwargs):
        variable = tk.StringVar()
        tk.Label.__init__(self, parent, text=variable, font=styling.font, justify=styling.justify)
        Element.__init__(self, parent, *args, **kwargs)
        self.grid(row=locator.row, column=locator.column, padx=5, pady=5)
        self.variable = variable
        self.parser = parser

    @property
    def value(self): return self.variable.get()
    @value.setter
    def value(self, value):
        string = self.parser(value) if value is not None else ""
        self.variable.set(string)


class Button(Element, tk.Button):
    def __init__(self, parent, *args, styling, locator, **kwargs):
        tk.Button.__init__(self, parent, text=styling.text, font=styling.font, justify=styling.justify)
        Element.__init__(self, parent, *args, **kwargs)
        self.grid(row=locator.row, column=locator.column, padx=10, pady=5)

    @property
    def state(self): return bool(self["state"] is tk.NORMAL)
    @state.setter
    def state(self, state):
        assert isinstance(state, bool)
        state = tk.NORMAL if bool(state) else tk.DISABLED
        self["state"] = state


class Scroll(Element, tk.Scrollbar):
    def __init__(self, parent, *args, styling, locator, **kwargs):
        tk.Scrollbar.__init__(self, parent, orient=styling.orientation)
        Element.__init__(self, parent, *args, **kwargs)
        self.grid(row=locator.row, column=locator.column)


class Window(Container, tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent)
        Container.__init__(self, parent, *args, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)


class Frame(Container, tk.Frame):
    def __init__(self, parent, *args, locator, **kwargs):
        tk.Frame.__init__(self, parent, borderwidth=5)
        Container.__init__(self, parent, *args, **kwargs)
        self.grid(row=locator.row, column=locator.column, padx=10, pady=10)


class Notebook(Container, ttk.Notebook):
    def __init__(self, parent, *args, **kwargs):
        ttk.Notebook.__init__(self, parent)
        Container.__init__(self, parent, *args, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)

    def create(self, widgets):
        elements = Container.create(self, widgets)
        for identity, element in elements.items():
            self.add(element, text=str(identity))
        return elements


class Table(Element, ttk.Treeview):
    def __init__(self, parent, *args, styling, locator, **kwargs):
        ttk.Treeview.__init__(self, parent, show="headings")
        Container.__init__(self, parent, *args, **kwargs)
        if getattr(styling, "vertical", False):
            vertical = parent["vertical"]
            vertical.configure(command=self.yview)
            self.configure(yscrollcommand=vertical.set)
        if getattr(styling, "horizontal", False):
            horizontal = parent["horizontal"]
            horizontal.configure(command=self.xview)
            self.configure(xscrollcommand=horizontal.set)
        self.grid(row=locator.row, column=locator.column)
        self.parsers = ODict()

    def create(self, widgets):
        self["columns"] = list(widgets.keys())
        for identity, widget in widgets.items():
            self.column(identity, width=int(widget.styling.width), anchor=tk.CENTER)
            self.heading(identity, text=str(widget.styling.text), anchor=tk.CENTER)
            self.parsers[identity] = widget.parser
        return {}

    def erase(self):
        for index in self.get_children():
            self.delete(index)

    def draw(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        for index, series in dataframe.iterrows():
            row = [function(series) for key, function in self.parsers.items()]
            self.insert("", tk.END, iid=index, values=tuple(row))

    @property
    def selected(self): return super().selection()
    @property
    def columns(self): return list(self["columns"])
    @property
    def index(self): return list(self.get_children())


class ApplicationMeta(SingletonMeta):
    def __init__(cls, *args, **kwargs):
        super(ApplicationMeta, cls).__init__(*args, **kwargs)
        cls.heading = kwargs.get("heading", getattr(cls, "heading", None))
        cls.window = kwargs.get("window", getattr(cls, "window", None))

    def __call__(cls, *args, **kwargs):
        instance = super(ApplicationMeta, cls).__call__(*args, **kwargs)
        instance.title(cls.heading)
        cls.window(instance.root, *args, identity=None, **kwargs)
        return instance


class Application(tk.Tk, metaclass=ApplicationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, **kwargs):
        super().__init__()
        root = tk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        self.__controllers = dict()
        self.__root = root

    def __setitem__(self, key, value): self.controllers[key] = value
    def __getitem__(self, key): return self.controllers[key]
    def __call__(self, *args, **kwargs):
        self.mainloop()
        for controller in self.controllers.values():
            controller(*args, **kwargs)

    @property
    def controllers(self): return self.__controllers
    @property
    def root(self): return self.__root


class Model(ABC):
    def __init__(self, *args, **kwargs):
        self.__controller = None

    @property
    def controller(self): return self.__controller
    @controller.setter
    def controller(self, controller): self.__controller = controller


class View(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__controller = None

    @property
    def controller(self): return self.__controller
    @controller.setter
    def controller(self, controller): self.__controller = controller


class Controller(ABC):
    def __init_subclass__(cls, *args, model, view, **kwargs):
        cls.Model = Model
        cls.View = View

    def __new__(cls, model, view, *args, **kwargs):
        assert isinstance(model, cls.Model)
        assert isinstance(view, cls.View)
        instance = super().__new__(cls)
        model.controller = instance
        view.controller = instance
        return instance

    def __init__(self, model, view, *args, wait, **kwargs):
        self.__wait = int(wait)
        self.__model = model
        self.__view = view

    def __call__(self, *args, **kwargs):
        self.execute(*args, **kwargs)
        callback = lambda: self(*args, **kwargs)
        self.after(self.wait, callback)

    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def model(self): return self.__model
    @property
    def view(self): return self.__view
    @property
    def wait(self): return self.__wait


class Events:
    Keyboard = Keyboard
    Status = Status
    Table = Table

    class Mouse:
        class Click: Single = MouseClickSingle, Double = MouseClickDouble
        Movement = MouseMovement
        Wheel = MouseWheel

    class Handler(object):
        def __init__(self, *events):
            self.__events = list(events)
            self.__callback = None

        def __call__(self, function):
            def wrapper(event):
                element = event.widget
                controller = self.controller(element)
                function(element, controller)

            setattr(wrapper, "handler", self)
            update_wrapper(wrapper, function)
            self.callback = wrapper
            return wrapper

        def register(self, element):
            for event in self.events:
                element.bind(str(event), self.callback)

        def controller(self, element):
            if isinstance(element, View):
                return element.controller
            return self.controller(element.parent)

        @property
        def events(self): return self.__events
        @property
        def callback(self): return self.__callback
        @callback.setter
        def callback(self, callback): self.__callback = callback


class MVC:
    Controller = Controller
    Model = Model
    View = View

class Stencils:
    Window = Window
    Notebook = Notebook
    Variable = Variable
    Label = Label
    Button = Button
    Frame = Frame
    Window = Window
    Table = Table
    Scroll = Scroll



