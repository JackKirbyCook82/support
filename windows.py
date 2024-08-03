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
from functools import update_wrapper
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.meta import SingletonMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Application", "Stencils", "Widget", "Events"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


MouseSingleClick = StrEnum("MouseSingleClick", {"LEFT": "<Button-1>", "RIGHT": "<Button-3>"})
MouseDoubleClick = StrEnum("MouseDoubleClick", {"LEFT": "<Double-Button-1>", "RIGHT": "<Double-Button-3>"})
KeyBoardPress = StrEnum("KeyBoardPress", {"RETURN": "<Return>"})
TableVirtuals = StrEnum("TableVirtuals", {"SELECT": "<<TreeviewSelect>>"})


class Events:
    class Mouse: Single = MouseSingleClick, Double = MouseDoubleClick
    class Keyboard: Press = KeyBoardPress
    class Table: Virtual = TableVirtuals

    class Handler(object):
        def __init__(self, *arguments, **parameters):
            self.__parameters = dict(parameters)
            self.__arguments = list(arguments)
            self.__callback = None

        def __call__(self, function):
            def wrapper(event):
                element = event.widget
                function(element, element.controller, **self.parameters)
            setattr(wrapper, "handler", self)
            update_wrapper(wrapper, function)
            self.callback = wrapper
            return wrapper

        def register(self, element):
            for argument in self.arguments:
                element.bind(str(argument), self.callback)

        @property
        def parameters(self): return self.__parameters
        @property
        def arguments(self): return self.__arguments
        @property
        def callback(self): return self.__callback
        @callback.setter
        def callback(self, callback): self.__callback = callback


class Widget(ntuple("Widget", "element styling locator parser")):
    def __new__(cls, element=None, locator=(0, 0), parser=None, **parameters):
        assert isinstance(locator, tuple) and len(locator) == 2
        Styling = ntuple("Styling", list(parameters.keys()))
        styling = Styling(*list(parameters.values()))
        return super().__new__(cls, element, styling, locator, parser)

    def __call__(self, parent, identity, controller):
        parameters = dict(styling=self.styling, locator=self.locator, parser=self.parser)
        instance = self.element(parent, identity=identity, controller=controller, **parameters)
        return instance


class ActionMeta(type):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(ActionMeta, cls).__init__(name, bases, attrs)
        events = [value for value in attrs.values() if hasattr(value, "handler")]
        cls.events = getattr(cls, "events", []) | events

    def __call__(cls, *args, **kwargs):
        instance = super(ActionMeta, cls).__call__(*args, **kwargs)
        for event in cls.events:
            event.handler.register(instance)
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

    def __call__(cls, *args, controller, **kwargs):
        instance = super(ContainerMeta, cls).__call__(*args, controller=controller, **kwargs)
        elements = instance.create(cls.widgets, controller)
        assert isinstance(elements, dict)
        for identity, element in elements.items():
            instance[identity] = element
        return instance


class Element(object):
    def __init__(self, parent, *args, identity, controller, **kwargs):
        self.__mutex = multiprocessing.RLock()
        self.__controller = controller
        self.__identity = identity
        self.__parent = parent

    @property
    def controller(self): return self.__controller
    @property
    def identity(self): return self.__identity
    @property
    def parent(self): return self.__parent
    @property
    def mutex(self): return self.__mutex


class Action(Element, metaclass=ActionMeta): pass
class Container(Element, metaclass=ContainerMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements = {}

    def __setitem__(self, key, value): self.elements[key] = value
    def __getitem__(self, key): return self.elements[key]
    def __iter__(self): return iter(list(self.elements.items()))

    def create(self, widgets, controller):
        function = lambda widget, identity: widget(self, identity, controller)
        elements = {identity: function(widget, identity) for identity, widget in widgets.items()}
        return elements


class Label(Element, tk.Label):
    def __init__(self, parent, *args, styling, locator, **kwargs):
        tk.Label.__init__(self, parent, text=styling.text, font=styling.font, justify=styling.justify)
        Element.__init__(self, parent, *args, **kwargs)
        self.grid(row=locator.row, column=locator.column, padx=5, pady=5)


class Variable(Element, tk.Label):
    def __init__(self, parent, *args, styling, locator, **kwargs):
        variable = tk.StringVar()
        tk.Label.__init__(self, parent, text=variable, font=styling.font, justify=styling.justify)
        Element.__init__(self, parent, *args, **kwargs)
        self.grid(row=locator.row, column=locator.column, padx=5, pady=5)
        self.variable = variable

    @property
    def text(self): return self.variable.get()
    @text.setter
    def text(self, text): self.variable.set(text)


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
    def __init__(self, parent, *args, styling, **kwargs):
        tk.Frame.__init__(self, parent, borderwidth=5)
        Container.__init__(self, parent, *args, **kwargs)
        self.grid(row=styling.locator.row, column=styling.locator.column, padx=10, pady=10)


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


class Table(Container, ttk.Treeview):
    def __init__(self, parent, *args, styling, **kwargs):
        ttk.Treeview.__init__(self, parent, show="headings")
        Container.__init__(self, parent, *args, **kwargs)
        self.grid(row=styling.locator.row, column=styling.locator.column)
        self.parsers = ODict()

    def create(self, widgets):
        self["columns"] = list(widgets.keys())
        for identity, widget in widgets.items():
            self.column(identity, width=int(widget.styling.width), anchor=tk.CENTER)
            self.heading(identity, text=str(widget.styling.text), anchor=tk.CENTER)
            self.parsers[identity] = widget.parser
        return {}

    def erase(self, *args, **kwargs):
        for index in self.get_children():
            self.delete(index)

    def draw(self, *args, dataframe, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        for index, series in dataframe.iterrows():
            row = [function(series) for key, function in self.parsers.items()]
            self.insert("", tk.END, iid=index, values=tuple(row))


class ApplicationMeta(SingletonMeta):
    def __init__(cls, *args, **kwargs):
        super(ApplicationMeta, cls).__init__(*args, **kwargs)
        cls.heading = kwargs.get("heading", getattr(cls, "heading", None))
        cls.window = kwargs.get("window", getattr(cls, "window", None))

    def __call__(cls, *args, **kwargs):
        instance = super(ApplicationMeta, cls).__call__(*args, **kwargs)
        instance.title(cls.heading)
        window = cls.window(instance.root, *args, controller=instance.controller, identifier=None, **kwargs)
        instance.create(window, *args, **kwargs)
        return instance


class Application(tk.Tk, metaclass=ApplicationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, *args, wait, **kwargs):
        super().__init__()
        root = tk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        self.__controller = self
        self.__models = dict()
        self.__views = dict()
        self.__wait = int(wait)
        self.__root = root

    def __call__(self, *args, **kwargs):
        self.execute(*args, **kwargs)
        self.mainloop()

    def create(self, window, *args, **kwargs): pass
    def execute(self, *args, **kwargs): pass

    @property
    def controller(self): return self.__controller
    @property
    def models(self): return self.__models
    @property
    def views(self): return self.__views
    @property
    def root(self): return self.__root
    @property
    def wait(self): return self.__wait


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



