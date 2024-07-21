# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 18 2024
@name:   Window Objects
@author: Jack Kirby Cook

"""

import tkinter as tk
from tkinter import ttk

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Text", "Column", "Button", "Table", "Frame", "Window", "Application"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Text(object):
    pass


class Column(object):
    pass


class Button(tk.Button):
    def __init__(self, parent):
        super().__init__(parent)


class Table(ttk.Treeview):
    def __init__(self, parent):
        super().__init__(parent)


class Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)


class Window(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)


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

    def show(self, window):
        window = self.windows[window]
        window.tkraise()

    @property
    def windows(self): return self.__windows



