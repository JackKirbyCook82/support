# -*- coding: utf-8 -*-
"""
Created on Weds Jan 10 2024
@name:   Windows Objects
@author: Jack Kirby Cook

"""

import PySimpleGUI as gui
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


tab = lambda key, name, layout: gui.Tab(name, layout, key=key)
tabgroup = lambda key, tabs: gui.TabGroup(tabs, key=key)
table = lambda key, header, layout: gui.Table(layout, header, key=key, enable_events=True)


class Window(ABC):
    def __repr__(self): return self.name
    def __init__(self, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)

    def __call__(self, *args, **kwargs):
        window = gui.Window(repr(self), [[]], resizable=True, finalize=True)
        while True:
            event, values = window.read()
            if event == gui.WINDOW_CLOSED:
                break
        window.close()

    @abstractmethod
    def process(self, *args, **kwargs): pass
    @abstractmethod
    def execute(self, *args, **kwargs): pass

    @property
    def name(self): return self.__name



