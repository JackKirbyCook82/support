# -*- coding: utf-8 -*-
"""
Created on Tues Oct 5 2021
@name:   Observer Objects
@author: Jack Kirby Cook

"""

from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Event", "Subscriber", "Publisher"]
__copyright__ = "Copyright 2020, Jack Kirby Cook"
__license__ = ""


class Event(ABC):
    def __init__(self, publisher, *args, **kwargs):
        self.__name = kwargs.get("name", self.__class__.__name__)
        self.__publisher = publisher

    def __repr__(self):
        return "{}|{}".format(self.name, repr(self.publisher))

    @property
    def publisher(self): return self.__publisher
    @property
    def name(self): return self.__name


class Observer(ABC):
    def __init_subclass__(cls, *args, **kwargs):
        events = set(getattr(cls, "__events__", []))
        update = set(kwargs.get("events", []))
        assert not any([not issubclass(key, Event) for key in update])
        cls.__events__ = events | update

    def __init__(self, *args, **kwargs):
        self.__name = str(kwargs.get("name", self.__class__.__name__))
        self.__events = set(self.__class__.__events__)

    def __repr__(self):
        return "{}".format(self.name)

    @property
    def events(self): return self.__events
    @property
    def name(self): return self.__name


class Publisher(Observer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__subscribers = set()

    def __contains__(self, subscriber): return subscriber in self.subscribers
    def __iter__(self): return (subscriber for subscriber in self.subscribers)

    def register(self, *subscribers):
        assert all([isinstance(subscriber, Subscriber) for subscriber in subscribers])
        for subscriber in subscribers:
            self.subscribers.add(subscriber)
            subscriber.publishers.add(self)

    def unregister(self, *subscribers):
        assert all([isinstance(subscriber, Subscriber) for subscriber in subscribers])
        for subscriber in subscribers:
            self.subscribers.discard(subscriber)
            subscriber.publishers.discard(self)

    def publish(self, event, *args, **kwargs):
        assert issubclass(event, Event)
        if event not in self.events:
            return
        instance = event(self, *args, **kwargs)
        for subscriber in self.subscribers:
            subscriber.observe(instance, *args, **kwargs)

    @property
    def subscribers(self): return self.__subscribers


class Subscriber(Observer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__publishers = set()

    def __contains__(self, publisher): return publisher in self.publishers
    def __iter__(self): return (publisher for publisher in self.publishers)

    def register(self, *publishers):
        assert all([isinstance(publisher, Publisher) for publisher in publishers])
        for publisher in publishers:
            self.publishers.add(publisher)
            publisher.subscribers.add(self)

    def unregister(self, *publishers):
        assert all([isinstance(publisher, Publisher) for publisher in publishers])
        for publisher in publishers:
            self.publishers.discard(publisher)
            publisher.subscribers.discard(self)

    def observe(self, event, *args, **kwargs):
        assert isinstance(event, Event)
        if type(event) not in self.events:
            return
        self.execute(event, *args, **kwargs)

    @abstractmethod
    def execute(self, event, *args, **kwargs):
        pass

    @property
    def publishers(self): return self.__publishers




