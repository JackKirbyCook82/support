# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 2021
@name:   MetaTypes
@author: Jack Kirby Cook
@file:   support/meta.py

"""

from abc import ABCMeta
from itertools import product as iterprod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["CounterMeta", "SingletonMeta", "AttributeMeta", "RegistryMeta", "NamespaceMeta", "TreeMeta", "CollectionMeta", "Meta"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"

from support.mixins import Mixin


class MetaError(Exception): pass
class Meta(ABCMeta):
    def __init_subclass__(mcs, **kwargs):
        super().__init_subclass__(**kwargs)
        mro = mcs.__mro__
        assert isinstance(mro, tuple)
        index = mro.index(Mixin)
        remaining = mro[index + 1:]
        if remaining != (ABCMeta, type, object): raise MetaError()

    def __new__(mcs, name, bases, attrs, **kwargs):
        return super().__new__(mcs, name, bases, attrs, **kwargs)

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)


class CounterMeta(Meta):
    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs, **kwargs)
        cls.__current__ = 1

    @property
    def counter(cls):
        current = cls.current
        cls.current = current + 1
        return current

    @property
    def current(cls): return cls.__current__
    @current.setter
    def current(cls, current): cls.__current__ = current


class SingletonMeta(Meta):
    instances = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in type(cls).instances.keys():
            instance = super().__call__(*args, **kwargs)
            type(cls).instances[cls] = instance
        return type(cls).instances[cls]


class TreeMeta(Meta):
    def __repr__(cls): return str(cls.__name__)
    def __str__(cls): return str(cls.__key__)

    def __new__(mcs, name, bases, attrs, **kwargs):
        function = lambda value: type(value) is TreeMeta or issubclass(type(value), TreeMeta)
        exclude = [key for key, value in attrs.items() if function(value)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super().__new__(mcs, name, bases, attrs, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, **kwargs):
        function = lambda value: type(value) is TreeMeta or issubclass(type(value), TreeMeta)
        super().__init__(name, bases, attrs, **kwargs)
        primary = {str(child): child for child in attrs.values() if function(child)}
        secondary = {str(child): child for child in kwargs.get("dependents", [])}
        cls.__dependents__ = getattr(cls, "__dependents__", {}) | dict(primary) | dict(secondary)
        cls.__key__ = kwargs.get("key", getattr(cls, "__key__", None))

    @property
    def dependents(cls): return cls.__dependents__
    @property
    def key(cls): return cls.__key__


class CollectionMeta(Meta):
    def __iter__(cls): return iter(cls.collection.items())
    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs, **kwargs)
        collection = {key: value for key, value in attrs.items() if not key.startswith("__") and not callable(value)}
        cls.__collection__ = getattr(cls, "__collection__", {}) | collection

    @property
    def collection(cls): return cls.__collection__


class NamespaceError(Exception): pass
class NamespaceMeta(Meta):
    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs, **kwargs)
        existing = getattr(cls, "namespace", {})
        updated = kwargs.get("namespace", {})
        overlap = existing.keys() & updated.keys()
        if overlap: raise NamespaceError()
        namespace = existing | updated
        for key in list(namespace.keys()):
            namespace[key] = kwargs.get(key, namespace[key])
        cls.__namespace__ = namespace

    def __call__(cls, *args, **kwargs):
        parameters = {key: kwargs.pop(key, value) for key, value in cls.namespace.items()}
        instance = super().__call__(*args, **parameters, **kwargs)
        return instance

    @property
    def namespace(cls): return cls.__namespace__


class AttributeMeta(Meta):
    def __new__(mcs, name, bases, attrs, **kwargs):
        assert RegistryMeta not in bases
        return super().__new__(mcs, name, bases, attrs, **kwargs)

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs, **kwargs)
        if not any([isinstance(base, AttributeMeta) for base in bases]): cls.root = cls
        attributes = [kwargs.get("attribute", None)] + kwargs.get("attributes", [])
        attributes = list(filter(lambda attribute: attribute is not None, attributes))
        assert all([isinstance(attribute, str) for attribute in attributes])
        for attribute in attributes: setattr(cls.root, attribute, cls)
        if bool(kwargs.get("root", False)): cls.root = cls

    @property
    def root(cls): return cls.__root__
    @root.setter
    def root(cls, root): cls.__root__ = root


class RegistryMeta(Meta):
    registries = dict()

    def __new__(mcs, name, bases, attrs, **kwargs):
        assert AttributeMeta not in bases
        return super().__new__(mcs, name, bases, attrs, **kwargs)

    def __init__(cls, name, bases, attrs, register=None, **kwargs):
        super().__init__(name, bases, attrs, **kwargs)
        register = [register] if not isinstance(register, list) else register
        register = list(filter(lambda value: value is not None, register))
        if not any([isinstance(base, RegistryMeta) for base in bases]):
            root = f"{cls.__module__}.{cls.__qualname__}"
            registry = RegistryMeta.registries.get(root, {})
            RegistryMeta.registries[root] = registry
            for key in register: registry[key] = cls
            cls.root = root
            return
        if bool(kwargs.get("root", False)):
            root = f"{cls.__module__}.{cls.__qualname__}"
            registry = RegistryMeta.registries.get(root, {})
            RegistryMeta.registries[root] = registry
            cls.root = root
        parents = [base for base in bases if isinstance(base, RegistryMeta)]
        for (base, key) in iterprod(parents, register):
            registry = RegistryMeta.registries[base.root]
            registry[key] = cls

    def __getitem__(cls, key): return RegistryMeta.registries[cls.root][key]
    def __setitem__(cls, key, value): RegistryMeta.registries[cls.root][key] = value
    def __iter__(cls): return iter(RegistryMeta.registries[cls.root].items())

    @property
    def root(cls): return cls.__root__
    @root.setter
    def root(cls, root): cls.__root__ = root


