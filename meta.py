# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 2021
@name:   MetaTypes
@author: Jack Kirby Cook

"""

from abc import ABCMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SingletonMeta", "AttributeMeta", "RegistryMeta", "MappingMeta", "TreeMeta"]
__copyright__ = "Copyright 2021, Jack Kirby Cook"
__license__ = "MIT License"


class Meta(ABCMeta):
    def __init_subclass__(mcs, *args, **kwargs):
        try: return super(Meta, mcs).__init_subclass__(*args, **kwargs)
        except TypeError: return super(Meta, mcs).__init_subclass__()

    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        try: return super(Meta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        except TypeError: return super(Meta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, *args, **kwargs):
        try: super(Meta, cls).__init__(*args, **kwargs)
        except TypeError: super(Meta, cls).__init__()


class SingletonMeta(Meta):
    instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in SingletonMeta.instances.keys():
            instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
            SingletonMeta.instances[cls] = instance
        return SingletonMeta.instances[cls]


class TreeMeta(Meta):
    def __repr__(cls): return str(cls.__name__)
    def __str__(cls): return str(cls.__key__)

    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        function = lambda value: type(value) is TreeMeta or issubclass(type(value), TreeMeta)
        exclude = [key for key, value in attrs.items() if function(value)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(TreeMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, dependents=[], **kwargs):
        function = lambda value: type(value) is TreeMeta or issubclass(type(value), TreeMeta)
        assert isinstance(dependents, list)
        assert all([function(child) for child in dependents])
        super(TreeMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        primary = {str(child): child for child in attrs.values() if function(child)}
        secondary = {str(child): child for child in list(dependents)}
        cls.__dependents__ = getattr(cls, "__dependents__", {}) | dict(primary) | dict(secondary)
        cls.__key__ = kwargs.get("key", getattr(cls, "__key__", None))

    @property
    def dependents(cls): return cls.__dependents__


class RegistryMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(RegistryMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        function = lambda base: type(base) is RegistryMeta or issubclass(type(base), RegistryMeta)
        if not any([function(base) for base in bases]):
            assert "register" not in kwargs.keys()
            cls.__registry__ = dict()
            return
        register = kwargs.get("register", [])
        register = [register] if not isinstance(register, list) else register
        register = list(filter(lambda value: value is not None, register))
        for key in register: cls[key] = cls

    def __setitem__(cls, key, value): cls.registry[key] = value
    def __getitem__(cls, key): return cls.registry[key]
    def __iter__(cls): return iter(cls.registry.items())

    @property
    def registry(cls): return cls.__registry__


class AttributeMeta(Meta):
    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(AttributeMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        function = lambda base: type(base) is AttributeMeta or issubclass(type(base), AttributeMeta)
        if not any([function(base) for base in bases]) or bool(kwargs.get("root", False)):
            assert "attribute" not in kwargs.keys() and "attributes" not in kwargs.keys()
            cls.__root__ = cls
        attributes = [kwargs.get("attribute", None)] + kwargs.get("attributes", [])
        attributes = list(filter(lambda attribute: attribute is not None, attributes))
        assert all([isinstance(attribute, str) for attribute in attributes])
        for attribute in attributes: setattr(cls.root, attribute, cls)

    @property
    def root(cls): return cls.__root__


class MappingMeta(Meta):
    def __iter__(cls): return iter(cls.mapping.items())
    def __contains__(cls, key): return bool(key in cls.mapping.keys())
    def __getitem__(cls, key): return cls.mapping[key]
    def __setitem__(cls, key, value): cls.mapping[key] = value

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(MappingMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        dunder = lambda key: str(key).startswith('__') and str(key).endswith('__')
        function = lambda value: isinstance(value, (bool, str, int, float, tuple, set, list, dict))
        mapping = {key: value for key, value in attrs.items() if not dunder(key) and function(value)}
        cls.__mapping__ = mapping

    @property
    def mapping(cls): return cls.__mapping__






