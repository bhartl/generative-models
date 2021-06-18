import inspect
from pydoc import locate
from copy import deepcopy
from inspect import getmembers, isclass

# possible useful globals for make
import numpy as np
from numpy import *
from gym.spaces import *


class Representable(object):

    def __init__(self, repr_fields=(), **kwargs):
        self.repr_fields = repr_fields

    def to_dict(self):
        module_ = inspect.getmodule(self).__name__
        class_ = self.__class__.__name__
        env_repr = dict(
            class_=class_,
            module_=module_,
        )

        for f in self.repr_fields:
            attr = getattr(self, f)
            env_repr[f] = attr.to_dict() if hasattr(attr, 'to_dict') else attr

        return env_repr

    @classmethod
    def from_dict(cls, dict_repr):
        dict_repr = deepcopy(dict_repr)
        repr_cls = dict_repr.pop('class_', cls)
        repr_module = dict_repr.pop('module_', None)

        if isinstance(repr_cls, str):
            if repr_module is not None:
                repr_cls = locate('.'.join([repr_module, repr_cls]))
            else:
                repr_cls = locate(repr_cls)

        assert isinstance(repr_cls, type), f'type of repr {repr_cls} not understood: {type(repr_cls)}'
        return repr_cls(**dict_repr)

    def __repr__(self):
        dict_repr = self.to_dict()
        cls = dict_repr.pop('class_')
        _ = dict_repr.pop('module_', None)
        kwargs = [f'{k}={repr(v)}' for k, v in dict_repr.items()]
        return f"{cls}({', '.join(kwargs)})"

    @classmethod
    def make(cls, repr_obj, **partial_local):
        if isinstance(repr_obj, cls):
            return repr_obj

        for k, v in partial_local.items():
            locals()[k] = v

        if cls.__name__ not in locals():
            locals()[cls.__name__] = cls

        if isinstance(repr_obj, dict):
            return cls.from_dict(repr_obj)

        try:
            return eval(repr_obj)
        except NameError:
            from gym import spaces
            partial_make = {name: class_ for name, class_ in getmembers(spaces, isclass)}
            assert not all(k in partial_local for k in partial_make.keys())
            return cls.make(repr_obj, **{**partial_local, **partial_make})


