from gym.spaces import *
from examples.world_model.util.representable import Representable
from inspect import getmembers, isclass
import numpy as np
from collections import Iterable
from typing import Union


class Agent(Representable):

    REPR_FIELDS = ('action_space', 'default_action')

    def __init__(self,
                 action_space: Space,
                 default_action: Union[np.ndarray, Iterable, int, float] = None
                 ):

        Representable.__init__(self, repr_fields=self.REPR_FIELDS)
        self.action_space = action_space
        self.default_action = np.asarray(default_action)

    @property
    def action_space(self) -> (Space, None):
        return self._action_space

    @action_space.setter
    def action_space(self, value: (Space, str, None)):
        if isinstance(value, str):
            value = eval(value)

        assert isinstance(value, Space) or value is None, f"observation_space type {type(value)} not supported"
        self._action_space = value

    @classmethod
    def make(cls, repr_obj, **partial_local):
        try:
            return super(Agent, cls).make(repr_obj, **partial_local)
        except NameError:
            # add gym spaces to locals() of cls.make ...
            from gym import spaces
            partial_make = {name: class_ for name, class_ in getmembers(spaces, isclass)}
            return super(Agent, cls).make(repr_obj, **{**partial_make, **partial_local})

    def get_action(self, observation: Space) -> object:
        pass

    def get_default_action(self) -> object:
        return self.default_action

