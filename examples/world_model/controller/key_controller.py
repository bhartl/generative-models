from pyglet.window import key
from numpy import zeros, unique


DEFAULT_KEY_MAP = {
    "LEFT": (0, -1, 0.),
    "RIGHT": (0, +1, 0.),
    "UP": (1, +1., 0.),
    "DOWN": (2, +0.8, 0.),
}
""" default action mapping: {`pyglet key`: (action_idx, on press value, on release value)-tuple} """

RESTART_KEY = 0xFF0D


class KeyController(object):
    """ KeyController instance with `press` and `release` methods,
        which update an action `a` and a `restart` property """

    def __init__(self, restart_key=RESTART_KEY, **action_map):
        """ Constructs a KeyController instance

        :param restart_key: string or pyglet key for restarting the environment
        :param action_map: key to action mapping of the form
                           {`pyglet key`: (action_idx, on press value, on release value)-tuple}
        """
        if action_map == {}:
            action_map = DEFAULT_KEY_MAP

        self.restart = False
        self.restart_key = (getattr(key, restart_key) if isinstance(restart_key, str) else restart_key)

        self.action_map = {(getattr(key, k) if isinstance(k, str) else k): v for k, v in action_map.items()}
        self.a = zeros(self.a_dim)
        [self.release(k, k) for k in self.action_map.keys()]

    @property
    def a_dim(self) -> int:
        """ dimension property of the action space """
        return len(unique([idx for idx, press_val, release_val in self.action_map.values()]))

    def press(self, k, mod):
        """ on press method, the k's action value is updated with the corresponding press value """
        if k == self.restart_key:  # on enter
            self.restart = True
            return

        idx, press_val, __ = self.action_map[k]
        self.a[idx] = press_val

    def release(self, k, mod):
        """ on release method, the k's action value is updated with the corresponding release value """
        if k == self.restart_key:  # on enter
            return

        idx, __, release_val = self.action_map[k]
        self.a[idx] = release_val
