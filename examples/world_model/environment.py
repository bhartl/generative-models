import argh
from gym import Env
import inspect
from gym import spaces
from examples.world_model.util.representable import Representable


class Environment(Env, Representable):
    """

    - Every environment comes with an `action_space` and an `observation_space`.
      These attributes are of type Space, and they describe the format of valid actions and observations

    from [1]: https://gym.openai.com/docs/
    """

    REPR_FIELDS = ('observation_space', 'reward', 'steps')

    def __init__(self,
                 observation_space: (spaces.Space, str),
                 reward: float = 0.,
                 steps: int = 0,
                 **kwargs
                 ):

        Representable.__init__(self, repr_fields=self.REPR_FIELDS)
        Env.__init__(self)

        self._observation_space = None
        self.observation_space = observation_space

        self.reward = reward
        self.steps = steps

    @property
    def observation_space(self) -> (spaces.Space, None):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: (spaces.Space, str, None)):
        if isinstance(value, str):
            value = eval(value)

        assert isinstance(value, spaces.Space) or value is None, f"observation_space type {type(value)} not supported"
        self._observation_space = value

    def step(self, action: object) -> [object, float, bool, dict]:
        """ The implementation of the classic “agent-environment loop”.

        Each time-step, the agent chooses an action, and the environment returns an observation and a reward.

        The environment’s step function returns four values:

        - observation (object): an environment-specific object representing your observation of the environment.
        - reward (float): amount of reward achieved by the previous action.
        - done (boolean): whether it’s time to reset the environment again.
        - info (dict): diagnostic information useful for debugging.
        """
        return None, self.reward, True, {}

    def reset(self, action: object = None) -> object:
        """
        :param action: (object) default action for initial step (defaults to None)
        :returns: initial observation
        """
        self.reward = 0.
        self.steps = 0

        return self.step(action)[0]

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


def gym_registry():
    from gym import envs
    print(envs.registry.all())


def test_repr():
    print('generate `Environment` with discrete observation space: ', end='')
    d = spaces.Discrete(5)
    e = Environment(observation_space=d)
    print(repr(e))

    print('check identity return of `make_env`', end='')
    e_id = Environment.make(e)
    assert e_id is e
    print('\rcheck identity return of `make_env`: success')

    print('check `to_dict` and reload through `from_dict` and `make_env`', end='')
    e_dict = e.to_dict()
    e_redict = Environment.make(e_dict)

    for k, v in e_dict.items():
        if k == 'class_':
            assert e.__class__.__name__ == v
            assert e_redict.__class__.__name__ == v

        elif k == 'module_':
            assert inspect.getmodule(e).__name__ == v
            assert inspect.getmodule(e_redict).__name__ == v

        else:
            assert getattr(e, k) == v
            assert getattr(e_redict, k) == v

    print('\rcheck `to_dict` and reload through `from_dict` and `make_env`: success')

    print('check `repr` and reload through `make_env`', end='')
    e_repr = repr(e)
    e_reval = Environment.make(e_repr)
    for k, v in e_reval.to_dict().items():
        assert e_dict[k] == v
    print('\rcheck `repr` and reload through `make_env`: success')


if __name__ == '__main__':
    argh.dispatch_commands([
        gym_registry,
        test_repr,
    ])

