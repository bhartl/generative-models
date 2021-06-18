from examples.world_model.agent import Agent
from gym.spaces import Space


class RandomWalker(Agent):

    REPR_FIELDS = ('randomize_interval', *Agent.REPR_FIELDS)

    def __init__(self, randomize_interval: int = 1, **kwargs):
        super(RandomWalker, self).__init__(**kwargs)
        self.randomize_interval = randomize_interval
        self._random_action = self.default_action
        self._action_counter = 0

    def get_action(self, observation: Space = None) -> object:
        if self._action_counter > 0 and not self._action_counter % self.randomize_interval:
            self._random_action = self.action_space.sample()
        self._action_counter += 1
        return self._random_action

    def get_default_action(self) -> object:
        self._action_counter = 0
        return self.default_action
