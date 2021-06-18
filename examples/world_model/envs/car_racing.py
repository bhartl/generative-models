"""

adapted from https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/env.py
"""

import numpy as np
from skimage.transform import resize
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing
from examples.world_model.environment import Environment

SCREEN_X = 64
SCREEN_Y = 64


KEY_MAP = {
    "LEFT": (0, -1, 0.),
    "RIGHT": (0, +1, 0.),
    "UP": (1, +1., 0.),
    "DOWN": (2, +0.8, 0.),
}
""" default action mapping: {`pyglet key`: (action_idx, on press value, on release value)-tuple} """

RESTART_KEY = 0xFF0D


class CarRacingWrapper(CarRacing, Environment):

    REPR_FIELDS = ('verbose', 'full_episode', 'steps')

    def __init__(self, full_episode=False, verbose=1, steps=0):
        Environment.__init__(self, observation_space=None)
        CarRacing.__init__(self, verbose=verbose)
        self.full_episode = full_episode
        self.observation_space = Box(low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3))  # , dtype=np.uint8
        self.steps = 0

    def step(self, action):
        try:
            obs, reward, done, _ = super(CarRacingWrapper, self).step(action)
            if self.full_episode:
                return self._process_frame(obs), reward, False, {}
            return self._process_frame(obs), reward, done, {}
        finally:
            self.steps += 1

    @staticmethod
    def _process_frame(frame):
        obs = frame[0:84, :, :].astype(float) / 255.0
        obs = resize(obs, (64, 64))
        obs = ((1.0 - obs) * 255).round().astype(np.uint8)
        return obs

    @staticmethod
    def make_env(env_name, seed=-1, render_mode=False, full_episode=False):
        env = CarRacingWrapper(full_episode=full_episode)

        if seed >= 0:
            env.seed(seed)

        return env

    @staticmethod
    def details():
        env = CarRacingWrapper(full_episode=False)

        print("environment details:")
        print("- env.action_space", env.action_space)
        print("  high, low", env.action_space.high, env.action_space.low)
        print("- env.observation_space", env.observation_space)
        print("  high, low", env.observation_space.high, env.observation_space.low)
        return

    def reset(self, action=None):
        self.steps = 0
        return CarRacing.reset(self)


def gym_demo(record_video=False, environment='CarRacing'):
    """ CarRacing demo, keyboard controlled

    adapted from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
    """

    from examples.world_model.controller.key_controller import KeyController
    controller = KeyController(restart_key=RESTART_KEY, **KEY_MAP)

    env = CarRacing() if environment == 'CarRacing' else CarRacingWrapper()
    env.render()
    env.viewer.window.on_key_press = controller.press
    env.viewer.window.on_key_release = controller.release

    try:
        if record_video:
            from gym.wrappers.monitor import Monitor
            env = Monitor(env, "/tmp/video-test", force=True)

        is_open = True
        while is_open:
            env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                s, r, done, info = env.step(controller.a)
                total_reward += r
                if steps % 200 == 0 or done:
                    print("\naction " + str(["{:+0.2f}".format(x) for x in controller.a]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))

                steps += 1
                is_open = env.render()
                if done or restart or is_open is False:
                    break

    except KeyboardInterrupt:
        print('Stopped via KeyboardInterrupt')

    finally:
        env.close()


if __name__ == "__main__":
    import argh
    argh.dispatch_commands([
        gym_demo,
    ])
