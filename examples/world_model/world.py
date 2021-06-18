from examples.world_model.environment import Environment
from examples.world_model.agent import Agent
from examples.world_model.util.representable import Representable
from examples.world_model.util.logger import Logger
from time import time


DEFAULT_LOG_FIELDS = ()  # ('episode', 'reward', 'done')


class World(Representable, Logger):
    """ The World class represents the stage where agents can interact with their environments """

    REPR_FIELDS = ('env', 'agent', 'n_episodes', 'verbose', 'render', 'max_steps', 'load_steps', 'log_fields', 'log_foos')

    def __init__(self,
                 env: (Environment, dict),
                 agent: (Agent, dict),
                 n_episodes: int = 10,
                 max_steps: (int, None) = None,
                 load_steps: int = 0,
                 verbose: bool = False,
                 render: bool = True,
                 log_fields: (tuple, list) = DEFAULT_LOG_FIELDS,
                 log_foos: (dict, None) = None
                 ):
        """ Constructs a World instance

        :load_steps: Number of steps where no actions are applied and no logging is performed - this might be
                       necessary for some environments, which have a load-screen etc. (defaults to 0).
        :param verbose: bool (or integer) specifying whether (or in which intervals) rollout information is printed
        :param log_fields: tuple or list of strings, specifying
                    (i) variables in the rollout method (i.e., 'observation', 'action', 'reward', 'info'),
                    (ii) properties or fields in the `World` instance (e.g., 'episode', ...)
                    (iii) or keys in the `log_lambdas` dictionary,
                    which should be logged in a dictionary for each rollout.
                    One dict with lists for each log-field is generated per rollout.
                    The logs can be retrieved via the world's `log` property
        :param log_foos: key - value pairs which specify additional log-functions, where the values are either
                         `callable`s or string-code-snippets which can be subjected to the `compile` function.
                         If a log-field (in the `log` argument) corresponds to a key in the `log_foo` dict, the
                         value is executed (if it is callable) or subjected to `eval` otherwise.
                         The result is stored in the world's `log` property.
                         Such a log-foo key value pair could be `steps="self.env.steps"``, to log the step counts
                         of the environment.
        """

        Representable.__init__(self, repr_fields=self.REPR_FIELDS)
        Logger.__init__(self, log_fields=log_fields, log_foos=log_foos)

        self.env = Environment.make(env)
        self.agent = Agent.make(agent)

        self.episode = 0
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.load_steps = load_steps
        self.log_steps = self.max_steps - self.load_steps

        self.verbose = verbose
        self.render = render

    def vprint(self, *args, vprint_step=0, **kwargs):
        if self.verbose and (isinstance(self.verbose, bool) or not vprint_step % self.verbose):
            print(*args, **kwargs)

    def __enter__(self):
        return self.make(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.env.close()

        except:
            print('could not close environment.')
            pass

    def rollout(self):
        """ A rollout of one episode of the environment """

        self.vprint("Rollout episode {}".format(self.episode))
        start = time()

        action = self.agent.get_default_action()
        observation = self.env.reset(action=action)
        done = False
        if self.load_steps == 0:
            self.log(observation=observation, reward=self.env.reward_model, done=done, info={}, action=action)

        total_reward = 0
        steps = 0
        while not done:
            if self.render:
                self.env.render()

            if self.load_steps <= steps:
                action = self.agent.get_action(observation)

            observation, reward, done, info = self.env.step(action)
            steps += 1

            if self.load_steps <= steps:
                total_reward += reward
                self.log(observation=observation, reward=reward, done=done, info={}, action=action)
                self.vprint(f"\rstep {steps}:, total reward: {total_reward}     ", end='', vprint_step=steps)

            if steps >= self.max_steps:
                break

        self.vprint(f"\rstep {steps}:, total reward: {total_reward}     ")

        self.vprint("Episode {} finished after {} time-steps ({:.3f} seconds).".format(
            self.episode,
            steps + 1,
            time()-start,
        ))

        self.episode += 1
        if self.log_history != []:
            return self.wrap_np(self.log_history[-1])

        return total_reward

    @staticmethod
    def make_rollout(world_repr):
        if isinstance(world_repr, World):
            world_repr = repr(world_repr)

        with World.make(world_repr) as w:
            return w.rollout()

    @staticmethod
    def make_async_rollouts(world_obj, n_worker=1, output_path='examples/dataset/world_model/history.h5', async_quiet=True, verbose=True):
        """
        :param world_obj: world object or list of world objects. If a single world object is given, all specified
                          episodes are evaluated asynchronously in parallel.
        :param n_worker: The size of the multiprocessing pool. If a number < 0 is given, all available cores are used.
        :param output_path: File-path for rollout results, written in hdf5 format.
        :param async_quiet: Boolean specifying whether the asynchronous execution of the different episodes
                            (in case of a single world object being provided) is quiet or verbose.
        :param verbose: Boolean specifying whether the callbacks of the asynchronous tasks are called.
        """
        from multiprocessing import Pool, cpu_count

        if verbose:
            print(f'Start async world rollouts.')

        async_episodes = not isinstance(world_obj, (list, tuple))
        assert async_episodes, "Dumping of log-history of multiple rollouts not implemented."

        world = World.make(world_obj)
        n_episodes = world.n_episodes
        world.n_episodes = 1
        world.verbose = not async_quiet
        world_obj = [repr(world)] * n_episodes
        world.n_episodes = n_episodes

        n_worker = n_worker if n_worker > 1 else cpu_count()

        class async_callback():
            def __init__(self, verbose=True):
                self.verbose = verbose
                self.counter = 0

            def __call__(self, x):
                self.counter += 1
                if self.verbose:
                    print(f'Working on episode {self.counter}.')

        with Pool(processes=n_worker) as pool:
            callback = async_callback(verbose)
            jobs = [pool.apply_async(World.make_rollout, (wo,), callback=callback) for wo in world_obj]
            rollout_results = [job.get() for job in jobs]
            pool.terminate()

            if verbose:
                print(f'Finished with async work-load.')
                print(f'Saving rollout history to file `{output_path}`.')

            world.log_history = rollout_results
            world.dump_history(output_path, exist_ok=True)

            print('Done.')

            return world


if __name__ == '__main__':
    env_repr = dict(class_='CarRacingWrapper',  # 'CarRacing',
                    module_='examples.world_model.envs.car_racing',  # 'gym.envs.box2d.car_racing',
                    verbose=False,
                    )

    agent_repr = dict(class_='RandomWalker',
                      module_='examples.world_model.controller.random_walker',
                      action_space='Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32)',
                      default_action=[0., 0., 0.],
                      randomize_interval=5,
                      )

    world_repr = dict(env=env_repr,
                      agent=agent_repr,
                      n_episodes=8,
                      verbose=100,
                      max_steps=100,
                      load_steps=50,
                      render=False,
                      log_fields=('action', 'reward', 'observation', ),
                      log_foos={f'observation': "observation.transpose(2, 0, 1)"}
                      )

    world = World.make_async_rollouts(world_obj=world_repr,
                                      n_worker=1,
                                      output_path='examples/dataset/world_model/car_racing-random_walker.h5',
                                      )

    h = world.load_history('examples/dataset/world_model/car_racing-random_walker.h5')
    print(f'reloaded log-fields {list(h[0].keys())} for {len(h)} episodes.')
