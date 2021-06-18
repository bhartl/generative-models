import argh
import gym
from examples.world_model.controller.random_walker import RandomWalker


def demo(steps=1000):
    env = gym.make('CartPole-v0')
    obs = env.reset()
    agent = RandomWalker(action_space=env.action_space)

    for _ in range(steps):
        env.render()
        random_action = agent.get_action(obs)  # take a random action
        obs, reward, done, info = env.step(random_action)
    env.close()


if __name__ == '__main__':
    argh.dispatch_commands([
        demo,
    ])