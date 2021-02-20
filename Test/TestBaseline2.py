import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

env = gym.make('CartPole-v1')

check_env(env)
