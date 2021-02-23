import math
import random

import gym

from stable_baselines.common.policies import MlpPolicy, CnnPolicy, ActorCriticPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# Parallel environments
import CnnExtractor
from CustomAgents.A2CAgent import A2CAgent
from game.simulator import Simulator

levels = [i for i in range(1, 111)]

random.shuffle(levels)

total_size = len(levels)

train_size = int(math.floor(total_size * 0.70))

train_levels = levels[:train_size]
test_levels = levels[-(total_size-train_size):]

start_lvl = random.choice(train_levels)

sim = Simulator(host="http://localhost:8090")
env = gym.make('lg-competition-v0', level=start_lvl, simulator=sim)

# env = make_vec_env('CartPole-v1', n_envs=4)
model = A2CAgent(CnnExtractor.CnnExtractor, env, verbose=1, train_levels=train_levels, tensorboard_log="./logs/a2c")
model.learn(total_timesteps=25000)

# del model # remove to demonstrate saving and loading
#
# model = A2C.load("a2c_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()