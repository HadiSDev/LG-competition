import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import numpy as np
from Agents.PPOAgent import PPOAgent
from game.simulator import Simulator
import random
import math
# Loading in simulator and environment
sim = Simulator(host="http://localhost:8090")
env = gym.make('lg-competition-v0', level=1, simulator=sim)

# To start a new playthrough, reset environment first to get initial obs. space
obs = env.reset(seed=42)

levels = [i for i in range(1, 111)]

random.shuffle(levels)

total_size = len(levels)

train_size = int(math.floor(total_size * 0.70))

train_levels = levels[:train_size]
test_levels = levels[-(total_size-train_size):]

model = PPOAgent(MlpPolicy, env, train_levels, verbose=1, tensorboard_log="./testlog/")

model.learn(total_timesteps=10000, tb_log_name="first_run")

model.save("test")
