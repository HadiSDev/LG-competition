# Loading in simulator and environment
import random
import time

import gym
import math
from tensorforce import Agent

from Custom.Environment import Environment2
from game.simulator import Simulator

levels = [i for i in range(1, 111)]

random.shuffle(levels)

total_size = len(levels)

train_size = int(math.floor(total_size * 0.70))

train_levels = levels[:train_size]
test_levels = levels[-(total_size-train_size):]

start_lvl = random.choice(train_levels)

sim = Simulator(host="http://localhost:8090")

environment = Environment2.create(environment='lg-competition-v0', simulator=sim)
environment.set_level(start_lvl)

agent = Agent.create(agent='../configs/a2c.json', environment=environment)
#
# # To start a new playthrough, reset environment first to get initial obs. space
# obs = env.reset(seed=time.time())


# Train for 100 episodes
for _ in range(100):
    states = environment.reset()
    terminal = False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)


