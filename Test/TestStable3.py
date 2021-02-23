import datetime
import math
import random
import time
import pandas as pd
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.a2c import CnnPolicy
from stable_baselines3.common.env_checker import check_env
from torch import nn

from Agents.CustomA2C import CustomA2C
from Policies.CustomCNN import CustomCNN
from game.lilys_garden_env import LilysGardenEnv
from game.simulator import Simulator

levels = [i for i in range(1, 111)]

random.shuffle(levels)

total_size = len(levels)

train_size = int(math.floor(total_size * 0.70))

train_levels = levels[:train_size]
test_levels = levels[-(total_size - train_size):]

start_lvl = random.choice(train_levels)

sim = Simulator(host="http://localhost:8090")
env = gym.make('lg-competition-v0', level=start_lvl, simulator=sim)
env.reset(seed=42)
check_env(env)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    net_arch=[64, dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=nn.ReLU,
    normalize_images=False
)

model = CustomA2C(CnnPolicy, env,
                  tensorboard_log="./logs/a2c",
                  policy_kwargs=policy_kwargs,
                  verbose=1,
                  gamma=0.98,
                  ent_coef=0.01,
                  vf_coef=0.25,
                  learning_rate=1e-4,
                  n_steps=256,
                  levels=train_levels
                  )
model.learn(total_timesteps=35000000, )

model.set_levels(None)

for level in levels:
    if isinstance(model.env, LilysGardenEnv):
        model.env.set_level(level)
        model.env.reset(time.time())
        model.learn(100000, tb_log_name=f'A2C-{level}')

# Evaluate

EVAL_EPISODES = 100

Episodes = []
Rewards = []
Levels = []

for level in levels:
    model.env.set_level(level)
    obs = env.reset(time.time())
    cur_episode = 1
    while cur_episode <= EVAL_EPISODES:

        Sum = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            Sum += sum(rewards)
            if np.alltrue(dones):
                Episodes.append(cur_episode)
                Rewards.append(rewards)
                Levels.append(level)
                cur_episode += 1
                break

CSV = {
    "Episodes": Episodes,
    "Levels": Levels,
    "Rewards": Rewards

}

df = pd.DataFrame(CSV)

df.to_csv(f"A2C_Eval_{datetime.datetime.now()}")
