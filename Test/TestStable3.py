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
#check_env(env)

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
model.learn(total_timesteps=100, )

model.set_levels(None)
environment = model.env.envs[0]

for level in levels[:1]:

    if isinstance(environment, LilysGardenEnv):
        environment.set_level(level)
        environment.reset(time.time())
        model.learn(10, tb_log_name=f'A2C-{level}')

# Evaluate

EVAL_EPISODES = 5

Rewards = []
Levels = []
valid_steps = []
total_steps = []
new_progress = []
successful_clicks = []
goal_reached = []

for level in levels[:5]:
    environment.set_level(level)
    obs = environment.reset(time.time())
    cur_episode = 1
    episode_rew = 0
    cur_valid_steps = 0
    cur_total_steps = 0
    cur_new_progress = 0
    cur_successful_click = 0
    cur_goal_reached = 0
    while cur_episode <= EVAL_EPISODES:

        Sum = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = environment.step(action)
            Sum += rewards

            if np.alltrue(dones):
                episode_rew += Sum

                if info['goal_reached']:
                    cur_goal_reached += 1

                if info['successful_click']:
                    cur_valid_steps += 1

                cur_new_progress += info['new_progress']
                cur_valid_steps += info['valid_steps']
                cur_total_steps += info['total_steps']

                cur_episode += 1
                break

    Rewards.append(episode_rew/EVAL_EPISODES)
    valid_steps.append(cur_valid_steps/EVAL_EPISODES)
    total_steps.append(cur_total_steps / EVAL_EPISODES)
    new_progress.append(cur_new_progress / EVAL_EPISODES)
    successful_clicks.append(cur_successful_click / EVAL_EPISODES)
    goal_reached.append(cur_goal_reached / EVAL_EPISODES)

    Levels.append(level)

CSV = {
    "Levels": Levels,
    "Rewards": Rewards,
    "total_steps": total_steps,
    "new_progress": new_progress,
    "successful_clicks": successful_clicks,
    "goal_reached": goal_reached,
    "valid_steps": valid_steps


}

df = pd.DataFrame(CSV)

df.to_csv(f"A2C_Eval")
