import gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import TensorBoardOutputFormat

from Networks.CNNNetwork import CustomCNN
from Networks.CustomNN import CustomActorCriticPolicy, CustomNetwork
from game.simulator import Simulator
import numpy as np

sim = Simulator(host="http://localhost:8090")
env = gym.make('lg-competition-v0', level=1, simulator=sim)
env.reset(seed=42)
check_env(env)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=64),
)

model = A2C(CustomActorCriticPolicy, env, tensorboard_log="./logs/a2c", policy_kwargs=policy_kwargs, verbose=1).learn(
    total_timesteps=1000)
