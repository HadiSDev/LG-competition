import math
import random
import time
import pandas as pd
import numpy as np
from stable_baselines3.a2c import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from torch import nn

from Agents.CustomA2C import CustomA2C
from Policies.CustomCNN import CustomCNN
from game.lilys_garden_env import LilysGardenEnv


def run_A2C():
    levels = [i for i in range(1, 111)]

    random.shuffle(levels)

    total_size = len(levels)

    train_size = int(math.floor(total_size * 0.70))

    train_levels = levels[:train_size]
    test_levels = levels[-(total_size - train_size):]

    start_lvl = random.choice(train_levels)

    argu = {
        'level': start_lvl,
    }

    env = LilysGardenEnv(level=start_lvl)
    env.reset()

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix='a2c_model_check')

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        net_arch=[512, dict(pi=[256, 64], vf=[256, 64])],
        activation_fn=nn.ReLU,
        normalize_images=True
    )

    model = CustomA2C(CnnPolicy, env,
                      tensorboard_log="./logs/a2c",
                      policy_kwargs=policy_kwargs,
                      verbose=1,
                      device='cuda',
                      )

    model.learn(total_timesteps=1000,
                callback=checkpoint_callback)

    model.save("./logs/A2C_model_final")
    # model.load("A2C_model")

    model.set_levels(None)
    environment = model.env

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
    completion_rates = []

    for level in levels[:1]:
        environment.set_level(level)

        cur_episode = 1
        episode_rew = 0
        cur_valid_steps = 0
        cur_total_steps = 0
        cur_new_progress = 0
        cur_successful_click = 0
        cur_goal_reached = 0
        cur_completion_rate = 0
        while cur_episode <= EVAL_EPISODES:

            Sum = 0
            obs = environment.reset(seed=time.time())
            start_time = time.time()
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
            end_time = time.time()
            delta_time = end_time - start_time
            cur_completion_rate += delta_time

        Rewards.append(episode_rew / EVAL_EPISODES)
        valid_steps.append(cur_valid_steps / EVAL_EPISODES)
        total_steps.append(cur_total_steps / EVAL_EPISODES)
        new_progress.append(cur_new_progress / EVAL_EPISODES)
        successful_clicks.append(cur_successful_click / EVAL_EPISODES)
        goal_reached.append(cur_goal_reached / EVAL_EPISODES)
        completion_rates.append(cur_completion_rate / EVAL_EPISODES)

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
