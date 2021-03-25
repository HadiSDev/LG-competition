from stable_baselines3 import A2C

from Agents.CustomA2C import CustomA2C

model = CustomA2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=10000)