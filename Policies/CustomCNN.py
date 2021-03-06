import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 18, kernel_size=2, stride=1),
            nn.Conv2d(18, 64, kernel_size=2, stride=1),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = preprocess_obs(th.as_tensor(observation_space.sample()[None]).float(),
                                    observation_space, False)

            n_flatten = self.cnn(
                reorder(sample)
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = self.cnn(reorder(observations))
        return self.linear(features)


class CustomCNNLarge(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 18, kernel_size=2, stride=1),
            nn.Conv2d(18, 64, kernel_size=2, stride=1),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = preprocess_obs(th.as_tensor(observation_space.sample()[None]).float(),
                                    observation_space, False)

            n_flatten = self.cnn(
                reorder(sample)
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = self.cnn(reorder(observations))
        return self.linear(features)


def reorder(observations: th.Tensor):
    return th.movedim(observations, -1, 1)