import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box

from MCTS.Trainer import Trainer
from game.lilys_garden_env import LilysGardenEnv


class MCTSPolicy(nn.Module):

    def __init__(self, observation_space: Box, n_actions, device, n_input_channels=24):
        super(MCTSPolicy, self).__init__()
        self.device = device
        self.n_obs = observation_space.shape[0]
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 18, kernel_size=2, stride=1),
            nn.Conv2d(18, 64, kernel_size=2, stride=1),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.Flatten(), )

        cnn_output = self.calc_fcnn_input(self.n_obs, 2, 0, 1)
        cnn_output = self.calc_fcnn_input(cnn_output, 2, 0, 1)
        cnn_output = self.calc_fcnn_input(cnn_output, 2, 0, 1)

        self.fcnn = nn.Sequential(
            nn.Linear(3840, 512),
            nn.ReLU(),
        )

        self.p_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        self.v_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Linear(64, 1),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            sample = sample.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample
                                 ).shape[1]

    @staticmethod
    def calc_fcnn_input(W, F, P, S):
        return (W - F + 2 * P) / S + 1

    def reorder(self, observations: th.Tensor):
        return th.movedim(observations, -1, 1)

    def forward(self, obs):

        features = self.cnn(obs)
        fcnn = self.fcnn(features)
        logits = self.p_net(fcnn)
        value = self.v_net(fcnn).view(-1)

        policy = F.softmax(logits, dim=1)

        return logits, policy, value

    def step(self, obs):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        obs = preproccess_obs(obs)
        obs.to(device="cuda:0")
        _, pi, v = self.forward(obs)
        return pi.detach().cpu().numpy(), v.detach().cpu().numpy()


def preproccess_obs(obs):
    mean = np.mean(obs)
    std = np.std(obs)

    obs = (obs - mean) / std

    obs = th.as_tensor(obs, device="cuda").float()
    obs = th.unsqueeze(obs, 0)
    obs = obs.permute(0, 3, 1, 2)
    return obs


if __name__ == '__main__':
    env = LilysGardenEnv()
    env.reset()
    observation, reward, done, info_dict = env.step(1)

    model = MCTSPolicy(env.observation_space, env.action_space.n, env.channels)

    trainer = Trainer(model)

    trainer.train(np.random.rand(13, 9, 24), np.random.rand(1, 117), np.random.rand((1)))
