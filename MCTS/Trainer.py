import logging

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Trainer for an MCTS policy network. Trains the network to minimize
    the difference between the value estimate and the actual returns and
    the difference between the policy estimate and the refined policy estimates
    derived via the tree search.
    """

    def __init__(self, policy, writer: SummaryWriter, learning_rate=0.1):
        self.step_model = policy

        value_criterion = nn.MSELoss()

        optimizer = th.optim.Adam(self.step_model.parameters(),
                                  lr=learning_rate)
        self.writer = writer

        def train(obs, search_pis, returns, epochs=1):
            p_loss = []
            v_loss = []
            total_loss = []
            obs = preproccess_obs(obs)
            obs.to(device="cuda:0")
            search_pis = th.as_tensor(search_pis).to(device="cuda:0")
            returns = th.as_tensor(returns).to(device="cuda:0")
            for epoch in range(epochs):
                optimizer.zero_grad()
                logits, policy, value = self.step_model(obs)

                logsoftmax = nn.LogSoftmax(dim=1)
                policy_loss = th.mean(th.sum(-search_pis * logsoftmax(logits), dim=1))
                value_loss = value_criterion(value.float(), returns.float())
                loss = policy_loss + value_loss

                p_loss.append(policy_loss)
                v_loss.append(value_loss)
                total_loss.append(loss)

                loss.backward()
                optimizer.step()

                logging.debug(f"Epoch: {epoch}, loss: {loss}, policy_loss: {policy_loss}, value_loss: {value_loss}")
                print(f"Epoch: {epoch}, loss: {loss}, policy_loss: {policy_loss}, value_loss: {value_loss}")

            length = len(total_loss)
            return sum(total_loss)/length, sum(p_loss)/length, sum(v_loss)/length

        self.train = train


def preproccess_obs(obs):
    mean = np.mean(obs)
    std = np.std(obs)

    obs = (obs - mean) / std

    obs = th.as_tensor(obs, device="cuda").float()
    obs = obs.permute(0, 3, 1, 2)

    return obs
