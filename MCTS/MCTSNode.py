import random
from typing import List

import numpy as np

from game.lilys_garden_env import LilysGardenEnv


class MCTSNode:

    def __init__(self, actions: List[int], seed: int, env: LilysGardenEnv, parent: 'MCTSNode'):
        self.ressq = 0
        self.weight = None
        self.actions = actions
        self.seed = seed
        self.env = env
        self.visits = 0
        self.children: List[MCTSNode] = []
        self.sp_utc = 0.0
        self.trajectory = None
        self.parent = parent

    def get_state(self):
        self.restore()
        return self.trajectory[0]

    def restore(self):

        if self.actions == self.env.current_actions and self.trajectory is not None:
            return

        self.trajectory = self.env.restore_state(self.actions, self.seed)

    def get_action(self):
        return self.actions[-1]

    def is_terminal(self):
        if self.trajectory is None:
            self.restore()
            return self.is_terminal()
        return self.trajectory[2]

    def set_weight(self, weight):
        self.weight = weight

    def is_expanded(self):
        return len(self.children) > 0

    def add_child(self, child: 'MCTSNode'):
        self.children.append(child)
        child.parent = self

    def expand(self):
        obs = self.get_state()
        mask = self.env.create_action_mask(obs)

        if mask is not None:
            actions = np.where(mask == 1)[0]
            actions = actions.tolist()
        else:
            actions = [i for i in range(self.env.action_space.n+1)]

        for action in actions:
            all_actions = [] + self.actions
            all_actions.append(action)

            child = MCTSNode(all_actions, self.seed, self.env, parent=self)
            self.add_child(child)
        return self

    def _select_random_action(self, obs):
        mask = self.env.create_action_mask(obs)

        if mask is not None:
            actions = np.where(mask == 1)[0]
            actions = actions.tolist()
        else:
            actions = [i for i in range(self.env.action_space.n+1)]

        return random.choice(actions)

    def simulate(self):
        total_acc_reward = 0
        self.restore()
        observation = self.get_state()
        while True:
            if self.is_terminal():
                total_acc_reward += self.trajectory[1]  # get reward
                return total_acc_reward

            observation, reward, done, info_dict = self.env.step(self._select_random_action(observation))

            total_acc_reward += reward

            if done:
                return total_acc_reward

    def __eq__(self, other):

        if isinstance(other, MCTSNode):
            return other.actions == self.actions
        else:
            return False
