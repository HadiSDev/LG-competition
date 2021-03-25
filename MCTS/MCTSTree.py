from typing import List

from MCTS.MCTSNode import MCTSNode
from MCTS.MCTSPolicy import MCTSPolicy
from MCTS.Trainer import Trainer
from game.lilys_garden_env import LilysGardenEnv
import numpy as np


class MCTSTree:

    def __init__(self, node: MCTSNode, env: LilysGardenEnv, trainer: Trainer, verbose: bool = False, C=1.5, T=0.3):
        self.T = T
        self.C = C
        self.root = node
        self.verbose = verbose
        self.trainer = trainer
        self.env = env

    def eval_UTC(self, node: MCTSNode):
        c = 0.5
        n = node.visits

        obs = node.get_state()
        pi, vi = self.trainer.step_model.step(obs)

        if node.parent is None:
            t = node.visits
        else:
            t = node.parent.visits

        utc = vi + c * np.sqrt(np.log(t) / n) * pi[0][node.get_action() - 1]
        D = 10000.
        modification = np.sqrt((node.ressq ** 2 - n * vi ** 2 + D) / n)

        node.sputc = utc + modification
        return node.sputc

    def utc_select(self, node: MCTSNode):
        assert all([child_p.is_expanded() for child_p in node.children])
        return max(node.children, key=self.eval_UTC)

    def simulate(self, node: MCTSNode, num_iter=1):

        avg_acc_total_reward = node.simulate()

        for n in range(num_iter):
            r = node.simulate()
            avg_acc_total_reward += r

        avg_acc_total_reward /= num_iter
        return avg_acc_total_reward

    def selection(self, node: MCTSNode):
        path = []
        while True:
            path.append(node)
            if node.visits == 0 or node.is_terminal():
                return path

            unexplored_children = [child for child in node.children if child.visits == 0]

            if unexplored_children:
                n = unexplored_children.pop()
                path.append(n)
                return path

            node = self.utc_select(node)

    def expand(self, node: MCTSNode):
        assert node.is_terminal() is False

        if node.is_expanded() is True:
            return node

        return node.expand()

    def rollout(self, node: MCTSNode):
        path = self.selection(node)
        selected_node = path[-1]
        self.expand(selected_node)
        reward = self.simulate(selected_node, num_iter=100)

        obs = selected_node.get_state()
        action_prob = np.zeros(self.env.action_space.n)
        action_prob[node.get_action()] = 1

        self.trainer.train(obs, action_prob, np.array([reward]))

        self.back_propogate(path, reward)

    def back_propogate(self, path: List[MCTSNode], reward):
        for node in reversed(path):
            node.visits += 1
            node.ressq += reward ** 2

    def run_episode(self, iter=100):

        for i in range(iter):
            self.rollout(root)


if __name__ == "__main__":
    env = LilysGardenEnv()
    state = env.reset()
    observation, reward, done, info_dict = env.step(1)

    model = MCTSPolicy(env.observation_space, env.action_space.n, env.channels)

    trainer = Trainer(model)

    pi, _ = trainer.step_model.step(observation)
    action = np.argmax(pi)
    root = MCTSNode([action], env.seed(), env, None)
    tree = MCTSTree(root, env, trainer)

    for i in range(2000):
        tree.rollout(root)
