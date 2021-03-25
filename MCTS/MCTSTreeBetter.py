import random
import time
from typing import List

from torch.utils.tensorboard import SummaryWriter

from MCTS.MCTSNode import MCTSNode
from MCTS.MCTSPolicy import MCTSPolicy
from MCTS.ReplayMemory import ReplayMemory
from MCTS.Trainer import Trainer
from game.lilys_garden_env import LilysGardenEnv
import numpy as np
import matplotlib.pyplot as plt


class MCTSTreeBetter:

    def __init__(self, node: MCTSNode, env: LilysGardenEnv, trainer: Trainer, max_size, verbose: bool = False, C=1.5,
                 T=0.3):
        self.T = T
        self.C = C
        self.root = node
        self.verbose = verbose
        self.trainer = trainer
        self.env = env
        self.max_size = max_size
        self.size_count = 1

        self.data_p = []
        self.data_v = []
        self.data_o = []

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

        self.data_p.append(action_prob)
        self.data_o.append(obs)
        self.data_v.append(reward)
        self.size_count += 1
        self.back_propogate(path, reward)

    def back_propogate(self, path: List[MCTSNode], reward):
        for node in reversed(path):
            node.visits += 1
            node.ressq += reward ** 2

    def run_episode(self, ):

        while True:

            if self.size_count >= self.max_size:
                break

            self.rollout(self.root)

        return self.data_o, self.data_p, self.data_v


if __name__ == "__main__":

    writer = SummaryWriter()

    BATCH_SIZE = 32
    env = LilysGardenEnv()

    model = MCTSPolicy(env.observation_space, env.action_space.n, env.channels)
    model.to("cuda")
    trainer = Trainer(model, writer)

    mem = ReplayMemory(100000,
                       {"ob": np.long,
                        "pi": np.float32,
                        "return": np.float32},
                       {"ob": [],
                        "pi": [env.action_space.n],
                        "return": []}, batch_size=BATCH_SIZE)

    levels = [i for i in range(1, 110)]

    random.shuffle(levels)
    test_env = LilysGardenEnv()


    def test_agent():
        total_rew = 0

        level = random.choice(levels[:10])
        test_env.set_level(level)
        obs = test_env.reset(time.time())
        step_idx = 0
        start = time.time()
        while True:
            p, _ = model.step(obs)
            action = np.argmax(p)
            obs, reward, done, info = test_env.step(action)

            step_idx += 1
            total_rew += reward

            if done:
                valid_moves = info.valid_steps
                total_moves = info.total_steps
                end = time.time()
                completion_time = end-start
                break

        return total_rew, step_idx, level, valid_moves, total_moves, completion_time




    level = random.choice(levels[:10])
    env.set_level(level)
    iteration = 0
    mcts_acc_searches = 0
    while True:
        observation = env.reset(time.time())
        pi, _ = trainer.step_model.step(observation)
        action = np.argmax(pi)

        actions = [action]
        root = MCTSNode(actions, env.sim_seed, env, None)

        tree = MCTSTreeBetter(root, env, trainer, BATCH_SIZE * 2)

        obs, pis, vs = tree.run_episode()
        mcts_acc_searches += 64

        mem.add_all({"ob": obs, "pi": pis, "return": vs})

        batch = mem.get_minibatch()

        total_loss, p_loss, v_loss = trainer.train(batch["ob"], batch["pi"], batch["return"])

        writer.add_scalar(f"avg_total_loss / iteration", total_loss, iteration)
        writer.add_scalar(f"avg_p_loss / iteration", p_loss, iteration)
        writer.add_scalar(f"avg_v_loss / iteration", v_loss, iteration)

        if iteration % 50 is 0:
            num_eval_episodes = 5
            avg_reward = []
            avg_valid_moves = []
            avg_total_moves = []
            avg_completion_rate = []
            for _ in range(num_eval_episodes):
                total_rew, step_idx, level, valid_moves, total_moves, completion_time = test_agent()
                avg_reward.append(total_rew)
                avg_valid_moves.append(valid_moves)
                avg_total_moves.append(total_moves)
                avg_completion_rate.append(completion_time)

                writer.add_scalar(f"eval_total_reward_level_{level} / iteration", total_rew, iteration)
                writer.add_scalar(f"eval_total_reward_level_{level} / tree_acc_searches", total_rew, mcts_acc_searches)

                writer.add_scalar(f"eval_valid_moves_level_{level} / iteration", valid_moves, iteration)
                writer.add_scalar(f"eval_valid_moves_level_{level} / tree_acc_searches", valid_moves, mcts_acc_searches)

                writer.add_scalar(f"eval_total_moves_level_{level} / iteration", total_moves, iteration)
                writer.add_scalar(f"eval_total_moves_level_{level} / tree_acc_searches", total_moves, mcts_acc_searches)

                writer.add_scalar(f"eval_completion_time_level_{level} / iteration", completion_time, iteration)
                writer.add_scalar(f"eval_completion_time_level_{level} / tree_acc_searches", completion_time, mcts_acc_searches)

            avg_total_moves = sum(avg_total_moves)/len(avg_total_moves)
            avg_valid_moves = sum(avg_valid_moves) / len(avg_valid_moves)
            avg_reward = sum(avg_reward) / len(avg_reward)
            avg_reward = sum(avg_completion_rate) / len(avg_completion_rate)

            writer.add_scalar(f"eval_total_reward_avg / iteration", avg_reward, iteration)
            writer.add_scalar(f"eval_valid_moves_avg / iteration", avg_valid_moves, iteration)
            writer.add_scalar(f"eval_total_moves_avg / iteration", avg_total_moves, iteration)
            writer.add_scalar(f"eval_completion_rate_avg / iteration", avg_total_moves, iteration)

        writer.flush()

        iteration += 1



