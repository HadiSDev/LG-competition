import random
import time
from datetime import date
from pathlib import Path
from typing import List
import logging
from torch.utils.tensorboard import SummaryWriter
import torch as tc
from MCTS.MCTSNode import MCTSNode
from MCTS.MCTSPolicy import MCTSPolicy
from MCTS.ReplayBuffer import ReplayBuffer, Transition
from MCTS.Trainer import Trainer
from game.lilys_garden_env import LilysGardenEnv
import numpy as np


class MCTSTreeBetter:

    def __init__(self, node: MCTSNode, env: LilysGardenEnv, trainer: Trainer, max_size, memory: ReplayBuffer, verbose: bool = False, C=1.5,
                 T=0.3):
        self.T = T
        self.C = C
        self.root = node
        self.verbose = verbose
        self.trainer = trainer
        self.env = env
        self.max_size = max_size
        self.size_count = 1

        self.memory = memory

    def eval_UTC(self, node: MCTSNode):
        c = 0.5
        n = node.visits

        obs = node.get_state()
        pi, vi = self.trainer.step_model.step(obs)
        print(f"Predicted {pi[0][node.get_action()]} for action {node.get_action()}")
        if node.parent is None:
            t = node.visits
        else:
            t = node.parent.visits

        utc = vi + c * np.sqrt(np.log(t) / n) * pi[0][node.get_action()]
        D = 10000.
        modification = np.sqrt((node.ressq ** 2 - n * vi ** 2 + D) / n)

        node.sputc = utc + modification
        return node.sputc

    def utc_select(self, node: MCTSNode):
        assert all([child_p.is_expanded() for child_p in node.children])
        return max(node.children, key=self.eval_UTC)

    def simulate(self, node: MCTSNode, num_iter=1):
        print("Simulating")
        logging.debug(f"Simulating")
        avg_acc_total_reward = node.simulate()

        for n in range(num_iter):
            r = node.simulate()
            avg_acc_total_reward += r

        avg_acc_total_reward /= num_iter
        return avg_acc_total_reward

    def selection(self, node: MCTSNode):
        print("selecting")
        logging.debug(f"Selecting")
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
        print("Expanding")
        logging.debug(f"Expanding")
        if node.is_expanded() is True:
            return node

        return node.expand()

    def rollout(self, node: MCTSNode):
        path = self.selection(node)
        selected_node = path[-1]
        self.expand(selected_node)
        reward = self.simulate(selected_node, num_iter=50)

        obs = selected_node.get_state()
        action_prob = np.zeros(self.env.action_space.n)
        action_prob[node.get_action()] = 1

        self.memory.push(obs, action_prob, reward)

        self.size_count += 1
        self.back_propogate(path, reward)

    def back_propogate(self, path: List[MCTSNode], reward):
        for node in reversed(path):
            node.visits += 1
            node.ressq += reward ** 2

    def run_episode(self, ):

        while True:
            print(f"Tree seach: {self.size_count}")
            logging.debug(f"Tree seach: {self.size_count}")

            if self.size_count >= self.max_size:
                break

            self.rollout(self.root)


if __name__ == "__main__":

    writer = SummaryWriter(log_dir="mcts_logs/mcts")
    BATCH_SIZE = 32
    env = LilysGardenEnv()

    model = MCTSPolicy(env.observation_space, env.action_space.n, env.channels)

    model.to("cuda")
    trainer = Trainer(model, writer)

    mem = ReplayBuffer(10000)

    levels = [i for i in range(1, 110)]
    levels = levels[:10]
    random.shuffle(levels)
    test_env = LilysGardenEnv()

    Path("mcts_models").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename="logs/logs.log", filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    def test_agent():
        total_rew = 0

        level = random.choice(levels)
        test_env.set_level(level)
        obs = test_env.reset(time.time())
        step_idx = 0
        start = time.time()
        actions_sorted = []
        while True:
            p, _ = model.step(obs)

            if len(actions_sorted) is 0:
                actions_sorted: List[int] = np.argsort(-p).tolist()[0]

            action = actions_sorted[0]
            obs, reward, done, info = test_env.step(action)

            step_idx += 1
            total_rew += reward

            if info['successful_click'] is False:
                actions_sorted.pop(0)
            else:
                actions_sorted = []

            if done:
                valid_moves = info['valid_steps']
                total_moves = info['total_steps']
                end = time.time()
                completion_time = end-start
                break

        return total_rew, step_idx, level, valid_moves, total_moves, completion_time


    level = random.choice(levels)
    env.set_level(level)
    iteration = 0
    mcts_acc_searches = 0
    while True:
        observation = env.reset(time.time())
        pi, _ = trainer.step_model.step(observation)
        action = np.argmax(pi)

        actions = [action]
        root = MCTSNode(actions, env.sim_seed, env, None)

        tree = MCTSTreeBetter(root, env, trainer, BATCH_SIZE * 2, mem)

        tree.run_episode()
        mcts_acc_searches += 64

        batch = mem.sample(batch_size=BATCH_SIZE)
        batch = Transition(*zip(*batch))
        total_loss, p_loss, v_loss = trainer.train(batch.obs, batch.pi, batch.vi)

        writer.add_scalar(f"avg_total_loss / iteration", total_loss, iteration)
        writer.add_scalar(f"avg_p_loss / iteration", p_loss, iteration)
        writer.add_scalar(f"avg_v_loss / iteration", v_loss, iteration)
        writer.flush()

        if iteration % 2 is 0:
            print("Evaluation started")
            logging.debug("Evaluation started")
            num_eval_episodes = 5
            avg_reward = []
            avg_valid_moves = []
            avg_total_moves = []
            avg_completion_rate = []
            for i in range(num_eval_episodes):
                print(f"Eval epiosde {i}")
                logging.debug(f"Eval epiosde {i}")
                total_rew, step_idx, level, valid_moves, total_moves, completion_time = test_agent()
                print(f"Eval epiosde {i} finished on level {level}")
                logging.debug(f"Eval epiosde {i} finished on level {level}")
                avg_reward.append(total_rew)
                avg_valid_moves.append(valid_moves)
                avg_total_moves.append(total_moves)
                avg_completion_rate.append(completion_time)

                writer.add_scalar(f"eval_total_reward_level_{level} / iteration", total_rew, iteration)
                writer.add_scalar(f"eval_total_reward_level_{level} / tree_acc_searches", total_rew, mcts_acc_searches)
                writer.flush()
                writer.add_scalar(f"eval_valid_moves_level_{level} / iteration", valid_moves, iteration)
                writer.add_scalar(f"eval_valid_moves_level_{level} / tree_acc_searches", valid_moves, mcts_acc_searches)
                writer.flush()
                writer.add_scalar(f"eval_total_moves_level_{level} / iteration", total_moves, iteration)
                writer.add_scalar(f"eval_total_moves_level_{level} / tree_acc_searches", total_moves, mcts_acc_searches)
                writer.flush()
                writer.add_scalar(f"eval_completion_time_level_{level} / iteration", completion_time, iteration)
                writer.add_scalar(f"eval_completion_time_level_{level} / tree_acc_searches", completion_time, mcts_acc_searches)
                writer.flush()

            avg_total_moves = sum(avg_total_moves)/len(avg_total_moves)
            avg_valid_moves = sum(avg_valid_moves) / len(avg_valid_moves)
            avg_reward = sum(avg_reward) / len(avg_reward)
            avg_completion_rate = sum(avg_completion_rate) / len(avg_completion_rate)
            print(f"eval average reward {avg_reward}")
            writer.add_scalar(f"eval_total_reward_avg / iteration", avg_reward, iteration)
            writer.flush()
            writer.add_scalar(f"eval_valid_moves_avg / iteration", avg_valid_moves, iteration)
            writer.flush()
            writer.add_scalar(f"eval_total_moves_avg / iteration", avg_total_moves, iteration)
            writer.flush()
            writer.add_scalar(f"eval_completion_rate_avg / iteration", avg_completion_rate, iteration)
            writer.flush()


        if iteration % 1 is 0:
            tc.save(model.state_dict(), f"mcts_models/mcts_net")

        iteration += 1



