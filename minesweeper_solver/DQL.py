from collections import namedtuple
from functools import wraps
from itertools import product
import os
import random
from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .minefield import CellState, GameState, MineField


def exit_env(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        finally:
            self._env.close_env()

    return wrapper


ActionTuple = namedtuple("Action", ["x", "y"])
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])


class MinesweepperEnv:
    """Provides the environment for the RL agent"""

    def __init__(
        self, height: int = 0, width: int = 0, mines: int = 0, state: str = "", headless=False, *args, **kwargs
    ):
        self._state = state
        self._mf = MineField(headless, width, height, mines, state=state, *args, **kwargs)

        self._height = self._mf.height
        self._width = self._mf.width
        self._n_mines = self._mf._n_mines

        self.actions = [ActionTuple(x, y) for y, x in product(range(self._height), range(self._width))]
        self.n_actions = len(self.actions)
        self.obs_dim = self._height * self._width

        # These change
        self._grid = self._mf.initial_grid
        self._game_state = GameState.PLAYING
        self._mines_left = self._mf.initial_mines_left
        self._seconds = self._mf.initial_seconds
        self._clicks = 0

    def take_action(self, x: int, y: int):
        """Takes the specified action and returns the resulting information"""
        out = self._mf.execute_action_and_get_info(x, y, 0)

        cell_before = out[0]
        cell_after = out[1]
        self._grid = out[2]
        self._game_state = out[3]
        self._mines_left = out[4]
        self._seconds = out[5]

        return cell_before, cell_after

    def reset(self):
        """Restarts the game."""
        if self._state:
            self._mf._import_gamestate()
        else:
            self._mf.restart()
        self._clicks = 0
        self._grid = self._mf.initial_grid
        self._game_state = GameState.PLAYING
        self._mines_left = self._mf.initial_mines_left
        self._seconds = self._mf.initial_seconds
        return self._encode_state(self._mf.initial_grid)

    def step(self, action_ind: int):
        """Makes the given action and calculate the reward.

        Logic for the reward:

        Every click on a cell that isn't unopened gives minus one point.
        Every click on a unopened cell gets one point.
        Win gets 100 points
        """

        self._clicks += 1

        reward = -0.5

        action = self.actions[action_ind]
        x = action.x
        y = action.y

        cell_before, cell_after = self.take_action(x, y)

        if cell_after in (cell_before, CellState.FLAG):
            raise Exception(f"Useless selection! x={x}, y={y}")
        if cell_after not in (cell_before, CellState.BOMBDEATH, CellState.FLAG):
            # Reward for opening a usefull cell
            reward += 1

        # Check if terminated
        terminated = False
        if self._game_state == GameState.WIN:
            print("BIG WIN !!!")
            terminated = True
            reward += 20
        elif self._game_state == GameState.LOST:
            terminated = True
            reward -= 10

        # Check if truncated
        truncated = False
        if self._clicks == self._height * self._width - self._n_mines:
            truncated = True

        return self._encode_state(self._grid), reward, terminated, truncated

    def close_env(self):
        self._mf.end_session()

    def _encode_state(self, grid):
        """Separates the minefield into  matrixes, other contains numbered cells, other the unopened cells"""
        encoded = np.zeros((2, self._height, self._width), dtype=np.float32)

        grid = np.array(grid)

        encoded[0] = np.where(grid < 9, grid / 8.0, 0)
        encoded[1] = (grid == 9).astype(np.float32)

        return encoded


class Memory(list):
    """A list with a limited size and items are appended to the front"""

    def __init__(self, size=10_000):
        super().__init__(())
        self._size = size

    def add(self, *values):
        """Adds a single item to the front of the list"""
        if len(self) == self._size:
            self.pop()
        super().insert(0, Transition(*values))

    def choice(self, n):
        """Randomly chooses n instances from the memory"""
        return random.sample(self, n)


class ConvolutionalNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        qmap = self.out(x)

        return qmap.view(x.size(0), -1)


class DQL:
    """Deep Q-learning model"""

    def __init__(
        self,
        episodes: int,
        batch_size: int,
        epsilon: Callable,
        gamma: float,
        lr: float,
        w_interval: int,
        state: str = "",
        env_args: List = [],
        env_kwargs: dict = {},
    ):
        """

        Args:
            episodes (int): Number of episodes to go through
            batch_size (int): How many datapoints are used in each optimization round
            epsilon (callable): A function that takes in the amount of steps as an argument and returns what
                the value of epsilon should be at that point in time. The image of epsilon should be [0, 1].
                With epsilon 0 all actions are policy driven, with 1 all are random.
            gamma (float): The future discount factor for the rewards.
            lr (float): Learning rate for the optimizer
            w_interval (int): After how many steps should the target networks weights be updated.
        """
        self._device = "cpu"

        self._episodes = episodes
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._gamma = gamma
        self._lr = lr
        self._w_interval = w_interval

        self._env = MinesweepperEnv(*env_args, **env_kwargs)

        self._set_rnd_seed()

        self._n_actions = self._env.n_actions
        self._dim_obs_space = self._env.obs_dim

        self._policy_net = ConvolutionalNet()
        self._target_net = ConvolutionalNet()

        self._update_target_net()

        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=self._lr, amsgrad=True)
        self._steps_taken = 0
        self._episodes_ran = 0

        self._plotter = Plotter()

    def save_model(self):
        state = {
            "episodes_ran": self._episodes,
            "state_dict_policy": self._policy_net.state_dict(),
            "state_dict_target": self._target_net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        torch.save(state, os.path.join("data", "model.pt"))

    def load_model(self):
        state = torch.load(os.path.join("data", "model.pt"))
        self._policy_net.load_state_dict(state["state_dict_policy"])
        self._target_net.load_state_dict(state["state_dict_target"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._episodes_ran = state["episodes_ran"]

    def _update_target_net(self):
        """Updates the target net to match the policy net"""
        self._target_net.load_state_dict(self._policy_net.state_dict())

    def _set_rnd_seed(self):
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _select_action(self, state: torch.Tensor):
        """Chooses an action in the given state using an epsilon greedy strategy."""
        if np.random.random() < self._epsilon(self._steps_taken):
            # Random action
            viable_actions = state[0][1].nonzero()
            action_xy = viable_actions[np.random.choice(len(viable_actions))]
            ind = self._env.actions.index((int(action_xy[1].item()), int(action_xy[0].item())))  # type: ignore
            return torch.tensor([[ind]], dtype=torch.int32, device=self._device)
        # Choose the action that maximises the future rewards according to the policy network
        with torch.no_grad():
            output: torch.Tensor = self._policy_net(state)
            output.masked_fill_(state[0][1].bool().logical_not_().flatten(), float("-inf"))
            return output.max(1).indices.view(1, 1)

    def _optim_batch(self, memory):
        """Performs one round of backpropagation and optimization"""
        if len(memory) < self._batch_size:
            return

        batch = memory.choice(self._batch_size)
        batch = Transition(*zip(*batch))
        states = torch.cat(batch.state)
        rewards = torch.cat(batch.reward)
        actions = torch.cat(batch.action)
        next_states = torch.cat([i for i in batch.next_state if i is not None])

        # The scores of the actions that the nn would have made
        Q_i = self._policy_net(states).gather(1, actions)

        # compute the actual scores of the next states
        mask = torch.tensor([i is not None for i in batch.next_state], dtype=torch.bool)
        target = rewards
        with torch.no_grad():
            target[mask] += self._gamma * self._target_net(next_states).max(1).values

        # Take one step towards the negative gradient
        loss_fn = F.huber_loss(Q_i.squeeze(1), target)
        self._optimizer.zero_grad()
        loss_fn.backward()
        self._optimizer.step()

    @exit_env
    def train(self):
        """Trains the model"""
        memory = Memory()
        episode_scores = []
        episode_clicks = []
        for _ in range(self._episodes):
            state = self._env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)

            terminated, truncated = False, False
            cum_rew = 0
            for _ in range(self._env._width * self._env._height):
                # Select an action, take it and store the transition
                action = self._select_action(state)  # type: ignore
                observation, reward, terminated, truncated = self._env.step(action.item())  # type: ignore
                self._steps_taken += 1

                cum_rew += reward

                if terminated:
                    observation = None
                else:
                    observation = torch.tensor(observation, dtype=torch.float32, device=self._device).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32, device=self._device)

                memory.add(state, action, reward, observation)

                # Update the current state
                state = observation

                # Once in a while update the target net
                if self._steps_taken % self._w_interval == 0:
                    print("steps taken:", self._steps_taken)
                    self._update_target_net()

                # Get some training in
                self._optim_batch(memory)

                if terminated or truncated:
                    # Time to stop the fun
                    episode_scores.append(cum_rew)
                    episode_clicks.append(self._env._clicks)
                    self._plotter.update(episode_scores)
                    break

            self._episodes_ran += 1

        self._plotter.finalise()
        self.save_model()


class Plotter:
    def __init__(self):
        plt.ion()
        self.mav = [0.0 for _ in range(50)]

    def finalise(self):
        plt.ioff()

    def update(self, scores):

        plt.figure(1)
        plt.clf()
        plt.title("Result")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.plot(scores)

        if len(scores) > 50:
            mav = scores[-50:]
            mav = sum(mav) / len(mav)
            self.mav.append(mav)

            plt.plot(self.mav)

        plt.pause(0.001)


"""if __name__ == "__main__":
    plt.ion()

    env_args = [9, 9, 10]

    batch_size = 128
    episodes = 200

    def epsilon(step: int):
        start = 0.9
        end = 0.01
        tc = 1000
        return end + (start - end) * np.exp(-step / tc)

    gamma = 0.99
    lr = 0.0003
    w_update_interval = 1000

    agent = DQL(episodes, batch_size, epsilon, gamma, lr, w_update_interval, env_args=env_args)

    try:
        a = agent.train()
    except Exception as e:
        agent._env.close_env()
        raise e

    plt.ioff()
    plt.show()"""
