from collections import namedtuple
from datetime import datetime
from enum import Enum
from functools import wraps
from itertools import product
import os
import random
from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
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


class Op(Enum):
    OPEN = 0
    FLAG = 2


ActionTuple = namedtuple("Action", ["op", "x", "y"])
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])


class MinesweepperEnv:
    """Provides the environment for the RL agent"""

    def __init__(
        self,
        height: int = 0,
        width: int = 0,
        mines: int = 0,
        state: str = "",
        flags_allowed: bool = False,
        headless: bool = False,
    ):
        self._state = state
        self._mf = MineField(headless, width, height, mines, state=state)

        self._height = self._mf.height
        self._width = self._mf.width
        self._n_mines = self._mf._n_mines

        self._flags_allowed = flags_allowed

        operations = list(Op) if self._flags_allowed else [Op.OPEN]

        self.actions = [
            ActionTuple(op, x, y) for op, y, x in product(operations, range(self._height), range(self._width))
        ]
        self.n_actions = len(self.actions)
        self.obs_dim = self._height * self._width

        # These change
        self._grid = self._mf.initial_grid
        self._game_state = GameState.PLAYING
        self._mines_left = self._mf.initial_mines_left
        self._seconds = self._mf.initial_seconds
        self._clicks = 0

    def take_action(self, op: Op, x: int, y: int):
        """Takes the specified action and returns the resulting information"""
        out = self._mf.execute_action_and_get_info(x, y, op.value)

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
        op = action.op
        x = action.x
        y = action.y

        cell_before, cell_after = self.take_action(op, x, y)

        if cell_after == cell_before:
            raise Exception(f"Useless selection! x={x}, y={y}, this shouldn't be possible.")

        if cell_before == CellState.FLAG and op == Op.FLAG:
            raise Exception(f"Tried to reflag x={x}, y={y}, this shouldn't be possible.")

        if cell_after not in (cell_before, CellState.BOMBDEATH, CellState.FLAG):
            # Reward for opening a usefull cell
            reward += 1.0

        if cell_after == CellState.FLAG:
            # Should see here if it was a valid choice, hardcoded punishment for now
            reward -= 5.0

        # Check if terminated
        terminated = False
        if self._game_state == GameState.WIN:
            print("BIG WIN !!!")
            terminated = True
            reward += 20.0
        elif self._game_state == GameState.LOST:
            terminated = True
            reward -= 10.0
        elif self._mines_left <= 0:  # Too many mines flagged and now stuck
            print("too many mines")
            terminated = True
            reward -= 20.0

        # Check if truncated
        truncated = False
        if self._clicks == self._height * self._width - self._n_mines:
            truncated = True

        next_state = self._encode_state(self._grid)

        return next_state, reward, terminated, truncated

    def close_env(self):
        self._mf.end_session()

    def _encode_state(self, grid) -> NDArray:
        """Separates the minefield into  matrixes, other contains numbered cells, other the unopened cells (and possible flags)"""

        encoded = np.zeros((2, self._height, self._width), dtype=np.float32)
        grid = np.array(grid)
        encoded[0] = np.where(grid < 9, grid / 8.0, 0)
        encoded[1] = (grid == CellState.UNOPENED.num()).astype(np.float32)

        if self._flags_allowed:
            encoded = np.append(encoded, [(grid == CellState.FLAG.num()).astype(np.float32)], axis=0)

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
    def __init__(self, flags_allowed: bool):
        super().__init__()

        in_channels = 3 if flags_allowed else 2
        out_channels = 2 if flags_allowed else 1

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

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
        epsilon: Callable[[int], float],
        gamma: float,
        lr: float,
        w_interval: int,
        flags_allowed: bool,
        save_model: bool,
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
        self._flags_allowed = flags_allowed
        self._save_model = save_model

        self._env = MinesweepperEnv(*env_args, flags_allowed=flags_allowed, **env_kwargs)

        self._set_rnd_seed()

        self._n_actions = self._env.n_actions
        self._dim_obs_space = self._env.obs_dim

        self._policy_net = ConvolutionalNet(flags_allowed)
        self._target_net = ConvolutionalNet(flags_allowed)

        self._update_target_net()

        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=self._lr, amsgrad=True)
        self._steps_taken = 0
        self._episodes_ran = 0

        self._flagging_prob = self._env._n_mines / (self._env._width * self._env._height)

        self._plotter = Plotter()

    def save_model(self):
        if not self._save_model:
            return
        state = {
            "episodes_ran": self._episodes,
            "state_dict_policy": self._policy_net.state_dict(),
            "state_dict_target": self._target_net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "flags_allowed": self._flags_allowed,
        }
        timestamp = datetime.now().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
        torch.save(state, os.path.join("models", f"model_{timestamp}.pt"))

    def load_model(self, filepath):
        state = torch.load(filepath)
        self._policy_net.load_state_dict(state["state_dict_policy"])
        self._target_net.load_state_dict(state["state_dict_target"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._episodes_ran = state["episodes_ran"]
        self._flags_allowed = state["flags_allowed"]

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
            # Take random action

            unopened_cells = state.squeeze(0)[1]
            viable_xy = unopened_cells.nonzero()
            action_xy = viable_xy[np.random.choice(len(viable_xy))]
            x = int(action_xy[1].item())
            y = int(action_xy[0].item())

            if self._flags_allowed and np.random.random() < self._flagging_prob:
                ind = self._env.actions.index((Op.FLAG, x, y))  # type: ignore
            else:
                ind = self._env.actions.index((Op.OPEN, x, y))  # type: ignore

            return torch.tensor([[ind]], dtype=torch.int32, device=self._device)

        # Choose the action that maximises the future rewards according to the policy network
        with torch.no_grad():
            output: torch.Tensor = self._policy_net(state).squeeze(0)

        mask_not_unopened = state.squeeze(0)[1].bool().logical_not_().flatten()

        if self._flags_allowed:
            mask_not_unopened = torch.cat((mask_not_unopened, mask_not_unopened))

        output.masked_fill_(mask_not_unopened, float("-inf"))
        return output.max(0).indices.view(1, 1)

    def _optim_batch(self, memory: Memory):
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
