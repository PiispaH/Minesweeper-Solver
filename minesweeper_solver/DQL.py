from collections import namedtuple
from enum import Enum
from itertools import product
import random
from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .minefield import CellState, GameState, MineField
from .utils import FlattenedGrid


class Operations(Enum):
    OPEN = 0
    FLAG = 1


Action = namedtuple("Action", ["op", "x", "y"])
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
        self._mines = self._mf._n_mines

        self.actions = [
            Action(Operations.OPEN, x, y)
            for x, y in product(range(self._width), range(self._height))
            # Action(op, x, y) for op, x, y in product(list(Operations), range(self._width), range(self._height))
        ]
        self.n_actions = len(self.actions)
        self.obs_dim = self._height * self._width

        # These need to be reset
        self._current_state = FlattenedGrid([[]])
        self._update_current_env_state()
        self._clicks = 0

    def _index_of_cell_at(self, x: int, y: int):
        return self._height * x + y

    def _update_single_cell(self, x: int, y: int):
        self._current_state[x, y] = self._mf.get_cell_state(x, y).num()

    def _update_current_env_state(self):
        """Returns the state of the environment as a vector"""
        self._current_state = FlattenedGrid(self._mf.get_grid())

    def _get_cell_state(self, x: int, y: int):
        return self._mf.get_cell_state(x, y)

    def reset(self):
        """Restarts the game."""
        if self._state:
            self._mf._import_gamestate()
        else:
            self._mf.restart()
        self._clicks = 0
        return self.encode_board()

    def _open_cell(self, x, y):
        """Opens the given cell. Returns whether the uncovered cell is empty (cell 0)"""
        self._mf.open_cell(x, y)

    def _flag_cell(self, x: int, y: int):
        self._mf.flag_cell(x, y)

    def _mines_left(self):
        """Returns the amount of mines left in the field by the UI counter"""
        return self._mf.get_mines()

    def step(self, action_ind: int):
        """Makes the given action and calculate the reward.

        Logic for the reward:

        Every click on a cell that isn't unopened gives minus one point.
        Every click on a unopened cell gets one point.
        Win gets 100 points
        """

        self._clicks += 1

        reward = 0

        action = self.actions[action_ind]
        x = action.x
        y = action.y
        op = action.op

        cell_state = self._get_cell_state(x, y)

        # whole_grid_update = False

        if op == Operations.OPEN:

            if cell_state == CellState.UNOPENED:
                # Give points for opening a unopened cell (even if it was a bomb).
                reward += 5
                # whole_grid_update = self._open_cell(x, y)
                self._open_cell(x, y)
            else:
                # Give minus points since this is not a valid move...
                reward -= 1

        # elif op == Operations.FLAG:
        #     if cell_state == CellState.UNOPENED:
        #         pass
        #     else:
        #         # Give minus points since either this is not a valid move or just repetition
        #         reward -= 1
        #
        #     self._flag_cell(x, y)

        game_state = self._mf.get_gamestate()

        # Check if terminated
        terminated = False
        if game_state == GameState.WIN:
            terminated = True
            reward += 100
        elif game_state == GameState.LOST:
            terminated = True
            reward -= 50

        # Check if truncated
        truncated = False
        if self._clicks == self._height * self._width:
            truncated = True

        # if not terminated and whole_grid_update:
        #     # Only do this when needed, it's costly...
        #     self._update_current_env_state()
        #     # mines_left = self._mf.get_mines() # this can be also optimised by counting the flags and mines ourself

        return self.encode_board(), reward, terminated, truncated

    def close_env(self):
        self._mf.end_session()

    def encode_board(self):

        encoded = np.zeros((2, self._height, self._width), dtype=np.float32)

        grid = np.array(self._mf.get_grid())

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

        # Basic convolutional feature extractor
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        # Output layer:
        # 1x1 conv so every spatial location corresponds to an action
        # num_actions_per_cell = 1 for "open"
        # num_actions_per_cell = 2 for "open" + "flag"
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        qmap = self.out(x)  # shape (batch, A, H, W)

        # Flatten to (batch, A*H*W)
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

        self._env = MinesweepperEnv(*env_args, state=state, **env_kwargs)

        self._set_rnd_seed()

        self._n_actions = self._env.n_actions
        self._dim_obs_space = self._env.obs_dim

        self._policy_nn = ConvolutionalNet()
        self._target_nn = ConvolutionalNet()

        self._update_target_net()

        self._optimizer = optim.AdamW(self._policy_nn.parameters(), lr=self._lr, amsgrad=True)
        self._steps_taken = 0

        self._plotter = Plotter()

    def _update_target_net(self):
        """Updates the target net to match the policy net"""
        self._target_nn.load_state_dict(self._policy_nn.state_dict())

    def _set_rnd_seed(self):
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # Somehow set the seed for the minesweepper here... (doubt thats gonna happen :D)

    def _select_action(self, state):
        """Chooses an action in the given state using an epsilon greedy strategy."""
        if np.random.random() < self._epsilon(self._steps_taken):
            # Random action
            return torch.tensor([[np.random.choice(range(self._n_actions))]], dtype=torch.int32, device=self._device)
        # Choose the action that maximises the future rewards according to the policy network
        with torch.no_grad():
            return self._policy_nn(state).max(1).indices.view(1, 1)

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
        Q_i = self._policy_nn(states).gather(1, actions)

        # compute the actual scores of the next states
        mask = torch.tensor([i is not None for i in batch.next_state], dtype=torch.bool)
        target = rewards
        with torch.no_grad():
            target[mask] += self._gamma * self._target_nn(next_states).max(1).values

        # Take one step towards the negative gradient
        loss_fn = F.huber_loss(Q_i.squeeze(1), target)
        self._optimizer.zero_grad()
        loss_fn.backward()
        self._optimizer.step()

    def run(self):
        """Runs the model"""
        memory = Memory()
        episode_scores = []
        episode_clicks = []
        for _ in range(self._episodes):
            state = self._env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)

            terminated, truncated = False, False

            cum_rewards = 0

            for _ in range(self._env._width * self._env._height):
                self._steps_taken += 1
                # Select an action, take it and store the transition
                action = self._select_action(state)
                observation, reward, terminated, truncated = self._env.step(action.item())  # type: ignore

                cum_rewards += reward

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
                    episode_scores.append(cum_rewards)
                    episode_clicks.append(self._env._clicks)
                    self._plotter.update(episode_scores)
                    break

        self._plotter.finalise()
        return episode_scores


class Plotter:
    def __init__(self):
        plt.ion()
        self.mav = []

    def finalise(self):
        plt.ioff()

    def update(self, scores):

        plt.figure(1)
        plt.clf()
        plt.title("Result")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.plot(scores)

        mav = scores[-30:]
        mav = sum(mav) / len(mav)
        self.mav.append(mav)

        plt.plot(self.mav)

        plt.pause(0.001)


if __name__ == "__main__":
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
        a = agent.run()
    except Exception as e:
        agent._env.close_env()
        raise e

    plt.ioff()
    plt.show()
