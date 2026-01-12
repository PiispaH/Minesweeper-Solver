from collections import namedtuple
from datetime import datetime
from functools import wraps
from itertools import product
import os
import random
from typing import Callable, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from minesweeper import Action, CellState, GameState, Interaction, MinesweeperHeadless


def exit_training(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        except (Exception, KeyboardInterrupt) as e:
            self.save_model()
            raise e

    return wrapper


ActionTuple = namedtuple("Action", ["op", "x", "y"])
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])


class MinesweeperEnv:
    """Provides the environment for the RL agent"""

    def __init__(
        self,
        height: int = 9,
        width: int = 9,
        n_mines: int = 10,
        rnd_seed: Union[int, None] = None,
        flags_allowed: bool = False,
    ):
        self._ms = MinesweeperHeadless(width, height, n_mines, rnd_seed=rnd_seed)

        self._height = height
        self._width = width
        self._n_mines = n_mines

        self._flags_allowed = flags_allowed

        self._available_actions = [Action.OPEN, Action.FLAG] if self._flags_allowed else [Action.OPEN]

        self.actions = [
            Interaction(x, y, act)
            for act, y, x in product(self._available_actions, range(self._height), range(self._width))
        ]
        self.n_actions = len(self.actions)

        # These change
        self._grid = 9 * np.ones((height, width), dtype=np.int64)
        self._visible = 9 * np.ones((height, width), dtype=np.int64)
        self._n_actions_taken = 0

    @property
    def unopened(self) -> NDArray:
        return self._ms._unopened

    @property
    def flagged(self) -> NDArray:
        return self._ms._flagged

    @property
    def gamestate(self) -> GameState:
        return self._ms.gamestate

    def take_action(self, act: Interaction) -> Tuple[int, int]:
        """Takes the specified action and returns the state of the affected cell before and after"""
        if act.action not in self._available_actions:
            raise ValueError(f"inproper action: {act.action}")

        cell_before = self._visible[act.y][act.x]

        if self.gamestate == GameState.NOT_STARTED:
            self._ms.make_interaction(act)
            for j, row in enumerate(self._ms.get_grid()):
                for i, value in enumerate(row):
                    self._grid[j][i] = value.num()
        elif self.gamestate == GameState.PLAYING:
            self._ms.make_interaction(act)

        if self.gamestate == GameState.LOST:
            return cell_before, CellState.MINE.num()

        cell_before = self._visible[act.y][act.x]

        self._visible = np.where(self.unopened, CellState.UNOPENED.num(), self._grid)
        self._visible = np.where(self.flagged, CellState.FLAG.num(), self._visible)
        self._state = np.concat([self._visible, self.unopened, self.flagged])

        cell_after = self._visible[act.y][act.x]

        return cell_before, cell_after

    def reset(self):
        """Restarts the game."""
        self._ms._new_game()
        self._n_actions_taken = 0
        self._grid.fill(CellState.UNOPENED.num())
        self._visible.fill(CellState.UNOPENED.num())
        self._game_state = GameState.PLAYING
        return self._encode_state()

    def step(self, action_ind: int):
        """Makes the given action and calculates the corresponding reward"""

        self._n_actions_taken += 1

        reward = -1.0

        action = self.actions[action_ind]
        act = action.action
        x = action.x
        y = action.y

        nbr_cells = [self._visible[x][y] for x, y in self._ms._nbr_inds(x, y)]
        nbr_unique = np.unique(nbr_cells)

        cell_before, cell_after = self.take_action(action)
        next_state = self._encode_state()
        terminated = next_state is None

        if cell_after == cell_before:
            raise Exception(
                f"Useless selection: x={x}, y={y}, act={act} on cell {cell_before}."
                f"Happened on action Nr. {self._n_actions_taken}."
            )

        if cell_before == CellState.FLAG and act == Action.FLAG:
            raise Exception(f"Tried to deflag x={x}, y={y}, this shouldn't be possible.")

        if act == Action.OPEN and cell_after not in (cell_before, CellState.MINE):
            if len(nbr_unique) == 1:
                # Random guess
                reward -= 5.0
            else:
                # Atleast somewhat informed choice
                reward += 2.0

        if cell_after == CellState.FLAG:
            if self._n_mines < len(np.transpose(next_state[2].nonzero())):
                # Flagged too many cells
                terminated = True
                reward -= 20.0
            elif self._ms.get_grid()[y][x] == CellState.MINE:
                # Flag placed correctly
                reward += 2.0
            else:
                # Incorrectly flagged cell
                reward -= 5.0

        # Check if win or loss
        if self.gamestate == GameState.WON:
            print("BIG WIN !!!")
            terminated = True
            reward += 10.0
        elif self.gamestate == GameState.LOST:
            terminated = True
            next_state = None
            reward -= 10.0

        # Check if truncated
        truncated = False
        if self._n_actions_taken == self._height * self._width - self._n_mines:
            truncated = True

        return next_state, reward, terminated, truncated

    def _encode_state(self) -> NDArray[np.float32]:
        """Returns the viisble information about the gamestate

        The first matrix in the returned array contains the currently visible numbered cells, the second contains the unopened cells,
        and the third one contains the flagged cells.
        """
        encoded = np.zeros((2, self._height, self._width), dtype=np.float32)
        encoded[0] = np.where((self._grid < 9) & np.logical_not(self.unopened), self._grid / 8.0, 0)
        encoded[1] = self.unopened.astype(np.float32)

        if self._flags_allowed:
            encoded = np.append(encoded, [self.flagged.astype(np.float32)], axis=0)  # type: ignore

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

        seed = 42

        self._env = MinesweeperEnv(*env_args, rnd_seed=seed, flags_allowed=flags_allowed)

        self._set_rnd_seed(seed)

        self._n_actions = self._env.n_actions

        self._policy_net = ConvolutionalNet(flags_allowed)
        self._target_net = ConvolutionalNet(flags_allowed)

        self._update_target_net()

        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=self._lr, amsgrad=True)
        self._steps_taken = 0
        self._episodes_ran = 0

        self._flagging_prob = self._env._n_mines / (self._env._width * self._env._height) if flags_allowed else 0.0

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

    def _set_rnd_seed(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _select_action(self, state: torch.Tensor):
        """Chooses an action in the given state using an epsilon greedy strategy."""
        if np.random.random() < self._epsilon(self._steps_taken):
            # Choose random action

            not_numbered_cells = state.squeeze(0)[1].bool()

            if self._flags_allowed:
                flags = state.squeeze(0)[2].bool()
                unopened_coords = torch.logical_and(not_numbered_cells, flags.logical_not()).nonzero()
            else:
                unopened_coords = not_numbered_cells.nonzero()

            if self._env._n_actions_taken != 0 and np.random.random() < self._flagging_prob:
                act = Action.FLAG
            else:
                act = Action.OPEN

            action_coords = unopened_coords[np.random.choice(len(unopened_coords))]
            x = int(action_coords[1].item())
            y = int(action_coords[0].item())
            ind = self._env.actions.index(Interaction(x, y, act))
            return torch.tensor([[ind]], dtype=torch.int32, device=self._device)

        # Choose the action that maximises the future rewards according to the policy network
        with torch.no_grad():
            output: torch.Tensor = self._policy_net(state).squeeze(0)

        numbered = state.squeeze(0)[1].bool().logical_not().flatten()
        mask = numbered
        if self._flags_allowed:  # Need to also add mask for flag actions
            flags = state.squeeze(0)[2].bool().flatten()
            flag_mask = torch.logical_or(numbered, flags)
            open_mask = torch.logical_or(numbered, flags)
            if self._env._n_actions_taken == 0:  # Can't flag on the first step
                flag_mask = torch.ones(len(flags), dtype=torch.bool)
            mask = torch.cat((open_mask, flag_mask))
        output.masked_fill_(mask, float("-inf"))
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

    @exit_training
    def train(self):
        """Trains the model"""
        memory = Memory()
        episode_scores = []
        episode_clicks = []
        for _ in range(self._episodes):
            state = self._env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)

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
                    episode_clicks.append(self._env._n_actions_taken)
                    self._plotter.update(episode_scores)
                    break

            self._episodes_ran += 1

        self._plotter.finalise()
        self.save_model()


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

        if len(scores) > 50:
            mav = scores[-50:]
            mav = sum(mav) / len(mav)

            if len(self.mav) == 0:
                self.mav = [mav for _ in range(50)]

            self.mav.append(mav)

            plt.plot(self.mav)

        plt.pause(0.001)
