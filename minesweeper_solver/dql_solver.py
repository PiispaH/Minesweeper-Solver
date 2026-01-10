from itertools import product
import os
import numpy as np
import torch
from minesweeper import Action, CellState, GameState, Interaction
from .DQL import ConvolutionalNet
from .solver import SolverBase


class SolverDQL(SolverBase):
    """A reinforcement learning model trained with a 9x9 grid with 10 mines, aka beginner mode."""

    def __init__(self, *args, filepath="", **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found. Train a DQL model first.")

        state = torch.load(os.path.join(filepath))
        self._flags_allowed = state["flags_allowed"]

        self._cnn = ConvolutionalNet(self._flags_allowed)
        self._cnn.load_state_dict(state["state_dict_policy"])
        self._actions = [(x, y) for y, x in product(range(self._width), range(self._height))]

    def logic(self):
        if self.gamestate == GameState.WON:
            return None

        self._visible_grid = np.array([[value.num() for value in row] for row in self._ui_grid])  # type: ignore

        grid = np.array(self._visible_grid)
        encoded = np.zeros((2, grid.shape[0], grid.shape[1]), dtype=np.float32)
        encoded[0] = np.where(grid < 9, grid / 8.0, 0)
        encoded[1] = (grid == CellState.UNOPENED.num()).astype(np.float32)

        mask = torch.tensor((encoded[1] == 0).flatten(), dtype=torch.bool, device="cpu")
        if self._flags_allowed:
            encoded = np.append(encoded, [(grid == CellState.FLAG.num()).astype(np.float32)], axis=0)
            mask = torch.cat((mask, mask))

        encoded = torch.tensor(encoded, dtype=torch.float32, device="cpu").unsqueeze(0)

        output: torch.Tensor = self._cnn(encoded)
        output.masked_fill_(mask, float("-inf"))
        a = output.max(1).indices.view(1, 1)

        try:
            action = self._actions[a]
            return Interaction(action[0], action[1], Action.OPEN)
        except IndexError:
            n_cells = grid.shape[0] * grid.shape[1]
            action = self._actions[a - n_cells]
            return Interaction(action[0], action[1], Action.FLAG)
