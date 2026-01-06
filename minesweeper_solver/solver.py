from itertools import product
import os
from random import choice, randint
from time import time
from typing import Set, Tuple
import numpy as np
from numpy.typing import NDArray
import torch
from .DQL import ConvolutionalNet
from .minefield import CellState, SmileyState, MineField


class SolverBase:
    """Base class for different solvers"""

    def __init__(self, width: int = 30, height: int = 16, mines: int = 99, state: str = "", headless: bool = False):
        # If nothin given, go with expert board.
        if state:
            self._mf = MineField(headless, state=state)
        else:
            self._mf = MineField(headless, width, height, mines)

        self._headless = headless

    def _open_cell(self, x: int, y: int):
        """Opens the cell with the given indices"""
        self._mf.open_cell(x, y)

    def _flag_cell(self, x: int, y: int):
        """Flags the cell with the given indices"""
        self._mf.flag_cell(x, y)

    def _check_if_won(self):
        return self._mf.get_gamestate() == SmileyState.WIN

    def _check_if_lost(self):
        return self._mf.get_gamestate() == SmileyState.LOST

    def _new_game(self):
        self._mf.restart()
        self._open_cell(self._mf.width // 2, self._mf.height // 2)

    def run(self, max_tries: int) -> float:
        """Starts up the solver

        Args:
            max_tries: The amount of games the solver will play before quitting. No limit with value 0.
        """
        self._open_cell(self._mf.width // 2, self._mf.height // 2)

        tries = 0
        start = time()
        while True:
            if tries == max_tries:
                print("Didn't win this time.")
                return 0.0

            self._step()

            if self._check_if_lost():
                tries += 1
                start = time()
                self._new_game()
            elif self._check_if_won():
                print("Win!!!")
                return time() - start

    def _step(self):
        """This method should be overwritten to provide the solver logic"""
        raise NotImplementedError

    def quit(self):
        self._mf.end_session()


class SolverRandom(SolverBase):

    def _step(self):
        """Just randomly clicks"""

        self._open_cell(randint(0, self._mf.width), randint(0, self._mf.height))


class SolverNaive(SolverBase):
    """Clicks safe cells and if there aren't any, opens the one that is least likely to be a mine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_need_to_check = set()  # A set of indices that contain no information anymore
        self._current_grid = self._mf.get_grid()
        self._odds = 1

    def _new_game(self):
        super()._new_game()
        self._no_need_to_check = set()
        self._current_grid = self._mf.get_grid()
        self._odds = 1
        print()

    def _check_if_won(self):
        won = super()._check_if_won()
        if won:
            print(f"Odds of winning this game: {self._odds*100:.2f}%")
        return won

    def _informative_numbered_cells(self):
        """Returns the indices of cells with values 1-8 that have larger amount of unopened near than their value."""
        certain_safe = set()
        certain_bombs = set()
        maybe = set()

        for y, row in enumerate(self._current_grid):
            for x, c in enumerate(row):
                if (x, y) in self._no_need_to_check:
                    continue
                elif not 0 < c < 10:  # Empty cells out
                    self._no_need_to_check.add((x, y))
                elif c == self._number_of_cells_near(x, y, CellState.FLAG.num()):
                    self._no_need_to_check.add((x, y))
                    certain_safe.add((x, y))
                elif c == self._number_of_cells_near(x, y, CellState.UNOPENED.num()) + self._number_of_cells_near(
                    x, y, CellState.FLAG.num()
                ):
                    self._no_need_to_check.add((x, y))
                    certain_bombs.add((x, y))
                elif c == 9:  # No to unopened
                    continue
                else:
                    maybe.add((x, y))

        return certain_safe, certain_bombs, maybe

    def _select_the_safest_bet(self, inds: Set[Tuple[int, int]]):
        lowest = (1.0, -1, -1)
        for ind in inds:
            ratio = (self._neighbourhood_mine_ratio(*ind), *ind)
            lowest = min(ratio, lowest, key=lambda x: x[0])

        if lowest[1] == -1:
            raise ValueError

        unopened = list(self._get_nbr_inds_of_types(*lowest[1:], CellState.UNOPENED))
        rndm = choice(unopened)
        print(f"With {lowest[0]:.3f} probability of failure, opening cell {rndm}")
        self._odds *= 1 - lowest[0]
        self._open_cell(*rndm)

    def _neighbourhood_mine_ratio(self, x: int, y: int) -> float:
        """The naive likelyhood of a random click on a neighbour ending up as a mine"""

        cell_value = self._current_grid[y][x]

        if cell_value == 0 or 8 < cell_value:
            raise ValueError(f"Cannot compute the mine ratio near cell at x={x}, y={y}")

        mines_found = self._number_of_cells_near(x, y, CellState.FLAG.num())
        unopened = self._number_of_cells_near(x, y, CellState.UNOPENED.num())
        return float((cell_value - mines_found) / unopened)

    def _number_of_cells_near(self, x: int, y: int, cellvalue: int):
        """Returns the amount of the given celltype in the 3x3 are surrounding the given coordinates"""
        arr = self._neighbours(x, y)
        nbs = -1 if arr[1][1] == cellvalue else 0
        for row in arr:
            for c in row:
                if c == cellvalue:
                    nbs += 1
        return nbs

    def _get_nbr_inds_of_types(self, x: int, y: int, celltype: CellState) -> Set[Tuple[int, int]]:
        """Returns a set of the neighbouring indices with the given celltype"""
        inds: Set[Tuple[int, int]] = set()
        for dy, row in enumerate(self._neighbours(x, y), start=-1):
            for dx, nbr in enumerate(row, start=-1):
                if (dy == dx == 0) or (nbr != celltype.num()):
                    continue
                inds.add((x + dx, y + dy))
        return inds

    def _neighbours(self, x: int, y: int) -> NDArray[np.int32]:
        """Returns a 3x3 matrix of the neighbours surrounding the given cell."""
        nbs = np.empty((3, 3), dtype=np.int32)
        for dx, dy in product((-1, 0, 1), repeat=2):
            i = x + dx
            j = y + dy

            if -1 in (i, j) or i == self._mf.width or j == self._mf.height:
                value = CellState.WALL.num()
            else:
                value = self._current_grid[y + dy][x + dx]

            nbs[dy + 1][dx + 1] = value

        return nbs

    def _handle_all_mines(self, certain_inds: Set[Tuple[int, int]]):
        """Handles cases where all unopeded neighbour cells are mines

        Args:
            certain_inds: A set of coordinates that mark cells whichs surrouning unopened cells are to be flagged
        """

        flagged = set()

        for x, y in certain_inds:
            for i, j in self._get_nbr_inds_of_types(x, y, CellState.UNOPENED):
                if (i, j) not in flagged:
                    self._flag_cell(i, j)
                    flagged.add((i, j))

    def _handle_safe_cells(self, certain_inds: Set[Tuple[int, int]]):
        """Opens neighbours of cells that have all mines already descovered.

        Args:
            inds: indicises to check if can be opened
        """

        for x, y in certain_inds:
            safe_inds = self._get_nbr_inds_of_types(x, y, CellState.UNOPENED)
            for i, j in safe_inds:
                self._open_cell(i, j)
                if self._check_if_lost():
                    raise SystemExit(f"Lost due to a mistake near cell x={i} y={j}.")

    def _step(self):
        """Looks at 3x3 neighbourhoods to open or flag, if that is not possible, opens the safest cell by chance."""

        self._current_grid = self._mf.get_grid()
        possibly_safe, certain_minefields, maybe = self._informative_numbered_cells()
        self._handle_all_mines(certain_minefields)
        self._handle_safe_cells(possibly_safe)
        changed = bool(possibly_safe | certain_minefields)
        if not changed:
            self._select_the_safest_bet(maybe)


class SolverDQL(SolverBase):
    """A reinforcement learning model trained with a 9x9 grid with 10 mines, aka beginner mode."""

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found. Train a DQL model first.")

        state = torch.load(os.path.join(filepath))
        self._flags_allowed = state["flags_allowed"]

        self._cnn = ConvolutionalNet(self._flags_allowed)
        self._cnn.load_state_dict(state["state_dict_policy"])
        self._actions = [(x, y) for y, x in product(range(kwargs["width"]), range(kwargs["height"]))]

    def _step(self):
        grid = np.array(self._mf.get_grid())
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
            self._open_cell(action[0], action[1])
        except IndexError:
            n_cells = grid.shape[0] * grid.shape[1]
            action = self._actions[a - n_cells]
            self._flag_cell(action[0], action[1])
