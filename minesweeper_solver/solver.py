from itertools import product
import os
import random
import time
from typing import Set, Tuple, Union
from minesweeper import Action, CellState, GameState, Interaction, Minesweeper
import numpy as np
from numpy.typing import NDArray
import torch
from minesweeper_solver.DQL import ConvolutionalNet


class SolverBase(Minesweeper):
    """Base class for minesweeper solver implementations"""

    def __init__(
        self, width: int, height: int, n_mines: int, tries: int, action_delay=0.0, rnd_seed: int | None = None
    ):
        super().__init__(width, height, n_mines, rnd_seed)

        if rnd_seed is not None:
            random.seed(rnd_seed)

        self._max_tries = tries
        self._current_attempt_n = 0
        self.fps = 60

        self._delay = action_delay
        self._next_action_time = 0.0

    def _get_interaction(self):
        ui_action = super()._get_interaction()
        if ui_action is not None and ui_action.action == Action.EXIT:
            # Makes it possible to exit the program during execution
            return Interaction(-1, -1, Action.EXIT)

        now = time.time()
        if now < self._next_action_time:
            return None

        self._next_action_time = now + self._delay

        if self._current_attempt_n < self._max_tries:
            if self.gamestate != GameState.LOST:
                return self.logic()
            else:
                self._current_attempt_n += 1
                return Interaction(-1, -1, Action.NEW_GAME)

        return Interaction(-1, -1, Action.EXIT)

    def logic(self) -> Union[Interaction, None]:
        raise NotImplementedError


class SolverRandom(SolverBase):
    """Randomly opens cells"""

    def logic(self) -> Union[Interaction, None]:
        x, y = random.randint(0, self._width - 1), random.randint(0, self._height - 1)

        return Interaction(x, y, Action.OPEN)


class SolverNaive(SolverBase):
    """Clicks safe cells and if there aren't any, opens the one that is least likely to be a mine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_need_to_check = set()  # A set of indices that contain no information anymore
        self._odds = [1.0]
        self._actions = []
        self._visible_grid = np.ones((self._height, self._width), dtype=np.int64)
        self._start = time.time()

    def _get_interaction(self):
        """This reimplementation first exhausts the known safe actions and then looks for more"""
        if bool(len(self._actions)):
            return self._actions.pop()
        return super()._get_interaction()

    def _new_game(self):
        super()._new_game()
        self._no_need_to_check.clear()
        self._actions = []
        self._odds = [1.0]
        self._start = time.time()

    def _handle_loss(self):
        print(f"With overall odds of getting this far {self._odds[-2]:.3f}")

    def _handle_win(self):
        super()._handle_win()
        print(f"With overall odds {self._odds[-1]:.3f}")
        print(f"With gametime: {time.time() - self._start:.3f}s")

    def logic(self) -> Union[Interaction, None]:
        """Looks at 3x3 neighbourhoods to open or flag, if that is not possible, opens the safest cell by chance."""
        if self.gamestate == GameState.WON:
            return None

        self._visible_grid = np.array([[value.num() for value in row] for row in self._ui_grid])  # type: ignore

        vals = np.unique(self._visible_grid)

        if len(vals) == 1 and vals[0] == CellState.UNOPENED:
            return Interaction(self._width // 2, self._height // 2, Action.OPEN)

        certain_safe, certain_minefields, maybe = self._informative_numbered_cells()
        self._handle_all_mines(certain_minefields)
        self._handle_safe_cells(certain_safe)

        if len(self._actions) == 0:
            if self._mines_left == 0:  # All mines cleared, just open the rest
                self._handle_safe_cells(maybe)
            else:
                self._select_the_safest_bet(maybe)

        act = self._actions.pop()
        return act

    def _informative_numbered_cells(self):
        """Returns the indices of cells with values 1-8 that have larger amount of unopened near than their value."""
        certain_safe = set()
        certain_bombs = set()
        maybe = set()

        for y, row in enumerate(self._visible_grid):
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
            raise ValueError("Could not figure out the lowest probability of a mine.")

        unopened = list(self._get_nbr_inds_of_types(*lowest[1:], CellState.UNOPENED))
        rndm = random.choice(unopened)
        print(f"With {lowest[0]:.3f} probability of failure, opening cell {rndm}")

        ind = len(self._odds)
        self._odds.append((1 - lowest[0]) * self._odds[ind - 1])

        self._actions.append(Interaction(*rndm, Action.OPEN))

    def _neighbourhood_mine_ratio(self, x: int, y: int) -> float:
        """The naive likelyhood of a random click on a neighbour ending up as a mine"""

        cell_value = self._visible_grid[y][x]

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

            if -1 in (i, j) or i == self._width or j == self._height:
                value = CellState.WALL.num()
            else:
                value = self._visible_grid[y + dy][x + dx]

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
                    self._actions.append(Interaction(i, j, Action.FLAG))
                    flagged.add((i, j))

    def _handle_safe_cells(self, certain_inds: Set[Tuple[int, int]]):
        """Opens neighbours of cells that have all mines already descovered.

        Args:
            inds: indicises to check if can be opened
        """

        for x, y in certain_inds:
            safe_inds = self._get_nbr_inds_of_types(x, y, CellState.UNOPENED)
            for i, j in safe_inds:
                self._actions.append(Interaction(i, j, Action.OPEN))


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
