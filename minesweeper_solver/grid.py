from copy import deepcopy
from enum import Enum
from itertools import product
from typing import Set, Tuple
import numpy as np
from .minesweeper_ui import MinesweeperUI

np.set_printoptions(linewidth=200)  # For printing the minefield


class CellState(Enum):
    """Enumeration for the possible states of the cells"""

    BOMB = -1
    UNOPENED = 0  # This is the lighter rim color, the center is the same as opened
    NEAR_1 = 1
    NEAR_2 = 2
    NEAR_3 = 3
    NEAR_4 = 4
    NEAR_5 = 5
    NEAR_6 = 6
    NEAR_7 = 7
    NEAR_8 = 8
    OPENED = 9


COLOR_TO_STATE = {
    (126, 126, 126): CellState.OPENED,
    (170, 170, 170): CellState.UNOPENED,
    (0, 0, 170): CellState.NEAR_1,
    (0, 82, 0): CellState.NEAR_2,
    (170, 0, 0): CellState.NEAR_3,
    (0, 0, 82): CellState.NEAR_4,
    (82, 0, 0): CellState.NEAR_5,
}
"""Maps the RGB value to a cell state"""


class Cell:
    """Represents a single cell in the minefield"""

    def __init__(self, grid_position: Tuple[int, int], screen_coords: Tuple[int, int]):
        self._grid_position = grid_position
        self._screen_coords = screen_coords
        self._state = CellState.UNOPENED
        self._unopened_near = 8

    @property
    def screen_pos(self) -> Tuple[int, int]:
        return self._screen_coords

    def __repr__(self) -> str:
        return f"{self._state.value}"

    @property
    def state(self) -> CellState:
        return self._state

    @state.setter
    def state(self, value: CellState):
        self._state = value


class Minefield:
    """Holds all the cells in the minefield"""

    def __init__(self, ul: Tuple[int, int], rows: int, columns: int, ms_ui: MinesweeperUI):
        self._ul_x = ul[0]
        self._ul_y = ul[1]
        self._cell_size = 32
        self._grid = np.array(
            [[Cell((i, j), self._cell_center_coords(i, j)) for i in range(columns)] for j in range(rows)], dtype=Cell
        )
        self._ms_ui = ms_ui
        self._need_to_check = set(product(range(columns), range(rows)))

    @property
    def grid(self) -> np.ndarray:
        return deepcopy(self._grid)

    def cell_at(self, i: int, j: int) -> Cell:
        return self._grid[j][i]

    def _cell_center_coords(self, i: int, j: int) -> Tuple[int, int]:
        """Gives the pixel coordinates of the center of the cell with index i, j"""
        return self._ul_x + i * self._cell_size, self._ul_y + j * self._cell_size

    def update_cell_states(self):
        """Update the status of the cells"""

        self._ms_ui.take_a_screenshot()
        no_need_to_check: Set[Tuple[int, int]] = set()

        for x, y in self._need_to_check:
            color = self._ms_ui.get_pixel_color(x, y)
            if COLOR_TO_STATE[color] == CellState.OPENED:  # Could still be oth opened or unopened
                if COLOR_TO_STATE[self._ms_ui.get_rim_color(x, y)] == CellState.OPENED:
                    self._grid[y][x].state = CellState.OPENED
                else:
                    # Still unopened, so no action is taken
                    continue
            else:
                # Some number was found
                self._grid[y][x].state = COLOR_TO_STATE[color]

            no_need_to_check.add((x, y))

        self._need_to_check = self._need_to_check.difference(no_need_to_check)
