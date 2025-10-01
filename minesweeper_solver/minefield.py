from __future__ import annotations
from copy import deepcopy
from enum import Enum
from itertools import product
from typing import Any, Dict, List, Set, Tuple
import numpy as np
from numpy.typing import NDArray
from .minesweeper_ui import MinesweeperUI

np.set_printoptions(linewidth=200)  # For printing the minefield


class CellState(Enum):
    """Enumeration for the possible states of the cells"""

    WALL = -1  # Not an actual cell but a wall
    OPENED = 0
    NEAR_1 = 1
    NEAR_2 = 2
    NEAR_3 = 3
    NEAR_4 = 4
    NEAR_5 = 5
    NEAR_6 = 6
    NEAR_7 = 7
    NEAR_8 = 8
    UNOPENED = 9  # This is the lighter rim color, the center is the same as opened
    BOMB = 10


COLOR_TO_STATE = {
    (170, 170, 170): CellState.UNOPENED,
    (0, 0, 170): CellState.NEAR_1,
    (0, 82, 0): CellState.NEAR_2,
    (170, 0, 0): CellState.NEAR_3,
    (0, 0, 82): CellState.NEAR_4,
    (82, 0, 0): CellState.NEAR_5,
    (0, 82, 82): CellState.NEAR_6,
    (0, 0, 0): CellState.NEAR_7,
    (126, 126, 126): CellState.OPENED,
}
"""Maps the RGB value to a cell state"""


class Cell:
    """Represents a single cell in the minefield"""

    def __init__(self, grid_position: Tuple[int, int], screen_coords: Tuple[int, int]):
        self._grid_position = grid_position
        self._screen_coords = screen_coords
        self.state = CellState.UNOPENED

    @property
    def screen_pos(self) -> Tuple[int, int]:
        return self._screen_coords

    def __repr__(self) -> str:
        return f"{self.state.value}"


class Wall(Cell):
    """A cell that represents a wall of the minefield"""

    def __init__(self, grid_position: Tuple[int, int], screen_coords: Tuple[int, int]):
        super().__init__(grid_position, screen_coords)
        self.state = CellState.WALL


class Minefield:
    """Holds all the cells in the minefield"""

    def __init__(self, ul: Tuple[int, int], rows: int, columns: int, mines: int, ms_ui: MinesweeperUI):
        self._ul_x = ul[0]
        self._ul_y = ul[1]
        self.height = rows
        self.width = columns
        self.mines = mines
        self._cell_size = 32
        self._ms_ui = ms_ui
        self._grid = self._init_minefield()
        self._unopened = self._init_unopened_cells()
        self._neighbours: Dict[Tuple[int, int], NDArray[Any]] = {}  # The cells surrounding a point

    @property
    def grid(self) -> np.ndarray:
        return deepcopy(self._grid)

    @property
    def unopened(self) -> Set[tuple[int, int]]:
        """Cells that are still unopened"""
        return deepcopy(self._unopened)

    def prepare_new_minefield(self):
        """Resets the minefield"""
        self._grid = self._init_minefield()
        self._unopened = self._init_unopened_cells()

    def cell_at(self, i: int, j: int) -> Cell:
        return self._grid[j + 1][i + 1]

    def update_minefield(self):
        """Updates the minefield to reflect the latest changes"""

        self._ms_ui.take_a_screenshot()

        dont_check = self._update_cell_status()
        self._unopened = self._unopened.difference(dont_check)

    def unopened_near(self, i: int, j: int) -> int:
        """Returns the amount of unopened cells nearby"""
        nbs = np.sum([[1 if x.state == CellState.UNOPENED.value else 0 for x in y] for y in self.neighbours(i, j)])
        nbs -= 1 if self.cell_at(i, j).state == CellState.UNOPENED else 0
        return nbs

    def bomb_ratio(self, i: int, j: int) -> float:
        """The naive likelyhood of a random click on a neighbor ending up as a bomb"""
        cell = self.cell_at(i, j)
        if cell.state == CellState.UNOPENED:
            ratio = float("inf")
        else:
            ratio = float(cell.state.value / self.unopened_near(i, j))
        return ratio

    def neighbours(self, i: int, j: int) -> NDArray[Any]:
        """Returns a 3x3 matrix of the neighbours surrounding the given cell."""
        nbs = self._neighbours.get((i, j), np.empty((3, 3), dtype=object))
        if not nbs.any():
            nbs = np.empty((3, 3), dtype=object)
            for di, dj in product((-1, 0, 1), repeat=2):
                nbs[dj][di] = self.cell_at(i + di, j + dj)
        return nbs

    def _init_unopened_cells(self) -> Set[Tuple[int, int]]:
        """Returns a set of fresh unopened indices"""
        return set(product(range(self.width), range(self.height)))

    def _init_minefield(self) -> np.ndarray[Any]:
        """Creates the initial minefield with walls and unopened cells"""
        grid: List[List[Cell]] = []
        for j in range(-1, self.height + 1):
            row: List[Cell] = []
            for i in range(-1, self.width + 1):
                if i < 0 or j < 0 or i == self.width or j == self.height:
                    row.append(Wall((i, j), (-1, -1)))
                else:
                    row.append(Cell((i, j), self._cell_center_coords(i, j)))
            grid.append(row)
        return np.array(grid)

    def _cell_center_coords(self, i: int, j: int) -> Tuple[int, int]:
        """Gives the pixel coordinates of the center of the cell with index i, j"""
        return self._ul_x + i * self._cell_size, self._ul_y + j * self._cell_size

    def _update_cell_status(self) -> Set[Tuple[int, int]]:
        no_need_to_check: Set[Tuple[int, int]] = set()

        for i, j in self._unopened:
            color = self._ms_ui.get_pixel_color(i, j)
            if COLOR_TO_STATE[color] == CellState.OPENED:  # Could still be both opened or unopened
                rim_color = COLOR_TO_STATE[self._ms_ui.get_rim_color(i, j)]
                if rim_color == CellState.OPENED:
                    # Highly likely empty, but could also be a seven.
                    self.cell_at(i, j).state = COLOR_TO_STATE[self._ms_ui.get_slight_center_offset_color(i, j)]
                else:
                    # Is unopened, so no action is taken
                    continue
            else:
                # A number was found
                try:
                    self.cell_at(i, j).state = COLOR_TO_STATE[color]
                except KeyError:
                    print("new color found:")
                    print(i, j, color)
                    raise KeyError

            no_need_to_check.add((i, j))
        return no_need_to_check
