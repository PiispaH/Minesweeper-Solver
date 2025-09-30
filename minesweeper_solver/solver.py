from .grid import Minefield
from .minesweeper_ui import MinesweeperUI


class Solver:
    """The solver that decides what to do and acts based on the decisions"""

    def __init__(self, minefield: Minefield, ms_ui: MinesweeperUI):
        self._minefield = minefield
        self._ms_ui = ms_ui

    def open_cell(self, i: int, j: int):
        """Opens the cell with the given indices"""
        cell = self._minefield.cell_at(i, j)
        self._ms_ui.click_at_pos(*cell.screen_pos)
        self._minefield.update_cell_states()
