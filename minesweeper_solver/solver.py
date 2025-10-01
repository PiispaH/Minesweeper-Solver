from time import time
from .minefield import Minefield
from .minesweeper_ui import MinesweeperUI


class BaseSolver:
    """Base class for different solvers"""

    def __init__(self, minefield: Minefield, ms_ui: MinesweeperUI):
        self._minefield = minefield
        self._ms_ui = ms_ui

    def _open_cell(self, i: int, j: int):
        """Opens the cell with the given indices"""
        cell = self._minefield.cell_at(i, j)
        self._ms_ui.click_at_pos(*cell.screen_pos, 0.4)

    def _check_if_won(self):
        return not bool(self._minefield.unopened)

    def _new_game(self):
        self._minefield.prepare_new_minefield()
        self._ms_ui.click_at_pos(*self._ms_ui.smiley, 1.0)

    def run(self) -> float:
        """Starts up the solver"""
        return self._run()

    def _run(self) -> float:
        """This method should be overwritten to provide the solver logic"""
        raise NotImplementedError


class Solver(BaseSolver):
    """Clicks safe cells and if there aren't any, opens the one that is least likely to be a bomb."""

    def _run(self):

        self._open_cell(self._minefield.width // 2, self._minefield.height // 2)

        start = time()
        while True:
            self._minefield.update_minefield()
            if self._check_if_won():
                break

            # TODO:
            # Determine safety of unopened cells
            # If safe cells:
            #   Open the safe ones. Also mark certain bomb cells
            #   as bombs and add them to the don't check list
            # If no safe cells:
            #   Open one that has the lowest bomb ratio
            #   After the random selection, have to check if the game has been lost
        return time() - start


class SolverRandom(BaseSolver):

    def _run(self):
        """Just randomly clicks"""
        from random import choice

        start = time()
        while True:
            cell = self._minefield.cell_at(*choice(list(self._minefield.unopened)))
            self._ms_ui.click_at_pos(*cell.screen_pos)
            self._minefield.update_minefield()
            if self._ms_ui.check_if_lost():
                self._new_game()
                start = time()
                continue
            if self._check_if_won():
                print("How did this happen...")
                break
        return time() - start
