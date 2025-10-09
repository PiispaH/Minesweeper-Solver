from copy import copy
from itertools import product
from time import time
from typing import Set, Tuple
import numpy as np
from .minefield import CellState, Minefield
from .minesweeper_ui import MinesweeperUI
from random import choice


class SolverBase:
    """Base class for different solvers"""

    def __init__(self, minefield: Minefield, ms_ui: MinesweeperUI):
        self._minefield = minefield
        self._ms_ui = ms_ui

    def _open_cell(self, i: int, j: int, dur: float = 0.0):
        """Opens the cell with the given indices"""
        cell = self._minefield.cell_at(i, j)
        self._ms_ui.click_at_pos(*cell.screen_pos, dur)

    def _move_to_cell(self, i: int, j: int, dur: float = 0.0):
        cell = self._minefield.cell_at(i, j)
        self._ms_ui.move_to(*cell.screen_pos, dur)

    def _check_if_won(self):
        return not bool(self._minefield.unopened)

    def _new_game(self):
        self._minefield.prepare_new_minefield()
        self._ms_ui.click_at_pos(*self._ms_ui.smiley, 1.0)
        self._open_cell(self._minefield.width // 2, self._minefield.height // 2)

    def run(self) -> float:
        """Starts up the solver"""
        self._open_cell(self._minefield.width // 2, self._minefield.height // 2)
        return self._run()

    def _run(self) -> float:
        """This method should be overwritten to provide the solver logic"""
        raise NotImplementedError

    def _select_the_safest_bet(self):
        lowest = (1.0, int(-1), int(-1))
        for inds in self._minefield._numbered_cells.values():
            for i, j in inds:
                ratio = (self._mine_ratio(i, j), i, j)
                lowest = min(ratio, lowest, key=lambda x: x[0])
        unopened = list(self._get_nbr_inds_of_types(lowest[1], lowest[2], CellState.UNOPENED))
        rndm = choice(unopened)
        print(f"With {lowest[0]:.3f} probability of failure, opening cell {rndm}")
        self._open_cell(*rndm, 5.0)

    def _mine_ratio(self, i: int, j: int) -> float:
        """The naive likelyhood of a random click on a neighbor ending up as a mine"""
        cell = self._minefield.cell_at(i, j)
        ratio = float(cell.state.value / max(self._number_of_cells_near(i, j, CellState.UNOPENED), 0.01))
        return ratio

    def _number_of_cells_near(self, i: int, j: int, celltype: CellState):
        """Returns the amount of the given celltype in the 3x3 are surrounding the given coordinates"""
        nbs = sum([sum([1 if x.state == celltype else 0 for x in y]) for y in self._minefield.neighbours(i, j)])
        nbs -= 1 if self._minefield.cell_at(i, j).state == CellState else 0
        return nbs

    def _mark_cells_at_inds_as_cellstate(self, inds: Set[Tuple[int, int]], cell_state: CellState):
        """Marks the cells in the given indices as the give state"""
        for i, j in inds:
            self._minefield.cell_at(i, j).state = cell_state

    def _get_nbr_inds_of_types(self, i: int, j: int, celltype: CellState) -> Set[Tuple[int, int]]:
        """Returns a set of the neighboring indices with the given celltype"""
        inds: Set[Tuple[int, int]] = set()
        for dj, row in enumerate(self._minefield.neighbours(i, j), start=-1):
            for di, nbr in enumerate(row, start=-1):
                if (dj == di == 0) or (nbr.state != celltype):
                    continue
                inds.add((di + i, dj + j))
        return inds


class SolverNaive(SolverBase):
    """Clicks safe cells and if there aren't any, opens the one that is least likely to be a mine.

    Uses a helper grid where 50/50 cases get stored. Sometimes misplaces a mine and loses because of that.
    """

    def __init__(self, minefield: Minefield, ms_ui: MinesweeperUI):
        super().__init__(minefield, ms_ui)
        self._joker_id = 1
        self._joker_grid = [[set() for _ in range(minefield.width + 2)] for _ in range(minefield.height + 2)]

    def _new_game(self):
        super()._new_game()
        self._joker_grid = [
            [set() for _ in range(self._minefield.width + 2)] for _ in range(self._minefield.height + 2)
        ]

    def _mark_cells_at_inds_as_cellstate(self, inds: Set[Tuple[int, int]], cell_state: CellState):
        """Marks the cells in the given indices as the give state, also removes the affected jokers"""
        for i, j in inds:
            jokers = copy(self._joker_grid[j + 1][i + 1])
            for joker in jokers:
                for di, dj in {(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)}:
                    self._joker_grid[j + dj + 1][i + di + 1].discard(joker)
            self._minefield.cell_at(i, j).state = cell_state

    def _get_joker_neighbours(self, i: int, j: int):
        """Returns a 3x3 array of the joker neighbourhood"""
        arr = np.empty((3, 3), dtype=set)
        for di, dj in product((-1, 0, 1), repeat=2):
            arr[1 + dj][1 + di] = self._joker_grid[j + dj + 1][i + di + 1]
        return arr

    def _get_paired_up_joker_ids(self, i: int, j: int) -> Set[int]:
        """Returns the joker ids in the 3x3 box that are paired up."""
        joker_nbrs = self._get_joker_neighbours(i, j)

        ids = {}
        # Count the occurrence of the ids
        for set_ in joker_nbrs.flatten():
            for ind_ in set_:
                if ind_ not in ids:
                    ids[ind_] = 1
                else:
                    ids[ind_] += 1

        res = set()
        # Return ids that are included two times
        for key, count in ids.items():
            if count == 2:
                res.add(key)
        return res

    def _get_joker_indices_from_ids(self, ids: Set) -> Set[Tuple[int, int]]:
        inds = set()
        for id_ in ids:
            for j, row in enumerate(self._joker_grid, start=-1):
                for i, x in enumerate(row, start=-1):
                    if id_ in x:
                        inds.add((i, j))
        return inds

    def _handle_all_mines(self, i: int, j: int, cell_state: CellState, to_remove: Set[Tuple[int, int]]) -> bool:
        """Handles cases where all unopeded neighbour cells are mines

        Args:
            i (int): The x coordinate of the cell
            j (int): The y coordinate of the cell
            cell_state (CellState): The state of the cell at the given coordinates
            to_remove (set(tuple(int, int))): A set containing the modified indices
        Returns:
            (bool): True if any changes were made
        """
        mines = self._number_of_cells_near(i, j, CellState.MINE)
        near_unopened = self._number_of_cells_near(i, j, CellState.UNOPENED)
        joker_ids = self._get_paired_up_joker_ids(i, j)

        if cell_state.value - mines == near_unopened - len(joker_ids):
            inds = self._get_nbr_inds_of_types(i, j, CellState.UNOPENED)
            joker_inds = self._get_joker_indices_from_ids(joker_ids)
            inds = inds - joker_inds
            if not inds:
                return False
            self._mark_cells_at_inds_as_cellstate(inds, CellState.MINE)
            to_remove.add((i, j))
            return True
        return False

    def _handle_safe_cells(self, i: int, j: int, cell_state: CellState, to_remove: Set[Tuple[int, int]]) -> bool:
        """Handles cases where safe cells can be opened

        Args:
            i (int): The x coordinate of the cell
            j (int): The y coordinate of the cell
            cell_state (CellState): The state of the cell at the given coordinates
            to_remove (set(tuple(int, int))): A set containing the modified indices
        Returns:
            (bool): True if any changes were made
        """
        changed = False
        joker_ids = self._get_paired_up_joker_ids(i, j)
        mines_near = self._number_of_cells_near(i, j, CellState.MINE)
        if cell_state.value == mines_near + len(joker_ids):
            inds = self._get_nbr_inds_of_types(i, j, CellState.UNOPENED)
            joker_inds = self._get_joker_indices_from_ids(joker_ids)
            inds = inds - joker_inds
            for x, y in inds:
                self._move_to_cell(x, y, 0.0)
                self._open_cell(x, y, 0.0)
                self._mark_cells_at_inds_as_cellstate({(x, y)}, CellState.OPENED)
                changed = True
                if self._ms_ui.check_if_lost():
                    self._minefield.print_minefield()
                    self._print_joker_grid()
                    raise SystemExit("Lost")
                to_remove.add((i, j))
        return changed

    def _print_joker_grid(self):
        for row in self._joker_grid:
            print(" ".join("0" if not x else str(x) for x in row))

    def _mark_joker_cells(self, i: int, j: int, cell_state: CellState) -> bool:
        """Handles cases where there are 50/50 scenarios

        Args:
            i (int): The x coordinate of the cell
            j (int): The y coordinate of the cell
            cell_state (CellState): The state of the cell at the given coordinates
            to_remove (set(tuple(int, int))): A set containing the modified indices
        Returns:
            (bool): True if any changes were made
        """
        joker_nbrs = self._get_joker_neighbours(i, j)
        if np.count_nonzero(joker_nbrs) > 1:
            # Don't mark the same jokers twice
            return False
        unopened = self._get_nbr_inds_of_types(i, j, CellState.UNOPENED)
        mines = self._number_of_cells_near(i, j, CellState.MINE)
        if cell_state.value - mines + 1 == len(unopened) == 2:  # Case where a single mine can be in only two places.
            for i_, j_ in unopened:
                self._joker_grid[j_ + 1][i_ + 1].add(self._joker_id)
            self._joker_id += 1
            return True
        return False

    def _run(self):
        """Runs the solver"""

        start = time()
        while True:
            changed = False
            self._minefield.update_minefield()  # Because openings aren't checked for often enough, empty clicks occur
            if self._check_if_won() or self._minefield.mines_left() == 0:
                break
            for cell_state, values in self._minefield._numbered_cells.items():
                to_remove_indices = set()
                for i, j in values:
                    changed |= self._mark_joker_cells(i, j, cell_state)
                    changed |= self._handle_all_mines(i, j, cell_state, to_remove_indices)
                    changed |= self._handle_safe_cells(i, j, cell_state, to_remove_indices)

                for i, j in to_remove_indices:
                    self._minefield._numbered_cells[cell_state].remove((i, j))

            if not changed:
                self._select_the_safest_bet()
                if self._ms_ui.check_if_lost():
                    self._new_game()

        return time() - start


class SolverRandom(SolverBase):

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
