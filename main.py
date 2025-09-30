#!/usr/bin/env python3

from minesweeper_solver.solver import Solver
from minesweeper_solver.grid import Minefield
from minesweeper_solver.minesweeper_ui import MinesweeperUI


def main():
    rows = 16
    columns = 30

    ms_ui = MinesweeperUI(columns, rows)
    ul = ms_ui.get_upper_left_cell()

    grid = Minefield(ul, rows, columns, ms_ui)

    solver = Solver(grid, ms_ui)

    solver.open_cell(0, 0)
    print(grid.grid)


if __name__ == "__main__":
    main()
