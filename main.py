#!/usr/bin/env python3

import threading
from time import sleep, time
from typing import Callable
import numpy as np
from pynput import keyboard
from minesweeper_solver.minefield import Minefield
from minesweeper_solver.minesweeper_ui import MinesweeperUI
from minesweeper_solver.solver import SolverNaive, SolverRandom


def listener(func: Callable[[], None]):
    def on_release(key: keyboard.Key) -> bool:
        if key == keyboard.Key.esc:
            return False
        return True

    thread = threading.Thread(target=func, daemon=True)
    thread.start()

    with keyboard.Listener(on_release=on_release) as listener:  # type: ignore
        listener.join()


def main():

    rows = 16
    columns = 30
    mines = 99

    ms_ui = MinesweeperUI(columns, rows)
    ul = ms_ui.get_upper_left_cell()

    grid = Minefield(ul, rows, columns, mines, ms_ui)

    # solver = SolverRandom(grid, ms_ui)
    solver = SolverNaive(grid, ms_ui)

    start = time()
    game_time = solver.run()
    print("\n\n---------- You won!!! ----------\n\n")
    print(f"Overall time taken for solve: {time()-start:.2g} s.")
    print(f"Game time for solve: {game_time:.2g} s.")

    # This frees the listener
    keyboard.Controller().press(keyboard.Key.esc)
    keyboard.Controller().release(keyboard.Key.esc)


if __name__ == "__main__":
    listener(main)
