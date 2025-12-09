import unittest
from minesweeper_solver.minefield import CellState, GameState, MineField
from minesweeper_solver.utils import get_gamestate


class TestMineField(unittest.TestCase):

    def tearDown(self):
        """Closes the webbrowser even if the test fails"""
        if hasattr(self, "minefield"):
            self.minefield.end_session()

    def test_importing_gamestate(self):
        self.minefield = MineField(True, state=get_gamestate(1))

        grid = [
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 2, 1, 2, 1, 1],
            [9, 9, 9, 9, 1, 0, 0, 0, 0],
            [9, 9, 9, 9, 1, 0, 0, 0, 0],
            [9, 9, 9, 9, 2, 1, 0, 1, 1],
            [9, 9, 9, 9, 9, 1, 0, 1, 9],
            [9, 9, 9, 9, 9, 1, 0, 1, 9],
        ]

        self.assertEqual(self.minefield.get_grid(), grid)

    def test_init(self):
        a = MineField(True, 1, 1, 1)
        a.end_session()
        a = MineField(True, state=get_gamestate(1))
        a.end_session()

        self.assertRaises(ValueError, lambda: MineField(True, width=1))
        self.assertRaises(ValueError, lambda: MineField(True, width=1, height=1, n_mines=1, state="1"))
        self.assertRaises(ValueError, lambda: MineField(True, width=1, height=1, n_mines=-1))

    def test_execute_action_and_get_info(self):
        expected_grid = [
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 2, 1, 2, 1, 1],
            [9, 9, 9, 9, 1, 0, 0, 0, 0],
            [9, 9, 9, 9, 1, 0, 0, 0, 0],
            [9, 9, 9, 9, 2, 1, 0, 1, 1],
            [9, 9, 9, 9, 9, 1, 0, 1, 9],
            [9, 9, 9, 9, 9, 1, 0, 1, 1],
        ]
        self.minefield = MineField(True, state=get_gamestate(1))
        cell_before, cell_after, grid, game_state, mines, seconds = self.minefield.execute_action_and_get_info(8, 8, 0)
        self.assertEqual(cell_before, CellState.UNOPENED)
        self.assertEqual(cell_after, CellState.CELL_1)
        self.assertEqual(grid, expected_grid)
        self.assertEqual(game_state, GameState.PLAYING)
        self.assertTrue(isinstance(mines, int))
        self.assertEqual(mines, 10)
        self.assertTrue(isinstance(seconds, int))
        self.assertEqual(seconds, 3)
