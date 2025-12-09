import unittest
import numpy as np
from minesweeper_solver.DQL import MinesweepperEnv, Action
from minesweeper_solver.minefield import CellState
from minesweeper_solver.utils import get_gamestate


class TestMinesweeperEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = 9
        cls.w = 9
        cls.m = 10
        cls.env = MinesweepperEnv(cls.h, cls.w, cls.m, headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.env.close_env()

    def tearDown(self):
        self.env.reset()

    def test_env_constructed_properly(self):
        grid = [[9 for _ in range(self.w)] for _ in range(self.h)]
        self.assertEqual(self.env._grid, grid)
        self.assertEqual(self.env._mines_left, self.m)
        self.assertEqual(self.env._n_mines, self.m)
        self.assertEqual(self.env._seconds, 0)

    def test_opening_cells(self):
        before, after = self.env.take_action(1, 0, Action.OPEN)
        self.assertEqual(before, CellState.UNOPENED)
        self.assertEqual(after, CellState.CELL_0)
        self.assertEqual(self.env._grid[0][1], CellState.CELL_0.num())

    def test_flagging_cells(self):
        first = self.env._mines_left
        self.assertEqual(self.env._grid[0][0], CellState.UNOPENED.num())
        before, after = self.env.take_action(0, 0, Action.FLAG)
        self.assertEqual(self.env._mines_left + 1, first)
        self.assertEqual(before, CellState.UNOPENED)
        self.assertTrue(self.env._grid[0][0] == CellState.FLAG.num() == after.num())

        before, after = self.env.take_action(0, 0, Action.FLAG)
        self.assertEqual(self.env._grid[0][0], CellState.UNOPENED.num())

    def test_reset(self):
        intial_state = self.env._grid
        self.env.take_action(0, 0, Action.OPEN)
        altered_state = self.env._grid
        _ = self.env.reset()
        self.assertFalse(np.allclose(self.env._grid, altered_state))
        self.assertTrue(np.allclose(self.env._grid, intial_state))


class TestMinesweeperEnv2(unittest.TestCase):

    def tearDown(self):
        self.env.close_env()

    def test_encode_state(self):
        self.env = MinesweepperEnv(state=get_gamestate(4), headless=True)
        state = self.env._encode_state(self.env._grid)

        expected = [[[x for _ in range(9)] for _ in range(9)] for x in (0, 1)]
        self.assertTrue(np.allclose(state, expected))

        self.env.take_action(0, 0, Action.OPEN)
        expected = [
            [
                [0, 0, 0, 1 / 8, 0, 0, 0, 0, 0],
                [0, 0, 0, 1 / 8, 0, 0, 0, 0, 0],
                [0, 0, 0, 1 / 8, 0, 0, 0, 0, 0],
                [0, 1 / 8, 1 / 8, 1 / 8, 0, 0, 0, 0, 0],
                [0, 2 / 8, 0, 0, 0, 0, 0, 0, 0],
                [1 / 8, 3 / 8, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        ]
        state = self.env._encode_state(self.env._grid)
        self.assertTrue(np.allclose(state, expected))


class TestActionSpace(unittest.TestCase):
    def test_action_space(self):
        self.env = MinesweepperEnv(3, 3, 2, headless=True)
        # Impossible to create 3x3 grid, the website automatically makes it 3x8

        self.assertEqual(self.env._width, 8)
        self.assertEqual(self.env._height, 3)
        self.assertEqual(self.env._n_mines, 2)
        self.assertEqual(len(self.env.actions), self.env._width * self.env._height)  # * 2 if flags are allowed
        self.env.close_env()
