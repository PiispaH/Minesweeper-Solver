import unittest
from minesweeper_solver.DQL import MinesweepperEnv
from minesweeper_solver.minefield import CellState
import numpy as np


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

    def test_env_constructed_properly(self):
        self.assertEqual(self.env._current_state.size(), (self.h, self.w))
        self.assertEqual(self.env._current_state.data, [9 for _ in range(self.h * self.w)])
        self.assertEqual(self.env._mf.get_mines(), self.m)

    def test_opening_cells(self):
        self.env._open_cell(0, 0)
        self.env._update_current_env_state()
        self.assertEqual(self.env._current_state[0, 0], CellState("square open0").num())
        self.env.reset()

    def test_flagging_cells(self):
        first = self.env._mines_left()
        self.env._flag_cell(0, 0)
        self.assertEqual(self.env._mines_left() + 1, first)

        self.env._update_current_env_state()
        self.assertEqual(self.env._current_state[0, 0], CellState("square bombflagged").num())
        self.assertEqual(self.env._current_state[0, 0], self.env._get_cell_state(0, 0).num())

        self.env._flag_cell(0, 0)
        self.env._update_current_env_state()
        self.assertEqual(self.env._current_state[0, 0], CellState("square blank").num())
        self.env.reset()

    def test_reset(self):
        intial_state = self.env.encode_board()
        self.env._open_cell(0, 0)
        old_state = self.env.encode_board()
        obs = self.env.reset()
        self.assertFalse(np.allclose(obs, old_state))
        self.assertTrue(np.allclose(obs, intial_state))


class TestActionSpace(unittest.TestCase):
    def test_action_space(self):
        self.env = MinesweepperEnv(3, 3, 2, headless=True)
        # Impossible to create 3x3 grid, the website automatically makes it 3x8

        self.assertEqual(self.env._width, 8)
        self.assertEqual(self.env._height, 3)
        self.assertEqual(self.env._mines, 2)
        self.assertEqual(len(self.env.actions), self.env._width * self.env._height)  # * 2 if flags are allowed
