import unittest
from minesweeper import Action, CellState, Interaction
import numpy as np
from minesweeper_solver.DQL import Action, MinesweeperEnv


class TestMinesweeperEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = 9
        cls.w = 9
        cls.m = 10
        cls.env = MinesweeperEnv(cls.h, cls.w, cls.m, rnd_seed=42, flags_allowed=True)

    def tearDown(self):
        self.env.reset()

    def assert_arrays_equal(self, a, b):
        for j, row in enumerate(a):
            for i, value in enumerate(row):
                self.assertEqual(value, b[j][i])

    def test_env_constructed_properly(self):
        grid = [[9 for _ in range(self.w)] for _ in range(self.h)]
        self.assert_arrays_equal(self.env._grid, grid)
        self.assertEqual(self.env._n_mines, self.m)

    def test_opening_cells_and_flagging(self):
        before, after = self.env.take_action(Interaction(1, 0, Action.OPEN))
        self.assertTrue(before == CellState.UNOPENED)
        self.assertEqual(after, CellState.CELL_0)
        self.assertEqual(self.env._visible[0][1], CellState.CELL_0)

        self.assertEqual(self.env._visible[2][0], CellState.UNOPENED)
        before, after = self.env.take_action(Interaction(0, 2, Action.FLAG))
        self.assertEqual(before, CellState.UNOPENED)
        self.assertTrue(self.env._visible[2][0] == CellState.FLAG == after)

        before, after = self.env.take_action(Interaction(0, 2, Action.FLAG))
        self.assertEqual(before, CellState.FLAG)
        self.assertEqual(self.env._visible[2][0], CellState.UNOPENED)

    def test_reset(self):
        intial_state = self.env._visible
        self.env.take_action(Interaction(0, 0, Action.OPEN))
        altered_state = self.env._visible.copy()
        _ = self.env.reset()
        self.assertFalse(np.allclose(self.env._visible, altered_state))
        self.assertTrue(np.allclose(self.env._visible, intial_state))


class TestMinesweeperEnvGamePlay(unittest.TestCase):
    """Class for testing that the gameplay through the environment is consistent"""

    def test_step(self):
        self.env = MinesweeperEnv(9, 9, 10, rnd_seed=42)

        expected = [[[x for _ in range(9)] for _ in range(9)] for x in (0, 1)]
        encoded = self.env._encode_state()
        self.assertTrue(np.allclose(encoded, expected))

        _1 = 1 / 8
        _2 = 2 / 8

        next_state, reward, terminated, truncated = self.env.step(
            self.env.actions.index((Interaction(0, 0, Action.OPEN)))
        )
        self.assertEqual(reward, 0.5)
        self.assertTrue(not terminated)
        self.assertTrue(not truncated)
        expected = [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, _1, 0.0, 0.0, 0.0],
                [_1, _1, 0.0, 0.0, 0.0, _1, 0.0, 0.0, 0.0],
                [0.0, _1, 0.0, _1, _1, _2, 0.0, 0.0, 0.0],
                [0.0, _1, 0.0, _1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, _1, _2, _2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        ]
        self.assertIsNotNone(next_state)
        self.assertTrue(np.allclose(next_state, expected))  # type:ignore

        # Should not be a valid action
        self.assertRaises(
            ValueError, lambda: self.env.step(self.env.actions.index((Interaction(-1, -1, Action.NEW_GAME))))
        )

        next_state, reward, terminated, truncated = self.env.step(
            self.env.actions.index((Interaction(0, 8, Action.OPEN)))
        )
        self.assertEqual(reward, 0.5)
        self.assertTrue(not terminated)
        self.assertTrue(not truncated)
        expected = [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, _1, 0.0, 0.0, 0.0],
                [_1, _1, 0.0, 0.0, 0.0, _1, 0.0, 0.0, 0.0],
                [0.0, _1, 0.0, _1, _1, _2, 0.0, 0.0, 0.0],
                [0.0, _1, 0.0, _1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, _1, _2, _2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, _2, _2, _1, _2, 0.0, 0.0, 0.0, 0.0],
                [_1, _1, 0.0, 0.0, _1, _1, _2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _1, 0.0, 0.0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
            ],
        ]
        self.assertIsNotNone(next_state)
        self.assertTrue(np.allclose(next_state, expected))  # type:ignore

        next_state, reward, terminated, truncated = self.env.step(
            self.env.actions.index((Interaction(8, 0, Action.OPEN)))
        )
        self.assertEqual(reward, 0.5)
        self.assertTrue(not terminated)
        self.assertTrue(not truncated)
        expected = [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, _1, 0.0, _1, 0.0],
                [_1, _1, 0.0, 0.0, 0.0, _1, 0.0, _1, 0.0],
                [0.0, _1, 0.0, _1, _1, _2, 0.0, _1, 0.0],
                [0.0, _1, 0.0, _1, 0.0, 0.0, 0.0, _1, 0.0],
                [0.0, _1, _2, _2, 0.0, 0.0, _2, _1, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _2, 0.0, 0.0],
                [0.0, _2, _2, _1, _2, 0.0, _2, _1, _1],
                [_1, _1, 0.0, 0.0, _1, _1, _2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _1, 0.0, 0.0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1, 1, 1, 0, 0],
                [1, 0, 0, 0, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
            ],
        ]
        self.assertIsNotNone(next_state)
        self.assertTrue(np.allclose(next_state, expected))  # type:ignore

        # Opening a mine
        next_state, reward, terminated, truncated = self.env.step(
            self.env.actions.index((Interaction(0, 2, Action.OPEN)))
        )
        self.assertEqual(reward, -10.5)
        self.assertTrue(terminated)
        self.assertIsNone(next_state)
        self.assertTrue(not truncated)
