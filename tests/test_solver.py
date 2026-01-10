import unittest
from unittest.mock import patch
import numpy as np
from minesweeper import Action, CellState, Interaction
from minesweeper_solver.solver import SolverNaive, SolverRandom

_U = CellState.UNOPENED.num()
_W = CellState.WALL.num()
_0 = CellState.CELL_0.num()
_1 = CellState.CELL_1.num()
_2 = CellState.CELL_2.num()
_3 = CellState.CELL_3.num()


class TestSolverRandom(unittest.TestCase):

    def test_solver_runs_and_stops(self):
        solver = SolverRandom(30, 16, 99, 1, rnd_seed=42)
        with patch("minesweeper_solver.solver.SolverRandom._handle_loss") as mock_method:
            solver.run()
            mock_method.assert_called_once()

        solver = SolverRandom(30, 16, 99, 2, rnd_seed=42)
        with patch("minesweeper_solver.solver.SolverRandom._handle_loss") as mock_method:
            solver.run()
            self.assertEqual(mock_method.call_count, 2)


class TestSolverNaive(unittest.TestCase):
    """Tests for SolverNaive with a class constructor"""

    def setUp(self):
        self.solver = SolverNaive(8, 6, 8, 1, rnd_seed=42)
        self.solver._visible_grid = np.array(
            [
                [_U, _U, _U, _U, _1, _0, _1, _U],
                [_U, _U, _2, _1, _1, _0, _2, _U],
                [_U, _U, _1, _0, _0, _0, _1, _U],
                [_U, _U, _2, _2, _2, _1, _2, _U],
                [_U, _U, _U, _U, _U, _U, _U, _U],
                [_U, _U, _U, _U, _U, _U, _U, _U],
            ]
        )

    def assertEqualArrays(self, a, b):
        """Checks that the all of the values in the given 2D arrays are equal."""
        for row1, row2 in zip(a, b):
            for value1, value2 in zip(row1, row2):
                self.assertEqual(value1, value2)

    def test_neighbours(self):
        nbs = self.solver._neighbours(0, 0)
        res = [
            [_W, _W, _W],
            [_W, _U, _U],
            [_W, _U, _U],
        ]
        self.assertEqualArrays(nbs, res)

        nbs = self.solver._neighbours(7, 0)
        res = [
            [_W, _W, _W],
            [_1, _U, _W],
            [_2, _U, _W],
        ]
        self.assertEqualArrays(nbs, res)

        nbs = self.solver._neighbours(7, 5)
        res = [
            [_U, _U, _W],
            [_U, _U, _W],
            [_W, _W, _W],
        ]
        self.assertEqualArrays(nbs, res)

        nbs = self.solver._neighbours(0, 5)
        res = [
            [_W, _U, _U],
            [_W, _U, _U],
            [_W, _W, _W],
        ]
        self.assertEqualArrays(nbs, res)

        nbs = self.solver._neighbours(4, 1)
        res = [
            [_U, _1, _0],
            [_1, _1, _0],
            [_0, _0, _0],
        ]
        self.assertEqualArrays(nbs, res)

    def test_get_nbr_inds_of_types(self):
        res = self.solver._get_nbr_inds_of_types(4, 3, CellState.CELL_0)
        expected = {(3, 2), (4, 2), (5, 2)}
        self.assertSetEqual(res, expected)

        res = self.solver._get_nbr_inds_of_types(4, 3, CellState.CELL_1)
        expected = {(5, 3)}
        self.assertSetEqual(res, expected)

        res = self.solver._get_nbr_inds_of_types(4, 3, CellState.CELL_2)
        expected = {(3, 3)}
        self.assertSetEqual(res, expected)

        res = self.solver._get_nbr_inds_of_types(4, 3, CellState.CELL_3)
        expected = set()
        self.assertSetEqual(res, expected)

        res = self.solver._get_nbr_inds_of_types(4, 3, CellState.UNOPENED)
        expected = {(3, 4), (4, 4), (5, 4)}
        self.assertSetEqual(res, expected)

        res = self.solver._get_nbr_inds_of_types(7, 0, CellState.CELL_2)
        expected = {(6, 1)}
        self.assertSetEqual(res, expected)

    def test_number_of_cells_near(self):
        res = self.solver._number_of_cells_near(4, 3, _0)
        expected = 3
        self.assertEqual(res, expected)

        res = self.solver._number_of_cells_near(4, 3, _1)
        expected = 1
        self.assertEqual(res, expected)

        res = self.solver._number_of_cells_near(4, 3, _2)
        expected = 1
        self.assertEqual(res, expected)

        res = self.solver._number_of_cells_near(4, 3, _3)
        expected = 0
        self.assertEqual(res, expected)

        res = self.solver._number_of_cells_near(4, 3, _U)
        expected = 3
        self.assertEqual(res, expected)

        res = self.solver._number_of_cells_near(7, 0, _2)
        expected = 1
        self.assertEqual(res, expected)

    def test_mine_ratio(self):
        self.assertRaises(ValueError, lambda: self.solver._neighbourhood_mine_ratio(0, 0))

        res = self.solver._neighbourhood_mine_ratio(4, 1)
        self.assertAlmostEqual(res, 1)

        res = self.solver._neighbourhood_mine_ratio(4, 3)
        self.assertAlmostEqual(res, 2 / 3)

        res = self.solver._neighbourhood_mine_ratio(3, 1)
        self.assertAlmostEqual(res, 1 / 2)

    def test_select_the_safest_bet(self):
        self.assertListEqual(self.solver._actions, list())
        with patch("minesweeper_solver.solver.random.choice") as mock_choice:
            mock_choice.return_value = (3, 0)
            self.solver._select_the_safest_bet({(3, 1), (4, 1)})
            mock_choice.assert_called_once_with([(2, 0), (3, 0)])
        self.assertListEqual(self.solver._actions, [Interaction(3, 0, Action.OPEN)])

    def test_informative_numbered_cells(self):
        _ = [
            [_U, _U, _U, _U, _1, _0, _1, _U],
            [_U, _U, _2, _1, _1, _0, _2, _U],
            [_U, _U, _1, _0, _0, _0, _1, _U],
            [_U, _U, _2, _2, _2, _1, _2, _U],
            [_U, _U, _U, _U, _U, _U, _U, _U],
            [_U, _U, _U, _U, _U, _U, _U, _U],
        ]
        res = self.solver._informative_numbered_cells()
        expected = (
            set(),
            {
                (4, 0),
                (4, 1),
            },
            {
                (6, 0),
                (2, 1),
                (3, 1),
                (6, 1),
                (2, 2),
                (6, 2),
                (2, 3),
                (3, 3),
                (4, 3),
                (5, 3),
                (6, 3),
            },
        )
        self.assertSetEqual(res[0], expected[0])
        self.assertSetEqual(res[1], expected[1])
        self.assertSetEqual(res[2], expected[2])


class TestSolverNaive2(unittest.TestCase):
    """Tests for SolverNaive without a class constructor"""

    def test_solver_runs_and_stops_when_win(self):

        def stop():
            solver._running = False

        solver = SolverNaive(30, 16, 99, 1, rnd_seed=52)
        with patch("minesweeper_solver.solver.SolverNaive._handle_win") as mock_handle_win:
            mock_handle_win.side_effect = stop
            with patch("minesweeper_solver.solver.random.choice") as mock_random:
                mock_random.side_effect = [
                    (6, 6),
                    (7, 15),
                    (6, 15),
                    (0, 0),
                    (2, 10),
                    (22, 6),
                    (23, 10),
                    (25, 7),
                    (26, 6),
                    (27, 6),
                    (28, 5),
                    (29, 5),
                ]

                solver.run()
            self.assertEqual(mock_handle_win.call_count, 1)
