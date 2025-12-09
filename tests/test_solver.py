import unittest
from unittest.mock import patch
from minesweeper_solver.minefield import CellState
from minesweeper_solver.solver import SolverBase, SolverNaive, SolverRandom
from minesweeper_solver.utils import get_gamestate


class TestSolverBase(unittest.TestCase):
    """Tests for SolverBase with a class constructor"""

    @classmethod
    def setUpClass(cls):
        state = get_gamestate(1)
        cls.solver = SolverBase(state=state, headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.solver.quit()

    def test_raises_when_run(self):
        self.assertRaises(NotImplementedError, self.solver.run)


class TestSolverRandom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = SolverRandom(9, 9, 10, headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.solver.quit()

    def test_solver_runs_and_stops(self):
        with patch("minesweeper_solver.solver.SolverRandom._check_if_lost") as mock_method:
            self.solver.run(1)
            mock_method.assert_called_once()

        with patch("minesweeper_solver.solver.SolverRandom._check_if_lost") as mock_method:
            self.solver.run(2)
            mock_method.assert_called()
            self.assertEqual(mock_method.call_count, 2)


class TestSolverNaive(unittest.TestCase):
    """Tests for SolverNaive with a class constructor"""

    @classmethod
    def setUpClass(cls):
        state = get_gamestate(1)
        cls.solver = SolverNaive(state=state, headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.solver.quit()

    def assertEqualArrays(self, a, b):
        """Checks that the all of the values in the given 2D arrays are equal."""
        for row1, row2 in zip(a, b):
            for value1, value2 in zip(row1, row2):
                self.assertEqual(value1, value2)

    def test_neighbours(self):
        nbs = self.solver._neighbours(4, 3)
        res = [
            [CellState.UNOPENED.num(), CellState.UNOPENED.num(), CellState.UNOPENED.num()],
            [CellState.UNOPENED.num(), CellState.CELL_2.num(), CellState.CELL_1.num()],
            [CellState.UNOPENED.num(), CellState.CELL_1.num(), CellState.CELL_0.num()],
        ]
        self.assertEqualArrays(nbs, res)

        nbs = self.solver._neighbours(6, 8)
        res = [
            [CellState.CELL_1.num(), CellState.CELL_0.num(), CellState.CELL_1.num()],
            [CellState.CELL_1.num(), CellState.CELL_0.num(), CellState.CELL_1.num()],
            [CellState.WALL.num(), CellState.WALL.num(), CellState.WALL.num()],
        ]
        self.assertEqualArrays(nbs, res)

        nbs = self.solver._neighbours(1, 1)
        res = [
            [CellState.UNOPENED.num(), CellState.UNOPENED.num(), CellState.UNOPENED.num()],
            [CellState.UNOPENED.num(), CellState.UNOPENED.num(), CellState.UNOPENED.num()],
            [CellState.UNOPENED.num(), CellState.UNOPENED.num(), CellState.UNOPENED.num()],
        ]
        self.assertEqualArrays(nbs, res)

        nbs = self.solver._neighbours(0, 0)
        res = [
            [CellState.WALL.num(), CellState.WALL.num(), CellState.WALL.num()],
            [CellState.WALL.num(), CellState.UNOPENED.num(), CellState.UNOPENED.num()],
            [CellState.WALL.num(), CellState.UNOPENED.num(), CellState.UNOPENED.num()],
        ]
        self.assertEqualArrays(nbs, res)

    def test_get_nbr_inds_of_types(self):
        res = self.solver._get_nbr_inds_of_types(4, 3, CellState.CELL_1)
        expected = {(5, 3), (4, 4)}
        self.assertSetEqual(res, expected)

        res = self.solver._get_nbr_inds_of_types(4, 3, CellState.CELL_2)
        expected = set()
        self.assertSetEqual(res, expected)

        res = self.solver._get_nbr_inds_of_types(4, 3, CellState.UNOPENED)
        expected = {(3, 2), (4, 2), (5, 2), (3, 3), (3, 4)}
        self.assertSetEqual(res, expected)

        res = self.solver._get_nbr_inds_of_types(4, 3, CellState.CELL_0)
        expected = {(5, 4)}
        self.assertSetEqual(res, expected)

        res = self.solver._get_nbr_inds_of_types(8, 6, CellState.CELL_1)
        expected = {(7, 6), (7, 7)}
        self.assertSetEqual(res, expected)

    def test_number_of_cells_near(self):
        res = self.solver._number_of_cells_near(4, 3, CellState.CELL_1.num())
        self.assertEqual(res, 2)

        res = self.solver._number_of_cells_near(4, 3, CellState.CELL_2.num())
        self.assertEqual(res, 0)

        res = self.solver._number_of_cells_near(4, 3, CellState.UNOPENED.num())
        self.assertEqual(res, 5)

        res = self.solver._number_of_cells_near(4, 3, CellState.CELL_0.num())
        self.assertEqual(res, 1)

        res = self.solver._number_of_cells_near(8, 6, CellState.CELL_1.num())
        self.assertEqual(res, 2)

    def test_mine_ratio(self):
        self.assertRaises(ValueError, lambda: self.solver._neighbourhood_mine_ratio(1, 1))

        res = self.solver._neighbourhood_mine_ratio(8, 3)
        self.assertAlmostEqual(res, 1 / 2)

        res = self.solver._neighbourhood_mine_ratio(7, 3)
        self.assertAlmostEqual(res, 1 / 3)

        res = self.solver._neighbourhood_mine_ratio(4, 3)
        self.assertAlmostEqual(res, 2 / 5)

        # Import another board state for more tests
        state = get_gamestate(3)
        self.solver._mf._state = state
        self.solver._mf._import_gamestate()
        self.solver._current_grid = self.solver._mf.get_grid()

        res = self.solver._neighbourhood_mine_ratio(0, 1)
        self.assertAlmostEqual(res, 1 / 2)

        res = self.solver._neighbourhood_mine_ratio(1, 2)
        self.assertAlmostEqual(res, 1 / 2)

        res = self.solver._neighbourhood_mine_ratio(2, 3)
        self.assertAlmostEqual(res, 1 / 2)

        # Back to the original board state for the sake of other tests
        state = get_gamestate(1)
        self.solver._mf._state = state
        self.solver._mf._import_gamestate()
        self.solver._current_grid = self.solver._mf.get_grid()

    def test_select_the_safest_bet(self):
        res = self.solver._neighbourhood_mine_ratio(4, 3)
        self.assertAlmostEqual(res, 2 / 5)

    def test_informative_numbered_cells(self):
        res = self.solver._informative_numbered_cells()
        expected = (
            set(),
            {
                (5, 6),
                (7, 6),
                (8, 6),
            },
            {
                (4, 3),
                (5, 3),
                (6, 3),
                (7, 3),
                (8, 3),
                (4, 4),
                (4, 5),
                (4, 6),
                (5, 7),
                (5, 8),
                (7, 7),
                (7, 8),
            },
        )
        self.assertSetEqual(res[0], expected[0])
        self.assertSetEqual(res[1], expected[1])
        self.assertSetEqual(res[2], expected[2])


class TestSolverNaive2(unittest.TestCase):
    """Tests for SolverNaive without a class constructor"""

    def test_select_the_safest_bet(self):
        state = get_gamestate(2)
        solver = SolverNaive(state=state, headless=True)

        exp_inds = list({(2, 7), (2, 8), (3, 7), (4, 7)})

        _, _, inds = solver._informative_numbered_cells()

        with patch("minesweeper_solver.solver.choice") as mock_random, patch(
            "minesweeper_solver.solver.SolverBase._open_cell"
        ) as mock_open:
            mock_random.return_value = (3, 7)
            solver._select_the_safest_bet(inds)
            mock_random.assert_called_once_with(exp_inds)
            mock_open.assert_called_once_with(3, 7)

        solver.quit()

    def test_solver_runs_and_stops_when_win(self):

        state = get_gamestate(1)
        solver = SolverNaive(state=state, headless=True)

        with patch("minesweeper_solver.solver.choice") as mock_random:
            mock_random.return_value = (6, 2)
            time = solver.run(19)
            self.assertSetEqual(set(mock_random.call_args[0][0]), set([(6, 2), (5, 2), (4, 2)]))

        self.assertTrue(bool(time))
        solver.quit()
