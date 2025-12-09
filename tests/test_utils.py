import unittest
from minesweeper_solver.utils import FlattenedGrid, get_gamestate


class TestFlattenedGrid(unittest.TestCase):
    def setUp(self):
        pass

    def test_flattened_grid_get_and_set_item(self):
        rows = 3
        columns = 2
        data = [[(j, i) for j in range(columns)] for i in range(rows)]
        grid = FlattenedGrid(data)

        self.assertEqual(grid[0, 0], (0, 0))
        self.assertEqual(grid[0, 1], (0, 1))
        self.assertEqual(grid[1, 2], (1, 2))

        self.assertRaises(IndexError, lambda: grid[2, 1])
        self.assertRaises(IndexError, lambda: grid[2, 2])
        with self.assertRaises(IndexError):
            grid[2, 1] = 3

        grid[1, 1] = (99, 99)

        self.assertEqual(grid[1, 1], (99, 99))
        self.assertEqual(1, str(grid).count("(99, 99)"))

        rows = 3
        columns = 8
        data = [[(j, i) for j in range(columns)] for i in range(rows)]
        grid = FlattenedGrid(data)
        self.assertEqual(grid[5, 2], (5, 2))


class TestGetState(unittest.TestCase):
    def test_get_gamestate(self):
        state1 = get_gamestate(1)
        state2 = get_gamestate(2)
        self.assertNotEqual(state1, state2)
