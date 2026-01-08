import unittest
from minesweeper_solver.utils import get_gamestate


class TestGetState(unittest.TestCase):
    def test_get_gamestate(self):
        state1 = get_gamestate(1)
        state2 = get_gamestate(2)
        self.assertNotEqual(state1, state2)
