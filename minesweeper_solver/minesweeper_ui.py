from typing import Any, Tuple, cast
from PIL.Image import Image
import pyautogui
import screeninfo


class MinesweeperUI:
    """Provides ways to read the state of the game and to interact with it."""

    def __init__(self, width: int, height: int) -> None:
        self._latest_image: Image | Any = None
        self._width = width
        self._height = height

    def check_game_status(self) -> bool:
        """Returns False when game is in lost state."""
        return pyautogui.pixel(948, 1311) == (0, 0, 0)

    def get_upper_left_cell(self) -> Tuple[int, int]:
        """Gets the coordinates for the center of the upper left cell"""
        x = 476 + 16  # Hardcoded at the moment. Can't use with any other grid sizes
        y = 1358 + 16

        if len(screeninfo.get_monitors()) == 1:
            y -= 1080

        return x, y

    def click_at_pos(self, x_pos: int, y_pos: int):
        """Left clicks on the screen at a given point"""
        pyautogui.leftClick(x_pos, y_pos, duration=0.1)

    def take_a_screenshot(self):
        """Takes a snapshot of the board state"""
        self._latest_image = pyautogui.screenshot(region=(476, 1358, 32 * self._width, 32 * self._height))

    def get_pixel_color(self, i: int, j: int) -> Tuple[int, int, int]:
        """Returns the oclor of the center of the cell at indexes i, j"""
        return cast(Tuple[int, int, int], self._latest_image.getpixel((16 + i * 32, 16 + j * 32)))

    def get_rim_color(self, i: int, j: int) -> Tuple[int, int, int]:
        """Returns the oclor of the upper left corner of the cell at indexes i, j"""
        return cast(Tuple[int, int, int], self._latest_image.getpixel((2 + i * 32, 2 + j * 32)))
