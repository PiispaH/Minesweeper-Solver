from typing import Any, Tuple, cast
from PIL.Image import Image
import pyautogui
import screeninfo


class MinesweeperUI:
    """Provides ways to read the state of the game and to interact with it.

    actually might be a lot faster to use the html if fetching that is quicker than taking images.
    """

    def __init__(self, width: int, height: int) -> None:
        self._latest_image: Image | Any = None
        self._width = width
        self._height = height
        self._screen_reduction = 1080 if len(screeninfo.get_monitors()) == 1 else 0

    @property
    def smiley(self):

        return 948, 1310 - self._screen_reduction

    def check_if_lost(self) -> bool:
        """Returns True when game is in lost state."""
        pixel1 = pyautogui.pixel(*self.smiley)
        pixel2 = pyautogui.pixel(self.smiley[0], self.smiley[1] + 4)
        return pixel1 == (170, 170, 0) and pixel2 == (0, 0, 0)

    def check_if_won(self) -> bool:
        """Returns True when game is in won state. (This might be too slow)"""
        pixel1 = pyautogui.pixel(*self.smiley)
        pixel2 = pyautogui.pixel(self.smiley[0] + 4, self.smiley[1] + 4)
        return pixel1 == (170, 170, 0) and pixel2 == (0, 0, 0)

    def get_upper_left_cell(self) -> Tuple[int, int]:
        """Gets the coordinates for the center of the upper left cell"""
        x = 476 + 16  # Hardcoded at the moment. Can't use with any other grid sizes
        y = 1358 + 16 - self._screen_reduction

        return x, y

    def click_at_pos(self, x_pos: int, y_pos: int, dur: float = 0.0):
        """Left clicks on the screen at a given point"""
        pyautogui.leftClick(x_pos, y_pos, duration=dur)

    def move_to(self, x: int, y: int, dur: float = 0.0):
        """Moves the cursor to the given point"""
        pyautogui.moveTo(x, y, duration=dur)

    def take_a_screenshot(self):
        """Takes a snapshot of the board state"""
        self._latest_image = pyautogui.screenshot(
            region=(476, 1358 - self._screen_reduction, 32 * self._width, 32 * self._height)
        )

    def get_pixel_color(self, i: int, j: int) -> Tuple[int, int, int]:
        """Returns the color of the center of the cell at indexes i, j"""
        return cast(Tuple[int, int, int], self._latest_image.getpixel((16 + i * 32, 16 + j * 32)))

    def get_rim_color(self, i: int, j: int) -> Tuple[int, int, int]:
        """Returns the color of the upper left corner of the cell at indexes i, j"""
        return cast(Tuple[int, int, int], self._latest_image.getpixel((2 + i * 32, 2 + j * 32)))

    def get_slight_center_offset_color(self, i: int, j: int) -> Tuple[int, int, int]:
        """Returns the color of the upper left corner of the cell at indexes i, j"""
        return cast(Tuple[int, int, int], self._latest_image.getpixel((16 + i * 32, 8 + j * 32)))

    def take_image(self):
        """Useful for debugging"""
        s = pyautogui.screenshot(region=(948, 1314 - self._screen_reduction, 16, 16))
        s.save("pic.png")
