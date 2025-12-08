from functools import wraps
import time
from typing import Tuple


def func_timer(func):
    """Can be used as a decorator to see the execution time of the function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"Time spent in {func.__qualname__}: {time.time()-start:.2f} s.")
        return res

    return wrapper


class FlattenedGrid:
    """Flattened representation of a two dimensional grid."""

    def __init__(self, data):
        """
        Args:
            data: An N x M array of data of any consistent type.
        """
        self._height = len(data)
        self._width = len(data[0])

        self._data = []
        for row in list(data):
            self._data.extend(list(row))

    @property
    def data(self):
        return self._data

    def __setitem__(self, key, value):
        if key[0] >= self._width:
            raise IndexError(f"index x={key[0]} is out of range.")
        elif key[1] >= self._height:
            raise IndexError(f"index y={key[1]} is out of range.")
        ind = self._width * key[1] + key[0]
        self._data[ind] = value

    def __getitem__(self, key: Tuple[int, int]):
        if key[0] >= self._width:
            raise IndexError(f"index x={key[0]} is out of range.")
        elif key[1] >= self._height:
            raise IndexError(f"index y={key[1]} is out of range.")
        ind = self._width * key[1] + key[0]
        return self._data[ind]

    def size(self) -> Tuple[int, int]:
        """
        Returns:
           The size of the grid as the tuple (height, width)
        """
        return self._height, self._width

    def __repr__(self) -> str:
        return str(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        return self._data == other
