from functools import wraps
import os
import time
from typing import Callable


def get_gamestate(number: int) -> str:
    """Returns the corresponding saved gamestate form the resources -folder."""
    with open(os.path.join("data", f"game_state_{number}.txt"), "r") as f:
        return f.read()


def func_timer(func: Callable) -> Callable:
    """Can be used as a decorator to see the execution time of the function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"Time spent in {func.__qualname__}: {time.time()-start:.2f} s.")
        return res

    return wrapper
