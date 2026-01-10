from functools import wraps
import time
from typing import Callable


def func_timer(func: Callable) -> Callable:
    """Can be used as a decorator to see the execution time of the function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"Time spent in {func.__qualname__}: {time.time()-start:.2f} s.")
        return res

    return wrapper
