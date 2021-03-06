import sys

from functools import update_wrapper
from typing import Any, Callable

import logging


def get_logger(name: str) -> logging.Logger:
    '''
    Utility function used in each module to
    initialize a logger

    Parameters
    ----------
    name : str
        Name of the logger (usually, the name of the module)

    Returns
    -------
    logging.Logger
        Logger
    '''
    
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

    
class CountCalls:
    '''
    Decorator class, that counts the number of calls to the decorated function
    '''
    
    def __init__(self, func: Callable) -> None:
        update_wrapper(self, func)
        self.func = func
        self.n_calls = 0
        
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.n_calls += 1
        return self.func(*args, **kwargs)
