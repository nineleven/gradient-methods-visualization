import sys

from functools import update_wrapper

import logging


def get_logger(name):
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

    
class CountCalls:
    
    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func
        self.n_calls = 0
        
    def __call__(self, *args, **kwargs):
        self.n_calls += 1
        return self.func(*args, **kwargs)
