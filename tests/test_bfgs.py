import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pytest

import numpy as np

from src.bfgs import bfgs


def test_quadratic():
    def grad(x):
        return 2 * x
    x0 = [1, 1]
    epsilon = 1e-5
    res = bfgs(grad, x0, epsilon)
    x = res['x']

    assert np.linalg.norm(x) < 1e-3

def test_ravine():
    def grad(x):
        print(x.shape)
        return [2 * x[0], 120 * x[1]]
    x0 = [-40, 80]
    epsilon = 1e-5
    res = bfgs(grad, x0, epsilon)
    x = res['x']

    assert np.linalg.norm(x) < 1e-3
