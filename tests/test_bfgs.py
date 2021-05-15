import numpy as np

from src.bfgs import bfgs


def test_quadratic() -> None:
    def grad(x: np.ndarray) -> np.ndarray:
        return 2 * x
    x0 = np.array([1, 1])
    epsilon = 1e-5
    res, _ = bfgs(grad, x0, epsilon)
    x = res['x']  # type: ignore

    assert np.linalg.norm(x) < 1e-3


def test_ravine() -> None:
    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([2 * x[0], 120 * x[1]])
    x0 = np.array([-40, 80])
    epsilon = 1e-5
    res, _ = bfgs(grad, x0, epsilon)
    x = res['x']  # type: ignore

    assert np.linalg.norm(x) < 1e-3
